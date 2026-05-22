from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.core.task_registry import TaskInfo, TaskRegistry, TaskStatus, TaskType
from app.core.config import Settings
from app.domain.extraction import (
    ChunkingRequiredError,
    InitialContextRequiredError,
    InitialContext,
    PatchDraftPrerequisiteError,
    PatchRecord,
    apply_merge_patch,
    extract_initial_context_from_data_package,
    initialize_draft_from_initial_context,
    patch_draft_from_content_chunks,
)
from app.domain.extraction.review_resolution import (
    PatchReviewDecision,
    PatchReviewItem,
    resolve_patch_review_items,
)
from app.domain.extraction.sanitizers import sanitize_document_against_schema
from app.domain.extraction.sanitizers import normalize_review_draft
from app.domain.profiles import validate_document_against_profile
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService


class ExtractionService:
    def __init__(
        self,
        profile_service: ProfileService,
        settings: Settings,
        datasource_service: DataSourceService | None = None,
        ollama_client: OllamaClientWrapper | None = None,
        output_repository: ExtractionOutputRepository | None = None,
        task_registry: TaskRegistry | None = None,
    ):
        self.profile_service = profile_service
        self.settings = settings
        self.datasource_service = datasource_service
        self.ollama_client = ollama_client
        self.output_repository = output_repository
        self.task_registry = task_registry

    async def extract_initial_context(
        self,
        *,
        data_package_id: str,
        max_files_to_read: int = 12,
        max_chars_per_file: int = 3000,
    ) -> InitialContext:
        if self.datasource_service is None or self.ollama_client is None:
            raise RuntimeError(
                "ExtractionService requires datasource_service and ollama_client "
                "to run extraction agents."
            )

        data_package = self.datasource_service.get_data_package(data_package_id)
        initial_context = await extract_initial_context_from_data_package(
            data_package=data_package,
            model=self.ollama_client.agent_model,
            max_files_to_read=max_files_to_read,
            max_chars_per_file=max_chars_per_file,
        )
        if self.output_repository is not None:
            self.output_repository.save_initial_context(
                workflow_id=data_package_id,
                initial_context=initial_context,
            )
        return initial_context

    async def extract_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
    ) -> dict[str, Any]:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "and output_repository to run extraction agents."
            )

        data_package = self.datasource_service.get_data_package(data_package_id)
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
        try:
            initial_context = self.output_repository.load_initial_context(data_package_id)
        except FileNotFoundError as exc:
            raise InitialContextRequiredError(
                "Initial context output not found for workflow "
                f"'{data_package_id}'. Run /api/v1/extraction/initial-context "
                "before /api/v1/extraction/initial-draft."
            ) from exc

        initial_draft = await initialize_draft_from_initial_context(
            initial_context=initial_context,
            data_package=data_package,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            model=self.ollama_client.agent_model,
        )
        initial_draft = normalize_review_draft(
            initial_draft,
            dataset_id=str(initial_draft.get("id") or data_package_id),
        )
        self.output_repository.save_initial_draft(
            workflow_id=data_package_id,
            initial_draft=initial_draft,
        )
        self.output_repository.clear_patch_artifacts(workflow_id=data_package_id)
        if self.task_registry is not None:
            await self.task_registry.remove_task(
                self._patch_draft_task_name(data_package_id),
            )
        return initial_draft

    async def patch_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        num_chunks_per_turn: int,
    ) -> tuple[dict[str, Any], TaskStatus]:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
            or self.task_registry is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "output_repository, and task_registry to run extraction agents."
            )

        try:
            initial_draft = self.output_repository.load_initial_draft(data_package_id)
        except FileNotFoundError as exc:
            raise PatchDraftPrerequisiteError(
                "Patch extraction requires existing initial_context.json and "
                "initial_draft.json workflow outputs. Run /api/v1/extraction/"
                "initial-context and /api/v1/extraction/initial-draft first."
            ) from exc

        task_name = self._patch_draft_task_name(data_package_id)
        task_info: TaskInfo | None = self.task_registry.get_task_info(task_name)

        if (
            task_info is None
            or task_info.status in {TaskStatus.CANCELLED, TaskStatus.CRASHED}
            or self._is_stale_completed_patch_task(data_package_id, task_info)
        ):
            # Validate prerequisites before creating the background task so request
            # errors are still returned directly by this endpoint.
            try:
                self.output_repository.load_initial_context(data_package_id)
            except FileNotFoundError as exc:
                raise PatchDraftPrerequisiteError(
                    "Patch extraction requires existing initial_context.json and "
                    "initial_draft.json workflow outputs. Run /api/v1/extraction/"
                    "initial-context and /api/v1/extraction/initial-draft first."
                ) from exc

            self.datasource_service.get_data_package(data_package_id)
            self.profile_service.get_profile(profile_identifier)
            self.profile_service.load_json_schema(profile_identifier)

            content_chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
                data_package_id
            )
            if not content_chunks_by_file:
                raise ChunkingRequiredError(
                    "Patch extraction requires completed datasource chunking. "
                    "Run /api/v1/datasources/chunk until it returns completed chunks first."
                )

            current_draft = self._load_current_draft_or_initial(
                data_package_id=data_package_id,
                initial_draft=initial_draft,
            )
            self.output_repository.save_draft(
                workflow_id=data_package_id,
                draft=current_draft,
            )
            await self.task_registry.create_task(
                coro=self._run_patch_initial_draft(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    num_chunks_per_turn=num_chunks_per_turn,
                ),
                type=TaskType.WORKFLOW,
                name=task_name,
            )
            return self.output_repository.load_draft(data_package_id), TaskStatus.RUNNING

        current_draft = self._load_current_draft_or_initial(
            data_package_id=data_package_id,
            initial_draft=initial_draft,
        )
        return current_draft, task_info.status

    async def _run_patch_initial_draft(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        num_chunks_per_turn: int,
    ) -> dict[str, Any]:
        if (
            self.datasource_service is None
            or self.ollama_client is None
            or self.output_repository is None
        ):
            raise RuntimeError(
                "ExtractionService requires datasource_service, ollama_client, "
                "and output_repository to run extraction agents."
            )

        self.datasource_service.get_data_package(data_package_id)
        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)

        try:
            initial_context = self.output_repository.load_initial_context(data_package_id)
            initial_draft = self.output_repository.load_initial_draft(data_package_id)
        except FileNotFoundError as exc:
            raise PatchDraftPrerequisiteError(
                "Patch extraction requires existing initial_context.json and "
                "initial_draft.json workflow outputs. Run /api/v1/extraction/"
                "initial-context and /api/v1/extraction/initial-draft first."
            ) from exc

        content_chunks_by_file = self.datasource_service.get_completed_content_chunks_by_file(
            data_package_id
        )
        if not content_chunks_by_file:
            raise ChunkingRequiredError(
                "Patch extraction requires completed datasource chunking. "
                "Run /api/v1/datasources/chunk until it returns completed chunks first."
            )

        current_draft = self._load_current_draft_or_initial(
            data_package_id=data_package_id,
            initial_draft=initial_draft,
        )
        self.output_repository.save_draft(
            workflow_id=data_package_id,
            draft=current_draft,
        )

        protected_fields = self.output_repository.load_protected_fields(data_package_id)

        def load_protected_fields() -> list[str]:
            return self.output_repository.load_protected_fields(data_package_id) # type: ignore

        async def save_progress(
            draft: dict[str, Any],
            patch_record: PatchRecord,
            batch_no: int,
            total_batches: int,
        ) -> None:
            self._save_patch_progress(
                workflow_id=data_package_id,
                draft=draft,
                patch_record=patch_record,
                batch_no=batch_no,
                total_batches=total_batches,
            )

        result = await patch_draft_from_content_chunks(
            initial_context=initial_context,
            initial_draft=current_draft,
            content_chunks_by_file=content_chunks_by_file,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            model=self.ollama_client.agent_model,
            num_chunks_per_turn=num_chunks_per_turn,
            on_patch_processed=save_progress,
            protected_fields=protected_fields,
            protected_fields_loader=load_protected_fields,
            completed_patch_file_names=self.output_repository.load_completed_patch_file_names(
                data_package_id,
            ),
        )
        next_draft = normalize_review_draft(
            result.draft,
            dataset_id=str(result.draft.get("id") or data_package_id),
        )

        self.output_repository.save_draft(
            workflow_id=data_package_id,
            draft=next_draft,
        )

        return next_draft

    def _save_patch_progress(
        self,
        *,
        workflow_id: str,
        draft: dict[str, Any],
        patch_record: PatchRecord,
        batch_no: int,
        total_batches: int,
    ) -> None:
        if self.output_repository is None:
            raise RuntimeError("ExtractionService requires output_repository.")

        self.output_repository.save_draft(
            workflow_id=workflow_id,
            draft=normalize_review_draft(
                draft,
                dataset_id=str(draft.get("id") or workflow_id),
            ),
        )

        # Update task registry progress
        if self.task_registry is not None:
            task_name = self._patch_draft_task_name(workflow_id)
            self.task_registry.update_progress(
                task_name,
                {
                    "batch_no": batch_no,
                    "total_batches": total_batches,
                    "file_name": patch_record.file_name,
                    "accepted_fields": patch_record.accepted_fields,
                    "total_candidates": len(patch_record.candidates),
                    "validation_errors": patch_record.validation_errors or [],
                },
            )

        # Save the field-level candidates for revision agent support.
        if patch_record.candidates:
            self.output_repository.save_candidates(
                workflow_id=workflow_id,
                patch_file_name=patch_record.file_name,
                candidates=patch_record.candidates,
            )

        # Always save the merged patch as the canonical patch file.
        self.output_repository.save_patch(
            workflow_id=workflow_id,
            patch_file_name=patch_record.file_name,
            patch=patch_record.merged_patch,
        )

        if patch_record.accepted_fields:
            # Save the accepted merged patch (may differ from raw if revised).
            self.output_repository.save_accepted_patch(
                workflow_id=workflow_id,
                patch_file_name=patch_record.file_name,
                patch=patch_record.merged_patch,
            )

        if patch_record.quality_report is not None:
            self.output_repository.save_quality_report(
                workflow_id=workflow_id,
                patch_file_name=patch_record.file_name,
                quality_report=patch_record.quality_report,
            )

        if (
            patch_record.quality_report is not None
            and patch_record.quality_report.unmapped_facts
        ):
            self.output_repository.save_unmapped_facts(
                workflow_id=workflow_id,
                patch_file_name=patch_record.file_name,
                unmapped_facts=patch_record.quality_report.unmapped_facts,
            )

    def _load_current_draft_or_initial(
        self,
        *,
        data_package_id: str,
        initial_draft: dict[str, Any],
    ) -> dict[str, Any]:
        if self.output_repository is None:
            raise RuntimeError("ExtractionService requires output_repository.")

        try:
            return self.output_repository.load_draft(data_package_id)
        except FileNotFoundError:
            return copy.deepcopy(initial_draft)

    @staticmethod
    def _patch_draft_task_name(data_package_id: str) -> str:
        return f"patching:draft:{data_package_id}"

    def _is_stale_completed_patch_task(
        self,
        data_package_id: str,
        task_info: TaskInfo | None,
    ) -> bool:
        if (
            task_info is None
            or task_info.status != TaskStatus.COMPLETED
            or self.output_repository is None
        ):
            return False
        return not self.output_repository.load_completed_patch_file_names(data_package_id)

    async def get_existing_initial_context(
        self,
        *,
        data_package_id: str,
    ) -> InitialContext | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_initial_context(data_package_id)
        except FileNotFoundError:
            return None

    async def get_existing_initial_draft(
        self,
        *,
        data_package_id: str,
    ) -> dict[str, Any] | None:
        if self.output_repository is None:
            return None
        try:
            return self.output_repository.load_initial_draft(data_package_id)
        except FileNotFoundError:
            return None

    async def save_initial_draft(
        self,
        *,
        data_package_id: str,
        draft: dict[str, Any],
    ) -> None:
        if self.output_repository is None:
            raise RuntimeError(
                "ExtractionService requires output_repository to save drafts."
            )
        self.output_repository.save_initial_draft(
            workflow_id=data_package_id,
            initial_draft=normalize_review_draft(
                draft,
                dataset_id=str(draft.get("id") or data_package_id),
            ),
        )

    async def get_protected_fields(self, data_package_id: str) -> list[str]:
        if self.output_repository is None:
            return []
        return self.output_repository.load_protected_fields(data_package_id)

    async def set_protected_fields(
        self,
        *,
        data_package_id: str,
        protected_fields: list[str],
    ) -> None:
        if self.output_repository is None:
            return
        self.output_repository.save_protected_fields(
            workflow_id=data_package_id,
            protected_fields=protected_fields,
        )

    async def get_patch_review_state(self, data_package_id: str) -> dict[str, Any]:
        if self.output_repository is None:
            return {
                "resolved_item_ids": [],
                "unmapped_assignments": {},
                "resolution_notes": {},
                "resolved_at": {},
            }
        return self.output_repository.load_patch_review_state(data_package_id)

    async def save_patch_review_state(
        self,
        *,
        data_package_id: str,
        review_state: dict[str, Any],
    ) -> dict[str, Any]:
        if self.output_repository is None:
            return review_state
        normalized = {
            "resolved_item_ids": list(review_state.get("resolved_item_ids", [])),
            "unmapped_assignments": dict(review_state.get("unmapped_assignments", {})),
            "resolution_notes": dict(review_state.get("resolution_notes", {})),
            "resolved_at": dict(review_state.get("resolved_at", {})),
        }
        self.output_repository.save_patch_review_state(
            workflow_id=data_package_id,
            review_state=normalized,
        )
        return normalized

    async def resolve_patch_review_items(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        review_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.ollama_client is None or self.output_repository is None:
            raise RuntimeError(
                "ExtractionService requires ollama_client and output_repository "
                "to run the patch review resolution agent."
            )

        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
        try:
            current_draft = self.output_repository.load_draft(data_package_id)
        except FileNotFoundError:
            current_draft = self.output_repository.load_initial_draft(data_package_id)
        current_draft = sanitize_document_against_schema(
            document=current_draft,
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        current_draft = normalize_review_draft(
            current_draft,
            dataset_id=str(current_draft.get("id") or data_package_id),
        )

        existing_state = self.output_repository.load_patch_review_state(data_package_id)
        parsed_items = [PatchReviewItem.model_validate(item) for item in review_items]
        resolution = await resolve_patch_review_items(
            current_draft=current_draft,
            review_items=parsed_items,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            existing_review_state=existing_state,
            model=self.ollama_client.agent_model,
        )

        next_draft = resolution.final_draft or current_draft
        validation = validate_document_against_profile(
            document=next_draft,
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        if not validation.valid:
            return {
                "draft": current_draft,
                "review_state": existing_state,
                "resolved_count": 0,
                "unresolved_item_ids": [item.id for item in parsed_items],
                "validation_errors": [
                    f"{issue.path}: {issue.message}" for issue in validation.errors
                ],
                "resolution_decisions": [],
            }

        requested_item_ids = {item.id for item in parsed_items}
        decisions_by_id = self._review_decisions_by_id(
            resolution.item_decisions,
            requested_item_ids,
        )
        resolved_now = [
            item_id
            for item_id, decision in decisions_by_id.items()
            if decision.outcome in {"included", "already_present", "excluded"}
        ]
        resolved_ids = list(dict.fromkeys([
            *existing_state.get("resolved_item_ids", []),
            *resolved_now,
        ]))
        resolved_id_set = set(resolved_ids)
        remaining_unresolved_ids = [
            item.id for item in parsed_items if item.id not in resolved_id_set
        ]
        resolved_at = dict(existing_state.get("resolved_at", {}))
        timestamp = datetime.now(UTC).isoformat()
        for item_id in resolved_now:
            resolved_at.setdefault(item_id, timestamp)
        next_state = {
            "resolved_item_ids": resolved_ids,
            "unmapped_assignments": {
                **dict(existing_state.get("unmapped_assignments", {})),
                **resolution.unmapped_assignments,
            },
            "resolution_notes": {
                **dict(existing_state.get("resolution_notes", {})),
                **{
                    item_id: self._format_review_decision_note(decision)
                    for item_id, decision in decisions_by_id.items()
                    if decision.outcome in {"included", "already_present", "excluded"}
                },
            },
            "resolved_at": resolved_at,
        }
        self.output_repository.save_draft(
            workflow_id=data_package_id,
            draft=next_draft,
        )
        self.output_repository.save_initial_draft(
            workflow_id=data_package_id,
            initial_draft=next_draft,
        )
        self.output_repository.save_patch_review_state(
            workflow_id=data_package_id,
            review_state=next_state,
        )
        return {
            "draft": next_draft,
            "review_state": next_state,
            "resolved_count": len(resolved_now),
            "unresolved_item_ids": remaining_unresolved_ids,
            "validation_errors": [],
            "resolution_decisions": [
                decision.model_dump(mode="json")
                for decision in decisions_by_id.values()
            ],
        }

    @staticmethod
    def _review_decisions_by_id(
        decisions: list[PatchReviewDecision],
        requested_item_ids: set[str],
    ) -> dict[str, PatchReviewDecision]:
        result: dict[str, PatchReviewDecision] = {}
        for decision in decisions:
            if decision.id in requested_item_ids and decision.id not in result:
                result[decision.id] = decision
        return result

    @staticmethod
    def _format_review_decision_note(decision: PatchReviewDecision) -> str:
        target = f" Target: {decision.target_path}." if decision.target_path else ""
        return f"{decision.outcome}: {decision.note}{target}"

    async def get_patch_progress(
        self,
        data_package_id: str,
    ) -> tuple[TaskStatus, dict[str, Any] | None]:
        if self.task_registry is None:
            return TaskStatus.UNKNOWN, None
        task_name = self._patch_draft_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is None:
            return TaskStatus.UNKNOWN, None
        if self._is_stale_completed_patch_task(data_package_id, task_info):
            return TaskStatus.UNKNOWN, None
        return task_info.status, task_info.progress

    async def get_patch_artifacts(
        self,
        data_package_id: str,
    ) -> dict[str, Any]:
        if self.output_repository is None:
            return {"patches": [], "quality_reports": [], "unmapped_facts": []}
        return {
            "patches": self.output_repository.load_patch_files(data_package_id),
            "quality_reports": self.output_repository.load_patch_quality_reports(
                data_package_id,
            ),
            "unmapped_facts": self.output_repository.load_unmapped_facts(
                data_package_id,
            ),
        }

    async def get_patch_files(
        self,
        data_package_id: str,
    ) -> list[dict[str, Any]]:
        if self.output_repository is None:
            return []
        return self.output_repository.load_patch_files(data_package_id)

    async def get_patch_quality_reports(
        self,
        data_package_id: str,
    ) -> list[dict[str, Any]]:
        if self.output_repository is None:
            return []
        return self.output_repository.load_patch_quality_reports(data_package_id)

    async def get_unmapped_facts(
        self,
        data_package_id: str,
    ) -> list[dict[str, Any]]:
        if self.output_repository is None:
            return []
        return self.output_repository.load_unmapped_facts(data_package_id)
