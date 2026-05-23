from __future__ import annotations

import copy
import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

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


logger = logging.getLogger(__name__)
RESOLVED_REVIEW_OUTCOMES = {"included", "already_present", "excluded"}


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

        def record_token_usage(agent_name: str, usage: Any, operation_count: int = 1) -> None:
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name=agent_name,
                usage=usage,
                operation_count=operation_count,
            )

        initial_context = await extract_initial_context_from_data_package(
            data_package=data_package,
            model=self.ollama_client.agent_model,
            max_files_to_read=max_files_to_read,
            max_chars_per_file=max_chars_per_file,
            on_token_usage=record_token_usage,
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

        token_usage_totals: dict[str, dict[str, int]] = {}

        def record_token_usage(agent_name: str, usage: Any, operation_count: int = 1) -> None:
            self._record_token_usage(
                token_usage_totals,
                agent_name=agent_name,
                usage=usage,
                operation_count=operation_count,
            )

        initial_draft = await initialize_draft_from_initial_context(
            initial_context=initial_context,
            data_package=data_package,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            model=self.ollama_client.agent_model,
            on_token_usage=record_token_usage,
        )
        initial_draft = normalize_review_draft(
            initial_draft,
            dataset_id=str(initial_draft.get("id") or data_package_id),
        )
        self.output_repository.save_initial_draft(
            workflow_id=data_package_id,
            initial_draft=initial_draft,
        )
        self._replace_workflow_token_usage_agents(
            data_package_id=data_package_id,
            token_usage=token_usage_totals,
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
        num_chunks_per_turn: int = 1,
        auto_resolve: bool = False,
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
                    auto_resolve=auto_resolve,
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
        num_chunks_per_turn: int = 1,
        auto_resolve: bool = False,
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
        resolver_task: asyncio.Task[dict[str, Any] | None] | None = None
        token_usage_totals: dict[str, dict[str, int]] = {}

        def load_protected_fields() -> list[str]:
            return self.output_repository.load_protected_fields(data_package_id) # type: ignore

        def record_token_usage(agent_name: str, usage: Any, patch_count: int = 1) -> None:
            self._record_patch_token_usage(
                token_usage_totals,
                agent_name=agent_name,
                usage=usage,
                patch_count=patch_count,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name=agent_name,
                usage=usage,
                operation_count=1,
            )
            self._update_patch_token_usage_progress(
                data_package_id=data_package_id,
                token_usage=token_usage_totals,
            )

        def start_auto_resolve() -> None:
            nonlocal resolver_task
            if not auto_resolve:
                return
            if resolver_task is not None and not resolver_task.done():
                return
            resolver_task = asyncio.create_task(
                self._auto_resolve_review_items(
                    data_package_id=data_package_id,
                    profile_identifier=profile_identifier,
                    on_token_usage=record_token_usage,
                ),
                name=f"resolving:review:{data_package_id}",
            )

        async def finish_auto_resolve() -> None:
            nonlocal resolver_task
            if not auto_resolve:
                return
            if resolver_task is not None:
                await resolver_task
                resolver_task = None
            await self._auto_resolve_review_items(
                data_package_id=data_package_id,
                profile_identifier=profile_identifier,
                on_token_usage=record_token_usage,
            )

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
                token_usage=self._patch_token_usage_summary(token_usage_totals),
            )
            start_auto_resolve()

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
            on_token_usage=record_token_usage,
        )
        patching_draft = normalize_review_draft(
            result.draft,
            dataset_id=str(result.draft.get("id") or data_package_id),
        )
        next_draft = patching_draft
        if auto_resolve:
            latest_draft = self._load_latest_draft_for_resolution(
                data_package_id=data_package_id,
                fallback=current_draft,
            )
            if latest_draft != current_draft:
                patching_delta = self._json_merge_patch_diff(current_draft, patching_draft)
                next_draft = apply_merge_patch(latest_draft, patching_delta)
                validation = validate_document_against_profile(
                    document=next_draft,
                    json_schema=profile_json_schema,
                    target_class=profile_manifest.target_class,
                )
                if not validation.valid:
                    next_draft = patching_draft

        self.output_repository.save_draft(
            workflow_id=data_package_id,
            draft=next_draft,
        )

        await finish_auto_resolve()
        if auto_resolve:
            try:
                next_draft = self.output_repository.load_draft(data_package_id)
            except FileNotFoundError:
                next_draft = normalize_review_draft(
                    result.draft,
                    dataset_id=str(result.draft.get("id") or data_package_id),
                )

        return next_draft

    async def _auto_resolve_review_items(
        self,
        *,
        data_package_id: str,
        profile_identifier: str,
        on_token_usage: Callable[[str, Any, int], None] | None = None,
    ) -> dict[str, Any] | None:
        """Automatically resolve unresolved review items from saved patch artifacts."""
        resolution_log: list[str] = []

        def update_progress(
            result: dict[str, Any] | None = None,
            *,
            active: bool = True,
        ) -> None:
            self._update_patch_resolution_progress(
                data_package_id=data_package_id,
                resolution_log=resolution_log,
                resolution_result=result,
                resolution_active=active,
            )

        update_progress(active=True)

        if (
            self.ollama_client is None
            or self.output_repository is None
        ):
            self._record_resolution_log(
                resolution_log,
                data_package_id,
                "Auto-resolve skipped because the resolver dependencies are unavailable.",
            )
            update_progress(active=False)
            return None

        try:
            current_draft = self.output_repository.load_draft(data_package_id)
        except FileNotFoundError:
            current_draft = self.output_repository.load_initial_draft(data_package_id)

        profile_manifest = self.profile_service.get_profile(profile_identifier)
        profile_json_schema = self.profile_service.load_json_schema(profile_identifier)
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
        artifacts = await self.get_patch_artifacts(data_package_id)
        review_items = self._build_patch_review_items_from_artifacts(
            artifacts=artifacts,
            existing_state=existing_state,
        )

        if not review_items:
            self._record_resolution_log(
                resolution_log,
                data_package_id,
                "Auto-resolve found no unresolved patch review items.",
            )
            update_progress(active=False)
            return None

        parsed_items = [PatchReviewItem.model_validate(item) for item in review_items]
        result = await self._resolve_and_persist_review_items(
            data_package_id=data_package_id,
            current_draft=current_draft,
            parsed_items=parsed_items,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            existing_review_state=existing_state,
            resolution_log=resolution_log,
            progress_callback=update_progress,
            on_token_usage=on_token_usage,
        )
        update_progress(result, active=False)
        return result

    def _build_patch_review_items_from_artifacts(
        self,
        *,
        artifacts: dict[str, Any],
        existing_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resolved_ids = set(existing_state.get("resolved_item_ids", []))
        rating_by_patch_and_field: dict[str, dict[str, Any]] = {}

        for report in self._record_list(artifacts.get("quality_reports", [])):
            base_name = self._patch_artifact_base_name(str(report.get("file_name") or ""))
            report_content = self._record(report.get("content")) or {}
            for rating in self._record_list(report_content.get("candidate_ratings")):
                field_path = str(rating.get("field_path") or "")
                if field_path:
                    rating_by_patch_and_field[f"{base_name}:{field_path}"] = rating

        review_items: list[dict[str, Any]] = []
        for artifact in self._record_list(artifacts.get("patches", [])):
            if artifact.get("artifact_type") != "candidates":
                continue
            file_name = str(artifact.get("file_name") or "")
            base_name = self._patch_artifact_base_name(file_name)
            for candidate_index, candidate in enumerate(self._record_list(artifact.get("content"))):
                path = str(candidate.get("field_path") or "")
                if not path:
                    continue
                confidence = candidate.get("confidence")
                confidence_value = (
                    float(confidence)
                    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                    else None
                )
                rating = rating_by_patch_and_field.get(f"{base_name}:{path}", {})
                decision = str(rating.get("decision") or "")
                issues = [
                    self._format_quality_issue(issue)
                    for issue in self._record_list(rating.get("issues"))
                ]
                needs_review = (
                    confidence_value is None
                    or confidence_value < 0.8
                    or decision != "accept"
                    or len(issues) > 0
                )
                if not needs_review:
                    continue
                item_id = self._matched_review_item_id(base_name, path, candidate_index)
                if item_id in resolved_ids:
                    continue
                detail_parts = [
                    self._confidence_label(confidence_value),
                    f"Decision: {decision}" if decision else "",
                    f"{len(issues)} issue{'s' if len(issues) != 1 else ''}" if issues else "",
                    str(candidate.get("reasoning") or ""),
                ]
                review_items.append({
                    "id": item_id,
                    "kind": "matched",
                    "path": path,
                    "detail": " - ".join(part for part in detail_parts if part),
                    "issues": issues,
                    "evidence": self._text_list(candidate.get("source_evidence")),
                    "patch": candidate.get("patch") if isinstance(candidate.get("patch"), dict) else {},
                    "confidence": confidence_value,
                    "file_name": file_name,
                })

        for fact in self._record_list(artifacts.get("unmapped_facts", [])):
            item_id = self._unmapped_review_item_id(fact)
            if item_id in resolved_ids:
                continue
            source_hint = str(fact.get("source_hint") or "")
            review_items.append({
                "id": item_id,
                "kind": "unmapped",
                "path": "Unassigned",
                "detail": str(fact.get("fact") or fact.get("reason") or "Unmapped source fact"),
                "issues": [],
                "evidence": [source_hint] if source_hint else [],
                "fact": str(fact.get("fact") or ""),
                "reason": str(fact.get("reason") or ""),
                "file_name": str(fact.get("file_name") or ""),
            })

        return review_items

    async def _resolve_and_persist_review_items(
        self,
        *,
        data_package_id: str,
        current_draft: dict[str, Any],
        parsed_items: list[PatchReviewItem],
        profile_manifest: Any,
        profile_json_schema: dict[str, Any],
        existing_review_state: dict[str, Any],
        resolution_log: list[str],
        progress_callback: Callable[[], None] | None = None,
        on_token_usage: Callable[[str, Any, int], None] | None = None,
    ) -> dict[str, Any]:
        if self.ollama_client is None or self.output_repository is None:
            raise RuntimeError(
                "ExtractionService requires ollama_client and output_repository "
                "to run the patch review resolution agent."
            )

        kind_counts = {
            "matched": sum(1 for item in parsed_items if item.kind == "matched"),
            "unmapped": sum(1 for item in parsed_items if item.kind == "unmapped"),
        }
        self._record_resolution_log(
            resolution_log,
            data_package_id,
            "Starting review resolution for "
            f"{len(parsed_items)} items: {kind_counts['matched']} matched, "
            f"{kind_counts['unmapped']} unmapped.",
        )
        if progress_callback:
            progress_callback()

        resolution = await resolve_patch_review_items(
            current_draft=current_draft,
            review_items=parsed_items,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            existing_review_state=existing_review_state,
            model=self.ollama_client.agent_model,
            on_token_usage=on_token_usage,
        )
        self._record_resolution_log(
            resolution_log,
            data_package_id,
            f"Review agent returned {len(resolution.item_decisions)} decisions.",
        )
        if progress_callback:
            progress_callback()

        next_draft = resolution.final_draft or current_draft
        validation = validate_document_against_profile(
            document=next_draft,
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        )
        if not validation.valid:
            validation_errors = [
                f"{issue.path}: {issue.message}" for issue in validation.errors
            ]
            self._record_resolution_log(
                resolution_log,
                data_package_id,
                f"Final draft failed schema validation with {len(validation_errors)} errors; review state was not changed.",
            )
            if progress_callback:
                progress_callback()
            return {
                "draft": current_draft,
                "review_state": existing_review_state,
                "resolved_count": 0,
                "unresolved_item_ids": [item.id for item in parsed_items],
                "validation_errors": validation_errors,
                "resolution_decisions": [],
                "resolution_log": resolution_log,
            }

        self._record_resolution_log(
            resolution_log,
            data_package_id,
            "Final draft passed schema validation.",
        )

        latest_draft = self._load_latest_draft_for_resolution(
            data_package_id=data_package_id,
            fallback=current_draft,
        )
        if latest_draft != current_draft:
            draft_patch = self._json_merge_patch_diff(current_draft, next_draft)
            next_draft = apply_merge_patch(latest_draft, draft_patch)
            rebased_validation = validate_document_against_profile(
                document=next_draft,
                json_schema=profile_json_schema,
                target_class=profile_manifest.target_class,
            )
            if not rebased_validation.valid:
                validation_errors = [
                    f"{issue.path}: {issue.message}"
                    for issue in rebased_validation.errors
                ]
                self._record_resolution_log(
                    resolution_log,
                    data_package_id,
                    "Rebased resolver changes failed schema validation with "
                    f"{len(validation_errors)} errors; review state was not changed.",
                )
                if progress_callback:
                    progress_callback()
                return {
                    "draft": latest_draft,
                    "review_state": existing_review_state,
                    "resolved_count": 0,
                    "unresolved_item_ids": [item.id for item in parsed_items],
                    "validation_errors": validation_errors,
                    "resolution_decisions": [],
                    "resolution_log": resolution_log,
                }
            self._record_resolution_log(
                resolution_log,
                data_package_id,
                "Rebased resolver draft changes onto the latest patching draft.",
            )

        requested_item_ids = {item.id for item in parsed_items}
        decisions_by_id = self._review_decisions_by_id(
            resolution.item_decisions,
            requested_item_ids,
        )
        decisions_by_id, synthesized_count = self._normalize_review_decisions(
            decisions_by_id=decisions_by_id,
            parsed_items=parsed_items,
        )
        if synthesized_count:
            self._record_resolution_log(
                resolution_log,
                data_package_id,
                f"Converted {synthesized_count} missing or unresolved decisions to excluded.",
            )

        outcome_counts = {
            outcome: sum(
                1 for decision in decisions_by_id.values()
                if decision.outcome == outcome
            )
            for outcome in sorted(RESOLVED_REVIEW_OUTCOMES)
        }
        self._record_resolution_log(
            resolution_log,
            data_package_id,
            "Decision counts: "
            f"{outcome_counts.get('included', 0)} included, "
            f"{outcome_counts.get('already_present', 0)} already present, "
            f"{outcome_counts.get('excluded', 0)} excluded.",
        )

        resolved_now = list(decisions_by_id)
        resolved_ids = list(dict.fromkeys([
            *existing_review_state.get("resolved_item_ids", []),
            *resolved_now,
        ]))
        resolved_id_set = set(resolved_ids)
        remaining_unresolved_ids = [
            item.id for item in parsed_items if item.id not in resolved_id_set
        ]
        resolved_at = dict(existing_review_state.get("resolved_at", {}))
        timestamp = datetime.now(UTC).isoformat()
        for item_id in resolved_now:
            resolved_at.setdefault(item_id, timestamp)
        next_state = {
            "resolved_item_ids": resolved_ids,
            "unmapped_assignments": {
                **dict(existing_review_state.get("unmapped_assignments", {})),
                **resolution.unmapped_assignments,
            },
            "resolution_notes": {
                **dict(existing_review_state.get("resolution_notes", {})),
                **{
                    item_id: self._format_review_decision_note(decision)
                    for item_id, decision in decisions_by_id.items()
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
        self._record_resolution_log(
            resolution_log,
            data_package_id,
            f"Saved review state with {len(resolved_ids)} total resolved item IDs; "
            f"{len(remaining_unresolved_ids)} submitted items remain unresolved.",
        )
        if progress_callback:
            progress_callback()

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
            "resolution_log": resolution_log,
        }

    def _save_patch_progress(
        self,
        *,
        workflow_id: str,
        draft: dict[str, Any],
        patch_record: PatchRecord,
        batch_no: int,
        total_batches: int,
        token_usage: dict[str, Any] | None = None,
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
            task_info = self.task_registry.get_task_info(task_name)
            progress = dict(task_info.progress or {}) if task_info else {}
            progress.update(
                {
                    "batch_no": batch_no,
                    "total_batches": total_batches,
                    "file_name": patch_record.file_name,
                    "accepted_fields": patch_record.accepted_fields,
                    "total_candidates": len(patch_record.candidates),
                    "validation_errors": patch_record.validation_errors or [],
                }
            )
            if token_usage and token_usage.get("agents"):
                progress["token_usage"] = token_usage
            self.task_registry.update_progress(
                task_name,
                progress,
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
        token_usage_totals: dict[str, dict[str, int]] = {}

        def record_token_usage(agent_name: str, usage: Any, operation_count: int = 1) -> None:
            self._record_token_usage(
                token_usage_totals,
                agent_name=agent_name,
                usage=usage,
                operation_count=operation_count,
            )
            self._record_workflow_token_usage(
                data_package_id=data_package_id,
                agent_name=agent_name,
                usage=usage,
                operation_count=1,
            )

        result = await self._resolve_and_persist_review_items(
            data_package_id=data_package_id,
            current_draft=current_draft,
            parsed_items=parsed_items,
            profile_manifest=profile_manifest,
            profile_json_schema=profile_json_schema,
            existing_review_state=existing_state,
            resolution_log=[],
            on_token_usage=record_token_usage,
        )
        token_usage = self._token_usage_summary(token_usage_totals)
        if token_usage.get("agents"):
            result["token_usage"] = token_usage
        return result

    async def get_token_usage(self, data_package_id: str) -> dict[str, Any]:
        if self.output_repository is None:
            return {"agents": {}}
        return self._token_usage_summary(
            self.output_repository.load_token_usage(data_package_id)
        )

    @classmethod
    def _record_token_usage(
        cls,
        totals: dict[str, dict[str, int]],
        *,
        agent_name: str,
        usage: Any,
        operation_count: int = 1,
        patch_count: int | None = None,
    ) -> None:
        input_tokens = cls._usage_int(usage, "input_tokens")
        output_tokens = cls._usage_int(usage, "output_tokens")
        total_tokens = cls._usage_int(usage, "total_tokens") or (
            input_tokens + output_tokens
        )
        requests = cls._usage_int(usage, "requests")
        if not any((input_tokens, output_tokens, total_tokens, requests)):
            return

        entry = totals.setdefault(
            agent_name,
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "requests": 0,
                "operation_count": 0,
                "patch_count": 0,
            },
        )
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["total_tokens"] += total_tokens
        entry["requests"] += requests
        entry["operation_count"] += max(1, operation_count)
        if patch_count is not None:
            entry["patch_count"] += max(1, patch_count)

    @classmethod
    def _record_patch_token_usage(
        cls,
        totals: dict[str, dict[str, int]],
        *,
        agent_name: str,
        usage: Any,
        patch_count: int = 1,
    ) -> None:
        cls._record_token_usage(
            totals,
            agent_name=agent_name,
            usage=usage,
            operation_count=1,
            patch_count=patch_count,
        )

    def _record_workflow_token_usage(
        self,
        *,
        data_package_id: str,
        agent_name: str,
        usage: Any,
        operation_count: int = 1,
    ) -> None:
        if self.output_repository is None:
            return
        totals = self.output_repository.load_token_usage(data_package_id)
        self._record_token_usage(
            totals,
            agent_name=agent_name,
            usage=usage,
            operation_count=operation_count,
        )
        self.output_repository.save_token_usage(
            workflow_id=data_package_id,
            token_usage=totals,
        )

    def _replace_workflow_token_usage_agents(
        self,
        *,
        data_package_id: str,
        token_usage: dict[str, dict[str, int]],
    ) -> None:
        if self.output_repository is None or not token_usage:
            return
        totals = self.output_repository.load_token_usage(data_package_id)
        for agent_name, values in token_usage.items():
            totals[agent_name] = dict(values)
        self.output_repository.save_token_usage(
            workflow_id=data_package_id,
            token_usage=totals,
        )

    @staticmethod
    def _usage_int(usage: Any, field_name: str) -> int:
        value = getattr(usage, field_name, 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _token_usage_summary(
        cls,
        totals: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        agents = {
            agent_name: cls._token_usage_entry_summary(values)
            for agent_name, values in totals.items()
            if any(
                values.get(key, 0)
                for key in ("input_tokens", "output_tokens", "total_tokens", "requests")
            )
        }
        if not agents:
            return {"agents": {}}

        combined_values = {
            "input_tokens": sum(item["input_tokens"] for item in agents.values()),
            "output_tokens": sum(item["output_tokens"] for item in agents.values()),
            "total_tokens": sum(item["total_tokens"] for item in agents.values()),
            "requests": sum(item["requests"] for item in agents.values()),
            "operation_count": sum(item["operation_count"] for item in agents.values()),
        }
        return {
            "agents": agents,
            "combined": cls._token_usage_entry_summary(combined_values),
        }

    @classmethod
    def _patch_token_usage_summary(
        cls,
        totals: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        agents = {
            agent_name: cls._patch_token_usage_entry_summary(values)
            for agent_name, values in totals.items()
            if any(
                values.get(key, 0)
                for key in ("input_tokens", "output_tokens", "total_tokens", "requests")
            )
        }
        if not agents:
            return {"agents": {}}

        combined_values = {
            "input_tokens": sum(item["input_tokens"] for item in agents.values()),
            "output_tokens": sum(item["output_tokens"] for item in agents.values()),
            "total_tokens": sum(item["total_tokens"] for item in agents.values()),
            "requests": sum(item["requests"] for item in agents.values()),
            "patch_count": max(item["patch_count"] for item in agents.values()),
        }
        return {
            "agents": agents,
            "combined": cls._patch_token_usage_entry_summary(combined_values),
        }

    @classmethod
    def _token_usage_entry_summary(
        cls,
        values: dict[str, int],
    ) -> dict[str, int | float]:
        return cls._token_usage_entry_summary_for_count(
            values,
            count_key="operation_count",
            average_suffix="operation",
        )

    @staticmethod
    def _patch_token_usage_entry_summary(values: dict[str, int]) -> dict[str, int | float]:
        return ExtractionService._token_usage_entry_summary_for_count(
            values,
            count_key="patch_count",
            average_suffix="patch",
        )

    @staticmethod
    def _token_usage_entry_summary_for_count(
        values: dict[str, int],
        *,
        count_key: str,
        average_suffix: str,
    ) -> dict[str, int | float]:
        operation_count = max(1, int(values.get("operation_count", 0)))
        patch_count = max(1, int(values.get("patch_count", 0)))
        input_tokens = int(values.get("input_tokens", 0))
        output_tokens = int(values.get("output_tokens", 0))
        total_tokens = int(values.get("total_tokens", 0))
        denominator = max(1, int(values.get(count_key, 0)))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "requests": int(values.get("requests", 0)),
            "operation_count": operation_count,
            "patch_count": patch_count,
            f"average_input_tokens_per_{average_suffix}": round(input_tokens / denominator, 2),
            f"average_output_tokens_per_{average_suffix}": round(output_tokens / denominator, 2),
            f"average_total_tokens_per_{average_suffix}": round(total_tokens / denominator, 2),
        }

    def _update_patch_token_usage_progress(
        self,
        *,
        data_package_id: str,
        token_usage: dict[str, dict[str, int]],
    ) -> None:
        if self.task_registry is None:
            return
        task_name = self._patch_draft_task_name(data_package_id)
        task_info = self.task_registry.get_task_info(task_name)
        if task_info is None:
            return
        progress = dict(task_info.progress or {})
        summary = self._patch_token_usage_summary(token_usage)
        if summary.get("agents"):
            progress["token_usage"] = summary
            self.task_registry.update_progress(task_name, progress)

    @staticmethod
    def _record(value: Any) -> dict[str, Any] | None:
        return value if isinstance(value, dict) else None

    @classmethod
    def _record_list(cls, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [
            item
            for item in (cls._record(item) for item in value)
            if item is not None
        ]

    @staticmethod
    def _text_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [text for text in (str(item) for item in value) if text]

    @staticmethod
    def _confidence_label(confidence: float | None) -> str:
        if confidence is None:
            return "unknown confidence"
        return f"{round(confidence * 100)}% confidence"

    @staticmethod
    def _patch_artifact_base_name(file_name: str) -> str:
        suffixes = [
            ".quality_report.json",
            ".candidates.json",
            ".accepted.json",
            ".raw.json",
            ".unmapped_facts.json",
        ]
        for suffix in suffixes:
            if file_name.endswith(suffix):
                return f"{file_name[:-len(suffix)]}.json"
        return file_name

    @staticmethod
    def _matched_review_item_id(base_name: str, path: str, index: int) -> str:
        return f"matched:{base_name}:{path}:{index}"

    @staticmethod
    def _unmapped_review_item_id(fact: dict[str, Any]) -> str:
        key = "|".join(
            str(fact.get(field) or "")
            for field in ("file_name", "fact", "reason", "source_hint")
        )
        return f"unmapped:{key}"

    @staticmethod
    def _format_quality_issue(issue: dict[str, Any]) -> str:
        parts = [
            str(issue.get("issue_type") or ""),
            f"({issue.get('severity')})" if issue.get("severity") else "",
            str(issue.get("explanation") or ""),
            (
                f"Suggested field: {issue.get('suggested_target_path')}"
                if issue.get("suggested_target_path")
                else ""
            ),
        ]
        return " ".join(part for part in parts if part)

    def _load_latest_draft_for_resolution(
        self,
        *,
        data_package_id: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if self.output_repository is None:
            return fallback
        try:
            return self.output_repository.load_draft(data_package_id)
        except FileNotFoundError:
            return fallback

    @classmethod
    def _json_merge_patch_diff(cls, before: Any, after: Any) -> dict[str, Any]:
        if before == after:
            return {}
        if isinstance(before, dict) and isinstance(after, dict):
            patch: dict[str, Any] = {}
            for key in before.keys() - after.keys():
                patch[key] = None
            for key, after_value in after.items():
                before_value = before.get(key)
                if key not in before or before_value != after_value:
                    if isinstance(before_value, dict) and isinstance(after_value, dict):
                        nested_patch = cls._json_merge_patch_diff(
                            before_value,
                            after_value,
                        )
                        if nested_patch:
                            patch[key] = nested_patch
                    else:
                        patch[key] = copy.deepcopy(after_value)
            return patch
        return copy.deepcopy(after) if isinstance(after, dict) else {}

    @staticmethod
    def _normalize_review_decisions(
        *,
        decisions_by_id: dict[str, PatchReviewDecision],
        parsed_items: list[PatchReviewItem],
    ) -> tuple[dict[str, PatchReviewDecision], int]:
        normalized: dict[str, PatchReviewDecision] = {}
        synthesized_count = 0
        for item in parsed_items:
            decision = decisions_by_id.get(item.id)
            target_path = None if item.path == "Unassigned" else item.path
            if decision is None:
                synthesized_count += 1
                normalized[item.id] = PatchReviewDecision(
                    id=item.id,
                    outcome="excluded",
                    note=(
                        "The resolution agent did not return a decision for this "
                        "item, so it was excluded to complete delegated review "
                        "without applying unsupported metadata."
                    ),
                    target_path=target_path,
                )
            elif decision.outcome == "unresolved":
                synthesized_count += 1
                normalized[item.id] = PatchReviewDecision(
                    id=item.id,
                    outcome="excluded",
                    note=(
                        "The resolution agent marked this item unresolved, so it "
                        "was excluded to complete delegated review. Agent note: "
                        f"{decision.note}"
                    ),
                    target_path=decision.target_path or target_path,
                )
            else:
                normalized[item.id] = decision
        return normalized, synthesized_count

    @staticmethod
    def _record_resolution_log(
        resolution_log: list[str],
        data_package_id: str,
        message: str,
    ) -> None:
        logger.info(
            "Patch review resolution for %s - %s",
            data_package_id,
            message,
        )
        resolution_log.append(message)

    def _update_patch_resolution_progress(
        self,
        *,
        data_package_id: str,
        resolution_log: list[str],
        resolution_result: dict[str, Any] | None = None,
        resolution_active: bool | None = None,
    ) -> None:
        if self.task_registry is None:
            return
        task_info = self.task_registry.get_task_info(
            self._patch_draft_task_name(data_package_id),
        )
        if task_info is None:
            return
        progress = dict(task_info.progress or {})
        progress["resolution_log"] = list(resolution_log)
        if resolution_active is not None:
            progress["resolution_active"] = resolution_active
        if resolution_result is not None:
            progress["resolution_resolved_count"] = resolution_result.get(
                "resolved_count",
                0,
            )
            progress["resolution_unresolved_item_ids"] = list(
                resolution_result.get("unresolved_item_ids", []),
            )
        self.task_registry.update_progress(
            self._patch_draft_task_name(data_package_id),
            progress,
        )

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
