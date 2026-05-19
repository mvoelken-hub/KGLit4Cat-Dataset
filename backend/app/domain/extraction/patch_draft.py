from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.datasources import ContentChunk
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import FieldPatchResult, InitialContext, PatchCandidate
from app.domain.extraction.patch_quality import (
    PatchQualityReport,
    review_patch_semantic_quality,
    validate_candidate_draft,
)
from app.domain.extraction.schema_utils import FieldInfo, get_top_level_fields
from app.domain.profiles import ProfileManifest


# ---------------------------------------------------------------------------
# URI policy for patch extraction agents
# ---------------------------------------------------------------------------
URI_POLICY_INSTRUCTIONS = (
    "URI and identifier policy: Do NOT fabricate or invent URIs for id, "
    "rdf_type, has_quantity_type, unit, or similar vocabulary-controlled "
    "fields. Instead, use descriptive human-readable labels for titles and "
    "descriptions (e.g. 'NMR FID Acquisition') and stable local identifiers "
    "for id fields (e.g. 'activity-1', 'entity-hms-q11-p-10', "
    "'agent-bruker-biospin'). A later vocabulary resolution step will "
    "replace these labels and local IDs with proper URIs from curated "
    "vocabularies. Never use https://w3id.org/ or other namespace URIs "
    "unless they already appear in the current draft or InitialContext."
)

# ---------------------------------------------------------------------------
# Patch extraction agent instructions
# ---------------------------------------------------------------------------
PATCH_DRAFT_INSTRUCTIONS = (
    "You extract structured metadata from document chunks and return "
    "field-level patch candidates for a current DCAT dataset draft. "
    "For each top-level field of the Dataset schema, decide whether the "
    "current chunk batch contains information relevant to that field. "
    "If it does, produce a PatchCandidate with the field name, a merge "
    "patch scoped to just that field, your confidence (0.0–1.0), a brief "
    "reasoning explanation, and source evidence from the chunks. "
    "If a field has no relevant information in this chunk, omit it from "
    "the candidates list. "
    "Do not rewrite unchanged fields. Avoid duplicate information. "
    "Focus on data-generating activity, device or instrument details, "
    "entities, distributions, file roles, and qualitative or quantitative "
    "attributes extracted from the chunks. "
    "For objects in lists, include an id field with a stable local "
    "identifier. Use ids from the current draft to update existing "
    "objects. Do not use null to delete existing values; null values "
    "are ignored by the merge function. "
    + URI_POLICY_INSTRUCTIONS
)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class MergePatch(BaseModel):
    """Backward-compatible single merge patch (used by legacy code paths)."""

    patch: dict[str, Any] = Field(default_factory=dict)


class PatchDraftPrerequisiteError(Exception):
    """Raised when patch extraction is requested before prerequisites exist."""


class ChunkingRequiredError(Exception):
    """Raised when patch extraction is requested before chunking completed."""


@dataclass
class PatchDraftDeps:
    initial_context: InitialContext
    current_draft: dict[str, Any]
    chunk_batch: list[ContentChunk]
    batch_no: int
    total_batches: int
    document_file_path: str
    profile_manifest: ProfileManifest
    profile_json_schema: dict[str, Any]
    top_level_fields: list[FieldInfo]


@dataclass
class PatchRecord:
    file_name: str
    candidates: list[PatchCandidate]
    accepted_fields: list[str]
    merged_patch: dict[str, Any]
    quality_report: PatchQualityReport | None = None
    validation_errors: list[str] | None = None


@dataclass
class PatchDraftResult:
    draft: dict[str, Any]
    patches: list[PatchRecord]


PatchProgressCallback = Callable[[dict[str, Any], PatchRecord], Awaitable[None]]


# ---------------------------------------------------------------------------
# Patch extraction agent
# ---------------------------------------------------------------------------

def create_patch_draft_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[PatchDraftDeps, FieldPatchResult]:
    agent = Agent(
        model,
        deps_type=PatchDraftDeps,
        output_type=prompted_json_output(
            FieldPatchResult,
            name="FieldPatchResult",
            description=(
                "List of field-level patch candidates for a DCAT dataset draft."
            ),
        ),
        instructions=PATCH_DRAFT_INSTRUCTIONS,
        model_settings={
            "temperature": 0.0,
            "seed": 42,
        },
        output_retries=output_retries,
    )

    @agent.instructions
    def add_patch_extraction_context(ctx: RunContext[PatchDraftDeps]) -> str:
        chunk_texts = ""
        for index, chunk in enumerate(ctx.deps.chunk_batch, start=1):
            chunk_texts += (
                f"Chunk {index}/{len(ctx.deps.chunk_batch)}:\n"
                f"{_chunk_json_for_prompt(chunk)}\n\n"
            )

        field_descriptions = "\n".join(
            f"- {field.name}: "
            f"{'array of objects' if field.is_array and field.is_object else 'array' if field.is_array else 'object' if field.is_object else 'scalar'}"
            f"{', nullable' if field.is_nullable else ''}"
            for field in ctx.deps.top_level_fields
        )

        return (
            f"DCAT schema profile:\n{ctx.deps.profile_manifest.identifier}\n\n"
            f"Target class:\n{ctx.deps.profile_manifest.target_class}\n\n"
            "Profile JSON Schema:\n"
            f"{json.dumps(ctx.deps.profile_json_schema, indent=2)}\n\n"
            f"Document file path:\n{ctx.deps.document_file_path}\n\n"
            "Top-level Dataset fields to consider:\n"
            f"{field_descriptions}\n\n"
            "InitialContext JSON:\n"
            f"{ctx.deps.initial_context.model_dump_json(indent=2)}\n\n"
            "Current DCAT dataset draft JSON:\n"
            f"{json.dumps(ctx.deps.current_draft, indent=2)}\n\n"
            f"Chunk batch {ctx.deps.batch_no}/{ctx.deps.total_batches}:\n"
            f"{chunk_texts}\n\n"
            "Return a JSON object like {\"candidates\": [...]} where each "
            "candidate targets one top-level field. Omit fields with no "
            "relevant information in this chunk batch."
        )

    return agent


async def extract_field_patch_candidates(
    *,
    initial_context: InitialContext,
    draft: dict[str, Any],
    document_file_path: str,
    chunk_batch: list[ContentChunk],
    batch_no: int,
    total_batches: int,
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    top_level_fields: list[FieldInfo],
    model: Any,
) -> list[PatchCandidate]:
    """Extract field-level patch candidates from a chunk batch."""
    agent = create_patch_draft_agent(model=model)
    deps = PatchDraftDeps(
        initial_context=initial_context,
        current_draft=draft,
        chunk_batch=chunk_batch,
        batch_no=batch_no,
        total_batches=total_batches,
        document_file_path=document_file_path,
        profile_manifest=profile_manifest,
        profile_json_schema=profile_json_schema,
        top_level_fields=top_level_fields,
    )
    result = await agent.run(
        (
            f"Extract field-level patch candidates for chunk batch "
            f"{batch_no}/{total_batches}. For each top-level Dataset field "
            "where the chunks contain relevant information, produce a "
            "PatchCandidate. Omit fields with no relevant information."
        ),
        deps=deps,
    )
    return result.output.candidates


# ---------------------------------------------------------------------------
# Main patch loop
# ---------------------------------------------------------------------------

async def patch_draft_from_content_chunks(
    *,
    initial_context: InitialContext,
    initial_draft: dict[str, Any],
    content_chunks_by_file: list[list[ContentChunk]],
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    model: Any,
    num_chunks_per_turn: int,
    on_patch_processed: PatchProgressCallback | None = None,
) -> PatchDraftResult:
    draft = copy.deepcopy(initial_draft)
    patches: list[PatchRecord] = []

    top_level_fields = get_top_level_fields(
        profile_json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
    )

    for document_index, chunks in enumerate(content_chunks_by_file, start=1):
        if not chunks:
            continue

        effective_batch_size = max(1, min(num_chunks_per_turn, len(chunks)))
        total_batches = (len(chunks) + effective_batch_size - 1) // effective_batch_size
        document_file_path = chunks[0].file_path

        for batch_index, start_index in enumerate(
            range(0, len(chunks), effective_batch_size),
            start=1,
        ):
            chunk_batch = chunks[start_index : start_index + effective_batch_size]
            patch_file_name = (
                f"{document_index}_{ContentChunk.get_chunk_group_id_from_file_path(document_file_path)}"
                f"_patch_{batch_index}_{total_batches}.json"
            )

            # Step 1: Extract field-level patch candidates from the LLM agent.
            candidates = await extract_field_patch_candidates(
                initial_context=initial_context,
                draft=draft,
                document_file_path=document_file_path,
                chunk_batch=chunk_batch,
                batch_no=batch_index,
                total_batches=total_batches,
                profile_manifest=profile_manifest,
                profile_json_schema=profile_json_schema,
                top_level_fields=top_level_fields,
                model=model,
            )

            # Step 2: Merge accepted candidates into a single proposed patch.
            proposed_patch = _merge_candidates_into_patch(candidates)

            # Step 3: Apply the proposed patch to a candidate draft and validate.
            candidate = apply_merge_patch(draft, proposed_patch)

            schema_errors = validate_candidate_draft(
                candidate=candidate,
                profile_json_schema=profile_json_schema,
                profile_manifest=profile_manifest,
            )

            # Step 4: Semantic quality review (batch review of all candidates).
            quality_report = await review_patch_semantic_quality(
                initial_context=initial_context,
                current_draft=draft,
                proposed_patch=proposed_patch,
                candidate_draft=candidate,
                schema_validation_errors=schema_errors,
                chunk_batch=chunk_batch,
                document_file_path=document_file_path,
                profile_manifest=profile_manifest,
                profile_json_schema=profile_json_schema,
                candidates=candidates,
                model=model,
            )

            # Step 5: Apply per-candidate decisions.
            accepted_fields: list[str] = []
            merged_accepted_patch: dict[str, Any] = {}

            for rating in quality_report.candidate_ratings:
                candidate_obj = _find_candidate(
                    candidates, rating.field_path,
                )
                if candidate_obj is None:
                    continue

                if rating.decision == "accept":
                    accepted_fields.append(rating.field_path)
                    merged_accepted_patch = deep_merge(
                        merged_accepted_patch, candidate_obj.patch,
                    )
                elif rating.decision == "revise" and rating.revised_patch:
                    revised_candidate = apply_merge_patch(
                        draft, rating.revised_patch,
                    )
                    revised_errors = validate_candidate_draft(
                        candidate=revised_candidate,
                        profile_json_schema=profile_json_schema,
                        profile_manifest=profile_manifest,
                    )
                    if not revised_errors:
                        accepted_fields.append(rating.field_path)
                        merged_accepted_patch = deep_merge(
                            merged_accepted_patch, rating.revised_patch,
                        )
                # "reject" → skip

            # Step 6: Deterministic guard — validate the full merged draft.
            if accepted_fields:
                merged_draft = apply_merge_patch(draft, merged_accepted_patch)
                merged_errors = validate_candidate_draft(
                    candidate=merged_draft,
                    profile_json_schema=profile_json_schema,
                    profile_manifest=profile_manifest,
                )
                if merged_errors:
                    # Merged result is invalid — reject entire batch.
                    patch_record = PatchRecord(
                        file_name=patch_file_name,
                        candidates=candidates,
                        accepted_fields=[],
                        merged_patch=merged_accepted_patch,
                        quality_report=quality_report,
                        validation_errors=merged_errors,
                    )
                    patches.append(patch_record)
                    await _notify_patch_progress(
                        on_patch_processed, draft, patch_record,
                    )
                    continue

                draft = merged_draft
                patch_record = PatchRecord(
                    file_name=patch_file_name,
                    candidates=candidates,
                    accepted_fields=accepted_fields,
                    merged_patch=merged_accepted_patch,
                    quality_report=quality_report,
                    validation_errors=[],
                )
                patches.append(patch_record)
                await _notify_patch_progress(
                    on_patch_processed, draft, patch_record,
                )
            else:
                # No candidates accepted — record as empty batch.
                patch_record = PatchRecord(
                    file_name=patch_file_name,
                    candidates=candidates,
                    accepted_fields=[],
                    merged_patch={},
                    quality_report=quality_report,
                    validation_errors=schema_errors or [],
                )
                patches.append(patch_record)
                await _notify_patch_progress(
                    on_patch_processed, draft, patch_record,
                )

    return PatchDraftResult(draft=draft, patches=patches)


def _find_candidate(
    candidates: list[PatchCandidate],
    field_path: str,
) -> PatchCandidate | None:
    """Find a candidate by field_path, or return None."""
    for c in candidates:
        if c.field_path == field_path:
            return c
    return None


def _merge_candidates_into_patch(
    candidates: list[PatchCandidate],
) -> dict[str, Any]:
    """Merge all candidate patches into a single top-level merge patch."""
    result: dict[str, Any] = {}
    for candidate in candidates:
        result = deep_merge(result, candidate.patch)
    return result


# ---------------------------------------------------------------------------
# Progress notification
# ---------------------------------------------------------------------------

async def _notify_patch_progress(
    on_patch_processed: PatchProgressCallback | None,
    draft: dict[str, Any],
    patch_record: PatchRecord,
) -> None:
    if on_patch_processed is None:
        return
    await on_patch_processed(copy.deepcopy(draft), patch_record)


# ---------------------------------------------------------------------------
# Merge utilities (unchanged public API)
# ---------------------------------------------------------------------------

def apply_merge_patch(
    draft: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    draft_copy = copy.deepcopy(draft)
    patch_copy = copy.deepcopy(patch)
    return deep_merge(draft_copy, patch_copy)


def deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if value is None:
            continue

        dst_value = dst.get(key)
        if isinstance(value, dict) and isinstance(dst_value, dict):
            deep_merge(dst_value, value)
        elif isinstance(value, list) and isinstance(dst_value, list):
            dst[key] = _merge_lists(dst_value, value)
        else:
            dst[key] = value

    return dst


def _merge_lists(dst_value: list[Any], patch_value: list[Any]) -> list[Any]:
    if not patch_value:
        return dst_value
    if not dst_value:
        return patch_value

    if all(isinstance(item, dict) for item in patch_value) and all(
        isinstance(item, dict) for item in dst_value
    ):
        return _merge_list_of_dicts_by_id(dst_value, patch_value)

    if all(isinstance(item, str) for item in patch_value) and all(
        isinstance(item, str) for item in dst_value
    ):
        return _merge_list_of_strings(dst_value, patch_value)

    return patch_value


def _merge_list_of_dicts_by_id(
    dst_value: list[dict[str, Any]],
    patch_value: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for item in dst_value:
        item.setdefault("id", _new_patch_id())

    dst_items_by_id = {item["id"]: item for item in dst_value}
    for item in patch_value:
        item.setdefault("id", _new_patch_id())
        item_id = item["id"]
        if item_id in dst_items_by_id:
            deep_merge(dst_items_by_id[item_id], item)
        else:
            dst_value.append(item)

    return dst_value


def _merge_list_of_strings(
    dst_value: list[str],
    patch_value: list[str],
) -> list[str]:
    merged = list(dst_value)
    for item in patch_value:
        if item not in merged:
            merged.append(item)
    return merged


def _new_patch_id() -> str:
    return str(uuid4())[:8]


def _chunk_json_for_prompt(chunk: ContentChunk) -> str:
    return chunk.model_dump_json(
        exclude={"embedding"},
    )
