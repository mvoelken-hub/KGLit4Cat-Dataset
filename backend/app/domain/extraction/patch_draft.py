from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.datasources import ContentChunk
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import FieldPatchResult, InitialContext, PatchCandidate
from app.domain.extraction.patch_quality import (
    CandidateQualityRating,
    PatchQualityReport,
    PatchQualityIssue,
    validate_candidate_draft,
)
from app.domain.extraction.schema_utils import (
    FieldInfo,
    get_top_level_fields,
    json_pointer_top_level_field,
    resolve_json_pointer,
    slice_profile_json_schema,
)
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
    "patch scoped to just that field, your confidence (0.0-1.0), a brief "
    "reasoning explanation, and source evidence from the chunks. "
    "If a field has no relevant information in this chunk, omit it from "
    "the candidates list. "
    "Do not rewrite unchanged fields. Avoid duplicate information. "
    "Focus on data-generating activity, device or instrument details, "
    "entities, distributions, file roles, and qualitative or quantitative "
    "attributes extracted from the chunks. "
    "For objects in lists, include an id field with a stable local "
    "identifier scoped to the dataset or experiment, for example "
    "'1h-nmr-clean/activity/1h-nmr-acquisition'. Use ids from the current "
    "draft to update existing objects. Do not use null to delete existing values; null values "
    "are ignored by the merge function. "
    "If a field is listed as protected, skip it entirely and do not "
    "produce a candidate for it. "
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
    protected_fields: list[str] = field(default_factory=list)


@dataclass
class PatchDiscoveryDeps:
    current_draft: dict[str, Any]
    chunk_batch: list[ContentChunk]
    batch_no: int
    total_batches: int
    document_file_path: str
    protected_fields: list[str] = field(default_factory=list)


@dataclass
class SchemaPatchDeps:
    current_draft_slice: dict[str, Any]
    schema_slice: dict[str, Any]
    location_picks: list["PatchLocationPick"]
    information: str
    evidence: list[str]
    document_file_path: str


@dataclass
class SchemaRepairDeps(SchemaPatchDeps):
    invalid_patch: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)


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


class PatchLocationPick(BaseModel):
    path: str = Field(
        description=(
            "JSON Pointer to the best existing draft location for this "
            "information, e.g. /was_generated_by/0/has_quantitative_attribute."
        )
    )
    rationale: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class PatchDiscoveryResult(BaseModel):
    information: str | None = Field(
        default=None,
        description=(
            "Concise metadata information from the chunk that should be added "
            "to the draft, or null if there is nothing useful to add."
        ),
    )
    evidence: list[str] = Field(default_factory=list)
    location_picks: list[PatchLocationPick] = Field(default_factory=list)


class SchemaPatchResult(BaseModel):
    destination: str = Field(description="Single chosen JSON Pointer destination.")
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="Top-level JSON merge patch rooted at the Dataset object.",
    )
    reasoning: str | None = None


PatchProgressCallback = Callable[[dict[str, Any], PatchRecord, int, int], Awaitable[None]]
TokenUsageCallback = Callable[[str, Any, int], None]


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

        protected_note = ""
        if ctx.deps.protected_fields:
            protected_note = (
                "\n\nProtected fields (DO NOT modify these):\n"
                + "\n".join(f"- {f}" for f in ctx.deps.protected_fields)
                + "\n"
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
            f"{protected_note}"
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
    protected_fields: list[str] | None = None,
    on_token_usage: TokenUsageCallback | None = None,
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
        protected_fields=protected_fields or [],
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
    if on_token_usage is not None:
        on_token_usage("patch_extraction", result.usage, 1)
    return result.output.candidates


# ---------------------------------------------------------------------------
# Lean schema-late patch agents
# ---------------------------------------------------------------------------

PATCH_DISCOVERY_INSTRUCTIONS = (
    "You inspect source chunks against the current metadata draft. "
    "Do not create a schema patch. Find one concise piece of new metadata "
    "that is supported by the chunks and not already represented in the draft. "
    "Return information=null when the chunks add nothing useful. "
    "When information exists, return ranked JSON Pointer location picks that "
    "point to reasonable existing places in the draft where this information "
    "should be stored. Prefer precise nested pointers over broad root fields. "
    "Do not invent facts."
)

SCHEMA_PATCH_INSTRUCTIONS = (
    "You convert discovered metadata information into one JSON merge patch "
    "for a Dataset draft. Use only the provided schema slice and draft slice. "
    "Choose exactly one destination from the provided JSON Pointer picks. "
    "Return a top-level merge patch rooted at the Dataset object, not a JSON "
    "Patch operation list. Do not use null to delete existing values. "
    + URI_POLICY_INSTRUCTIONS
)

SCHEMA_REPAIR_INSTRUCTIONS = (
    "You repair a JSON merge patch that failed full Dataset schema validation. "
    "Use the validation errors, schema slice, draft slice, and source evidence "
    "to return a corrected top-level JSON merge patch. Preserve the intended "
    "metadata information and do not invent facts or schema fields. "
    + URI_POLICY_INSTRUCTIONS
)


def create_patch_discovery_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[PatchDiscoveryDeps, PatchDiscoveryResult]:
    agent = Agent(
        model,
        deps_type=PatchDiscoveryDeps,
        output_type=prompted_json_output(
            PatchDiscoveryResult,
            name="PatchDiscoveryResult",
            description="New chunk information and ranked draft locations.",
        ),
        instructions=PATCH_DISCOVERY_INSTRUCTIONS,
        model_settings={"temperature": 0.0, "seed": 42},
        output_retries=output_retries,
    )

    @agent.instructions
    def add_discovery_context(ctx: RunContext[PatchDiscoveryDeps]) -> str:
        protected_note = ""
        if ctx.deps.protected_fields:
            protected_note = (
                "\n\nProtected fields (do not pick locations under these):\n"
                + "\n".join(f"- {field}" for field in ctx.deps.protected_fields)
                + "\n"
            )
        return (
            f"Document file path:\n{ctx.deps.document_file_path}\n\n"
            f"Patch batch:\n{ctx.deps.batch_no}/{ctx.deps.total_batches}\n\n"
            "Current draft JSON:\n"
            f"{json.dumps(ctx.deps.current_draft, indent=2, ensure_ascii=False)}\n\n"
            "Source chunk batch:\n"
            f"{json.dumps([chunk.model_dump(exclude={'embedding'}) for chunk in ctx.deps.chunk_batch], indent=2, ensure_ascii=False)}\n"
            f"{protected_note}"
        )

    return agent


def create_schema_patch_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[SchemaPatchDeps, SchemaPatchResult]:
    agent = Agent(
        model,
        deps_type=SchemaPatchDeps,
        output_type=prompted_json_output(
            SchemaPatchResult,
            name="SchemaPatchResult",
            description="A schema-conformant top-level JSON merge patch.",
        ),
        instructions=SCHEMA_PATCH_INSTRUCTIONS,
        model_settings={"temperature": 0.0, "seed": 42},
        output_retries=output_retries,
    )

    @agent.instructions
    def add_patch_context(ctx: RunContext[SchemaPatchDeps]) -> str:
        return _schema_patch_context(ctx.deps)

    return agent


def create_schema_repair_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[SchemaRepairDeps, SchemaPatchResult]:
    agent = Agent(
        model,
        deps_type=SchemaRepairDeps,
        output_type=prompted_json_output(
            SchemaPatchResult,
            name="SchemaPatchResult",
            description="A repaired schema-conformant top-level JSON merge patch.",
        ),
        instructions=SCHEMA_REPAIR_INSTRUCTIONS,
        model_settings={"temperature": 0.0, "seed": 42},
        output_retries=output_retries,
    )

    @agent.instructions
    def add_repair_context(ctx: RunContext[SchemaRepairDeps]) -> str:
        return (
            _schema_patch_context(ctx.deps)
            + "\nInvalid patch JSON:\n"
            + json.dumps(ctx.deps.invalid_patch, indent=2, ensure_ascii=False)
            + "\n\nValidation errors:\n"
            + json.dumps(ctx.deps.validation_errors, indent=2, ensure_ascii=False)
            + "\n"
        )

    return agent


def _schema_patch_context(deps: SchemaPatchDeps) -> str:
    return (
        "Information to add:\n"
        f"{deps.information}\n\n"
        "Evidence:\n"
        f"{json.dumps(deps.evidence, indent=2, ensure_ascii=False)}\n\n"
        "Location picks:\n"
        f"{json.dumps([pick.model_dump(mode='json') for pick in deps.location_picks], indent=2, ensure_ascii=False)}\n\n"
        f"Document file path:\n{deps.document_file_path}\n\n"
        "Current draft slice JSON:\n"
        f"{json.dumps(deps.current_draft_slice, indent=2, ensure_ascii=False)}\n\n"
        "Relevant schema slice JSON:\n"
        f"{json.dumps(deps.schema_slice, indent=2, ensure_ascii=False)}\n"
    )


async def discover_patch_information(
    *,
    draft: dict[str, Any],
    document_file_path: str,
    chunk_batch: list[ContentChunk],
    batch_no: int,
    total_batches: int,
    model: Any,
    protected_fields: list[str] | None = None,
    on_token_usage: TokenUsageCallback | None = None,
) -> PatchDiscoveryResult:
    agent = create_patch_discovery_agent(model=model)
    result = await agent.run(
        "Find new metadata information in this chunk batch and rank JSON "
        "Pointer locations in the current draft where it belongs.",
        deps=PatchDiscoveryDeps(
            current_draft=draft,
            chunk_batch=chunk_batch,
            batch_no=batch_no,
            total_batches=total_batches,
            document_file_path=document_file_path,
            protected_fields=protected_fields or [],
        ),
    )
    if on_token_usage is not None:
        on_token_usage("patch_discovery", result.usage, 1)
    return result.output


async def create_schema_bound_patch(
    *,
    discovery: PatchDiscoveryResult,
    current_draft_slice: dict[str, Any],
    schema_slice: dict[str, Any],
    location_picks: list[PatchLocationPick],
    document_file_path: str,
    model: Any,
    on_token_usage: TokenUsageCallback | None = None,
) -> SchemaPatchResult:
    agent = create_schema_patch_agent(model=model)
    result = await agent.run(
        "Create one top-level JSON merge patch for the discovered information.",
        deps=SchemaPatchDeps(
            current_draft_slice=current_draft_slice,
            schema_slice=schema_slice,
            location_picks=location_picks,
            information=discovery.information or "",
            evidence=discovery.evidence,
            document_file_path=document_file_path,
        ),
    )
    if on_token_usage is not None:
        on_token_usage("schema_patch_writer", result.usage, 1)
    return result.output


async def repair_schema_bound_patch(
    *,
    discovery: PatchDiscoveryResult,
    current_draft_slice: dict[str, Any],
    schema_slice: dict[str, Any],
    location_picks: list[PatchLocationPick],
    invalid_patch: dict[str, Any],
    validation_errors: list[str],
    document_file_path: str,
    model: Any,
    on_token_usage: TokenUsageCallback | None = None,
) -> SchemaPatchResult:
    agent = create_schema_repair_agent(model=model)
    result = await agent.run(
        "Repair the JSON merge patch so it satisfies the schema.",
        deps=SchemaRepairDeps(
            current_draft_slice=current_draft_slice,
            schema_slice=schema_slice,
            location_picks=location_picks,
            information=discovery.information or "",
            evidence=discovery.evidence,
            document_file_path=document_file_path,
            invalid_patch=invalid_patch,
            validation_errors=validation_errors,
        ),
    )
    if on_token_usage is not None:
        on_token_usage("schema_repair", result.usage, 1)
    return result.output


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
    protected_fields: list[str] | None = None,
    protected_fields_loader: Callable[[], list[str]] | None = None,
    completed_patch_file_names: set[str] | None = None,
    on_token_usage: TokenUsageCallback | None = None,
) -> PatchDraftResult:
    draft = copy.deepcopy(initial_draft)
    patches: list[PatchRecord] = []
    completed_patch_file_names = completed_patch_file_names or set()
    progress_batch_no = 0
    progress_total_batches = _count_progress_batches(
        content_chunks_by_file=content_chunks_by_file,
        num_chunks_per_turn=num_chunks_per_turn,
    )

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
            if patch_file_name in completed_patch_file_names:
                progress_batch_no += 1
                continue

            current_protected_fields = (
                protected_fields_loader()
                if protected_fields_loader is not None
                else protected_fields
            )
            protected_top_level_fields = _normalise_protected_fields(
                current_protected_fields,
            )

            discovery = await discover_patch_information(
                draft=draft,
                document_file_path=document_file_path,
                chunk_batch=chunk_batch,
                batch_no=batch_index,
                total_batches=total_batches,
                model=model,
                protected_fields=current_protected_fields,
                on_token_usage=on_token_usage,
            )
            location_picks = _valid_location_picks(
                discovery.location_picks,
                protected_fields=protected_top_level_fields,
                top_level_fields={field.name for field in top_level_fields},
            )
            if not _has_discovered_information(discovery) or not location_picks:
                patch_record = PatchRecord(
                    file_name=patch_file_name,
                    candidates=[],
                    accepted_fields=[],
                    merged_patch={},
                    quality_report=None,
                    validation_errors=[],
                )
                patches.append(patch_record)
                progress_batch_no += 1
                await _notify_patch_progress(
                    on_patch_processed,
                    draft,
                    patch_record,
                    progress_batch_no,
                    progress_total_batches,
                )
                continue

            target_fields = _top_level_fields_from_location_picks(location_picks)
            schema_slice = slice_profile_json_schema(
                profile_json_schema=profile_json_schema,
                target_class=profile_manifest.target_class,
                field_names=target_fields,
            )
            draft_slice = _draft_slice_for_location_picks(draft, location_picks)
            schema_patch = await create_schema_bound_patch(
                discovery=discovery,
                current_draft_slice=draft_slice,
                schema_slice=schema_slice,
                location_picks=location_picks,
                document_file_path=document_file_path,
                model=model,
                on_token_usage=on_token_usage,
            )
            merged_patch = schema_patch.patch
            validation_errors = _validate_generated_patch(
                patch=merged_patch,
                draft=draft,
                protected_fields=protected_top_level_fields,
                profile_json_schema=profile_json_schema,
                profile_manifest=profile_manifest,
            )
            repair_attempts = 0
            while validation_errors and repair_attempts < DEFAULT_OUTPUT_RETRIES:
                repair_attempts += 1
                schema_patch = await repair_schema_bound_patch(
                    discovery=discovery,
                    current_draft_slice=draft_slice,
                    schema_slice=schema_slice,
                    location_picks=location_picks,
                    invalid_patch=merged_patch,
                    validation_errors=validation_errors,
                    document_file_path=document_file_path,
                    model=model,
                    on_token_usage=on_token_usage,
                )
                merged_patch = schema_patch.patch
                validation_errors = _validate_generated_patch(
                    patch=merged_patch,
                    draft=draft,
                    protected_fields=protected_top_level_fields,
                    profile_json_schema=profile_json_schema,
                    profile_manifest=profile_manifest,
                )

            candidate_field = _patch_primary_field(merged_patch) or target_fields[0]
            patch_candidate = PatchCandidate(
                field_path=candidate_field,
                patch=merged_patch,
                confidence=_location_pick_confidence(location_picks),
                reasoning=schema_patch.reasoning
                or f"Schema-late patch for discovered information: {discovery.information}",
                source_evidence=discovery.evidence,
            )

            if validation_errors:
                patch_record = PatchRecord(
                    file_name=patch_file_name,
                    candidates=[patch_candidate],
                    accepted_fields=[],
                    merged_patch=merged_patch,
                    quality_report=_synthetic_quality_report(
                        candidate=patch_candidate,
                        accepted=False,
                        validation_errors=validation_errors,
                    ),
                    validation_errors=validation_errors,
                )
                patches.append(patch_record)
                progress_batch_no += 1
                await _notify_patch_progress(
                    on_patch_processed,
                    draft,
                    patch_record,
                    progress_batch_no,
                    progress_total_batches,
                )
                continue

            draft = apply_merge_patch(draft, merged_patch)
            patch_record = PatchRecord(
                file_name=patch_file_name,
                candidates=[patch_candidate],
                accepted_fields=[patch_candidate.field_path],
                merged_patch=merged_patch,
                quality_report=_synthetic_quality_report(
                    candidate=patch_candidate,
                    accepted=True,
                    validation_errors=[],
                ),
                validation_errors=[],
            )
            patches.append(patch_record)
            progress_batch_no += 1
            await _notify_patch_progress(
                on_patch_processed,
                draft,
                patch_record,
                progress_batch_no,
                progress_total_batches,
            )
            continue

    return PatchDraftResult(draft=draft, patches=patches)


def _has_discovered_information(discovery: PatchDiscoveryResult) -> bool:
    return bool(discovery.information and discovery.information.strip())


def _valid_location_picks(
    picks: list[PatchLocationPick],
    *,
    protected_fields: set[str],
    top_level_fields: set[str],
) -> list[PatchLocationPick]:
    valid: list[PatchLocationPick] = []
    seen: set[str] = set()
    for pick in picks:
        top_level_field = json_pointer_top_level_field(pick.path)
        if (
            top_level_field is None
            or top_level_field not in top_level_fields
            or top_level_field in protected_fields
            or pick.path in seen
        ):
            continue
        seen.add(pick.path)
        valid.append(pick)
    return valid


def _top_level_fields_from_location_picks(
    location_picks: list[PatchLocationPick],
) -> list[str]:
    fields: list[str] = []
    for pick in location_picks:
        field = json_pointer_top_level_field(pick.path)
        if field and field not in fields:
            fields.append(field)
    return fields


def _draft_slice_for_location_picks(
    draft: dict[str, Any],
    location_picks: list[PatchLocationPick],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _top_level_fields_from_location_picks(location_picks):
        if field in draft:
            result[field] = copy.deepcopy(draft[field])

    pointer_context: dict[str, Any] = {}
    for pick in location_picks:
        try:
            pointer_context[pick.path] = copy.deepcopy(
                resolve_json_pointer(draft, pick.path)
            )
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    if pointer_context:
        result["_selected_location_context"] = pointer_context
    return result


def _validate_generated_patch(
    *,
    patch: dict[str, Any],
    draft: dict[str, Any],
    protected_fields: set[str],
    profile_json_schema: dict[str, Any],
    profile_manifest: ProfileManifest,
) -> list[str]:
    if not patch:
        return ["Schema patch writer returned an empty patch."]
    if _patch_touches_protected_field(patch, protected_fields):
        return ["Generated patch touches a protected field."]
    candidate = apply_merge_patch(draft, patch)
    return validate_candidate_draft(
        candidate=candidate,
        profile_json_schema=profile_json_schema,
        profile_manifest=profile_manifest,
    )


def _patch_primary_field(patch: dict[str, Any]) -> str | None:
    for key in patch:
        return key
    return None


def _location_pick_confidence(location_picks: list[PatchLocationPick]) -> float:
    for pick in location_picks:
        if pick.confidence is not None:
            return pick.confidence
    return 0.8


def _synthetic_quality_report(
    *,
    candidate: PatchCandidate,
    accepted: bool,
    validation_errors: list[str],
) -> PatchQualityReport:
    if accepted:
        return PatchQualityReport(
            overall_decision="accept",
            candidate_ratings=[
                CandidateQualityRating(
                    field_path=candidate.field_path,
                    decision="accept",
                    issues=[],
                )
            ],
            summary="Schema-late patch passed full schema validation.",
        )

    return PatchQualityReport(
        overall_decision="reject",
        candidate_ratings=[
            CandidateQualityRating(
                field_path=candidate.field_path,
                decision="reject",
                issues=[
                    PatchQualityIssue(
                        path=candidate.field_path,
                        issue_type="schema_mismatch",
                        severity="major",
                        explanation="; ".join(validation_errors),
                    )
                ],
            )
        ],
        summary="Schema-late patch failed full schema validation.",
    )


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


def _count_progress_batches(
    *,
    content_chunks_by_file: list[list[ContentChunk]],
    num_chunks_per_turn: int,
) -> int:
    total = 0
    for chunks in content_chunks_by_file:
        if not chunks:
            continue
        effective_batch_size = max(1, min(num_chunks_per_turn, len(chunks)))
        total += (len(chunks) + effective_batch_size - 1) // effective_batch_size
    return total


def _normalise_protected_fields(
    protected_fields: list[str] | None,
) -> set[str]:
    return {
        field.split(".", 1)[0]
        for field in protected_fields or []
        if field and field.split(".", 1)[0]
    }


def _candidate_touches_protected_field(
    candidate: PatchCandidate,
    protected_fields: set[str],
) -> bool:
    return (
        _field_path_is_protected(candidate.field_path, protected_fields)
        or _patch_touches_protected_field(candidate.patch, protected_fields)
    )


def _field_path_is_protected(
    field_path: str,
    protected_fields: set[str],
) -> bool:
    if not protected_fields:
        return False
    return field_path.split(".", 1)[0] in protected_fields


def _patch_touches_protected_field(
    patch: dict[str, Any],
    protected_fields: set[str],
) -> bool:
    if not protected_fields:
        return False
    return any(key.split(".", 1)[0] in protected_fields for key in patch)


# ---------------------------------------------------------------------------
# Progress notification
# ---------------------------------------------------------------------------

async def _notify_patch_progress(
    on_patch_processed: PatchProgressCallback | None,
    draft: dict[str, Any],
    patch_record: PatchRecord,
    batch_no: int,
    total_batches: int,
) -> None:
    if on_patch_processed is None:
        return
    await on_patch_processed(
        copy.deepcopy(draft),
        patch_record,
        batch_no,
        total_batches,
    )


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
    dst_items_by_id = {
        item["id"]: item
        for item in dst_value
        if isinstance(item.get("id"), str) and item.get("id")
    }
    for item in patch_value:
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id in dst_items_by_id:
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
