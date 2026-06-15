from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.extraction.evidence_context import EvidenceContext
from app.domain.extraction.evidence_context import EvidenceNote
from app.domain.extraction.evidence_context import FileInventoryItem
from app.domain.extraction.vocabulary import ExtractionNormalization


PROFILE_PROJECTION_SYSTEM_PROMPT = """
You transform normalized scientific extraction context into the selected application profile JSON document.
Use the supplied target profile schema and profile metadata to choose where facts belong.
Prefer schema-valid concise metadata over exhaustive copying. Return only JSON that satisfies the target schema.
"""


PROFILE_TARGET_PLANNER_SYSTEM_PROMPT = """
You choose the most specific schema target for a group of validated scientific evidence notes.
Return only the structured target decision. Prefer specific scaffold objects or fields over broad description text.
Choose skip when the evidence is technical noise, redundant, or not profile-worthy.
Treat /description as a last-resort target for genuine dataset-level prose; do not use it as a dumping ground for structured method, agent, distribution, entity, measurement, identifier, type, or date evidence.
"""


PROFILE_PATCH_SYSTEM_PROMPT = """
You update a schema-valid scientific metadata profile document using one group of validated evidence notes.
Return only JSON Patch operations. Use add or replace for supported facts and remove only for clearly wrong placeholder values.
Patch only inside the selected target path unless updating a directly required parent identifier or title.
Do not invent unsupported facts. Keep the document schema-valid.
"""


PROFILE_TARGET_WRITER_SYSTEM_PROMPT = """
You update exactly one selected schema target in a scientific metadata profile document.
Return only the structured target write decision. When writing, return the complete replacement value for the selected target path, not JSON Patch operations.
Preserve existing supported values in the current target value, add only evidence-supported facts, and keep the returned value valid for the selected schema slice.
Match the schema shape exactly: array fields must remain arrays, object fields must remain objects, and scalar strings must not replace arrays.
Do not write raw instrument parameter keys or low-level acquisition settings to /keyword or /description.
Return skip when the evidence is redundant, too technical for the selected target, or cannot be represented without inventing facts.
"""


class ProfilePatchOperation(BaseModel):
    op: Literal["add", "replace", "remove"]
    path: str = Field(..., description="RFC 6902 JSON Pointer path.")
    value: Any | None = Field(
        default=None,
        description="Value for add/replace operations. Omit for remove.",
    )


class ProfilePatchDocument(BaseModel):
    operations: list[ProfilePatchOperation] = Field(default_factory=list)
    reason: str = ""


class ProfileTargetDecision(BaseModel):
    status: Literal["targeted", "skip"] = "targeted"
    target_path: str | None = None
    target_class: str | None = None
    target_label: str = ""
    reason: str = ""


class ProfileTargetWriteDocument(BaseModel):
    status: Literal["write", "skip"] = "write"
    value: Any = None
    reason: str = ""


class ProfileObjectPatchResult(BaseModel):
    object_identifier: str
    object_kind: str
    status: Literal["applied", "skipped", "failed"] = "skipped"
    operations: list[ProfilePatchOperation] = Field(default_factory=list)
    error: str | None = None
    reason: str = ""
    target_path: str | None = None
    target_class: str | None = None
    planner_status: str | None = None
    planner_reason: str | None = None
    target_value: Any = None
    schema_queries: list[dict[str, Any]] = Field(default_factory=list)
    candidate_paths: list[str] = Field(default_factory=list)
    selected_schema_branch: dict[str, Any] | None = None
    merge_status: str | None = None


def build_profile_target_planner_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    evidence_notes: list[EvidenceNote],
    target_catalog: list[dict[str, Any]],
    file_inventory: list[FileInventoryItem],
) -> str:
    return "".join(
        text
        for _, text in build_profile_target_planner_prompt_components(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_target_class,
            evidence_notes=evidence_notes,
            target_catalog=target_catalog,
            file_inventory=file_inventory,
        )
    )


def build_profile_target_planner_prompt_components(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    evidence_notes: list[EvidenceNote],
    target_catalog: list[dict[str, Any]],
    file_inventory: list[FileInventoryItem],
) -> list[tuple[str, str]]:
    return [
        (
            "profile_identifiers",
            f"Data package id: {data_package_id}\n"
            f"Profile identifier: {profile_identifier}\n"
            f"Profile target class: {profile_target_class}\n\n",
        ),
        (
            "evidence_notes",
            "Validated evidence note group JSON:\n"
            f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n",
        ),
        ("target_catalog", "Available projection targets JSON:\n" f"{target_catalog}\n\n"),
        (
            "file_inventory",
            "Deterministic package file inventory JSON (context only, not model evidence):\n"
            f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n",
        ),
        (
            "planner_instruction",
            "Return a target decision. Select one target_path from the catalog, or return status='skip'. "
            "Use /description only for genuinely dataset-level prose that cannot fit a more specific target. "
            "Prefer scaffold_status='unfilled' or structured category_affinities when evidence can populate them.",
        ),
    ]


def build_profile_target_write_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    target_path: str,
    target_class: str | None,
    target_label: str,
    current_target_value: Any,
    evidence_notes: list[EvidenceNote],
    file_inventory: list[FileInventoryItem],
    schema_slice: dict[str, Any],
) -> str:
    return "".join(
        text
        for _, text in build_profile_target_write_prompt_components(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_target_class,
            target_path=target_path,
            target_class=target_class,
            target_label=target_label,
            current_target_value=current_target_value,
            evidence_notes=evidence_notes,
            file_inventory=file_inventory,
            schema_slice=schema_slice,
        )
    )


def build_profile_target_write_prompt_components(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    target_path: str,
    target_class: str | None,
    target_label: str,
    current_target_value: Any,
    evidence_notes: list[EvidenceNote],
    file_inventory: list[FileInventoryItem],
    schema_slice: dict[str, Any],
) -> list[tuple[str, str]]:
    append_instruction = (
        "The selected target path ends with '/-', so return exactly one object for one new array item, not the whole array. "
        if target_path.endswith("/-")
        else ""
    )
    return [
        (
            "profile_identifiers",
            f"Data package id: {data_package_id}\n"
            f"Profile identifier: {profile_identifier}\n"
            f"Profile target class: {profile_target_class}\n\n",
        ),
        (
            "target_metadata",
            f"Selected target path: {target_path}\n"
            f"Selected target class: {target_class or ''}\n"
            f"Selected target label: {target_label}\n\n",
        ),
        ("current_target_value", "Current selected target value JSON:\n" f"{current_target_value}\n\n"),
        (
            "evidence_notes",
            "Validated evidence note group JSON:\n"
            f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n",
        ),
        (
            "file_inventory",
            "Deterministic package file inventory JSON (context only, not model evidence):\n"
            f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n",
        ),
        ("schema_slice", "Selected target schema slice JSON:\n" f"{schema_slice}\n\n"),
        (
            "writer_instruction",
            "Return JSON with status='write' and `value` set to the complete replacement value for the selected target path, "
            "or status='skip' when nothing should be written. Preserve schema-valid existing values unless the evidence clearly improves them. "
            f"{append_instruction}"
            "Use the current target value as the shape contract: keep arrays as arrays, objects as objects, null-capable object fields as null or objects, and strings as strings.",
        ),
    ]


def build_profile_patch_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    current_document: dict[str, Any],
    evidence_notes: list[EvidenceNote],
    file_inventory: list[FileInventoryItem],
    schema_slice: dict[str, Any],
    target_path: str,
    target_class: str | None = None,
) -> str:
    return "".join(
        text
        for _, text in build_profile_patch_prompt_components(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_target_class,
            current_document=current_document,
            evidence_notes=evidence_notes,
            file_inventory=file_inventory,
            schema_slice=schema_slice,
            target_path=target_path,
            target_class=target_class,
        )
    )


def build_profile_patch_prompt_components(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    current_document: dict[str, Any],
    evidence_notes: list[EvidenceNote],
    file_inventory: list[FileInventoryItem],
    schema_slice: dict[str, Any],
    target_path: str,
    target_class: str | None = None,
) -> list[tuple[str, str]]:
    return [
        (
            "profile_identifiers",
            f"Data package id: {data_package_id}\n"
            f"Profile identifier: {profile_identifier}\n"
            f"Profile target class: {profile_target_class}\n\n",
        ),
        (
            "target_metadata",
            f"Selected target path: {target_path}\n"
            f"Selected target class: {target_class or ''}\n\n",
        ),
        ("current_document", "Current profile document JSON:\n" f"{current_document}\n\n"),
        (
            "evidence_notes",
            "Validated evidence note group JSON:\n"
            f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n",
        ),
        (
            "file_inventory",
            "Deterministic package file inventory JSON (context only, not model evidence):\n"
            f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n",
        ),
        ("schema_slice", "Selected target schema slice JSON:\n" f"{schema_slice}\n\n"),
        (
            "patch_instruction",
            "Return JSON with an `operations` array of RFC 6902 JSON Patch operations. "
            "Patch only values supported by the evidence note evidence_text and observation. "
            "All operation paths must stay inside the selected target path unless updating /id, /title, or /identifier.",
        ),
    ]


def build_profile_projection_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    evidence_context: EvidenceContext,
    normalization: ExtractionNormalization,
    warnings: list[str],
    profile_schema: dict[str, Any] | None = None,
) -> str:
    # The target profile JSON Schema is delivered to the chat model via
    # Ollama's `format` parameter (and the structured prompt builder's
    # system-prompt example). Emitting the full schema again here would
    # double the body size and push the request over Ollama's per-request
    # body limit for the dcat-ap-plus profile. We keep only a short pointer
    # in the prompt text.
    del profile_schema
    return "".join(
        text
        for _, text in build_profile_projection_prompt_components(
            data_package_id=data_package_id,
            profile_identifier=profile_identifier,
            profile_target_class=profile_target_class,
            evidence_context=evidence_context,
            normalization=normalization,
            warnings=warnings,
        )
    )


def build_profile_projection_prompt_components(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    evidence_context: EvidenceContext,
    normalization: ExtractionNormalization,
    warnings: list[str],
) -> list[tuple[str, str]]:
    return [
        (
            "profile_identifiers",
            f"Data package id: {data_package_id}\n"
            f"Profile identifier: {profile_identifier}\n"
            f"Profile target class: {profile_target_class}\n\n",
        ),
        (
            "merged_evidence_context",
            "Merged EvidenceContext JSON:\n"
            f"{evidence_context.model_dump(mode='json')}\n\n",
        ),
        (
            "vocabulary_normalization",
            "Vocabulary normalization JSON:\n"
            f"{normalization.model_dump(mode='json')}\n\n",
        ),
        ("normalization_warnings", "Normalization warnings JSON:\n" f"{warnings}\n\n"),
        (
            "projection_instruction",
            "The target profile JSON Schema is enforced via the structured-output contract. "
            "Return exactly one JSON object that satisfies the schema. Use normalized vocabulary URIs "
            "where available, and include raw unmatched values only when schema-valid.",
        ),
    ]

