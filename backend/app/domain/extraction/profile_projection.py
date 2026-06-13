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


def build_profile_target_planner_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    evidence_notes: list[EvidenceNote],
    target_catalog: list[dict[str, Any]],
    file_inventory: list[FileInventoryItem],
) -> str:
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        "Validated evidence note group JSON:\n"
        f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n"
        "Available projection targets JSON:\n"
        f"{target_catalog}\n\n"
        "Deterministic package file inventory JSON (context only, not model evidence):\n"
        f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n"
        "Return a target decision. Select one target_path from the catalog, or return status='skip'. "
        "Use /description only for genuinely dataset-level prose that cannot fit a more specific target. "
        "Prefer scaffold_status='unfilled' or structured category_affinities when evidence can populate them."
    )


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
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        f"Selected target path: {target_path}\n"
        f"Selected target class: {target_class or ''}\n"
        f"Selected target label: {target_label}\n\n"
        "Current selected target value JSON:\n"
        f"{current_target_value}\n\n"
        "Validated evidence note group JSON:\n"
        f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n"
        "Deterministic package file inventory JSON (context only, not model evidence):\n"
        f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n"
        "Selected target schema slice JSON:\n"
        f"{schema_slice}\n\n"
        "Return JSON with status='write' and `value` set to the complete replacement value for the selected target path, "
        "or status='skip' when nothing should be written. Preserve schema-valid existing values unless the evidence clearly improves them."
    )


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
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        f"Selected target path: {target_path}\n"
        f"Selected target class: {target_class or ''}\n\n"
        "Current profile document JSON:\n"
        f"{current_document}\n\n"
        "Validated evidence note group JSON:\n"
        f"{[note.model_dump(mode='json') for note in evidence_notes]}\n\n"
        "Deterministic package file inventory JSON (context only, not model evidence):\n"
        f"{[item.model_dump(mode='json') for item in file_inventory]}\n\n"
        "Selected target schema slice JSON:\n"
        f"{schema_slice}\n\n"
        "Return JSON with an `operations` array of RFC 6902 JSON Patch operations. "
        "Patch only values supported by the evidence note evidence_text and observation. "
        "All operation paths must stay inside the selected target path unless updating /id, /title, or /identifier."
    )


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
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        "Merged EvidenceContext JSON:\n"
        f"{evidence_context.model_dump(mode='json')}\n\n"
        "Vocabulary normalization JSON:\n"
        f"{normalization.model_dump(mode='json')}\n\n"
        "Normalization warnings JSON:\n"
        f"{warnings}\n\n"
        "The target profile JSON Schema is enforced via the structured-output contract. "
        "Return exactly one JSON object that satisfies the schema. Use normalized vocabulary URIs "
        "where available, and include raw unmatched values only when schema-valid."
    )

