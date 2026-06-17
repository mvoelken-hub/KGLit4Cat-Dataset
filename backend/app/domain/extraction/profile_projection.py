from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.extraction.evidence_context import EvidenceContext
from app.domain.extraction.vocabulary import ExtractionNormalization


PROFILE_PROJECTION_SYSTEM_PROMPT = """
You transform normalized scientific extraction context into the selected application profile JSON document.
Use the supplied target profile schema and profile metadata to choose where facts belong.
Prefer schema-valid concise metadata over exhaustive copying. Return only JSON that satisfies the target schema.
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

