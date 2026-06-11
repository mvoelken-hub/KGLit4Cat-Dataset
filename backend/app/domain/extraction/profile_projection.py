from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.extraction.extraction_context import ExtractionContext
from app.domain.extraction.extraction_context import TracedExtractionObject
from app.domain.extraction.vocabulary import ExtractionNormalization


PROFILE_PROJECTION_SYSTEM_PROMPT = """
You transform normalized scientific extraction context into the selected application profile JSON document.
Use the supplied target profile schema and profile metadata to choose where facts belong.
Prefer schema-valid concise metadata over exhaustive copying. Return only JSON that satisfies the target schema.
"""


PROFILE_PATCH_SYSTEM_PROMPT = """
You update a schema-valid scientific metadata profile document using one traced extraction object as evidence.
Return only JSON Patch operations. Use add or replace for supported facts and remove only for clearly wrong placeholder values.
Do not invent unsupported facts. Keep the document schema-valid.
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


class ProfileObjectPatchResult(BaseModel):
    object_identifier: str
    object_kind: str
    status: Literal["applied", "skipped", "failed"] = "skipped"
    operations: list[ProfilePatchOperation] = Field(default_factory=list)
    error: str | None = None
    reason: str = ""


def build_profile_patch_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    current_document: dict[str, Any],
    extraction_object: TracedExtractionObject,
    schema_slice: dict[str, Any],
) -> str:
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        "Current profile document JSON:\n"
        f"{current_document}\n\n"
        "Patch content extraction object JSON:\n"
        f"{extraction_object.model_dump(mode='json')}\n\n"
        "Small target schema slice JSON:\n"
        f"{schema_slice}\n\n"
        "Return JSON with an `operations` array of RFC 6902 JSON Patch operations. "
        "Patch only values supported by the extraction object source_text and fields."
    )


def build_profile_projection_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    extraction_context: ExtractionContext,
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
        "Merged ExtractionContext JSON:\n"
        f"{extraction_context.model_dump(mode='json')}\n\n"
        "Vocabulary normalization JSON:\n"
        f"{normalization.model_dump(mode='json')}\n\n"
        "Normalization warnings JSON:\n"
        f"{warnings}\n\n"
        "The target profile JSON Schema is enforced via the structured-output contract. "
        "Return exactly one JSON object that satisfies the schema. Use normalized vocabulary URIs "
        "where available, and include raw unmatched values only when schema-valid."
    )

