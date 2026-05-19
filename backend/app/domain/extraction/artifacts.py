from typing import Any

from pydantic import BaseModel, Field


class FileRelationship(BaseModel):
    source_file: str = Field(description="File path containing or implying the relationship.")
    related_file: str | None = Field(
        default=None,
        description="Related file path, when a specific counterpart is identifiable.",
    )
    relationship_type: str = Field(
        description="Short relationship label such as metadata_for, derived_from, raw_data_for, or companion_file."
    )
    description: str = Field(description="Concise explanation of the relationship.")
    evidence: str | None = Field(
        default=None,
        description="Short textual evidence or filename clue supporting the relationship.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in the inferred relationship from 0 to 1.",
    )


class MetadataSource(BaseModel):
    file_path: str = Field(description="File path used as a metadata source.")
    source_type: str = Field(
        description="Kind of source, for example README, instrument export, table header, filename, or PDF report."
    )
    description: str = Field(description="What metadata this source contributes.")
    extracted_fields: list[str] = Field(
        default_factory=list,
        description="InitialContext fields or metadata facts supported by this source.",
    )
    evidence: str | None = Field(
        default=None,
        description="Short textual evidence from the file or file path.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this source interpretation from 0 to 1.",
    )


class InitialContext(BaseModel):
    device_name: str | None = None
    device_model: str | None = None
    entities_analyzed: list[str] = Field(
        default_factory=list,
        description="Sample IDs, compound names, or specimen identifiers",
    )
    analytical_technique: str | None = None
    file_relationships: list[FileRelationship] = Field(default_factory=list)
    metadata_sources: list[MetadataSource] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    summary: str


class PatchCandidate(BaseModel):
    """A field-level merge-patch candidate with provenance metadata."""

    field_path: str = Field(
        description=(
            "Top-level Dataset property this candidate targets, "
            "e.g. 'was_generated_by' or 'keyword'."
        ),
    )
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Merge-patch scoped to this field. Must be a valid partial "
            "update that can be merged at the top level of the draft."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this patch is correct and relevant (0.0–1.0).",
    )
    reasoning: str = Field(
        description="Brief justification for why this patch is warranted.",
    )
    source_evidence: list[str] = Field(
        default_factory=list,
        description="Relevant text snippets from the chunks supporting this patch.",
    )


class FieldPatchResult(BaseModel):
    """Agent output: a list of field-level patch candidates."""

    candidates: list[PatchCandidate] = Field(
        default_factory=list,
        description="Field-level patch candidates extracted from the chunk batch.",
    )
