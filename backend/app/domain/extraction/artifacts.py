from typing import Any, Literal

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


class ContextEntity(BaseModel):
    label: str = Field(
        description=(
            "Human-readable sample, compound, specimen, or other evaluated "
            "entity label suitable for Dataset.is_about_entity."
        )
    )
    role: Literal["sample", "compound", "specimen", "unknown"] = Field(
        default="unknown",
        description="Best compact role for the entity; use unknown when the source does not make the role clear.",
    )
    identifier: str | None = Field(
        default=None,
        description="Source-provided identifier for the entity when available; do not invent global URIs.",
    )
    evidence: str | None = Field(
        default=None,
        description="Short source text or filename evidence supporting this entity.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this entity extraction from 0 to 1.",
    )


class ContextAgent(BaseModel):
    name: str = Field(
        description=(
            "Name of a person, organization, instrument, or software that "
            "carried out or enabled data generation."
        )
    )
    role: Literal["instrument", "software", "organization", "person", "unknown"] = Field(
        default="unknown",
        description=(
            "Compact agent role; use instrument for devices and software for "
            "acquisition or processing software."
        ),
    )
    model: str | None = Field(
        default=None,
        description="Device model, software version, or similar model/version label when stated by the source.",
    )
    evidence: str | None = Field(
        default=None,
        description="Short source text or filename evidence supporting this agent.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this agent extraction from 0 to 1.",
    )


class ContextActivity(BaseModel):
    label: str | None = Field(
        default=None,
        description="Short label for the high-level data-generating activity, usually one root activity.",
    )
    technique: str | None = Field(
        default=None,
        description=(
            "Analytical or data-generating technique used by this activity, "
            "suitable for Dataset.was_generated_by."
        ),
    )
    agent_names: list[str] = Field(
        default_factory=list,
        description="Names of ContextAgent entries associated with this activity.",
    )
    evidence: str | None = Field(
        default=None,
        description="Short source text or filename evidence supporting this activity.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this activity extraction from 0 to 1.",
    )


class InitialContext(BaseModel):
    dataset_title: str | None = Field(
        default=None,
        description=(
            "Human-readable dataset title intended for Dataset.title; infer "
            "only from explicit metadata, filenames, or strong source context."
        ),
    )
    dataset_description: str | None = Field(
        default=None,
        description=(
            "Dataset.description-ready description of what the dataset contains; "
            "do not include extraction-process notes."
        ),
    )
    entities: list[ContextEntity] = Field(
        default_factory=list,
        description="Compact evaluated entities the dataset is about, such as samples, compounds, or specimens.",
    )
    agents: list[ContextAgent] = Field(
        default_factory=list,
        description=(
            "Compact agents involved in data generation, including instruments, "
            "software, people, or organizations."
        ),
    )
    activities: list[ContextActivity] = Field(
        default_factory=list,
        description=(
            "Compact high-level data-generating activities; prefer one root "
            "activity when the source supports it."
        ),
    )
    file_relationships: list[FileRelationship] = Field(default_factory=list)
    metadata_sources: list[MetadataSource] = Field(default_factory=list)
    keywords: list[str] = Field(
        default_factory=list,
        description="Concise dataset keywords or tags supported by source evidence.",
    )
    summary: str = Field(
        description=(
            "Short extraction summary for UI/debugging; not necessarily copied "
            "to Dataset.description when dataset_description is available."
        )
    )


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
