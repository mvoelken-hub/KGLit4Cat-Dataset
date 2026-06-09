from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ReferenceObject(BaseModel):
    object_type: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    required: bool = True


class ReferenceAttribute(BaseModel):
    title: str
    value: str
    object_type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    required: bool = True


class ReferenceVocabMapping(BaseModel):
    source_value: str
    mapping_type: Literal["quantity_kind", "unit", "qualitative"]
    acceptable_uris: list[str] = Field(default_factory=list)
    expected_unresolved: bool = False


class EvaluationReference(BaseModel):
    dataset_filename: str
    package_id: str | None = None
    description: str = ""
    relevant_files: list[str] = Field(default_factory=list)
    expected_objects: list[ReferenceObject] = Field(default_factory=list)
    expected_attributes: list[ReferenceAttribute] = Field(default_factory=list)
    expected_vocab_mappings: list[ReferenceVocabMapping] = Field(default_factory=list)
    required_profile_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


class AttributeMatch(BaseModel):
    expected: str
    matched: str | None = None
    status: Literal["matched", "missing"] = "missing"


class EvaluationRunManifest(BaseModel):
    dataset_filename: str
    package_id: str
    profile_identifier: str | None = None
    chat_model: str | None = None
    embedding_model: str | None = None
    max_context_length: int | None = None
    qualitative_vocab_identifiers: list[str] = Field(default_factory=list)
    chunking: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class EvaluationReport(BaseModel):
    reference: EvaluationReference
    manifest: EvaluationRunManifest | None = None
    schema_valid: bool
    file_ranking_top1_hit: bool
    file_ranking_top3_recall: float
    object_metrics: dict[str, MetricSummary]
    attribute_metrics: MetricSummary
    source_trace_coverage: float
    vocab_mapping_metrics: MetricSummary
    required_profile_field_coverage: float
    warnings_count: int
    token_usage: dict[str, Any] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    attribute_matches: list[AttributeMatch] = Field(default_factory=list)


class PartialEvaluationReport(BaseModel):
    dataset_filename: str
    package_id: str
    manifest: EvaluationRunManifest | None = None
    outcome: Literal["missing_output", "timeout", "crashed"] = "missing_output"
    status: str = "unknown"
    stage: str | None = None
    processed_chunks: int = 0
    total_chunks: int = 0
    completion_fraction: float = 0.0
    warnings_count: int = 0
    message: str = ""
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def load_reference(path: Path) -> EvaluationReference:
    return EvaluationReference.model_validate_json(path.read_text(encoding="utf-8"))
