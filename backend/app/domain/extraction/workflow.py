from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.task_registry import TaskStatus
from app.domain.extraction.evidence_context import (
    EvidenceCriticGranularity,
    FilteredEvidenceNote,
    RoutedEvidenceContext,
)
from app.domain.extraction.requirement_enrichment import RequirementReport
from app.domain.extraction.vocabulary import ExtractionNormalization
from app.domain.extraction.file_ranking import RankedFile
from app.domain.extraction.overview import (
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewStatus,
    InitialFileSummaryProgress,
    InitialFileSummaryStatus,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
)
from app.domain.profiles import ProfileValidationIssue
from app.domain.semantics import VocabQuery, VocabQueryResult


class ChunkingRequiredError(Exception):
    pass


class ExtractionResultNotFoundError(Exception):
    pass


class ExtractionValidationError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("Projected extraction result did not satisfy the selected profile.")
        self.errors = errors


class ExtractionChunkRef(BaseModel):
    chunk_index: int = Field(..., ge=0)
    file_path: str
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)


class ExtractionVocabQueryConfig(BaseModel):
    qualitative_vocab_identifiers: list[str] = Field(default_factory=list)
    vector_top_k: int = Field(default=12, ge=1)
    fulltext_top_k: int = Field(default=12, ge=1)
    seed_top_k: int = Field(default=6, ge=1)
    max_hops: int = Field(default=1, ge=0)
    max_statements_per_seed: int = Field(default=12, ge=1)
    traversal_direction: str = "undirected"
    vector_weight: float = Field(default=1.0, gt=0)
    fulltext_weight: float = Field(default=1.0, gt=0)
    rrf_k: int = Field(default=60, ge=1)
    quantitative_vector_top_k: int = Field(default=12, ge=1)
    quantitative_fulltext_top_k: int = Field(default=12, ge=1)
    quantitative_seed_top_k: int = Field(default=6, ge=1)
    quantitative_max_hops: int = Field(default=0, ge=0)
    quantitative_max_statements_per_seed: int = Field(default=12, ge=1)
    quantitative_traversal_direction: str = "undirected"
    quantitative_vector_weight: float = Field(default=1.0, gt=0)
    quantitative_fulltext_weight: float = Field(default=1.0, gt=0)
    quantitative_rrf_k: int = Field(default=60, ge=1)


class ExtractionVocabQueryRecord(BaseModel):
    query_id: str
    kind: str
    source_value: str
    source_context: dict[str, Any] = Field(default_factory=dict)
    vocabulary_identifier: str
    rdf_type: str
    query: VocabQuery
    status: str = Field("pending", pattern="^(pending|running|completed|failed)$")
    result: VocabQueryResult | None = None
    error: str | None = None
    duration_ms: float | None = None


class ExtractionChunkResult(ExtractionChunkRef):
    status: str = Field("pending", pattern="^(pending|running|repair_pending|completed|failed|skipped)$")
    evidence_context: RoutedEvidenceContext | None = None
    skip_reason: str | None = None
    error: str | None = None
    response_duration_ms: float | None = None
    context_tokens: int | None = None


DraftQualityState = Literal[
    "complete_final_draft",
    "imperfect_final_draft",
    "empty_profile_shell",
]
DraftValidationStatus = Literal["valid", "invalid", "not_run"]
ProjectionLedgerStatus = Literal[
    "pending",
    "running",
    "projected",
    "not_projected",
    "ambiguous",
    "user_edit_required",
]
FieldValidationStatus = Literal["valid", "invalid", "missing", "not_run"]
FieldEnrichmentStatus = Literal[
    "grounded",
    "not_grounded",
    "no_candidate",
    "ambiguous",
    "user_selected_vocab_term",
    "intentionally_unresolved",
]
CurationLedgerStatus = Literal[
    "unchanged",
    "user_modified",
    "user_removed",
    "user_selected_vocab_term",
    "intentionally_unresolved",
]
ChunkRepairMode = Literal["deferred", "immediate", "disabled"]


class DraftValidationResult(BaseModel):
    status: DraftValidationStatus = "not_run"
    errors: list[ProfileValidationIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class QualityIssue(BaseModel):
    code: str
    severity: Literal["blocking", "warning", "info"] = "warning"
    message: str
    requirement_id: str | None = None
    path: str | None = None


class DocumentQualityState(BaseModel):
    schema_valid: bool
    profile_conformant: bool | None = None
    evidence_grounded: bool | None = None
    semantic_valid: bool | None = None
    metadata_completeness_score: float | None = None
    operational_access_score: float | None = None
    fair_assessment: dict[str, Any] | None = None
    blocking_issues: list[QualityIssue] = Field(default_factory=list)
    warnings: list[QualityIssue] = Field(default_factory=list)


class EvidenceQueryLedgerEntry(BaseModel):
    query_id: str
    requirement_id: str
    target_path: str = ""
    query: dict[str, Any] = Field(default_factory=dict)
    result_evidence_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    rejected_result_reasons: dict[str, str] = Field(default_factory=dict)
    ranking_explanation: list[str] = Field(default_factory=list)


class ProjectionLedgerRecord(BaseModel):
    object_identifier: str
    object_kind: str
    source_evidence: str | None = None
    evidence_note_identifiers: list[str] = Field(default_factory=list)
    status: ProjectionLedgerStatus = "not_projected"
    projected_paths: list[str] = Field(default_factory=list)
    target_path: str | None = None
    target_class: str | None = None
    planner_status: str | None = None
    planner_reason: str | None = None
    evidence_quality: dict[str, Any] = Field(default_factory=dict)
    schema_queries: list[dict[str, Any]] = Field(default_factory=list)
    candidate_paths: list[str] = Field(default_factory=list)
    selected_schema_branch: dict[str, Any] | None = None
    merge_status: str | None = None
    reason: str = ""
    error: str | None = None


class FieldCompletionLedgerRecord(BaseModel):
    json_path: str
    field_name: str
    generated_value: Any = None
    curated_value: Any = None
    source_evidence: list[str] = Field(default_factory=list)
    validation_status: FieldValidationStatus = "not_run"
    enrichment_status: FieldEnrichmentStatus = "not_grounded"
    issue_categories: list[str] = Field(default_factory=list)
    edit_needed_reason: str = ""


class CurationLedgerRecord(BaseModel):
    json_path: str
    field_name: str
    generated_value: Any = None
    curated_value: Any = None
    source_evidence: list[str] = Field(default_factory=list)
    status: CurationLedgerStatus = "unchanged"
    reason: str = ""


class ExtractionRunState(BaseModel):
    profile_identifier: str | None = None
    chunking_strategy: str = "semantic"
    chunk_repair_mode: ChunkRepairMode = "deferred"
    evidence_critic_granularity: EvidenceCriticGranularity = "per_chunk"
    vocab_query_config: ExtractionVocabQueryConfig = Field(default_factory=ExtractionVocabQueryConfig)
    chat_model: str | None = None
    ranked_files: list[RankedFile] = Field(default_factory=list)
    initial_file_summaries: list[ExtractionFileSummary] = Field(default_factory=list)
    initial_file_summary_progress: InitialFileSummaryProgress | None = None
    initial_file_summary_status: InitialFileSummaryStatus | None = None
    initial_extraction_overview: ExtractionOverview | None = None
    initial_extraction_overview_status: ExtractionOverviewStatus | None = None
    initial_extraction_overview_diagnostic: (
        InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None
    ) = None
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)
    vocab_queries: list[ExtractionVocabQueryRecord] = Field(default_factory=list)
    generated_final_draft: dict[str, Any] | None = None
    curated_document: dict[str, Any] | None = None
    generated_initial_draft: dict[str, Any] | None = None
    requirement_report: RequirementReport | None = None
    document_quality_state: DocumentQualityState | None = None
    draft_quality_state: DraftQualityState | None = None
    validation: DraftValidationResult = Field(default_factory=DraftValidationResult)
    curated_validation: DraftValidationResult | None = None
    initial_draft_scaffold: dict[str, Any] = Field(default_factory=dict)
    projection_ledger: list[ProjectionLedgerRecord] = Field(default_factory=list)
    field_completion_ledger: list[FieldCompletionLedgerRecord] = Field(default_factory=list)
    evidence_query_ledger: list[EvidenceQueryLedgerEntry] = Field(default_factory=list)
    curation_ledger: list[CurationLedgerRecord] = Field(default_factory=list)
    filtered_evidence_notes: list[FilteredEvidenceNote] = Field(default_factory=list)


class ExtractionRunProgress(BaseModel):
    stage: str = "pending"
    chunk_repair_mode: ChunkRepairMode = "deferred"
    evidence_critic_granularity: EvidenceCriticGranularity = "per_chunk"
    processed_chunks: int = 0
    total_chunks: int = 0
    normalized_quantities: int = 0
    normalized_qualitative_attributes: int = 0
    interim_evidence_context: RoutedEvidenceContext | None = None
    vocab_query_config: ExtractionVocabQueryConfig = Field(default_factory=ExtractionVocabQueryConfig)
    ranked_files: list[RankedFile] = Field(default_factory=list)
    initial_file_summaries: list[ExtractionFileSummary] = Field(default_factory=list)
    initial_file_summary_progress: InitialFileSummaryProgress | None = None
    initial_file_summary_status: InitialFileSummaryStatus | None = None
    initial_extraction_overview: ExtractionOverview | None = None
    initial_extraction_overview_status: ExtractionOverviewStatus | None = None
    initial_extraction_overview_diagnostic: (
        InitialOverviewPromptDiagnostic | InitialOverviewFailureDiagnostic | None
    ) = None
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)
    vocab_queries: list[ExtractionVocabQueryRecord] = Field(default_factory=list)
    generated_final_draft: dict[str, Any] | None = None
    curated_document: dict[str, Any] | None = None
    generated_initial_draft: dict[str, Any] | None = None
    requirement_report: RequirementReport | None = None
    document_quality_state: DocumentQualityState | None = None
    draft_quality_state: DraftQualityState | None = None
    validation: DraftValidationResult = Field(default_factory=DraftValidationResult)
    curated_validation: DraftValidationResult | None = None
    initial_draft_scaffold: dict[str, Any] = Field(default_factory=dict)
    projection_ledger: list[ProjectionLedgerRecord] = Field(default_factory=list)
    field_completion_ledger: list[FieldCompletionLedgerRecord] = Field(default_factory=list)
    evidence_query_ledger: list[EvidenceQueryLedgerEntry] = Field(default_factory=list)
    curation_ledger: list[CurationLedgerRecord] = Field(default_factory=list)
    current_chunk: ExtractionChunkRef | None = None
    warnings: list[str] = Field(default_factory=list)


class CompleteWorkflowStepProgress(BaseModel):
    name: str
    status: TaskStatus = TaskStatus.UNKNOWN


class CompleteWorkflowProgress(BaseModel):
    stage: str = "pending"
    data_package_id: str
    profile_identifier: str | None = None
    steps: list[CompleteWorkflowStepProgress] = Field(default_factory=list)
    chunking_status: TaskStatus = TaskStatus.UNKNOWN
    extraction_status: TaskStatus = TaskStatus.UNKNOWN
    extraction_progress: ExtractionRunProgress | None = None
    warnings: list[str] = Field(default_factory=list)
    result_url: str | None = None


class ExtractionRunResult(BaseModel):
    generated_final_draft: dict[str, Any]
    machine_evidence_context: RoutedEvidenceContext
    generated_initial_draft: dict[str, Any] | None = None
    requirement_report: RequirementReport | None = None
    initial_file_summaries: list[ExtractionFileSummary] = Field(default_factory=list)
    initial_file_summary_status: InitialFileSummaryStatus | None = None
    initial_extraction_overview: ExtractionOverview | None = None
    initial_extraction_overview_status: ExtractionOverviewStatus | None = None
    curated_document: dict[str, Any] | None = None
    document_quality_state: DocumentQualityState | None = None
    draft_quality_state: DraftQualityState
    validation: DraftValidationResult
    curated_validation: DraftValidationResult | None = None
    initial_draft_scaffold: dict[str, Any] = Field(default_factory=dict)
    projection_ledger: list[ProjectionLedgerRecord] = Field(default_factory=list)
    field_completion_ledger: list[FieldCompletionLedgerRecord] = Field(default_factory=list)
    evidence_query_ledger: list[EvidenceQueryLedgerEntry] = Field(default_factory=list)
    curation_ledger: list[CurationLedgerRecord] = Field(default_factory=list)
    chat_model: str | None = None
    normalization: ExtractionNormalization | None = None
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, Any] = Field(default_factory=dict)
