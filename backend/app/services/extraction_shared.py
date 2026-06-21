from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass

import jsonpatch
from pydantic import ValidationError
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import Settings
from app.core.logging import logger
from app.core.task_registry import TaskRegistry, TaskStatus, TaskType
from app.domain.datasources import ContentChunk, FileType
from app.domain.extraction import (
    DEFAULT_QUALITATIVE_VOCAB_IDENTIFIERS,
    EVIDENCE_CONTEXT_SYSTEM_PROMPT,
    EVIDENCE_CRITIC_SYSTEM_PROMPT,
    EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT,
    EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT,
    EXTRACTION_OVERVIEW_SYSTEM_PROMPT,
    QUDT_QUANTITY_KIND_VOCAB,
    QUDT_QUANTITY_KIND_URI,
    QUDT_QUANTITY_URI,
    QUDT_SCHEMA_VOCAB,
    QUDT_UNIT_VOCAB,
    QUDT_UNIT_URI,
    VOCAB_CANDIDATE_SELECTION_SYSTEM_PROMPT,
    VOCAB_FALLBACK_QUERY_SYSTEM_PROMPT,
    VOCAB_QUERY_FORMULATION_SYSTEM_PROMPT,
    VOCAB_OBJECT_GROUNDING_SELECTION_SYSTEM_PROMPT,
    ChunkingRequiredError,
    CompleteWorkflowProgress,
    CompleteWorkflowStepProgress,
    CurationLedgerRecord,
    DefinedTerm,
    DocumentQualityState,
    DraftQualityState,
    DraftValidationResult,
    EVIDENCE_INSTANCE_BUILDER_SYSTEM_PROMPT,
    EVIDENCE_INSTANCE_REPAIR_SYSTEM_PROMPT,
    EVIDENCE_NOVELTY_EVALUATOR_SYSTEM_PROMPT,
    MEASUREMENT_SEMANTIC_ROUTER_SYSTEM_PROMPT,
    EvidenceNoveltyDecision,
    MeasurementSemanticRouteDecision,
    DCAT_AP_PLUS_COVERAGE_REQUIREMENTS,
    DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS,
    DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS,
    CoverageReport,
    REQUIREMENT_EVALUATOR_SYSTEM_PROMPT,
    REQUIREMENT_PATCH_SYSTEM_PROMPT,
    SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT,
    SEMANTIC_RECONSTRUCTION_SYSTEM_PROMPT,
    SEMANTIC_SYNTHESIS_SYSTEM_PROMPT,
    DcatRequirement,
    RequirementEvaluation,
    RequirementEvidenceItem,
    RequirementPatchAttempt,
    RequirementPatchResult,
    SchemaConstrainedPatchResult,
    SchemaConstrainedWrite,
    SchemaConstrainedPatchRoute,
    LegacyEvidencePatchRoute,
    SemanticReconstructionDefect,
    SemanticReconstructionDiagnosis,
    SemanticReconstructionPatchResult,
    SemanticReconstructionRecord,
    SourceTraceReport,
    RequirementReportItem,
    apply_evidence_instance,
    apply_schema_constrained_writes,
    assessment_for_report_item,
    build_requirement_report,
    build_context_window_for_note,
    build_instance_builder_prompt,
    build_instance_repair_prompt,
    build_measurement_semantic_route_prompt,
    build_novelty_evaluator_prompt,
    build_requirement_evaluation_prompt,
    build_requirement_patch_prompt,
    build_semantic_diagnosis_prompt,
    build_semantic_reconstruction_prompt,
    compact_requirement_evidence,
    builder_output_model_for_target,
    merge_requirement_assessment,
    normalized_requirement_evaluation,
    report_items_from_evaluation,
    route_evidence_note_to_target,
    parse_schema_constrained_patch_result,
    schema_for_json_pointer,
    compute_coverage_report,
    compute_source_trace_report,
    score_requirement_report,
    score_requirement_items,
    semantic_diagnosis_output_schema,
    select_requirement_evidence_packet,
    stable_evidence_id,
    ChunkRepairMode,
    EvidenceAssessment,
    EvidenceAssessmentContext,
    EvidenceCandidate,
    EvidenceCriticGranularity,
    EvidenceChunkContext,
    EvidenceChunkMetadata,
    EvidenceContext,
    EvidenceQueryLedgerEntry,
    FilteredEvidenceNote,
    FileInventoryItem,
    RoutedEvidenceContext,
    ExtractionChunkRef,
    ExtractionChunkResult,
    ExtractionContext,
    ExtractionFileContentWindow,
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewEdge,
    ExtractionOverviewFilePreview,
    ExtractionOverviewInspectedFile,
    ExtractionOverviewModelOutput,
    ExtractionOverviewNode,
    ExtractionOverviewStatus,
    InitialFileSummaryDiagnosticRecord,
    InitialFileSummaryDiagnostics,
    InitialFileSummaryProgress,
    InitialOverviewFailureDiagnostic,
    InitialOverviewPromptDiagnostic,
    ExtractionNormalization,
    ExtractionResultNotFoundError,
    ExtractionRunProgress,
    ExtractionRunResult,
    ExtractionRunState,
    ExtractionValidationError,
    ExtractionVocabQueryConfig,
    ExtractionVocabQueryRecord,
    FileContext,
    FileRankingResult,
    FieldCompletionLedgerRecord,
    GroundedExtractionObject,
    GroundingRolePolicy,
    ProfileFieldNormalization,
    ProfileObjectPatchResult,
    PromptTokenBudgeter,
    SchemaBranch,
    DATASET_LEVEL_PROJECTION_SYSTEM_PROMPT,
    DATASET_SUMMARY_SYSTEM_PROMPT,
    DatasetSummaryProjection,
    ShallowDatasetLevelProjection,
    ProjectionLedgerRecord,
    QualityIssue,
    QualitativeAttribute,
    QualitativeAttributeNormalization,
    QuantitativeAttribute,
    QuantityNormalization,
    Resource,
    TracedExtractionObject,
    VocabularyCandidateSelection,
    VocabularyFallbackQuery,
    VocabularyQueryFormulation,
    VocabularyTermMapping,
    build_dataset_level_projection_prompt_components,
    build_dataset_summary_prompt_components,
    build_evidence_critic_prompt_components,
    build_evidence_context_prompt,
    build_evidence_context_prompt_components,
    build_evidence_system_prompt_components_with_overview,
    build_evidence_system_prompt_with_overview,
    dedupe_repeated_evidence_notes,
    build_candidate_selection_prompt,
    build_candidate_selection_prompt_components,
    build_extraction_file_summary_prompt,
    build_extraction_overview_fallback_prompt,
    build_extraction_overview_prompt_components,
    build_extraction_overview_prompt,
    compact_seeded_overview_for_prompt,
    filtered_evidence_ledger,
    is_noisy_payload_chunk,
    build_fallback_query_prompt,
    build_fallback_query_prompt_components,
    build_query_formulation_prompt,
    build_query_formulation_prompt_components,
    build_object_grounding_selection_prompt,
    build_object_grounding_selection_prompt_components,
    ShallowDatasetProjection,
    overview_projection_record,
    overview_projection_repair_record,
    projection_stage_record,
    shallow_projection_to_dcat_document,
    shallow_required_skeleton,
    build_schema_branch_index,
    build_schema_search_query,
    build_qualitative_vocab_query,
    build_quantity_kind_vocab_query,
    build_unit_vocab_query,
    fallback_file_ranking,
    rank_summarized_files,
    merge_evidence_contexts,
    normalize_chunk_text_for_evidence_prompt,
    route_evidence_candidates,
    schema_branches_to_catalog,
    search_schema_branches,
    validate_evidence_candidates,
)
from app.domain.profiles import (
    ProfileValidationIssue,
    remove_null_values,
    validate_document_against_profile,
    validation_schema_for_target_class,
)
from app.domain.semantics import VocabQuery, VocabQueryResult
from app.ollama.completion import generate_structured, generate_text, repair_structured_output
from app.ollama.errors import CompletionError, MaxRetriesExceeded
from app.ollama.prompt_diagnostics import PromptCompletionDiagnostics, run_usage_to_dict
from app.ollama.usage import RunUsage
from app.repositories.extraction_output_repository import ExtractionOutputRepository

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper
    from app.services.datasource_service import DataSourceService
    from app.services.profile_service import ProfileService
    from app.services.semantic_service import SemanticService


ExtractionTargetStage = Literal["context", "profile", "grounding", "complete"]
INITIAL_OVERVIEW_TOP_FILE_LIMIT = 16
INITIAL_OVERVIEW_PREVIEW_LINE_LIMIT = 80
INITIAL_OVERVIEW_MAX_LINE_CHARS = 500
INITIAL_OVERVIEW_MIN_INPUT_TOKENS = 1200
INITIAL_OVERVIEW_EXPECTED_OUTPUT_TOKENS = 1000
INITIAL_OVERVIEW_MAX_OUTPUT_TOKENS = 1200
INITIAL_OVERVIEW_INPUT_SAFETY_MARGIN_TOKENS = 250
INITIAL_OVERVIEW_RANKED_FILE_BUDGET_RATIO = 0.05
INITIAL_OVERVIEW_SUMMARY_BUDGET_RATIO = 0.45
INITIAL_OVERVIEW_GRAPH_BUDGET_RATIO = 0.22
INITIAL_OVERVIEW_PREVIEW_BUDGET_RATIO = 0.20
INITIAL_OVERVIEW_SUMMARY_TOKEN_BUDGET = 80
INITIAL_OVERVIEW_FAILURE_EXCERPT_TOKENS = 300
INITIAL_FILE_SUMMARY_CONTEXT_RATIO = 0.35
ESTIMATED_CHARS_PER_TOKEN = 4
OVERVIEW_SUMMARY_LIST_LIMITS = {
    "purpose_evidence": 2,
    "metadata_signals": 3,
    "instrument_or_software_terms_and_settings": 4,
    "quantitative_signals": 4,
}
OVERVIEW_SUMMARY_REDUCTION_ORDER = (
    "quantitative_signals",
    "metadata_signals",
    "purpose_evidence",
    "instrument_or_software_terms_and_settings",
)
INITIAL_OVERVIEW_WEAK_EDGE_EVIDENCE = {
    "",
    "rank",
    "ranked",
    "listed",
    "top file",
    "top ranked",
    "ranked file",
    "ranked files",
    "file list",
    "listed file",
    "listed files",
}


@dataclass
class _InitialOverviewGroupRule:
    node_id: str
    label: str
    summary: str
    relation: str
    keywords: tuple[str, ...]


INITIAL_OVERVIEW_GROUP_RULES = (
    _InitialOverviewGroupRule(
        node_id="group:dataset_documentation",
        label="Dataset documentation",
        summary="Files that explicitly document the package or dataset.",
        relation="documents",
        keywords=(
            "dataset description",
            "dataset documentation",
            "data descriptor",
            "human-readable",
        ),
    ),
    _InitialOverviewGroupRule(
        node_id="group:audit_provenance",
        label="Audit and provenance",
        summary="Audit, provenance, log, history, and integrity-check resources.",
        relation="documents",
        keywords=("audit", "provenance", "trail", "log", "history", "hash"),
    ),
    _InitialOverviewGroupRule(
        node_id="group:acquisition_settings",
        label="Acquisition settings",
        summary="Configuration and parameter files for data acquisition.",
        relation="parameterizes",
        keywords=(
            "acquisition",
            "acquire",
            "measurement setting",
            "acquisition parameter",
        ),
    ),
    _InitialOverviewGroupRule(
        node_id="group:processing_settings",
        label="Processing settings",
        summary="Configuration and parameter files for data processing.",
        relation="parameterizes",
        keywords=(
            "processing",
            "process parameter",
            "processing parameter",
            "processed parameter",
        ),
    ),
    _InitialOverviewGroupRule(
        node_id="group:instrument_settings",
        label="Instrument settings",
        summary="Instrument tuning, calibration, reference, or setting resources.",
        relation="parameterizes",
        keywords=("instrument settings", "calibration", "reference", "settings"),
    ),
    _InitialOverviewGroupRule(
        node_id="group:method_program",
        label="Method or program logic",
        summary="Executable or declarative method/program logic for the experiment.",
        relation="configures",
        keywords=(
            "method program",
            "experiment program",
            "method logic",
            "protocol",
            "workflow",
        ),
    ),
    _InitialOverviewGroupRule(
        node_id="group:raw_data",
        label="Raw data",
        summary="Primary or raw measurement data resources.",
        relation="describes",
        keywords=("raw data", "raw measurements", "primary measurements"),
    ),
    _InitialOverviewGroupRule(
        node_id="group:processed_data",
        label="Processed data",
        summary="Processed data or transformed measurement output resources.",
        relation="describes",
        keywords=("processed data", "processed spectrum", "processed output"),
    ),
    _InitialOverviewGroupRule(
        node_id="group:derived_results",
        label="Derived results",
        summary="Derived result, peak, annotation, or result-table resources.",
        relation="describes",
        keywords=("derived result", "peak list", "result table", "annotation"),
    ),
    _InitialOverviewGroupRule(
        node_id="group:parameter_settings",
        label="Parameter settings",
        summary="Parameter/configuration resources whose role is not more specific.",
        relation="parameterizes",
        keywords=(
            "parameter file",
            "parameter values",
            "configuration",
            "settings",
            "key-value pairs",
            "parameters",
        ),
    ),
    _InitialOverviewGroupRule(
        node_id="group:ambiguous_supporting_resources",
        label="Ambiguous supporting resources",
        summary="Resources with unclear purpose that may still orient extraction.",
        relation="uncertain_relation",
        keywords=(
            "unclear",
            "not clear",
            "purpose cannot",
            "empty content",
            "arbitrary units",
        ),
    ),
)

INITIAL_OVERVIEW_SPECIFIC_SETTING_GROUPS = {
    "group:acquisition_settings",
    "group:processing_settings",
    "group:instrument_settings",
    "group:method_program",
}


@dataclass
class _QuantityCandidateDiscovery:
    quantity: QuantitativeAttribute
    quantity_kind_query_id: str
    unit_query_id: str


@dataclass
class _QualitativeCandidateDiscovery:
    attribute: QualitativeAttribute
    query_ids: list[str]


@dataclass
class _ObjectGroundingCandidateDiscovery:
    object_identifier: str
    object_kind: str
    raw_type: str
    source_context: dict[str, Any]
    query_ids: list[str]


@dataclass
class _ProfileFieldCandidateDiscovery:
    json_path: str
    field_name: str
    source_value: str
    vocabulary_identifier: str
    query_ids: list[str]
    role: str = ""


@dataclass
class _EvidenceProjectionGroup:
    group_id: str
    object_kind: str
    notes: list[EvidenceCandidate]
    target_hint: str
    target_class_hint: str | None


@dataclass
class _QuantitativeEvidenceGroup:
    group_id: str
    label: str
    value: float
    unit: str | None
    notes: list[Any]


def _resource_title(properties: dict[str, Any]) -> str | None:
    label_keys = (
        "label",
        "prefLabel",
        "skos__prefLabel",
        "preferred_label",
        "title",
        "skos__definition",
        "definition",
        "name",
        "symbol",
        "ucumCode",
    )
    for key in label_keys:
        value = properties.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return None


__all__ = [name for name in globals() if not name.startswith("__")]


