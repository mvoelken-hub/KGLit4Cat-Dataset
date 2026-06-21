import { apiBaseUrl, buildQuery, readJson } from './client';
import type { ChunkingStrategy } from './datasources';
import type { InitialContext } from './types';

export type WorkflowTaskStatus = 'unknown' | 'running' | 'completed' | 'cancelled' | 'crashed';
export type ExtractionTargetStage = 'context' | 'profile' | 'grounding' | 'complete';
export type ChunkRepairMode = 'deferred' | 'immediate' | 'disabled';
export type EvidenceCriticGranularity = 'per_chunk' | 'per_candidate' | 'disabled';
export type EvidenceRoute = 'portable_evidence' | 'contextual_evidence' | 'rejected_evidence';

export type EvidenceCandidate = {
  candidate_id: string;
  category: string;
  role: string;
  claim: string;
  evidence_text: string;
  source_context: string;
  uncertainty: string;
  file_path: string;
  start_idx: number;
  end_idx: number;
  evidence_match_score: number;
};

export type EvidenceAssessment = {
  candidate_id: string;
  groundedness: string;
  self_containedness: string;
  scope_clarity: string;
  portability: string;
  semantic_interpretability: string;
  environment_dependence: string;
  specificity: string;
  novelty: string;
  uncertainty: string;
  rationale: string;
};

export type RoutedEvidenceRecord = {
  route: EvidenceRoute;
  reason: string;
  candidate: EvidenceCandidate;
  assessment?: EvidenceAssessment | null;
  chunk_index?: number | null;
};

export type RoutedEvidenceContext = {
  portable_evidence: EvidenceCandidate[];
  contextual_evidence: EvidenceCandidate[];
  rejected_evidence: RoutedEvidenceRecord[];
  assessments: EvidenceAssessment[];
  file_inventory: Array<Record<string, unknown>>;
};

export type WorkflowTokenUsageEntry = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  requests: number;
  operation_count?: number;
  patch_count?: number;
  response_duration_ms?: number;
  total_duration_ms?: number;
  average_input_tokens_per_operation?: number;
  average_output_tokens_per_operation?: number;
  average_total_tokens_per_operation?: number;
  average_response_duration_ms_per_operation?: number;
  average_total_duration_ms_per_operation?: number;
  average_input_tokens_per_patch?: number;
  average_output_tokens_per_patch?: number;
  average_total_tokens_per_patch?: number;
  average_input_tokens_per_request?: number;
  average_output_tokens_per_request?: number;
  average_total_tokens_per_request?: number;
  average_response_duration_ms_per_request?: number;
  average_total_duration_ms_per_request?: number;
};

export type WorkflowTokenUsage = {
  agents?: Record<string, WorkflowTokenUsageEntry>;
  combined?: WorkflowTokenUsageEntry;
};

export type ExtractionRunProgress = {
  stage: string;
  chunk_repair_mode?: ChunkRepairMode;
  evidence_critic_granularity?: EvidenceCriticGranularity;
  processed_chunks: number;
  total_chunks: number;
  normalized_quantities: number;
  normalized_qualitative_attributes: number;
  interim_evidence_context?: RoutedEvidenceContext | null;
  generated_final_draft?: Record<string, unknown> | null;
  curated_document?: Record<string, unknown> | null;
  generated_initial_draft?: Record<string, unknown> | null;
  generated_patched_draft?: Record<string, unknown> | null;
  generated_reconstructed_draft?: Record<string, unknown> | null;
  requirement_report?: RequirementReport | null;
  document_quality_state?: DocumentQualityState | null;
  draft_quality_state?: DraftQualityState | null;
  validation?: DraftValidationResult | null;
  curated_validation?: DraftValidationResult | null;
  initial_draft_scaffold?: InitialDraftScaffold;
  projection_ledger?: ProjectionLedgerRecord[];
  field_completion_ledger?: FieldCompletionLedgerRecord[];
  evidence_query_ledger?: EvidenceQueryLedgerEntry[];
  curation_ledger?: CurationLedgerRecord[];
  initial_file_summaries?: ExtractionFileSummary[];
  initial_file_summary_progress?: InitialFileSummaryProgress | null;
  initial_file_summary_status?: InitialFileSummaryStatus | null;
  initial_extraction_overview?: ExtractionOverview | null;
  initial_extraction_overview_status?: ExtractionOverviewStatus | null;
  initial_extraction_overview_diagnostic?: InitialOverviewDiagnostic | null;
  dataset_summary?: string;
  vocab_query_config?: ExtractionVocabQueryConfig;
  vocab_queries?: ExtractionVocabQueryRecord[];
  ranked_files?: RankedExtractionFile[];
  chunk_results?: ExtractionChunkResult[];
  current_chunk?: ExtractionChunkRef | null;
  warnings: string[];
};

export type RequirementStatus = 'fulfilled' | 'partial' | 'missing' | 'not_applicable';

export type RequirementEvidenceItem = {
  evidence_id?: string;
  candidate_id: string;
  category: string;
  role?: string;
  claim: string;
  evidence_text: string;
  source_context?: string;
  file_path?: string;
  start_idx?: number;
  end_idx?: number;
};

export type RequirementPatchAttempt = {
  attempted: boolean;
  status: 'not_attempted' | 'applied' | 'failed' | 'rolled_back';
  target_path?: string | null;
  target_class?: string | null;
  validation_errors?: string[];
  reason?: string;
};

export type RequirementReportItem = {
  requirement_id: string;
  label: string;
  weight: number;
  status: RequirementStatus;
  applicable: boolean;
  quality: number;
  weighted_score: number;
  rationale?: string;
  target_paths?: string[];
  evidence_search_hints?: string[];
  selected_evidence?: RequirementEvidenceItem[];
  context_window?: RequirementEvidenceItem[];
  patch?: RequirementPatchAttempt;
};

export type CoverageFieldReport = {
  path: string;
  score: number;
  present: boolean;
  kind?: string;
  children?: CoverageFieldReport[];
};

export type CoverageReport = {
  score: number;
  filled_fields: number;
  total_fields: number;
  fields: CoverageFieldReport[];
};

export type SourceTraceReport = {
  score: number;
  used_evidence_count: number;
  evidence_ids: string[];
};

export type SemanticReconstructionRecord = {
  requirement_id: string;
  status: 'applied' | 'skipped' | 'failed' | 'rolled_back';
  target_paths: string[];
  changed_paths: string[];
  reason: string;
  validation_errors: string[];
};

export type RequirementReport = {
  schema_valid: boolean;
  coverage_score: number;
  semantic_requirements_score: number;
  source_trace_score: number;
  coverage: CoverageReport;
  semantic_requirements: RequirementReportItem[];
  source_trace: SourceTraceReport;
  coverage_patches: RequirementReportItem[];
  semantic_reconstructions: SemanticReconstructionRecord[];
};

export type EvidenceQueryLedgerEntry = {
  query_id: string;
  requirement_id: string;
  target_path?: string;
  query?: Record<string, unknown>;
  result_evidence_ids?: string[];
  selected_evidence_ids?: string[];
  rejected_result_reasons?: Record<string, string>;
  ranking_explanation?: string[];
};

export type QualityIssue = {
  code: string;
  severity: 'blocking' | 'warning' | 'info' | string;
  message: string;
  requirement_id?: string | null;
  path?: string | null;
};

export type DocumentQualityState = {
  schema_valid: boolean;
  profile_conformant?: boolean | null;
  evidence_grounded?: boolean | null;
  semantic_valid?: boolean | null;
  coverage_score?: number | null;
  semantic_requirements_score?: number | null;
  source_trace_score?: number | null;
  operational_access_score?: number | null;
  fair_assessment?: Record<string, unknown> | null;
  blocking_issues: QualityIssue[];
  warnings: QualityIssue[];
};

export type ExtractionVocabQueryConfig = {
  qualitative_vocab_identifiers: string[];
  vector_top_k: number;
  fulltext_top_k: number;
  seed_top_k: number;
  max_hops: number;
  max_statements_per_seed: number;
  traversal_direction: string;
  vector_weight: number;
  fulltext_weight: number;
  rrf_k: number;
  quantitative_vector_top_k: number;
  quantitative_fulltext_top_k: number;
  quantitative_seed_top_k: number;
  quantitative_max_hops: number;
  quantitative_max_statements_per_seed: number;
  quantitative_traversal_direction: string;
  quantitative_vector_weight: number;
  quantitative_fulltext_weight: number;
  quantitative_rrf_k: number;
};

export type RankedExtractionFile = {
  rank: number;
  file_path: string;
  score?: number | null;
  reasons?: string[];
};

export type InitialFileSummaryProgress = {
  total_files: number;
  processed_files: number;
  summarized_files: number;
  skipped_files: number;
  failed_files: number;
  current_file_path?: string | null;
};

export type InitialOverviewDiagnostic = {
  status?: 'structured_success' | 'structured_failure' | string;
  prompt_budget?: Record<string, unknown>;
  included_summary_paths?: string[];
  dropped_summary_paths?: string[];
  included_ranked_paths?: string[];
  dropped_ranked_paths?: string[];
  included_preview_paths?: string[];
  dropped_preview_paths?: string[];
  hard_truncated?: boolean;
  error_type?: string;
  message?: string;
  last_error_type?: string;
  last_error?: string;
  failed_response_excerpt?: string;
  usage?: Record<string, unknown>;
};

export type ExtractionChunkRef = {
  chunk_index: number;
  file_path: string;
  start_idx: number;
  end_idx: number;
};

export type ExtractionChunkResult = ExtractionChunkRef & {
  status: 'pending' | 'running' | 'repair_pending' | 'completed' | 'failed' | 'skipped' | string;
  evidence_context?: RoutedEvidenceContext | null;
  skip_reason?: string | null;
  error?: string | null;
  response_duration_ms?: number | null;
  context_tokens?: number | null;
};

export type InitialDraftScaffoldEntry = {
  path: string;
  value?: unknown;
  label?: string;
  target_class?: string | null;
  kind?: string;
  prune_if_unchanged?: boolean;
};

export type InitialDraftScaffold = {
  version?: number;
  evidence_categories?: string[];
  entries?: InitialDraftScaffoldEntry[];
};

export type ExtractionOverviewStatus = 'structured' | 'unstructured_fallback' | 'failed' | string;
export type ExtractionFileSummaryStatus = 'summarized' | 'failed' | string;
export type InitialFileSummaryStatus = 'completed' | 'partial' | 'failed' | string;

export type ExtractionFileSummary = {
  file_path: string;
  status: ExtractionFileSummaryStatus;
  data_format: string;
  explicit_purpose: string;
  purpose_evidence: string[];
  metadata_signals: string[];
  instrument_or_software_terms_and_settings: string[];
  quantitative_signals: string[];
};

export type ExtractionOverviewNodeKind = 'package' | 'directory' | 'file' | 'group' | string;
export type ExtractionOverviewRelation =
  | 'contains'
  | 'describes'
  | 'derives_from'
  | 'documents'
  | 'configures'
  | 'parameterizes'
  | 'generated_by'
  | 'related_to'
  | 'uncertain_relation'
  | string;

export type ExtractionOverviewInspectedFile = {
  file_path: string;
  byte_size?: number | null;
  chars_read: number;
  reason: string;
};

export type ExtractionOverviewNode = {
  node_id: string;
  label: string;
  kind: ExtractionOverviewNodeKind;
  file_path?: string | null;
  rank?: number | null;
  summary?: string;
};

export type ExtractionOverviewEdge = {
  edge_id: string;
  source: string;
  target: string;
  relation: ExtractionOverviewRelation;
  evidence: string[];
  note?: string;
};

export type ExtractionOverview = {
  source_fingerprint: string;
  source_file_paths: string[];
  inspected_files: ExtractionOverviewInspectedFile[];
  nodes: ExtractionOverviewNode[];
  edges: ExtractionOverviewEdge[];
  uncertainties: string[];
};

export type ExtractionVocabQueryRecord = {
  query_id: string;
  kind: string;
  source_value: string;
  source_context: Record<string, unknown>;
  vocabulary_identifier: string;
  rdf_type: string;
  query: Record<string, unknown>;
  status: 'pending' | 'running' | 'completed' | 'failed' | string;
  result?: Record<string, unknown> | null;
  error?: string | null;
  duration_ms?: number | null;
};

export type ProfilePatchOperation = {
  op: 'add' | 'replace' | 'remove' | string;
  path: string;
  value?: unknown;
};

export type ProfilePatchResult = {
  object_identifier: string;
  object_kind: string;
  status: 'applied' | 'skipped' | 'failed' | string;
  operations: ProfilePatchOperation[];
  error?: string | null;
  reason: string;
};

export type DraftQualityState = 'complete_final_draft' | 'imperfect_final_draft' | 'empty_profile_shell';

export type DraftValidationIssue = {
  path: string;
  message: string;
  schema_path: string;
};

export type DraftValidationResult = {
  status: 'valid' | 'invalid' | 'not_run' | string;
  errors: DraftValidationIssue[];
  warnings: string[];
};

export type ProjectionLedgerRecord = {
  object_identifier: string;
  object_kind: string;
  source_evidence?: string | null;
  evidence_note_identifiers?: string[];
  status: 'projected' | 'not_projected' | 'ambiguous' | 'user_edit_required' | string;
  projected_paths: string[];
  target_path?: string | null;
  target_class?: string | null;
  planner_status?: string | null;
  planner_reason?: string | null;
  evidence_quality?: {
    note_count?: number;
    routes?: Record<string, number>;
    [key: string]: unknown;
  };
  reason: string;
  error?: string | null;
};

export type FieldCompletionLedgerRecord = {
  json_path: string;
  field_name: string;
  generated_value?: unknown;
  curated_value?: unknown;
  source_evidence: string[];
  validation_status: 'valid' | 'invalid' | 'missing' | 'not_run' | string;
  enrichment_status: 'grounded' | 'not_grounded' | 'no_candidate' | 'ambiguous' | 'user_selected_vocab_term' | 'intentionally_unresolved' | string;
  issue_categories: string[];
  edit_needed_reason: string;
};

export type CurationLedgerRecord = {
  json_path: string;
  field_name: string;
  generated_value?: unknown;
  curated_value?: unknown;
  source_evidence: string[];
  status: 'unchanged' | 'user_modified' | 'user_removed' | 'user_selected_vocab_term' | 'intentionally_unresolved' | string;
  reason: string;
};

export type WorkflowProgress = ExtractionRunProgress & {
  batch_no?: number;
  total_batches?: number;
  file_name?: string;
  accepted_fields?: string[];
  total_candidates?: number;
  validation_errors?: string[];
  resolution_active?: boolean;
  resolution_log?: string[];
  resolution_resolved_count?: number;
  resolution_unresolved_item_ids?: string[];
  token_usage?: WorkflowTokenUsage;
};

export type ExtractionRunResult = {
  generated_final_draft: Record<string, unknown>;
  machine_evidence_context: RoutedEvidenceContext;
  generated_initial_draft?: Record<string, unknown> | null;
  generated_patched_draft?: Record<string, unknown> | null;
  generated_reconstructed_draft?: Record<string, unknown> | null;
  requirement_report?: RequirementReport | null;
  initial_file_summaries?: ExtractionFileSummary[];
  initial_file_summary_status?: InitialFileSummaryStatus | null;
  initial_extraction_overview?: ExtractionOverview | null;
  initial_extraction_overview_status?: ExtractionOverviewStatus | null;
  dataset_summary?: string;
  curated_document?: Record<string, unknown> | null;
  document_quality_state?: DocumentQualityState | null;
  draft_quality_state: DraftQualityState;
  validation: DraftValidationResult;
  curated_validation?: DraftValidationResult | null;
  initial_draft_scaffold?: InitialDraftScaffold;
  projection_ledger: ProjectionLedgerRecord[];
  field_completion_ledger: FieldCompletionLedgerRecord[];
  evidence_query_ledger?: EvidenceQueryLedgerEntry[];
  curation_ledger: CurationLedgerRecord[];
  normalization?: Record<string, unknown> | null;
  warnings: string[];
  token_usage: WorkflowTokenUsage;
};

export type ExtractionRunResponse = {
  status: WorkflowTaskStatus;
  result?: ExtractionRunResult | null;
  progress?: ExtractionRunProgress | null;
};

export async function runExtraction(input: {
  data_package_id: string;
  profile_identifier?: string | null;
  qualitative_vocab_identifiers?: string[] | null;
  resume?: boolean;
  force_profile_rebuild?: boolean;
  target_stage?: ExtractionTargetStage;
  chunking_strategy?: ChunkingStrategy;
  chat_model?: string | null;
  chunk_repair_mode?: ChunkRepairMode;
  evidence_critic_granularity?: EvidenceCriticGranularity;
}): Promise<ExtractionRunResponse> {
  return readJson(await fetch(apiBaseUrl + '/extraction/stages/evidence', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function runInitialContext(input: {
  data_package_id: string;
  force_rerun?: boolean;
}): Promise<{ status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null }> {
  return readJson(await fetch(apiBaseUrl + '/extraction/stages/orientation/' + encodeURIComponent(input.data_package_id), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force_rerun: input.force_rerun ?? false }),
  }));
}

export async function getInitialContextProgress(data_package_id: string): Promise<{ status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/stages/orientation/' + encodeURIComponent(data_package_id) + '/progress');
  const payload = await readJson(await response) as { status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ?? null };
}

export async function pauseExtraction(data_package_id: string): Promise<{ status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/workflows/' + encodeURIComponent(data_package_id) + '/pause', {
    method: 'POST',
  });
  const payload = await readJson(await response) as { status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ?? null };
}

export async function getExtractionResult(data_package_id: string, chunking_strategy?: ChunkingStrategy, chat_model?: string | null): Promise<ExtractionRunResult | null> {
  const response = await fetch(apiBaseUrl + '/extraction/results/' + encodeURIComponent(data_package_id) + buildQuery({ chunking_strategy, chat_model }));
  if (response.status === 404) return null;
  return readJson(await response);
}

export async function extractInitialContext(input: {
  data_package_id: string;
}): Promise<InitialContext> {
  const response = await runExtraction({
    data_package_id: input.data_package_id,
    target_stage: 'context',
  });
  const context = response.result?.machine_evidence_context || response.progress?.interim_evidence_context;
  return context
    ? initialContextFromEvidenceContext(context)
    : emptyInitialContext('Extraction is running.');
}

export async function saveCuratedDocument(data_package_id: string, document: object, profile_identifier?: string): Promise<Record<string, unknown>> {
  if (!profile_identifier) return document as Record<string, unknown>;
  const response = await fetch(apiBaseUrl + '/extraction/stages/curation/' + encodeURIComponent(data_package_id) + '/document', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile_identifier, document }),
  });
  const payload = await readJson(await response) as { progress?: ExtractionRunProgress | null };
  return payload.progress?.curated_document ?? (document as Record<string, unknown>);
}

export async function applyCurationFieldAction(input: {
  data_package_id: string;
  action: 'select_vocab_term' | 'mark_unresolved';
  json_path: string;
  selected_uri?: string | null;
  selected_title?: string | null;
  vocabulary_identifier?: string | null;
}): Promise<{ status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null }> {
  return readJson(await fetch(apiBaseUrl + '/extraction/stages/curation/' + encodeURIComponent(input.data_package_id) + '/field', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: input.action,
      json_path: input.json_path,
      selected_uri: input.selected_uri ?? null,
      selected_title: input.selected_title ?? null,
      vocabulary_identifier: input.vocabulary_identifier ?? null,
    }),
  }));
}

export type VocabularyGroundingResponse = {
  curated_document: Record<string, unknown>;
  status: WorkflowTaskStatus;
};

export async function runVocabularyGrounding(input: {
  data_package_id: string;
  profile_identifier: string;
  chunking_strategy?: ChunkingStrategy;
  chat_model?: string | null;
}): Promise<VocabularyGroundingResponse> {
  const response = await runExtraction({
    data_package_id: input.data_package_id,
    profile_identifier: input.profile_identifier,
    resume: true,
    target_stage: 'grounding',
    chunking_strategy: input.chunking_strategy,
    chat_model: input.chat_model,
  });
  return {
    curated_document: response.result?.curated_document ?? response.result?.generated_final_draft ?? response.progress?.curated_document ?? response.progress?.generated_final_draft ?? {},
    status: response.status,
  };
}

export async function getWorkflowProgress(data_package_id: string, chunking_strategy?: ChunkingStrategy, chat_model?: string | null): Promise<{ status: WorkflowTaskStatus; progress?: WorkflowProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/stages/evidence/' + encodeURIComponent(data_package_id) + '/progress' + buildQuery({ chunking_strategy, chat_model }));
  const payload = await readJson(await response) as { status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ? { ...payload.progress } : null };
}

export async function getTokenUsage(data_package_id: string, chunking_strategy?: ChunkingStrategy, chat_model?: string | null): Promise<WorkflowTokenUsage> {
  const response = await fetch(apiBaseUrl + '/extraction/workflows/' + encodeURIComponent(data_package_id) + '/token-usage' + buildQuery({ chunking_strategy, chat_model }));
  return readJson(await response);
}

export async function updateVocabQueryConfig(
  data_package_id: string,
  config: ExtractionVocabQueryConfig,
): Promise<{ status: WorkflowTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/stages/grounding/' + encodeURIComponent(data_package_id) + '/config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return readJson(await response);
}

export async function rerunAllVocabQueries(data_package_id: string): Promise<ExtractionRunResult> {
  return readJson(await fetch(apiBaseUrl + '/extraction/stages/grounding/' + encodeURIComponent(data_package_id) + '/rerun', {
    method: 'POST',
  }));
}

export async function rerunVocabQuery(data_package_id: string, query_id: string): Promise<ExtractionRunResult> {
  return readJson(await fetch(apiBaseUrl + '/extraction/stages/grounding/' + encodeURIComponent(data_package_id) + '/queries/' + encodeURIComponent(query_id) + '/rerun', {
    method: 'POST',
  }));
}

export function initialContextFromEvidenceContext(context: RoutedEvidenceContext): InitialContext {
  const notes = context.portable_evidence;
  const observations = notes
    .map((note) => note.claim)
    .filter((value): value is string => Boolean(value));
  const evidenceByCategory = (category: string) => notes
    .filter((note) => note.category === category)
    .map((note) => ({
      observation: note.claim || 'Evidence note',
      evidence: note.evidence_text || null,
    }));
  const activitySignals = evidenceByCategory('activity_signal');
  const agentSignals = [
    ...evidenceByCategory('software_signal'),
    ...evidenceByCategory('instrument_signal'),
  ];
  const methodSignals = evidenceByCategory('method_signal');
  const resourceSignals = evidenceByCategory('resource_signal');
  const surroundingSignals = evidenceByCategory('surrounding_signal');
  const titleSource = activitySignals[0] || resourceSignals[0] || methodSignals[0] || surroundingSignals[0] || null;
  const summary = observations.length
    ? observations.slice(0, 5).join(' ')
    : 'Evidence context generated.';
  return {
    dataset_title: titleSource?.observation || null,
    dataset_description: summary,
    entities: activitySignals.map((entity) => ({
      label: entity.observation,
      role: 'unknown',
      identifier: null,
      evidence: entity.evidence,
      confidence: null,
    })),
    agents: agentSignals.map((agent) => ({
      name: agent.observation,
      role: 'unknown',
      model: null,
      evidence: agent.evidence,
      confidence: null,
    })),
    activities: methodSignals.map((method) => ({
      label: method.observation,
      technique: null,
      agent_names: [],
      evidence: method.evidence,
      confidence: null,
    })),
    file_relationships: [],
    metadata_sources: [],
    keywords: [],
    summary,
  };
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function emptyInitialContext(summary: string): InitialContext {
  return {
    dataset_title: null,
    dataset_description: null,
    entities: [],
    agents: [],
    activities: [],
    file_relationships: [],
    metadata_sources: [],
    keywords: [],
    summary,
  };
}

function arrayOfRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object' && !Array.isArray(item))
    : [];
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && Boolean(item.trim()))
    : [];
}

