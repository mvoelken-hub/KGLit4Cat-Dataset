import { apiBaseUrl, readJson } from './client';
import type { InitialContext } from './types';

export type PatchTaskStatus = 'unknown' | 'running' | 'completed' | 'cancelled' | 'crashed';
export type ExtractionTargetStage = 'context' | 'profile' | 'grounding' | 'complete';
export type ChunkRepairMode = 'deferred' | 'immediate' | 'disabled';

export type PatchTokenUsageEntry = {
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

export type PatchTokenUsage = {
  agents?: Record<string, PatchTokenUsageEntry>;
  combined?: PatchTokenUsageEntry;
};

export type ExtractionRunProgress = {
  stage: string;
  chunk_repair_mode?: ChunkRepairMode;
  processed_chunks: number;
  total_chunks: number;
  normalized_quantities: number;
  normalized_qualitative_attributes: number;
  interim_evidence_context?: Record<string, unknown> | null;
  generated_final_draft?: Record<string, unknown> | null;
  curated_document?: Record<string, unknown> | null;
  draft_quality_state?: DraftQualityState | null;
  validation?: DraftValidationResult | null;
  curated_validation?: DraftValidationResult | null;
  initial_draft_scaffold?: InitialDraftScaffold;
  projection_ledger?: ProjectionLedgerRecord[];
  field_completion_ledger?: FieldCompletionLedgerRecord[];
  curation_ledger?: CurationLedgerRecord[];
  initial_file_summaries?: ExtractionFileSummary[];
  initial_file_summary_status?: InitialFileSummaryStatus | null;
  initial_extraction_overview?: ExtractionOverview | null;
  initial_extraction_overview_status?: ExtractionOverviewStatus | null;
  vocab_query_config?: ExtractionVocabQueryConfig;
  vocab_queries?: ExtractionVocabQueryRecord[];
  ranked_files?: RankedExtractionFile[];
  chunk_results?: ExtractionChunkResult[];
  current_chunk?: ExtractionChunkRef | null;
  warnings: string[];
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
};

export type ExtractionChunkRef = {
  chunk_index: number;
  file_path: string;
  start_idx: number;
  end_idx: number;
};

export type ExtractionChunkResult = ExtractionChunkRef & {
  status: 'pending' | 'running' | 'repair_pending' | 'completed' | 'failed' | 'skipped' | string;
  evidence_context?: Record<string, unknown> | null;
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
  source_fingerprint: string;
  file_path: string;
  rank: number;
  status: ExtractionFileSummaryStatus;
  data_format: string;
  data_characteristics: string[];
  explicit_purpose: string;
  purpose_evidence: string[];
  metadata_signals: string[];
  detected_identifiers: string[];
  instrument_or_software_terms: string[];
  parameter_terms: string[];
  uncertainty_notes: string[];
  known_traps: string[];
};

export type ExtractionOverviewFileRole = {
  file_path: string;
  role: string;
  extraction_notes: string[];
  read_reason?: string;
};

export type ExtractionOverviewInspectedFile = {
  file_path: string;
  byte_size?: number | null;
  chars_read: number;
  reason: string;
};

export type ExtractionOverview = {
  source_fingerprint: string;
  source_file_paths: string[];
  inspected_files: ExtractionOverviewInspectedFile[];
  file_roles: ExtractionOverviewFileRole[];
  observed_signals: string[];
  suggested_interpretations: string[];
  conflicts_or_uncertainties: string[];
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
    signal_level?: Record<string, number>;
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

export type PatchProgress = ExtractionRunProgress & {
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
  token_usage?: PatchTokenUsage;
};

export type ExtractionRunResult = {
  generated_final_draft: Record<string, unknown>;
  machine_evidence_context: Record<string, unknown>;
  initial_file_summaries?: ExtractionFileSummary[];
  initial_file_summary_status?: InitialFileSummaryStatus | null;
  initial_extraction_overview?: ExtractionOverview | null;
  initial_extraction_overview_status?: ExtractionOverviewStatus | null;
  curated_document?: Record<string, unknown> | null;
  draft_quality_state: DraftQualityState;
  validation: DraftValidationResult;
  curated_validation?: DraftValidationResult | null;
  initial_draft_scaffold?: InitialDraftScaffold;
  projection_ledger: ProjectionLedgerRecord[];
  field_completion_ledger: FieldCompletionLedgerRecord[];
  curation_ledger: CurationLedgerRecord[];
  normalization?: Record<string, unknown> | null;
  warnings: string[];
  token_usage: PatchTokenUsage;
};

export type ExtractionRunResponse = {
  status: PatchTaskStatus;
  result?: ExtractionRunResult | null;
  progress?: ExtractionRunProgress | null;
};

export type PatchArtifact = {
  file_name?: string;
  artifact_type?: string;
  content?: unknown;
  [key: string]: unknown;
};

export type PatchArtifacts = {
  patches: PatchArtifact[];
  quality_reports: PatchArtifact[];
  unmapped_facts: PatchArtifact[];
};

export type PatchReviewState = {
  resolved_item_ids: string[];
  unmapped_assignments: Record<string, string>;
  resolution_notes: Record<string, string>;
  resolved_at: Record<string, string>;
};

export type PatchReviewResolutionItem = {
  id: string;
  kind: 'matched' | 'unmapped';
  path: string;
  detail?: string;
  issues?: string[];
  evidence?: string[];
  patch?: Record<string, unknown>;
  fact?: string;
  reason?: string;
  confidence?: number;
  file_name?: string;
};

export type PatchReviewDecision = {
  id: string;
  outcome: 'included' | 'already_present' | 'excluded' | 'unresolved' | string;
  note: string;
  target_path?: string | null;
};

export type PatchReviewResolutionResponse = {
  curated_document: Record<string, unknown>;
  review_state: PatchReviewState;
  resolved_count: number;
  unresolved_item_ids: string[];
  validation_errors: string[];
  resolution_decisions: PatchReviewDecision[];
  resolution_log: string[];
  token_usage?: PatchTokenUsage | null;
};

export async function runExtraction(input: {
  data_package_id: string;
  profile_identifier?: string | null;
  qualitative_vocab_identifiers?: string[] | null;
  resume?: boolean;
  target_stage?: ExtractionTargetStage;
  chunk_repair_mode?: ChunkRepairMode;
}): Promise<ExtractionRunResponse> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function runInitialContext(input: {
  data_package_id: string;
  force_rerun?: boolean;
}): Promise<{ status: PatchTaskStatus; progress?: ExtractionRunProgress | null }> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(input.data_package_id) + '/initial-context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force_rerun: input.force_rerun ?? false }),
  }));
}

export async function getInitialContextProgress(data_package_id: string): Promise<{ status: PatchTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/initial-context/progress');
  const payload = await readJson(await response) as { status: PatchTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ?? null };
}

export async function pauseExtraction(data_package_id: string): Promise<{ status: PatchTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/pause', {
    method: 'POST',
  });
  const payload = await readJson(await response) as { status: PatchTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ?? null };
}

export async function getExtractionResult(data_package_id: string): Promise<ExtractionRunResult | null> {
  const response = await fetch(apiBaseUrl + '/extraction/result/' + encodeURIComponent(data_package_id));
  if (response.status === 404) return null;
  return readJson(await response);
}

export async function getExistingInitialContext(data_package_id: string): Promise<InitialContext | null> {
  const result = await getExtractionResult(data_package_id);
  return result ? initialContextFromEvidenceContext(result.machine_evidence_context) : null;
}

export async function getExistingGeneratedFinalDraft(data_package_id: string): Promise<Record<string, unknown> | null> {
  const result = await getExtractionResult(data_package_id);
  return result?.generated_final_draft ?? null;
}

export async function getExistingCuratedDocument(data_package_id: string): Promise<Record<string, unknown> | null> {
  const result = await getExtractionResult(data_package_id);
  return result?.curated_document ?? result?.generated_final_draft ?? null;
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
  const response = await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/curated-document', {
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
}): Promise<{ status: PatchTaskStatus; progress?: ExtractionRunProgress | null }> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(input.data_package_id) + '/curation/field', {
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
  status: PatchTaskStatus;
};

export async function runVocabularyGrounding(input: {
  data_package_id: string;
  profile_identifier: string;
}): Promise<VocabularyGroundingResponse> {
  const response = await runExtraction({
    data_package_id: input.data_package_id,
    profile_identifier: input.profile_identifier,
    resume: true,
    target_stage: 'grounding',
  });
  return {
    curated_document: response.result?.curated_document ?? response.result?.generated_final_draft ?? response.progress?.curated_document ?? response.progress?.generated_final_draft ?? {},
    status: response.status,
  };
}

export async function getPatchProgress(data_package_id: string): Promise<{ status: PatchTaskStatus; progress?: PatchProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/progress');
  const payload = await readJson(await response) as { status: PatchTaskStatus; progress?: ExtractionRunProgress | null };
  return { status: payload.status, progress: payload.progress ? { ...payload.progress } : null };
}

export async function getTokenUsage(data_package_id: string): Promise<PatchTokenUsage> {
  const response = await fetch(apiBaseUrl + '/extraction/' + encodeURIComponent(data_package_id) + '/token-usage');
  return readJson(await response);
}

export async function updateVocabQueryConfig(
  data_package_id: string,
  config: ExtractionVocabQueryConfig,
): Promise<{ status: PatchTaskStatus; progress?: ExtractionRunProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/vocab-query-config', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  return readJson(await response);
}

export async function rerunAllVocabQueries(data_package_id: string): Promise<ExtractionRunResult> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/vocab-queries/rerun', {
    method: 'POST',
  }));
}

export async function rerunVocabQuery(data_package_id: string, query_id: string): Promise<ExtractionRunResult> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run/' + encodeURIComponent(data_package_id) + '/vocab-queries/' + encodeURIComponent(query_id) + '/rerun', {
    method: 'POST',
  }));
}

export async function getPatchArtifacts(_data_package_id: string): Promise<PatchArtifacts> {
  return { patches: [], quality_reports: [], unmapped_facts: [] };
}

export async function getPatchFiles(_data_package_id: string): Promise<PatchArtifact[]> {
  return [];
}

export async function getPatchQualityReports(_data_package_id: string): Promise<PatchArtifact[]> {
  return [];
}

export async function getUnmappedFacts(_data_package_id: string): Promise<PatchArtifact[]> {
  return [];
}

export async function getPatchReviewState(_data_package_id: string): Promise<PatchReviewState> {
  return emptyReviewState();
}

export async function savePatchReviewState(_data_package_id: string, reviewState: PatchReviewState): Promise<PatchReviewState> {
  return reviewState;
}

export async function resolvePatchReview(input: {
  data_package_id: string;
  profile_identifier: string;
  review_items: PatchReviewResolutionItem[];
}): Promise<PatchReviewResolutionResponse> {
  return {
    curated_document: {},
    review_state: emptyReviewState(),
    resolved_count: 0,
    unresolved_item_ids: input.review_items.map((item) => item.id),
    validation_errors: [],
    resolution_decisions: [],
    resolution_log: ['Manual patch review was removed from the extraction workflow.'],
  };
}

export function initialContextFromEvidenceContext(context: Record<string, unknown>): InitialContext {
  const notes = arrayOfRecords(context.notes);
  const observations = notes
    .map((note) => stringValue(note.observation))
    .filter((value): value is string => Boolean(value));
  const evidenceByCategory = (category: string) => notes
    .filter((note) => stringValue(note.category) === category)
    .map((note) => ({
      observation: stringValue(note.observation) || 'Evidence note',
      evidence: stringValue(note.evidence_text) || null,
    }));
  const entitySignals = evidenceByCategory('entity_signal');
  const agentSignals = evidenceByCategory('agent_signal');
  const methodSignals = evidenceByCategory('method_signal');
  const resourceSignals = evidenceByCategory('resource_signal');
  const titleSource = entitySignals[0] || resourceSignals[0] || methodSignals[0] || null;
  const summary = observations.length
    ? observations.slice(0, 5).join(' ')
    : 'Evidence context generated.';
  return {
    dataset_title: titleSource?.observation || null,
    dataset_description: summary,
    entities: entitySignals.map((entity) => ({
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

function emptyReviewState(): PatchReviewState {
  return {
    resolved_item_ids: [],
    unmapped_assignments: {},
    resolution_notes: {},
    resolved_at: {},
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
