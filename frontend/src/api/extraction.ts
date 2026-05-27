import { apiBaseUrl, readJson } from './client';
import type { InitialContext, PatchDraftResponse } from './types';

export type PatchTaskStatus = 'unknown' | 'running' | 'completed' | 'cancelled' | 'crashed';

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
  processed_chunks: number;
  total_chunks: number;
  normalized_quantities: number;
  normalized_qualitative_attributes: number;
  interim_context?: Record<string, unknown> | null;
  ranked_files?: RankedExtractionFile[];
  chunk_results?: ExtractionChunkResult[];
  current_chunk?: ExtractionChunkRef | null;
  warnings: string[];
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
  status: 'pending' | 'running' | 'completed' | 'failed' | string;
  extraction_context?: Record<string, unknown> | null;
  error?: string | null;
  response_duration_ms?: number | null;
  context_tokens?: number | null;
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
  document: Record<string, unknown>;
  extraction_context: Record<string, unknown>;
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
  draft: object;
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
  profile_identifier: string;
  qualitative_vocab_identifiers?: string[] | null;
  resume?: boolean;
}): Promise<ExtractionRunResponse> {
  return readJson(await fetch(apiBaseUrl + '/extraction/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function getExtractionResult(data_package_id: string): Promise<ExtractionRunResult | null> {
  const response = await fetch(apiBaseUrl + '/extraction/result/' + encodeURIComponent(data_package_id));
  if (response.status === 404) return null;
  return readJson(await response);
}

export async function getExistingInitialContext(data_package_id: string): Promise<InitialContext | null> {
  const result = await getExtractionResult(data_package_id);
  return result ? initialContextFromExtractionContext(result.extraction_context) : null;
}

export async function getExistingInitialDraft(data_package_id: string): Promise<object | null> {
  const result = await getExtractionResult(data_package_id);
  return result?.document ?? null;
}

export async function extractInitialContext(input: {
  data_package_id: string;
  profile_identifier: string;
}): Promise<InitialContext> {
  const response = await runExtraction({
    data_package_id: input.data_package_id,
    profile_identifier: input.profile_identifier,
  });
  const context = response.result?.extraction_context || response.progress?.interim_context;
  return context
    ? initialContextFromExtractionContext(context)
    : emptyInitialContext('Extraction is running.');
}

export async function saveInitialContext(_data_package_id: string, context: InitialContext): Promise<InitialContext> {
  return context;
}

export async function extractInitialDraft(input: {
  data_package_id: string;
  profile_identifier: string;
}): Promise<object> {
  const response = await runExtraction(input);
  return response.result?.document ?? {};
}

export async function saveDraft(_data_package_id: string, draft: object): Promise<object> {
  return draft;
}

export async function patchDraft(input: {
  data_package_id: string;
  profile_identifier: string;
  num_chunks_per_turn: number;
  auto_resolve: boolean;
}): Promise<PatchDraftResponse> {
  const response = await runExtraction({
    data_package_id: input.data_package_id,
    profile_identifier: input.profile_identifier,
  });
  return {
    draft: response.result?.document ?? {},
    status: response.status,
  };
}

export async function getProtectedFields(_data_package_id: string): Promise<string[]> {
  return [];
}

export async function setProtectedFields(_data_package_id: string, fields: string[]): Promise<string[]> {
  return fields;
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
    draft: {},
    review_state: emptyReviewState(),
    resolved_count: 0,
    unresolved_item_ids: input.review_items.map((item) => item.id),
    validation_errors: [],
    resolution_decisions: [],
    resolution_log: ['Manual patch review was removed from the extraction workflow.'],
  };
}

export function initialContextFromExtractionContext(context: Record<string, unknown>): InitialContext {
  const resources = arrayOfRecords(context.resources);
  const activities = arrayOfRecords(context.data_generating_activities);
  const entities = arrayOfRecords(context.evaluated_entities);
  const agents = arrayOfRecords(context.agentic_entities);
  const dataset = resources.find((resource) => stringValue(resource.type)?.toLowerCase() === 'dataset') || resources[0] || {};
  return {
    dataset_title: stringValue(dataset.identifier) || null,
    dataset_description: stringValue(dataset.description) || null,
    entities: entities.map((entity) => ({
      label: stringValue(entity.identifier) || stringValue(entity.description) || 'Entity',
      role: 'unknown',
      identifier: stringValue(entity.identifier) || null,
      evidence: stringValue(entity.description) || null,
      confidence: null,
    })),
    agents: agents.map((agent) => ({
      name: stringValue(agent.identifier) || 'Agent',
      role: 'unknown',
      model: null,
      evidence: stringValue(agent.description) || null,
      confidence: null,
    })),
    activities: activities.map((activity) => ({
      label: stringValue(activity.identifier) || null,
      technique: null,
      agent_names: [],
      evidence: stringValue(activity.description) || null,
      confidence: null,
    })),
    file_relationships: [],
    metadata_sources: [],
    keywords: stringArray(dataset.keywords),
    summary: stringValue(dataset.description) || 'Extraction context generated.',
  };
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
