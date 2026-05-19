import { apiBaseUrl, readJson } from './client';
import type { InitialContext, PatchDraftResponse } from './types';

export type PatchTaskStatus = 'unknown' | 'running' | 'completed' | 'cancelled' | 'crashed';

export type PatchProgress = {
  batch_no?: number;
  total_batches?: number;
  file_name?: string;
  accepted_fields?: string[];
  total_candidates?: number;
  validation_errors?: string[];
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

export type PatchReviewResolutionResponse = {
  draft: object;
  review_state: PatchReviewState;
  resolved_count: number;
  unresolved_item_ids: string[];
  validation_errors: string[];
};

export async function getExistingInitialContext(data_package_id: string): Promise<InitialContext | null> {
  const response = await fetch(apiBaseUrl + '/extraction/initial-context/' + encodeURIComponent(data_package_id));
  if (response.status === 404) return null;
  return readJson(await response);
}

export async function getExistingInitialDraft(data_package_id: string): Promise<object | null> {
  const response = await fetch(apiBaseUrl + '/extraction/initial-draft/' + encodeURIComponent(data_package_id));
  if (response.status === 404) return null;
  return readJson(await response);
}

export async function extractInitialContext(input: {
  data_package_id: string;
  max_files_to_read?: number;
  max_chars_per_file?: number;
}): Promise<InitialContext> {
  return readJson(await fetch(apiBaseUrl + '/extraction/initial-context', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function extractInitialDraft(input: {
  data_package_id: string;
  profile_identifier: string;
}): Promise<object> {
  return readJson(await fetch(apiBaseUrl + '/extraction/initial-draft', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function saveDraft(data_package_id: string, draft: object): Promise<object> {
  return readJson(await fetch(apiBaseUrl + '/extraction/initial-draft/' + encodeURIComponent(data_package_id), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data_package_id, draft }),
  }));
}

export async function patchDraft(input: {
  data_package_id: string;
  profile_identifier: string;
  num_chunks_per_turn?: number;
}): Promise<PatchDraftResponse> {
  return readJson(await fetch(apiBaseUrl + '/extraction/patch-draft', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}

export async function getProtectedFields(data_package_id: string): Promise<string[]> {
  const response = await fetch(apiBaseUrl + '/extraction/initial-draft/' + encodeURIComponent(data_package_id) + '/protected-fields');
  if (!response.ok) return [];
  return readJson(await response);
}

export async function setProtectedFields(data_package_id: string, fields: string[]): Promise<string[]> {
  const response = await fetch(apiBaseUrl + '/extraction/initial-draft/' + encodeURIComponent(data_package_id) + '/protected-fields', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fields }),
  });
  return readJson(await response);
}

export async function getPatchProgress(data_package_id: string): Promise<{ status: PatchTaskStatus; progress?: PatchProgress | null }> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/progress');
  return readJson(await response);
}

export async function getPatchArtifacts(data_package_id: string): Promise<PatchArtifacts> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/artifacts');
  return readJson(await response);
}

export async function getPatchFiles(data_package_id: string): Promise<PatchArtifact[]> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/patches');
  return readJson(await response);
}

export async function getPatchQualityReports(data_package_id: string): Promise<PatchArtifact[]> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/quality-reports');
  return readJson(await response);
}

export async function getUnmappedFacts(data_package_id: string): Promise<PatchArtifact[]> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/unmapped-facts');
  return readJson(await response);
}

export async function getPatchReviewState(data_package_id: string): Promise<PatchReviewState> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/review-state');
  return readJson(await response);
}

export async function savePatchReviewState(data_package_id: string, reviewState: PatchReviewState): Promise<PatchReviewState> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(data_package_id) + '/review-state', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reviewState),
  });
  return readJson(await response);
}

export async function resolvePatchReview(input: {
  data_package_id: string;
  profile_identifier: string;
  review_items: PatchReviewResolutionItem[];
}): Promise<PatchReviewResolutionResponse> {
  const response = await fetch(apiBaseUrl + '/extraction/patch-draft/' + encodeURIComponent(input.data_package_id) + '/resolve-review', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      profile_identifier: input.profile_identifier,
      review_items: input.review_items,
    }),
  });
  return readJson(await response);
}
