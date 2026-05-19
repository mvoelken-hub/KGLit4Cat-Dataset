import { apiBaseUrl, readJson } from './client';
import type { InitialContext, PatchDraftResponse } from './types';

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
