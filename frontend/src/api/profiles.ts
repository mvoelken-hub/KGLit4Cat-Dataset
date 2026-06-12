import { apiBaseUrl, readJson } from './client';
import type { ProfileManifestResponse } from './types';

export interface RegisterProfileInput {
  identifier: string;
  target_class: string;
  schema_url?: string;
  schema_file?: File;
  version?: string;
  enrichable_fields?: string[];
}

export async function listProfiles(): Promise<ProfileManifestResponse[]> {
  return readJson(await fetch(apiBaseUrl + '/profiles'));
}

export async function getProfileJsonSchema(identifier: string): Promise<Record<string, unknown>> {
  return readJson(await fetch(apiBaseUrl + '/profiles/' + encodeURIComponent(identifier) + '/json-schema'));
}

export async function registerProfile(input: RegisterProfileInput): Promise<ProfileManifestResponse> {
  const form = new FormData();
  form.append('identifier', input.identifier);
  form.append('target_class', input.target_class || 'Dataset');
  if (input.schema_url) form.append('schema_url', input.schema_url);
  if (input.schema_file) form.append('schema_file', input.schema_file);
  if (input.version) form.append('version', input.version);
  for (const field of input.enrichable_fields ?? []) {
    form.append('enrichable_fields', field);
  }
  return readJson(await fetch(apiBaseUrl + '/profiles', { method: 'POST', body: form }));
}

export async function deleteProfile(identifier: string): Promise<void> {
  await readJson(await fetch(apiBaseUrl + '/profiles/' + encodeURIComponent(identifier), { method: 'DELETE' }));
}

export async function validateProfileDocument(identifier: string, document: object): Promise<{ valid: boolean; errors: Array<{ path: string; message: string; schema_path: string }> }> {
  return readJson(await fetch(apiBaseUrl + '/profiles/' + encodeURIComponent(identifier) + '/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document }),
  }));
}

export async function exportProfileDocumentJsonLd(identifier: string, document: object): Promise<{ document: Record<string, unknown>; triple_count: number }> {
  return readJson(await fetch(apiBaseUrl + '/profiles/' + encodeURIComponent(identifier) + '/jsonld', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document }),
  }));
}
