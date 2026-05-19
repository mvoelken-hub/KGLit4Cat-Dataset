import { apiBaseUrl, readJson } from './client';
import type { ProfileManifestResponse } from './types';

export async function listProfiles(): Promise<ProfileManifestResponse[]> {
  return readJson(await fetch(apiBaseUrl + '/profiles'));
}

export async function validateProfileDocument(identifier: string, document: object): Promise<{ valid: boolean; errors: Array<{ path: string; message: string; schema_path: string }> }> {
  return readJson(await fetch(apiBaseUrl + '/profiles/' + encodeURIComponent(identifier) + '/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document }),
  }));
}
