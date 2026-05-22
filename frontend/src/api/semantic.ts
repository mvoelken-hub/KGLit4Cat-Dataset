import { apiBaseUrl, readJson } from './client';
import type { VocabEmbeddingStatus, VocabSchemeInfo } from './types';

export async function listVocabularies(): Promise<string[]> {
  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies'));
}

export async function getVocabulary(identifier: string): Promise<VocabSchemeInfo> {
  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies/' + encodeURIComponent(identifier)));
}

export async function importVocabulary(params: { identifier: string; rdfSource: string }): Promise<VocabSchemeInfo> {
  const formData = new FormData();
  formData.append('identifier', params.identifier);
  formData.append('rdf_source', params.rdfSource);

  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies', {
    method: 'POST',
    body: formData,
  }));
}

export async function checkVocabularyEmbeddings(identifier: string): Promise<VocabEmbeddingStatus> {
  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies/embeddings/' + encodeURIComponent(identifier), { method: 'POST' }));
}

export async function deleteVocabulary(identifier: string): Promise<void> {
  const response = await fetch(apiBaseUrl + '/semantic/vocabularies/' + encodeURIComponent(identifier), { method: 'DELETE' });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Failed to delete vocabulary (${response.status})`);
  }
}
