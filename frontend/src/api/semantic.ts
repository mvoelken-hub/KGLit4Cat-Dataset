import { apiBaseUrl, readJson } from './client';
import type { VocabEmbeddingStatus, VocabQueryResult, VocabSchemeInfo } from './types';

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

export async function queryVocabulary(params: { identifier: string; rdfType: string; query: string }): Promise<VocabQueryResult> {
  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies/query/' + encodeURIComponent(params.identifier), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      rdf_type: params.rdfType,
      vector_query: params.query,
      fulltext_query: params.query,
      vector_top_k: 10,
      fulltext_top_k: 10,
      seed_top_k: 5,
      max_hops: 1,
      max_statements_per_seed: 25,
    }),
  }));
}

export async function deleteVocabulary(identifier: string): Promise<void> {
  const response = await fetch(apiBaseUrl + '/semantic/vocabularies/' + encodeURIComponent(identifier), { method: 'DELETE' });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Failed to delete vocabulary (${response.status})`);
  }
}
