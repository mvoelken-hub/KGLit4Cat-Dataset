import { apiBaseUrl, readJson } from './client';
import type { VocabEmbeddingStatus, VocabQueryParams, VocabQueryResult, VocabSchemeInfo } from './types';

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

export async function queryVocabulary(params: VocabQueryParams): Promise<VocabQueryResult> {
  const body: Record<string, unknown> = {
    rdf_type: params.rdfType,
    vector_top_k: params.vectorTopK,
    fulltext_top_k: params.fulltextTopK,
    seed_top_k: params.seedTopK,
    traversal_direction: params.traversalDirection,
    max_hops: params.maxHops,
    max_statements_per_seed: params.maxStatementsPerSeed,
    allowed_rel_types: params.allowedRelTypes,
    vector_weight: params.vectorWeight,
    fulltext_weight: params.fulltextWeight,
    rrf_k: params.rrfK,
  };

  if (params.searchMode === 'vector') {
    body.vector_query = params.vectorQuery;
  } else if (params.searchMode === 'fulltext') {
    body.fulltext_query = params.fulltextQuery;
  } else {
    body.vector_query = params.vectorQuery;
    body.fulltext_query = params.fulltextQuery;
  }

  return readJson(await fetch(apiBaseUrl + '/semantic/vocabularies/query/' + encodeURIComponent(params.identifier), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }));
}

export async function deleteVocabulary(identifier: string): Promise<void> {
  const response = await fetch(apiBaseUrl + '/semantic/vocabularies/' + encodeURIComponent(identifier), { method: 'DELETE' });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Failed to delete vocabulary (${response.status})`);
  }
}
