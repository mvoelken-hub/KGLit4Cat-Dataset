import { apiBaseUrl, buildQuery, readJson } from './client';
import type { ChunkRequestResponse, DataPackageResponse } from './types';

export async function listDataPackages(): Promise<DataPackageResponse[]> {
  return readJson(await fetch(apiBaseUrl + '/datasources'));
}

export async function uploadDataPackage(file: File): Promise<DataPackageResponse> {
  const form = new FormData();
  form.append('file', file);
  return readJson(await fetch(apiBaseUrl + '/datasources', { method: 'POST', body: form }));
}

export async function chunkDataPackage(input: {
  id: string;
  buffer_window_size?: number;
  embedding_batch_size?: number;
  semantic_chunking_threshold?: number;
  replace_existing_chunks?: boolean;
}): Promise<ChunkRequestResponse> {
  const query = buildQuery({
    id: input.id,
    buffer_window_size: input.buffer_window_size ?? 1,
    embedding_batch_size: input.embedding_batch_size ?? 32,
    semantic_chunking_threshold: input.semantic_chunking_threshold ?? 95,
    replace_existing_chunks: input.replace_existing_chunks ?? false,
  });
  return readJson(await fetch(apiBaseUrl + '/datasources/chunk' + query, { method: 'POST' }));
}

export async function getChunkStatus(data_package_id: string): Promise<{ has_chunks: boolean; file_count: number }> {
  return readJson(await fetch(apiBaseUrl + '/datasources/' + encodeURIComponent(data_package_id) + '/chunks/status'));
}
