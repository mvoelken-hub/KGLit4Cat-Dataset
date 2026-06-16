import { apiBaseUrl, buildQuery, readJson } from './client';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, TaskStatus, TextQualityConfig } from './types';

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
  semantic_chunking_threshold?: number;
  chunking_strategy?: 'semantic' | 'fixed_tokens';
  fixed_tokens_per_chunk?: number;
  replace_existing_chunks?: boolean;
  protected_line_indices?: Record<string, number[]>;
  text_quality_config?: TextQualityConfig;
  embedding_num_gpu?: number;
}): Promise<ChunkRequestResponse> {
  const query = buildQuery({
    id: input.id,
    buffer_window_size: input.buffer_window_size ?? 1,
    semantic_chunking_threshold: input.semantic_chunking_threshold ?? 95,
    chunking_strategy: input.chunking_strategy ?? 'semantic',
    fixed_tokens_per_chunk: input.fixed_tokens_per_chunk ?? 1024,
    replace_existing_chunks: input.replace_existing_chunks ?? false,
    protected_line_indices: input.protected_line_indices ? JSON.stringify(input.protected_line_indices) : undefined,
    text_quality_config: input.text_quality_config ? JSON.stringify(input.text_quality_config) : undefined,
    embedding_num_gpu: input.embedding_num_gpu,
  });
  return readJson(await fetch(apiBaseUrl + '/datasources/chunk' + query, { method: 'POST' }));
}

export async function getChunkStatus(data_package_id: string): Promise<{ has_chunks: boolean; file_count: number; status: TaskStatus }> {
  return readJson(await fetch(apiBaseUrl + '/datasources/' + encodeURIComponent(data_package_id) + '/chunks/status'));
}

export async function getDataPackageChunks(data_package_id: string): Promise<ChunkResponse[][]> {
  return readJson(await fetch(apiBaseUrl + '/datasources/' + encodeURIComponent(data_package_id) + '/chunks'));
}

export async function getFileEntryContent(data_package_id: string, file_path: string): Promise<{ file_path: string; file_name: string; file_extension: string; content: string }> {
  return readJson(await fetch(apiBaseUrl + '/datasources/' + encodeURIComponent(data_package_id) + '/files/' + encodeURIComponent(file_path)));
}

export async function deleteDataPackage(id: string): Promise<void> {
  const response = await fetch(apiBaseUrl + '/datasources/' + encodeURIComponent(id), { method: 'DELETE' });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Failed to delete dataset (${response.status})`);
  }
}
