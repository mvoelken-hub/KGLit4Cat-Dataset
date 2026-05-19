export type TaskStatus = 'running' | 'completed' | 'cancelled' | 'crashed' | 'unknown';

export interface FileEntryResponse {
  file_path: string;
  file_name: string;
  file_extension: string;
  byte_size: number;
}

export interface DataPackageResponse {
  file_name: string;
  id: string;
  files: FileEntryResponse[];
}

export interface ChunkResponse {
  content: string;
  data_package_id: string;
  file_path: string;
  start_idx: number;
  end_idx: number;
  summary?: string | null;
}

export interface ChunkRequestResponse {
  chunks: ChunkResponse[][];
  status: TaskStatus;
}

export interface MetadataSource {
  file_path: string;
  source_type: string;
  description: string;
  extracted_fields: string[];
  evidence?: string | null;
  confidence?: number | null;
}

export interface FileRelationship {
  source_file: string;
  related_file?: string | null;
  relationship_type: string;
  description: string;
  evidence?: string | null;
  confidence?: number | null;
}

export interface InitialContext {
  device_name?: string | null;
  device_model?: string | null;
  entities_analyzed: string[];
  analytical_technique?: string | null;
  file_relationships: FileRelationship[];
  metadata_sources: MetadataSource[];
  keywords: string[];
  summary: string;
}

export interface ProfileManifestResponse {
  identifier: string;
  source: string;
  source_type: string;
  target_class: string;
  checksum: string;
  enrichable_fields: string[];
}

export interface PatchDraftResponse {
  draft: object;
  status: TaskStatus;
}
