import { apiBaseUrl, readJson } from './client';

export type LlmBudget = {
  chat_model: string;
  max_context_length: number;
  input_token_budget?: number;
  input_target_ratio?: number;
  warning_threshold: number;
  danger_threshold: number;
};

export type RunningOllamaModel = {
  model?: string | null;
  size?: number | null;
  size_vram?: number | null;
  expires_at?: string | null;
};

export type OllamaConfig = {
  host: {
    base_url: string;
    is_local: boolean;
    server_settings_note: string;
    flash_attention: boolean;
    kv_cache_type: string;
  };
  runtime: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    input_token_budget: number;
    input_target_ratio: number;
    embedding_batch_size: number;
    embedding_num_gpu: number;
    embedding_gpu_label: string;
    resets_on_api_restart: boolean;
  };
  running: {
    available: boolean;
    error?: { type: string; message: string };
    models: RunningOllamaModel[];
    chat_model_loaded: boolean;
    embedding_model_loaded: boolean;
  };
  warnings: string[];
};

export type OllamaRuntimePatch = {
  chat_model?: string;
  embedding_model?: string;
  max_context_length?: number;
  embedding_batch_size?: number;
  embedding_num_gpu?: number;
};

export async function getLlmBudget(): Promise<LlmBudget> {
  return readJson(await fetch(apiBaseUrl + '/llm-budget'));
}

export async function getOllamaConfig(): Promise<OllamaConfig> {
  return readJson(await fetch(apiBaseUrl + '/ollama-config'));
}

export async function updateOllamaRuntimeConfig(input: OllamaRuntimePatch): Promise<OllamaConfig> {
  return readJson(await fetch(apiBaseUrl + '/ollama-config/runtime', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}
