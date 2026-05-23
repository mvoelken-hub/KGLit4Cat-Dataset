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
  name?: string | null;
  model?: string | null;
  size?: number | null;
  size_vram?: number | null;
  context_length?: number | null;
  processor?: string | null;
  gpu_percent?: number | null;
  cpu_percent?: number | null;
  expires_at?: string | null;
};

export type OllamaAvailableModel = {
  name?: string | null;
  model?: string | null;
  size?: number | null;
  digest?: string | null;
  modified_at?: string | null;
  details?: {
    parameter_size?: string | null;
    quantization_level?: string | null;
    family?: string | null;
    families?: string[] | null;
    format?: string | null;
  } | null;
  capabilities?: string[];
  kind?: 'chat' | 'embedding' | 'unknown' | string;
  is_cloud?: boolean;
};

export type OllamaDiagnostics = {
  status: string;
  summary: string;
  recommendations: string[];
  both_resident_after_final?: boolean;
  chat_displaced_by_embedding?: boolean;
  embedding_displaced_by_chat?: boolean;
};

export type OllamaConfig = {
  host: {
    base_url: string;
    is_local: boolean;
    server_settings_note: string;
    flash_attention?: boolean;
    kv_cache_type?: string;
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
  models: {
    available: boolean;
    error?: { type: string; message: string };
    models: OllamaAvailableModel[];
  };
  running: {
    available: boolean;
    error?: { type: string; message: string };
    models: RunningOllamaModel[];
    chat_model_loaded: boolean;
    embedding_model_loaded: boolean;
    both_configured_models_loaded: boolean;
  };
  diagnostics: OllamaDiagnostics;
  warnings: string[];
};

export type OllamaRuntimePatch = {
  chat_model?: string;
  embedding_model?: string;
  max_context_length?: number;
  embedding_batch_size?: number;
  embedding_num_gpu?: number;
};

export type OllamaPerformanceTest = {
  runtime: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_num_gpu: number;
  };
  initial: {
    available: boolean;
    error?: { type: string; message: string };
    models: RunningOllamaModel[];
  };
  steps: Array<{
    key: string;
    label: string;
    model: string;
    success: boolean;
    error?: string | null;
    load_duration_ms?: number | null;
    snapshot: {
      available: boolean;
      error?: { type: string; message: string };
      models: RunningOllamaModel[];
    };
  }>;
  diagnostics: OllamaDiagnostics;
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

export async function listOllamaModels(): Promise<OllamaConfig['models']> {
  return readJson(await fetch(apiBaseUrl + '/ollama-models'));
}

export async function pullOllamaModel(model: string): Promise<OllamaConfig['models']> {
  return readJson(await fetch(apiBaseUrl + '/ollama-models/pull', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  }));
}

export async function removeOllamaModel(model: string): Promise<OllamaConfig['models']> {
  return readJson(await fetch(apiBaseUrl + '/ollama-models/' + encodeURIComponent(model), {
    method: 'DELETE',
  }));
}

export async function runOllamaPerformanceTest(input: OllamaRuntimePatch = {}): Promise<OllamaPerformanceTest> {
  return readJson(await fetch(apiBaseUrl + '/ollama-config/runtime/performance-test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  }));
}
