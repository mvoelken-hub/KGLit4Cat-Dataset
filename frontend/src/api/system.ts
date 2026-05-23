import { apiBaseUrl, readJson } from './client';

export type LlmBudget = {
  chat_model: string;
  max_context_length: number;
  input_token_budget?: number;
  input_target_ratio?: number;
  warning_threshold: number;
  danger_threshold: number;
};

export async function getLlmBudget(): Promise<LlmBudget> {
  return readJson(await fetch(apiBaseUrl + '/llm-budget'));
}
