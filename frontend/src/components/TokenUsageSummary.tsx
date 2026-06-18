import type { ReactNode } from 'react';
import type { PatchTokenUsage, PatchTokenUsageEntry } from '../api/extraction';
import type { LlmBudget } from '../api/system';
import { formatDuration, formatTokenCount } from '../lib/format';
const tokenUsageLabels: Record<string, string> = {
  initial_context: 'Context extraction',
  file_ranking: 'File ranking',
  chunk_extraction: 'Chunk extraction',
  chunk_extraction_repair: 'Chunk extraction repair',
  quantity_vocab_selection: 'Quantity vocabulary',
  qualitative_vocab_selection: 'Qualitative vocabulary',
  profile_projection: 'Profile projection',
  dataset_summary: 'Dataset summary',
  dataset_level_projection: 'Dataset-level projection',
  dataset_level_projection_repair: 'Dataset-level repair',
  profile_target_planner: 'Projection target planner',
  profile_target_writer: 'Projection target writer',
  profile_patch: 'Profile projection',
  patch_discovery: 'Patch discovery',
  schema_patch_writer: 'Schema patch writer',
  schema_repair: 'Schema repair',
  patch_extraction: 'Patch extraction',
  patch_quality: 'Patch quality review',
};

type TokenAverageUnit = 'patch' | 'operation' | 'request';

function tokenAverageUnitLabel(unit: TokenAverageUnit): string {
  return unit === 'request' ? 'model call' : unit;
}

function usageAverage(usage: PatchTokenUsageEntry, unit: TokenAverageUnit, kind: 'input' | 'output' | 'total'): number {
  if (unit === 'request') return requestAverage(usage, kind);
  const suffix = unit === 'patch' ? 'patch' : 'operation';
  const key = `average_${kind}_tokens_per_${suffix}` as keyof PatchTokenUsageEntry;
  const preferred = usage[key];
  if (typeof preferred === 'number') return preferred;
  const fallbackKey = `average_${kind}_tokens_per_${unit === 'patch' ? 'operation' : 'patch'}` as keyof PatchTokenUsageEntry;
  const fallback = usage[fallbackKey];
  if (typeof fallback === 'number') return fallback;
  const totalKey = `${kind}_tokens` as keyof PatchTokenUsageEntry;
  const total = usage[totalKey];
  const count = unit === 'patch' ? usage.patch_count : usage.operation_count;
  return typeof total === 'number' ? total / Math.max(1, Number(count || 1)) : 0;
}

export function requestAverage(usage: PatchTokenUsageEntry, kind: 'input' | 'output' | 'total'): number {
  const key = `average_${kind}_tokens_per_request` as keyof PatchTokenUsageEntry;
  const preferred = usage[key];
  if (typeof preferred === 'number') return preferred;
  const totalKey = `${kind}_tokens` as keyof PatchTokenUsageEntry;
  const total = usage[totalKey];
  return typeof total === 'number' ? total / Math.max(1, Number(usage.requests || 1)) : 0;
}

function usageBudgetState(usage: PatchTokenUsageEntry, budget?: LlmBudget | null): 'ok' | 'warning' | 'danger' {
  if (!budget?.max_context_length) return 'ok';
  const avgInput = requestAverage(usage, 'input');
  if (avgInput >= budget.max_context_length * budget.danger_threshold) return 'danger';
  if (avgInput >= budget.max_context_length * budget.warning_threshold) return 'warning';
  return 'ok';
}

export function TokenUsageSummary({
  tokenUsage,
  averageUnit = 'patch',
  heading = 'Token usage',
  agentKeys,
  budget,
  notesByAgent,
}: {
  tokenUsage?: PatchTokenUsage | null;
  averageUnit?: TokenAverageUnit;
  heading?: string;
  agentKeys?: string[];
  budget?: LlmBudget | null;
  notesByAgent?: Record<string, { tone: 'warning' | 'danger'; message: string }>;
}) {
  const allowedAgents = agentKeys ? new Set(agentKeys) : null;
  const agentEntries = Object.entries(tokenUsage?.agents ?? {})
    .filter(([key]) => !allowedAgents || allowedAgents.has(key))
    .filter(([, usage]) => usage.total_tokens > 0)
    .sort(([left], [right]) => {
      const order = [
        'file_ranking',
        'chunk_extraction',
        'quantity_vocab_selection',
        'qualitative_vocab_selection',
        'profile_projection',
        'dataset_summary',
        'dataset_level_projection',
        'dataset_level_projection_repair',
        'profile_target_planner',
        'profile_target_writer',
        'profile_patch',
        'initial_context',
        'patch_discovery',
        'schema_patch_writer',
        'schema_repair',
        'patch_extraction',
        'patch_quality',
        'auto_resolve',
      ];
      return (order.indexOf(left) === -1 ? order.length : order.indexOf(left))
        - (order.indexOf(right) === -1 ? order.length : order.indexOf(right));
    });
  const combined = !allowedAgents && tokenUsage?.combined && tokenUsage.combined.total_tokens > 0
    ? tokenUsage.combined
    : null;

  if (!agentEntries.length && !combined) return null;

  const rows = [
    ...agentEntries.map(([key, usage]) => ({
      key,
      label: tokenUsageLabels[key] || key,
      usage,
    })),
    ...(combined ? [{ key: 'combined', label: 'Combined', usage: combined }] : []),
  ];

  return (
    <div className="token-usage-summary">
      <span>{heading}</span>
      <div>
        {rows.map((row) => (
          <div className={`token-usage-row ${row.key === 'combined' ? 'combined' : ''} ${usageBudgetState(row.usage, budget)}`} key={row.key}>
            <strong>{row.label}</strong>
            <span>{formatTokenCount(usageAverage(row.usage, averageUnit, 'total'))} avg total / {tokenAverageUnitLabel(averageUnit)}</span>
            <small>
              {formatTokenCount(requestAverage(row.usage, 'input'))} context tokens / {formatTokenCount(requestAverage(row.usage, 'output'))} output avg per model call
              {' - '}
              {formatTokenCount(requestAverage(row.usage, 'total'))} total avg per model call
              {' - '}
              {formatTokenCount(row.usage.requests)} model call{row.usage.requests === 1 ? '' : 's'}
            </small>
            {(row.usage.average_response_duration_ms_per_request || row.usage.response_duration_ms) ? (
              <small>
                {formatDuration(row.usage.average_response_duration_ms_per_request ?? row.usage.response_duration_ms)} avg response generation
                {(row.usage.average_total_duration_ms_per_request || row.usage.total_duration_ms) ? ` - ${formatDuration(row.usage.average_total_duration_ms_per_request ?? row.usage.total_duration_ms)} avg total request` : ''}
              </small>
            ) : null}
            {budget?.max_context_length ? (
              <small>
                limit: {formatTokenCount(budget.max_context_length)} input tokens
                {usageBudgetState(row.usage, budget) === 'warning' && ' - Average input per model call is near the configured context window.'}
                {usageBudgetState(row.usage, budget) === 'danger' && ' - Average input per model call exceeds the configured context window.'}
              </small>
            ) : null}
            {notesByAgent?.[row.key] ? (
              <small className={`token-usage-note ${notesByAgent[row.key].tone}`}>
                {notesByAgent[row.key].message}
              </small>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

