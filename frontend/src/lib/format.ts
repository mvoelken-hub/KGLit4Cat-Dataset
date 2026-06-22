export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatExtractionStage(stage: string): string {
  return stage
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatTokenCount(value?: number): string {
  const numberValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return Math.round(numberValue).toLocaleString();
}

export function formatMemory(value?: number | null): string {
  if (!value || !Number.isFinite(value)) return 'N/A';
  return `${Math.round(value / 1024 / 1024).toLocaleString()} MB`;
}

export function formatDuration(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'N/A';
  if (value >= 1000) return `${(value / 1000).toFixed(1)} s`;
  return `${Math.round(value)} ms`;
}

export function labelDy(lineCount: number): string {
  return lineCount > 1 ? '-0.3em' : '0.32em';
}
