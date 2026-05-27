import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { chunkDataPackage, deleteDataPackage, getChunkStatus, getDataPackageChunks, getFileEntryContent, listDataPackages, uploadDataPackage } from './api/datasources';
import {
  extractInitialDraft,
  getExistingInitialContext,
  getExistingInitialDraft,
  getExtractionResult,
  getPatchArtifacts,
  getPatchProgress,
  getPatchReviewState,
  getProtectedFields,
  getTokenUsage,
  initialContextFromExtractionContext,
  patchDraft,
  runExtraction,
  saveDraft,
  saveInitialContext,
  savePatchReviewState,
  setProtectedFields as apiSetProtectedFields,
} from './api/extraction';
import { deleteProfile, getProfileJsonSchema, listProfiles, registerProfile } from './api/profiles';
import {
  getLlmBudget,
  getOllamaConfig,
  pullOllamaModel,
  removeOllamaModel,
  runOllamaPerformanceTest,
  updateOllamaRuntimeConfig,
  type LlmBudget,
  type OllamaConfig,
  type OllamaPerformanceTest,
} from './api/system';
import { JsonEditor, type JsonObject, type JsonPatchMarker, type JsonSchemaDocument, type JsonValue, setValueAtPath } from './components/JsonEditor';
import { ChunkingDialog } from './components/ChunkingDialog';
import { VocabularyPanel } from './components/VocabularyPanel';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryResponse, InitialContext, ProfileManifestResponse, TextQualityConfig } from './api/types';
import type {
  ExtractionChunkRef,
  ExtractionChunkResult,
  PatchArtifacts,
  PatchProgress,
  PatchReviewState,
  PatchTaskStatus,
  PatchTokenUsage,
  PatchTokenUsageEntry,
  RankedExtractionFile,
} from './api/extraction';

type BusyKey = 'upload' | 'chunk' | 'context' | 'draft' | 'patch' | 'load' | 'profile' | 'profile-delete' | 'dataset-delete' | 'ollama';
type ReviewItem = JsonPatchMarker & { kind: 'matched' | 'unmapped'; targetPath?: string; fact?: string; reason?: string; outcome?: string; resolutionNote?: string };

const emptyReviewState: PatchReviewState = {
  resolved_item_ids: [],
  unmapped_assignments: {},
  resolution_notes: {},
  resolved_at: {},
};

const selectedPackageStorageKey = 'simone_selected_package_id';

function readStoredSelectedPackageId(): string {
  try {
    return localStorage.getItem(selectedPackageStorageKey) || '';
  } catch {
    return '';
  }
}

function persistSelectedPackageId(packageId: string) {
  try {
    if (packageId) {
      localStorage.setItem(selectedPackageStorageKey, packageId);
    } else {
      localStorage.removeItem(selectedPackageStorageKey);
    }
  } catch {
    // ignore storage errors
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function formatExtractionStage(stage: string): string {
  return stage
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function EditableContextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="context-field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function EditableContextTextArea({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="context-field context-field-wide">
      <span>{label}</span>
      <textarea value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function PersistedContextField({
  label,
  value,
  wide = false,
}: {
  label: string;
  value?: string | number | null;
  wide?: boolean;
}) {
  return (
    <div className={`context-field context-field-readonly ${wide ? 'context-field-wide' : ''}`}>
      <span>{label}</span>
      <strong>{value || '-'}</strong>
    </div>
  );
}

function PersistedContextChips({ label, values }: { label: string; values: string[] }) {
  return (
    <div className="context-field context-field-wide context-field-readonly context-chips-field">
      <span>{label}</span>
      {values.length ? (
        <div className="chips">
          {values.map((value) => <span key={value}>{value}</span>)}
        </div>
      ) : (
        <strong>-</strong>
      )}
    </div>
  );
}

function chunkResultKey(chunk: Pick<ExtractionChunkResult, 'file_path' | 'start_idx' | 'end_idx'>): string {
  return `${chunk.file_path}:${chunk.start_idx}:${chunk.end_idx}`;
}

function extractionContextItemTitle(item: Record<string, unknown>, fallback: string): string {
  return String(item.identifier || item.title || item.label || fallback);
}

function extractionContextItemDescription(item: Record<string, unknown>): string | null {
  const value = item.description || item.value;
  return typeof value === 'string' && value.trim() ? value : null;
}

function extractionContextKeywords(item: Record<string, unknown>): string[] {
  const value = item.keywords;
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === 'string' && Boolean(entry.trim()))
    : [];
}

function extractionContextTraces(context?: Record<string, unknown> | null) {
  return asRecordArray(context?.extraction_objects)
    .map((trace) => ({
      objectType: String(trace.object_type || 'unknown'),
      sourceText: typeof trace.source_text === 'string' ? trace.source_text : '',
      object: asRecord(trace.extracted_object),
    }))
    .filter((trace) => trace.object);
}

function extractionContextTraceLabel(trace: { objectType: string; object: Record<string, unknown> | null }) {
  const label = trace.object ? extractionContextItemTitle(trace.object, 'Extracted object') : 'Extracted object';
  return `${trace.objectType.replace(/_/g, ' ')}: ${label}`;
}

function ExtractionContextResultView({ context }: { context?: Record<string, unknown> | null }) {
  if (!context) return <p className="muted">No extraction result is available for this chunk yet.</p>;

  const traces = extractionContextTraces(context);
  const sections = [
    { key: 'resource', label: 'Resources' },
    { key: 'method', label: 'Methods' },
    { key: 'data_generating_activity', label: 'Activities' },
    { key: 'evaluated_entity', label: 'Entities' },
    { key: 'agentic_entity', label: 'Agents' },
  ].map((section) => ({
    ...section,
    items: traces.filter((trace) => trace.objectType === section.key).map((trace) => trace.object).filter((item): item is Record<string, unknown> => Boolean(item)),
  })).filter((section) => section.items.length > 0);

  if (!sections.length) {
    return <p className="muted">The model call completed, but this chunk did not yield structured extraction context items.</p>;
  }

  return (
    <div className="extraction-context-results">
      {sections.map((section) => (
        <section key={section.key} className="extraction-result-section">
          <span>{section.label}</span>
          <div>
            {section.items.map((item, index) => {
              const keywords = extractionContextKeywords(item);
              return (
                <div className="extraction-result-card" key={`${section.key}-${index}`}>
                  <strong>{extractionContextItemTitle(item, `${section.label} ${index + 1}`)}</strong>
                  {extractionContextItemDescription(item) && <p>{extractionContextItemDescription(item)}</p>}
                  {keywords.length > 0 && (
                    <div className="extraction-result-keywords">
                      {keywords.map((keyword) => <span key={keyword}>{keyword}</span>)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

function ChunkTraceModal({
  chunk,
  content,
  context,
  onClose,
}: {
  chunk: ExtractionChunkResult;
  content: string;
  context?: Record<string, unknown> | null;
  onClose: () => void;
}) {
  const [activePanel, setActivePanel] = useState<'chunk' | 'unmatched'>('chunk');
  const traces = extractionContextTraces(context);
  const segments = buildTraceSegments(content, traces);
  const matchedTraces = new Set(segments.flatMap((segment) => segment.traces));
  const unmatchedTraces = traces.filter((trace) => trace.sourceText && !matchedTraces.has(trace));
  return createPortal(
    <div className="vocab-dialog-overlay" onClick={onClose}>
      <div className="vocab-dialog chunk-trace-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="vocab-dialog-header">
          <strong>Chunk {chunk.chunk_index + 1}: {chunk.file_path}</strong>
          <button className="ghost" onClick={onClose}>Close</button>
        </div>
        <div className="vocab-dialog-body chunk-trace-body">
          <div className="chunk-trace-meta">
            <span>Lines {chunk.start_idx}-{chunk.end_idx}</span>
            {chunk.context_tokens ? <span>{formatTokenCount(chunk.context_tokens)} context tokens</span> : null}
            {chunk.response_duration_ms ? <span>{formatDuration(chunk.response_duration_ms)} response generation</span> : null}
          </div>
          <div className="chunk-trace-tabs">
            <button
              className={activePanel === 'chunk' ? 'active' : ''}
              type="button"
              onClick={() => setActivePanel('chunk')}
              aria-expanded={activePanel === 'chunk'}
            >
              Chunk text
            </button>
            {unmatchedTraces.length > 0 && (
              <button
                className={activePanel === 'unmatched' ? 'active' : ''}
                type="button"
                onClick={() => setActivePanel('unmatched')}
                aria-expanded={activePanel === 'unmatched'}
              >
                Unmatched traces ({unmatchedTraces.length})
              </button>
            )}
          </div>
          {activePanel === 'chunk' && (
            <pre className="chunk-trace-text">
              {segments.map((segment, index) => segment.traces.length ? (
                <span className={`chunk-trace-highlight ${segment.tooltipBelow ? 'below' : ''}`} key={index}>
                  {segment.text}
                  <span className="chunk-trace-tooltip">
                    {segment.traces.map((trace, traceIndex) => (
                      <span key={`${trace.objectType}-${traceIndex}`}>
                        <strong>{extractionContextTraceLabel(trace)}</strong>
                        {trace.object && extractionContextItemDescription(trace.object) ? <small>{extractionContextItemDescription(trace.object)}</small> : null}
                      </span>
                    ))}
                  </span>
                </span>
              ) : <span key={index}>{segment.text}</span>)}
            </pre>
          )}
          {activePanel === 'unmatched' && unmatchedTraces.length > 0 && (
            <div className="chunk-unmatched-traces">
              <div>
                {unmatchedTraces.map((trace, index) => (
                  <section key={`${trace.objectType}-${index}`}>
                    <strong>{extractionContextTraceLabel(trace)}</strong>
                    <small>{trace.sourceText}</small>
                  </section>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}

function buildTraceSegments(content: string, traces: ReturnType<typeof extractionContextTraces>) {
  const sixthLineStart = nthLineStart(content, 6);
  const matches = traces
    .flatMap((trace) => {
      const sourceText = trace.sourceText.trim();
      if (!sourceText) return [];
      const match = findTraceRange(content, sourceText);
      return match ? [{ ...match, trace }] : [];
    })
    .sort((left, right) => left.start - right.start || right.end - left.end);
  const segments: Array<{ text: string; traces: typeof traces; tooltipBelow?: boolean }> = [];
  let cursor = 0;
  for (const match of matches) {
    if (match.start < cursor) {
      const previous = segments[segments.length - 1];
      if (previous?.traces.length && match.end <= cursor) previous.traces.push(match.trace);
      continue;
    }
    if (match.start > cursor) segments.push({ text: content.slice(cursor, match.start), traces: [] });
    segments.push({
      text: content.slice(match.start, match.end),
      traces: [match.trace],
      tooltipBelow: match.start <= sixthLineStart,
    });
    cursor = match.end;
  }
  if (cursor < content.length) segments.push({ text: content.slice(cursor), traces: [] });
  return segments.length ? segments : [{ text: content || 'No chunk text available.', traces: [] }];
}

function findTraceRange(content: string, sourceText: string): { start: number; end: number } | null {
  const candidates = uniqueStrings([
    sourceText.trim(),
    decodeTraceEscapes(sourceText.trim()),
    stripTracePromptMetadata(sourceText),
    decodeTraceEscapes(stripTracePromptMetadata(sourceText)),
  ]).filter((candidate) => candidate.length > 0);

  for (const candidate of candidates) {
    const direct = findDirectRange(content, candidate);
    if (direct) return direct;
  }
  for (const candidate of candidates) {
    const flexible = findFlexibleWhitespaceRange(content, candidate);
    if (flexible) return flexible;
  }
  for (const candidate of candidates) {
    const lineSequence = findLineSequenceRange(content, candidate);
    if (lineSequence) return lineSequence;
  }
  return null;
}

function findDirectRange(content: string, sourceText: string): { start: number; end: number } | null {
  const exactStart = content.indexOf(sourceText);
  if (exactStart >= 0) return { start: exactStart, end: exactStart + sourceText.length };
  const caseInsensitiveStart = content.toLowerCase().indexOf(sourceText.toLowerCase());
  return caseInsensitiveStart >= 0 ? { start: caseInsensitiveStart, end: caseInsensitiveStart + sourceText.length } : null;
}

function findFlexibleWhitespaceRange(content: string, sourceText: string, fromIndex = 0): { start: number; end: number } | null {
  const normalizedContent = normalizeTraceSearchText(content.slice(fromIndex), true);
  const normalizedSource = normalizeTraceSearchText(sourceText, false);
  if (normalizedSource.text.length < 3) return null;
  const normalizedStart = normalizedContent.text.indexOf(normalizedSource.text);
  if (normalizedStart < 0) return null;
  return {
    start: fromIndex + (normalizedContent.startMap[normalizedStart] ?? 0),
    end: fromIndex + (normalizedContent.endMap[normalizedStart + normalizedSource.text.length - 1] ?? content.length),
  };
}

function findLineSequenceRange(content: string, sourceText: string): { start: number; end: number } | null {
  const lines = sourceText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !isTracePromptMetadataLine(line));
  if (lines.length < 2 || lines.join('').length < 24) return null;

  let firstStart: number | null = null;
  let lastEnd = 0;
  let searchFrom = 0;
  for (const line of lines) {
    const relativeDirect = findDirectRange(content.slice(searchFrom), line);
    const match = relativeDirect
      ? { start: searchFrom + relativeDirect.start, end: searchFrom + relativeDirect.end }
      : findFlexibleWhitespaceRange(content, line, searchFrom);
    if (!match) return null;
    firstStart ??= match.start;
    lastEnd = match.end;
    searchFrom = match.end;
  }
  return firstStart === null ? null : { start: firstStart, end: lastEnd };
}

function normalizeTraceSearchText(value: string, keepMap: boolean) {
  const textParts: string[] = [];
  const startMap: number[] = [];
  const endMap: number[] = [];
  let whitespaceStart: number | null = null;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (/\s/.test(character)) {
      whitespaceStart ??= index;
      continue;
    }
    if (whitespaceStart !== null) {
      textParts.push(' ');
      if (keepMap) {
        startMap.push(whitespaceStart);
        endMap.push(index);
      }
      whitespaceStart = null;
    }
    textParts.push(character.toLowerCase());
    if (keepMap) {
      startMap.push(index);
      endMap.push(index + 1);
    }
  }
  if (whitespaceStart !== null) {
    textParts.push(' ');
    if (keepMap) {
      startMap.push(whitespaceStart);
      endMap.push(value.length);
    }
  }

  let start = 0;
  let end = textParts.length;
  while (start < end && textParts[start] === ' ') start += 1;
  while (end > start && textParts[end - 1] === ' ') end -= 1;
  return {
    text: textParts.slice(start, end).join(''),
    startMap: keepMap ? startMap.slice(start, end) : [],
    endMap: keepMap ? endMap.slice(start, end) : [],
  };
}

function stripTracePromptMetadata(sourceText: string): string {
  const lines = sourceText.split(/\r?\n/).filter((line) => !isTracePromptMetadataLine(line.trim()));
  const stripped = lines.join('\n').trim();
  return stripped || sourceText.trim();
}

function isTracePromptMetadataLine(line: string): boolean {
  return /^(Dataset name:|File path:|Line-Index-Span:|Chunk content \(after text-quality line filtering\):)/i.test(line);
}

function decodeTraceEscapes(value: string): string {
  return value.replace(/\\r\\n/g, '\n').replace(/\\n/g, '\n').replace(/\\t/g, '\t');
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function nthLineStart(content: string, lineNumber: number) {
  let index = 0;
  for (let line = 1; line < lineNumber; line += 1) {
    const next = content.indexOf('\n', index);
    if (next < 0) return content.length;
    index = next + 1;
  }
  return index;
}

function ExtractionContextOverview({
  rankedFiles,
  chunkResults,
  currentChunk,
  chunksByFile,
  packageFiles,
  progress,
  status,
}: {
  rankedFiles: RankedExtractionFile[];
  chunkResults: ExtractionChunkResult[];
  currentChunk?: ExtractionChunkRef | null;
  chunksByFile: ChunkResponse[][];
  packageFiles: FileEntryResponse[];
  progress?: PatchProgress | null;
  status?: PatchTaskStatus | null;
}) {
  const [traceChunk, setTraceChunk] = useState<{ chunk: ExtractionChunkResult; content: string } | null>(null);
  const resultByKey = new Map(chunkResults.map((chunk) => [chunkResultKey(chunk), chunk]));
  const packageFileByPath = new Map(packageFiles.map((file) => [file.file_path, file]));
  const chunkGroupsByPath = new Map(
    chunksByFile
      .map((group): [string, ChunkResponse[]] => [group[0]?.file_path ?? '', group])
      .filter(([filePath]) => Boolean(filePath)),
  );
  const rankedFilePaths = rankedFiles
    .slice()
    .sort((left, right) => left.rank - right.rank)
    .map((file) => file.file_path);
  const fallbackFilePaths = [
    ...packageFiles.map((file) => file.file_path),
    ...chunksByFile.map((group) => group[0]?.file_path).filter((filePath): filePath is string => Boolean(filePath)),
    ...chunkResults.map((chunk) => chunk.file_path),
  ];
  const filePaths = Array.from(new Set([...rankedFilePaths, ...fallbackFilePaths]));
  const totalChunks = progress?.total_chunks || chunkResults.length || chunksByFile.flat().length;
  const completedChunks = chunkResults.filter((chunk) => chunk.status === 'completed').length;
  const runningChunks = chunkResults.filter((chunk) => chunk.status === 'running').length;
  const failedChunks = chunkResults.filter((chunk) => chunk.status === 'failed').length;

  if (!filePaths.length) {
    return (
      <div className="extraction-overview-empty">
        <strong>No extraction context overview yet.</strong>
        <p>Create chunks before starting extraction context extraction.</p>
      </div>
    );
  }

  return (
    <div className="extraction-overview">
      <div className="extraction-overview-summary">
        <div>
          <span>Status</span>
          <strong>{formatExtractionStage(status || 'unknown')}</strong>
        </div>
        <div>
          <span>Chunks</span>
          <strong>{completedChunks}/{totalChunks || 0} extracted</strong>
        </div>
        <div>
          <span>Current stage</span>
          <strong>{formatExtractionStage(progress?.stage || 'not started')}</strong>
        </div>
        <div>
          <span>Queue</span>
          <strong>{runningChunks ? `${runningChunks} running` : failedChunks ? `${failedChunks} failed` : `${Math.max(0, (totalChunks || 0) - completedChunks)} pending`}</strong>
        </div>
      </div>

      <div className="ranked-file-list">
        {filePaths.map((filePath, fileIndex) => {
          const rank = rankedFiles.find((file) => file.file_path === filePath)?.rank ?? fileIndex + 1;
          const file = packageFileByPath.get(filePath);
          const chunks = chunkGroupsByPath.get(filePath) ?? [];
          const persistedChunks = chunkResults.filter((chunk) => chunk.file_path === filePath);
          const chunkCards = chunks.length
            ? chunks.map((chunk, index) => {
              const result = resultByKey.get(chunkResultKey(chunk));
              const fallbackChunk: ExtractionChunkResult = {
                chunk_index: index,
                file_path: chunk.file_path,
                start_idx: chunk.start_idx,
                end_idx: chunk.end_idx,
                status: 'pending',
                extraction_context: null,
              };
              return result ?? fallbackChunk;
            })
            : persistedChunks;
          const hasRunningChunk = chunkCards.some((chunk) => chunk.status === 'running' || (currentChunk && chunkResultKey(currentChunk) === chunkResultKey(chunk)));

          return (
            <details className="ranked-file" key={filePath} open={hasRunningChunk || fileIndex === 0 ? true : undefined}>
              <summary>
                <span className="rank-badge">#{rank}</span>
                <strong>{filePath}</strong>
                <small>
                  {chunkCards.length} chunk{chunkCards.length === 1 ? '' : 's'}
                  {file ? ` - ${formatBytes(file.byte_size)}` : ''}
                </small>
              </summary>
              <div className="extraction-chunk-list">
                {chunkCards.length ? chunkCards.map((chunk) => {
                  const running = chunk.status === 'running' || (currentChunk && chunkResultKey(currentChunk) === chunkResultKey(chunk));
                  const statusClass = running ? 'running' : chunk.status === 'completed' ? 'completed' : chunk.status === 'failed' ? 'failed' : 'pending';
                  const sourceChunk = (chunkGroupsByPath.get(chunk.file_path) ?? []).find((item) => chunkResultKey(item) === chunkResultKey(chunk));
                  const chunkText = sourceChunk?.content ?? '';
                  return (
                    <details className={`extraction-chunk-card ${statusClass}`} key={chunkResultKey(chunk)} open={running ? true : undefined}>
                      <summary>
                        <div>
                          <strong>Chunk {chunk.chunk_index + 1}</strong>
                          <small>Lines {chunk.start_idx}-{chunk.end_idx}</small>
                        </div>
                        <span className="chunk-status">
                          {running && <i aria-hidden="true" />}
                          {running ? 'extracting' : chunk.status}
                        </span>
                      </summary>
                      {chunk.status === 'failed' && chunk.error && <p className="warning">{chunk.error}</p>}
                      {(chunk.status === 'completed' || chunkText) && (
                        <div className="chunk-call-meta">
                          {chunk.status === 'completed' && chunk.context_tokens ? <span>{formatTokenCount(chunk.context_tokens)} context tokens</span> : null}
                          {chunk.status === 'completed' && chunk.response_duration_ms ? <span>{formatDuration(chunk.response_duration_ms)} response generation</span> : null}
                          {chunkText ? <button className="small ghost" type="button" onClick={() => setTraceChunk({ chunk, content: chunkText })}>View chunk text</button> : null}
                        </div>
                      )}
                      <ExtractionContextResultView context={chunk.extraction_context} />
                    </details>
                  );
                }) : (
                  <p className="muted">No chunks are available for this ranked file.</p>
                )}
              </div>
            </details>
          );
        })}
      </div>
      {traceChunk && (
        <ChunkTraceModal
          chunk={traceChunk.chunk}
          content={traceChunk.content}
          context={traceChunk.chunk.extraction_context}
          onClose={() => setTraceChunk(null)}
        />
      )}
    </div>
  );
}

function StepPanel({
  number,
  title,
  description,
  children,
  actions,
  active = false,
}: {
  number: string;
  title: string;
  description: string;
  children: ReactNode;
  actions?: ReactNode;
  active?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const contentId = `step-panel-${number}`;

  return (
    <article className={`step-card ${active ? 'active' : ''} ${collapsed ? 'collapsed' : ''}`}>
      <div className="step-index">{number}</div>
      <div className="step-body">
        <div className="step-header">
          <button
            className="step-collapse-toggle"
            type="button"
            aria-expanded={!collapsed}
            aria-controls={contentId}
            onClick={() => setCollapsed((value) => !value)}
            title={collapsed ? `Expand ${title}` : `Collapse ${title}`}
          >
            {collapsed ? '+' : '-'}
          </button>
          <div className="step-title">
            <h2>{title}</h2>
            <p>{description}</p>
          </div>
          {actions && <div className="step-header-actions">{actions}</div>}
        </div>
        {!collapsed && (
          <div className="step-content" id={contentId}>
            {children}
          </div>
        )}
      </div>
    </article>
  );
}

function contextTechnique(context: InitialContext): string | null | undefined {
  return context.activities?.find((activity) => activity.technique)?.technique;
}

function contextActivityLabel(context: InitialContext): string | null {
  const labels = context.activities
    ?.map((activity) => activity.label || activity.technique)
    .filter(Boolean) as string[] | undefined;
  return labels?.length ? labels.join(', ') : null;
}

function contextAgentLabel(context: InitialContext): string | null {
  if (!context.agents?.length) return null;
  const labels = context.agents.map((agent) => agent.model ? `${agent.name} (${agent.model})` : agent.name).filter(Boolean);
  return labels.length ? labels.join(', ') : null;
}

function contextEntityLabel(context: InitialContext): string | null {
  const labels = context.entities?.map((entity) => entity.label).filter(Boolean) ?? [];
  return labels.length ? labels.join(', ') : null;
}

function commaList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function contextEntitiesInput(context: InitialContext): string {
  return (context.entities ?? []).map((entity) => entity.label).join(', ');
}

function updateContextDatasetTitle(context: InitialContext, value: string): InitialContext {
  return { ...context, dataset_title: value.trim() || null };
}

function updateContextTechnique(context: InitialContext, value: string): InitialContext {
  const activities = [...(context.activities ?? [])];
  const first = activities[0] ?? { label: null, technique: null, agent_names: [] };
  activities[0] = { ...first, technique: value.trim() || null };
  return { ...context, activities };
}

function updateContextAgentName(context: InitialContext, value: string): InitialContext {
  const agents = [...(context.agents ?? [])];
  const first = agents[0] ?? { name: '', role: 'unknown' as const };
  agents[0] = { ...first, name: value };
  return { ...context, agents };
}

function updateContextAgentModel(context: InitialContext, value: string): InitialContext {
  const agents = [...(context.agents ?? [])];
  const first = agents[0] ?? { name: '', role: 'unknown' as const };
  agents[0] = { ...first, model: value.trim() || null };
  return { ...context, agents };
}

function updateContextEntities(context: InitialContext, value: string): InitialContext {
  const labels = commaList(value);
  const entities = labels.map((label, index) => {
    const existing = context.entities?.[index];
    return existing ? { ...existing, label } : { label, role: 'unknown' as const };
  });
  return { ...context, entities };
}

function updateContextActivityLabel(context: InitialContext, value: string): InitialContext {
  const activities = [...(context.activities ?? [])];
  const first = activities[0] ?? { label: null, technique: null, agent_names: [] };
  activities[0] = { ...first, label: value.trim() || null };
  return { ...context, activities };
}

function updateContextDescription(context: InitialContext, value: string): InitialContext {
  return { ...context, dataset_description: value.trim() || null };
}

function updateContextKeywords(context: InitialContext, value: string): InitialContext {
  return { ...context, keywords: commaList(value) };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord).filter((item): item is Record<string, unknown> => Boolean(item)) : [];
}

function hasPatchArtifacts(artifacts: PatchArtifacts): boolean {
  return artifacts.patches.length > 0 || artifacts.quality_reports.length > 0 || artifacts.unmapped_facts.length > 0;
}

function patchArtifactBaseName(fileName?: string): string {
  return (fileName || '')
    .replace(/\.quality_report\.json$/, '.json')
    .replace(/\.candidates\.json$/, '.json')
    .replace(/\.accepted\.json$/, '.json')
    .replace(/\.raw\.json$/, '.json')
    .replace(/\.unmapped_facts\.json$/, '.json');
}

function confidenceLabel(confidence?: number): string {
  return confidence === undefined ? 'unknown confidence' : `${Math.round(confidence * 100)}% confidence`;
}

function textList(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)).filter(Boolean) : [];
}

function issueList(issues: Record<string, unknown>[]): string[] {
  return issues.map((issue) => [
    issue.issue_type ? String(issue.issue_type) : '',
    issue.severity ? `(${String(issue.severity)})` : '',
    issue.explanation ? String(issue.explanation) : '',
    issue.suggested_target_path ? `Suggested field: ${String(issue.suggested_target_path)}` : '',
  ].filter(Boolean).join(' '));
}

function unmappedFactKey(fact: Record<string, unknown>): string {
  return [fact.file_name, fact.fact, fact.reason, fact.source_hint].map((item) => String(item || '')).join('|');
}

function matchedReviewItemId(baseName: string, path: string, index: number): string {
  return `matched:${baseName}:${path}:${index}`;
}

function unmappedReviewItemId(fact: Record<string, unknown>): string {
  return `unmapped:${unmappedFactKey(fact)}`;
}

function reviewOutcomeFromNote(note?: string): string | undefined {
  const match = note?.match(/^(included|already_present|excluded|unresolved):/);
  return match?.[1];
}

function displayResolutionLogEntry(entry: string): string {
  return entry.replace(/^[0-9a-f]{64}:\s+/i, '');
}

function buildReviewItems(artifacts: PatchArtifacts | null, reviewState: PatchReviewState): ReviewItem[] {
  if (!artifacts) return [];
  const resolvedIds = new Set(reviewState.resolved_item_ids);
  const ratingByPatchAndField = new Map<string, Record<string, unknown>>();

  for (const report of artifacts.quality_reports) {
    const baseName = patchArtifactBaseName(report.file_name);
    const reportContent = asRecord(report.content);
    for (const rating of asRecordArray(reportContent?.candidate_ratings)) {
      const fieldPath = String(rating.field_path || '');
      if (fieldPath) ratingByPatchAndField.set(`${baseName}:${fieldPath}`, rating);
    }
  }

  const items: ReviewItem[] = [];
  for (const artifact of artifacts.patches) {
    if (artifact.artifact_type !== 'candidates') continue;
    const baseName = patchArtifactBaseName(artifact.file_name);
    asRecordArray(artifact.content).forEach((candidate, candidateIndex) => {
      const path = String(candidate.field_path || '');
      if (!path) return;
      const confidence = typeof candidate.confidence === 'number' ? candidate.confidence : undefined;
      const rating = ratingByPatchAndField.get(`${baseName}:${path}`);
      const decision = String(rating?.decision || '');
      const issues = asRecordArray(rating?.issues);
      const needsReview = confidence === undefined || confidence < 0.8 || decision !== 'accept' || issues.length > 0;
      const id = matchedReviewItemId(baseName, path, candidateIndex);
      const resolutionNote = reviewState.resolution_notes[id];
      items.push({
        id,
        kind: 'matched',
        path,
        status: needsReview ? 'needs_review' : 'accepted',
        label: needsReview ? 'Review' : 'Patch',
        resolved: !needsReview || resolvedIds.has(id),
        outcome: reviewOutcomeFromNote(resolutionNote),
        resolutionNote,
        confidence,
        fileName: String(artifact.file_name || ''),
        patch: candidate.patch,
        evidence: textList(candidate.source_evidence),
        issues: issueList(issues),
        detail: [
          confidenceLabel(confidence),
          decision ? `Decision: ${decision}` : '',
          issues.length ? `${issues.length} issue${issues.length === 1 ? '' : 's'}` : '',
          String(candidate.reasoning || ''),
        ].filter(Boolean).join(' - '),
      });
    });
  }

  for (const fact of artifacts.unmapped_facts) {
    const id = unmappedReviewItemId(fact);
    const assignedPath = reviewState.unmapped_assignments[id] || '';
    const resolutionNote = reviewState.resolution_notes[id];
    items.push({
      id,
      kind: 'unmapped',
      path: assignedPath || 'Unassigned',
      targetPath: assignedPath,
      status: 'unmapped',
      label: 'Unmapped',
      resolved: resolvedIds.has(id),
      outcome: reviewOutcomeFromNote(resolutionNote),
      resolutionNote,
      fileName: String(fact.file_name || ''),
      evidence: fact.source_hint ? [String(fact.source_hint)] : [],
      detail: String(fact.fact || fact.reason || 'Unmapped source fact'),
      fact: String(fact.fact || ''),
      reason: String(fact.reason || ''),
    });
  }

  return items;
}

function ResolutionLogList({ entries }: { entries: string[] }) {
  if (!entries.length) return null;
  return (
    <div className="resolution-log">
      <span>Resolution log</span>
      <ol>
        {entries.map((entry, index) => <li key={`${index}-${entry}`}>{displayResolutionLogEntry(entry)}</li>)}
      </ol>
    </div>
  );
}

const tokenUsageLabels: Record<string, string> = {
  initial_context: 'Context extraction',
  file_ranking: 'File ranking',
  chunk_extraction: 'Chunk extraction',
  chunk_extraction_repair: 'Chunk extraction repair',
  quantity_vocab_selection: 'Quantity vocabulary',
  qualitative_vocab_selection: 'Qualitative vocabulary',
  profile_projection: 'Profile projection',
  patch_discovery: 'Patch discovery',
  schema_patch_writer: 'Schema patch writer',
  schema_repair: 'Schema repair',
  patch_extraction: 'Patch extraction',
  patch_quality: 'Patch quality review',
  auto_resolve: 'Auto-resolve',
};

function formatTokenCount(value?: number): string {
  const numberValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return Math.round(numberValue).toLocaleString();
}

function usageAverage(usage: PatchTokenUsageEntry, unit: 'patch' | 'operation', kind: 'input' | 'output' | 'total'): number {
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

function requestAverage(usage: PatchTokenUsageEntry, kind: 'input' | 'output' | 'total'): number {
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

function TokenUsageSummary({
  tokenUsage,
  averageUnit = 'patch',
  heading = 'Token usage',
  agentKeys,
  budget,
}: {
  tokenUsage?: PatchTokenUsage | null;
  averageUnit?: 'patch' | 'operation';
  heading?: string;
  agentKeys?: string[];
  budget?: LlmBudget | null;
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
            <span>{formatTokenCount(usageAverage(row.usage, averageUnit, 'total'))} avg total / {averageUnit === 'patch' ? 'patch' : 'operation'}</span>
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
          </div>
        ))}
      </div>
    </div>
  );
}

function formatMemory(value?: number | null): string {
  if (!value || !Number.isFinite(value)) return 'N/A';
  return `${Math.round(value / 1024 / 1024).toLocaleString()} MB`;
}

function formatDuration(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return 'N/A';
  if (value >= 1000) return `${(value / 1000).toFixed(1)} s`;
  return `${Math.round(value)} ms`;
}

function OllamaSettingsPanel({
  config,
  budget,
  tokenUsage,
  patchTokenUsage,
  busy,
  onApply,
  onRefresh,
  onPullModel,
  onRemoveModel,
  onRunPerformanceTest,
}: {
  config: OllamaConfig | null;
  budget: LlmBudget | null;
  tokenUsage: PatchTokenUsage | null;
  patchTokenUsage?: PatchTokenUsage | null;
  busy: boolean;
  onApply: (values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_batch_size: number;
    embedding_num_gpu: number;
  }) => void;
  onRefresh: () => void;
  onPullModel: (model: string) => Promise<void>;
  onRemoveModel: (model: string) => Promise<void>;
  onRunPerformanceTest: (values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_num_gpu: number;
  }) => Promise<OllamaPerformanceTest>;
}) {
  const runtime = config?.runtime;
  const combinedUsage = patchTokenUsage?.combined ?? tokenUsage?.combined ?? null;
  const averageInput = combinedUsage ? requestAverage(combinedUsage, 'input') : 0;
  const [chatModel, setChatModel] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('');
  const [maxContextLength, setMaxContextLength] = useState(8192);
  const [embeddingBatchSize, setEmbeddingBatchSize] = useState(32);
  const [embeddingNumGpu, setEmbeddingNumGpu] = useState(-1);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [pullModelName, setPullModelName] = useState('');
  const [performanceTest, setPerformanceTest] = useState<OllamaPerformanceTest | null>(null);
  const [localMessage, setLocalMessage] = useState('');

  useEffect(() => {
    if (!runtime) return;
    setChatModel(runtime.chat_model);
    setEmbeddingModel(runtime.embedding_model);
    setMaxContextLength(runtime.max_context_length);
    setEmbeddingBatchSize(runtime.embedding_batch_size);
    setEmbeddingNumGpu(runtime.embedding_num_gpu);
  }, [runtime]);

  function submit(event: FormEvent) {
    event.preventDefault();
    onApply({
      chat_model: chatModel.trim(),
      embedding_model: embeddingModel.trim(),
      max_context_length: Math.max(512, maxContextLength),
      embedding_batch_size: Math.max(1, embeddingBatchSize),
      embedding_num_gpu: embeddingNumGpu,
    });
  }

  async function runSwitchTest() {
    if (!runtime) return;
    setLocalMessage('');
    try {
      const result = await onRunPerformanceTest({
        chat_model: chatModel.trim(),
        embedding_model: embeddingModel.trim(),
        max_context_length: Math.max(512, maxContextLength),
        embedding_num_gpu: embeddingNumGpu,
      });
      setPerformanceTest(result);
      setLocalMessage('Switch test completed.');
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Switch test failed.');
    }
  }

  async function pullModel(event: FormEvent) {
    event.preventDefault();
    const model = pullModelName.trim();
    if (!model) return;
    const host = config?.host.base_url ?? 'the configured Ollama host';
    if (!window.confirm(`Pull ${model} on ${host}? This mutates the Ollama host.`)) return;
    setLocalMessage('');
    try {
      await onPullModel(model);
      setPullModelName('');
      setLocalMessage(`Pull requested for ${model}.`);
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Could not pull model.');
    }
  }

  async function removeModel(model: string) {
    const host = config?.host.base_url ?? 'the configured Ollama host';
    if (!window.confirm(`Remove ${model} from ${host}? This mutates the Ollama host.`)) return;
    setLocalMessage('');
    try {
      await onRemoveModel(model);
      setLocalMessage(`Removed ${model}.`);
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Could not remove model.');
    }
  }

  const availableModels = config?.models.models ?? [];
  const chatModelOptions = Array.from(new Set([
    chatModel,
    ...availableModels
      .filter((model) => model.kind !== 'embedding')
      .map((model) => model.model || model.name || '')
      .filter(Boolean),
  ])).filter(Boolean);
  const embeddingModelOptions = Array.from(new Set([
    embeddingModel,
    ...availableModels
      .filter((model) => model.kind === 'embedding')
      .map((model) => model.model || model.name || '')
      .filter(Boolean),
  ])).filter(Boolean);
  const selectedChatModelInfo = availableModels.find((model) => (model.model || model.name) === chatModel);
  const selectedChatModelName = chatModel.toLowerCase();
  const selectedChatModelSize = selectedChatModelInfo?.size ?? null;
  const selectedChatModelIsCloud = Boolean(selectedChatModelInfo?.is_cloud) || selectedChatModelName.includes('cloud') || selectedChatModelSize === 0;
  const showPerformanceTest = Boolean(runtime && !selectedChatModelIsCloud);

  useEffect(() => {
    if (selectedChatModelIsCloud) setPerformanceTest(null);
  }, [selectedChatModelIsCloud]);

  return (
    <section className="ollama-panel" aria-label="Ollama runtime settings">
      <div className="ollama-panel-header">
        <div>
          <span>Ollama</span>
          <strong>{config?.host.base_url ?? 'Unavailable'}</strong>
          {config && <small className="ollama-mode-chip">{config.host.is_local ? 'Local' : 'Remote'}</small>}
        </div>
        <button
          className="ghost small"
          type="button"
          onClick={() => setDetailsOpen(true)}
        >
          Details
        </button>
      </div>

      <div className="ollama-summary-grid">
        <div>
          <span>Models</span>
          <strong>{runtime?.chat_model ?? 'No chat model'}</strong>
          <small>{runtime?.embedding_model ?? 'No embedding model'}</small>
        </div>
        <div>
          <span>Context budget</span>
          <strong>{formatTokenCount(budget?.input_token_budget ?? runtime?.input_token_budget)} input tokens</strong>
          <small>{formatTokenCount(budget?.max_context_length ?? runtime?.max_context_length)} total context tokens</small>
        </div>
      </div>

      {detailsOpen && createPortal((
        <div className="vocab-dialog-overlay" onClick={() => setDetailsOpen(false)}>
          <div className="vocab-dialog ollama-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <strong>Ollama Runtime Settings</strong>
              <div className="ollama-dialog-actions">
                <button className="ghost small" type="button" onClick={onRefresh} disabled={busy}>Refresh</button>
                <button className="ghost" type="button" onClick={() => setDetailsOpen(false)}>Close</button>
              </div>
            </div>
            <div className="vocab-dialog-body ollama-dialog-body">
              <div className="ollama-status-grid">
                <div>
                  <span>Host mode</span>
                  <strong>{config ? (config.host.is_local ? 'Local host' : 'Remote host') : 'Unknown'}</strong>
                  <small>{config?.host.is_local ? config.host.server_settings_note : 'Remote mode shows runtime controls and Ollama API diagnostics only.'}</small>
                </div>
                {config?.host.is_local && (
                  <div>
                    <span>Server memory settings</span>
                    <strong>flash {String(config.host.flash_attention ?? false)} / KV {config.host.kv_cache_type ?? 'unknown'}</strong>
                    <small>Change these on the Ollama host, not from the UI.</small>
                  </div>
                )}
                <div>
                  <span>Context budget</span>
                  <strong>{formatTokenCount(budget?.input_token_budget ?? runtime?.input_token_budget)} input tokens</strong>
                  <small>{formatTokenCount(budget?.max_context_length ?? runtime?.max_context_length)} total context tokens</small>
                </div>
                <div>
                  <span>Recent average input</span>
                  <strong>{formatTokenCount(averageInput)} tokens</strong>
                  <small>{averageInput && runtime && averageInput > runtime.input_token_budget * 0.8 ? 'Lower context usage before starting the next run.' : 'Use chunks per turn to tune call size.'}</small>
                </div>
                <div>
                  <span>Diagnostics</span>
                  <strong>{config?.diagnostics.status ?? 'unknown'}</strong>
                  <small>{config?.diagnostics.summary ?? 'Run the switch test for residency guidance.'}</small>
                </div>
              </div>

              <form className="ollama-runtime-form" onSubmit={submit}>
                <label>
                  <span>Chat model</span>
                  <select value={chatModel} onChange={(event) => setChatModel(event.target.value)} disabled={!runtime || busy || chatModelOptions.length === 0}>
                    {chatModelOptions.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Embedding model</span>
                  <select value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)} disabled={!runtime || busy || embeddingModelOptions.length === 0}>
                    {embeddingModelOptions.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Max context</span>
                  <input type="number" min={512} step={512} value={maxContextLength} onChange={(event) => setMaxContextLength(parseInt(event.target.value, 10) || 512)} disabled={!runtime || busy} />
                </label>
                <label>
                  <span>Embedding batch</span>
                  <input type="number" min={1} value={embeddingBatchSize} onChange={(event) => setEmbeddingBatchSize(parseInt(event.target.value, 10) || 1)} disabled={!runtime || busy} />
                </label>
                <label>
                  <span>Embedding GPU</span>
                  <select value={embeddingNumGpu} onChange={(event) => setEmbeddingNumGpu(parseInt(event.target.value, 10))} disabled={!runtime || busy}>
                    <option value={-1}>Auto</option>
                    <option value={0}>CPU only</option>
                    <option value={999}>GPU only</option>
                  </select>
                </label>
                <button type="submit" disabled={!runtime || busy}>Apply runtime settings</button>
              </form>

              {showPerformanceTest && (
                <div className="ollama-test-panel">
                  <div className="ollama-section-header">
                    <div>
                      <span>Performance test</span>
                      <strong>Embedding/chat switch test</strong>
                    </div>
                    <button className="ghost small" type="button" onClick={() => void runSwitchTest()} disabled={!runtime || busy}>
                      Run switch test
                    </button>
                  </div>
                  <small>Loads embedding -{'>'} chat -{'>'} embedding -{'>'} chat to detect reload pressure.</small>
                  {performanceTest && (
                    <div className="ollama-test-result">
                      <strong>{performanceTest.diagnostics.summary}</strong>
                      <div className="ollama-test-steps">
                        {performanceTest.steps.map((step) => (
                          <div key={step.key}>
                            <span>{step.label}</span>
                            <strong>{step.success ? formatDuration(step.load_duration_ms) : 'failed'}</strong>
                            <small>{step.success ? `${step.snapshot.models.length} loaded after call` : step.error}</small>
                          </div>
                        ))}
                      </div>
                      {performanceTest.diagnostics.recommendations.length > 0 && (
                        <div className="ollama-recommendations">
                          {performanceTest.diagnostics.recommendations.map((recommendation) => (
                            <small key={recommendation}>{recommendation}</small>
                          ))}
                          <div className="ollama-recommendation-actions">
                            <button className="ghost small" type="button" onClick={() => setEmbeddingNumGpu(0)} disabled={busy}>
                              Set embeddings to CPU
                            </button>
                            <button className="ghost small" type="button" onClick={() => setEmbeddingBatchSize(Math.max(1, Math.floor(embeddingBatchSize / 2)))} disabled={busy}>
                              Halve embed batch
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="ollama-loaded-models">
                <span>Loaded models</span>
                {!config?.running.available && <small>{config?.running.error?.message ?? 'Could not inspect loaded Ollama models.'}</small>}
                {config?.running.available && !config.running.models.length && <small>No models are currently resident.</small>}
                {config?.running.models.map((model, index) => (
                  <small key={model.model || index}>
                    {model.model || 'unknown'} - {model.processor ? `Processor ${model.processor}` : `VRAM ${formatMemory(model.size_vram)} / total ${formatMemory(model.size)}`}
                    {model.processor ? ` - VRAM ${formatMemory(model.size_vram)} / total ${formatMemory(model.size)}` : ''}
                    {model.context_length ? ` - context ${formatTokenCount(model.context_length)}` : ''}
                  </small>
                ))}
                {config?.running.available && (
                  <small>
                    Chat {config.running.chat_model_loaded ? 'resident' : 'not resident'}; embedding {config.running.embedding_model_loaded ? 'resident' : 'not resident'}.
                  </small>
                )}
              </div>

              <div className="ollama-model-overview">
                <div className="ollama-section-header">
                  <div>
                    <span>Available models</span>
                    <strong>{config?.models.available ? `${availableModels.length} installed` : 'Unavailable'}</strong>
                  </div>
                </div>
                {!config?.models.available && <small>{config?.models.error?.message ?? 'Could not inspect installed models.'}</small>}
                <form className="ollama-pull-form" onSubmit={(event) => void pullModel(event)}>
                  <input
                    value={pullModelName}
                    onChange={(event) => setPullModelName(event.target.value)}
                    placeholder="model:tag"
                    disabled={busy}
                  />
                  <button type="submit" disabled={busy || !pullModelName.trim()}>Pull model</button>
                </form>
                {availableModels.length > 0 && (
                  <div className="ollama-model-list">
                    {availableModels.map((model, index) => {
                      const name = model.model || model.name || '';
                      const canUseAsChat = model.kind !== 'embedding';
                      const canUseAsEmbedding = model.kind === 'embedding';
                      return (
                        <div className="ollama-model-row" key={name || index}>
                          <div>
                            <strong>{name || 'unknown'}</strong>
                            <small>
                              {formatMemory(model.size)}
                              {model.details?.parameter_size ? ` - ${model.details.parameter_size}` : ''}
                              {model.details?.quantization_level ? ` - ${model.details.quantization_level}` : ''}
                              {model.kind ? ` - ${model.kind}` : ''}
                              {model.is_cloud ? ' - Cloud' : ''}
                            </small>
                          </div>
                          <div className="ollama-model-actions">
                            {canUseAsChat && (
                              <button className="ghost small" type="button" onClick={() => setChatModel(name)} disabled={!name || busy}>Chat</button>
                            )}
                            {canUseAsEmbedding && (
                              <button className="ghost small" type="button" onClick={() => setEmbeddingModel(name)} disabled={!name || busy}>Embed</button>
                            )}
                            <button className="ghost small danger-button" type="button" onClick={() => void removeModel(name)} disabled={!name || busy}>Remove</button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {localMessage && <p className="muted">{localMessage}</p>}
              <p className="muted">ℹ️ Runtime edits affect future SIMONE calls only and reset when the API restarts.</p>
            </div>
          </div>
        </div>
      ), document.body)}
    </section>
  );
}

function FileViewer({ file, content, chunksByFile, onClose }: {
  file: FileEntryResponse;
  content: string;
  chunksByFile: ChunkResponse[][];
  onClose: () => void;
}) {
  const lines = content.split('\n');
  const fileChunks = chunksByFile.find((group) => group[0]?.file_path === file.file_path) ?? [];

  const chunkColors = [
    'rgba(99, 154, 0, 0.22)',
    'rgba(0, 120, 180, 0.18)',
    'rgba(180, 90, 0, 0.18)',
    'rgba(140, 60, 180, 0.18)',
    'rgba(200, 50, 80, 0.18)',
    'rgba(0, 160, 140, 0.18)',
  ];

  const getChunkInfo = (lineIndex: number): { inChunk: boolean; color: string; chunkIndex: number } => {
    for (let i = 0; i < fileChunks.length; i++) {
      const chunk = fileChunks[i];
      const indices = chunk.filtered_line_indices;
      if (indices && indices.length > 0) {
        if (indices.includes(lineIndex)) {
          return { inChunk: true, color: chunkColors[i % chunkColors.length], chunkIndex: i };
        }
      } else if (lineIndex >= chunk.start_idx && lineIndex <= chunk.end_idx) {
        return { inChunk: true, color: chunkColors[i % chunkColors.length], chunkIndex: i };
      }
    }
    return { inChunk: false, color: '', chunkIndex: -1 };
  };

  const viewer = (
    <div className="file-viewer-overlay" onClick={onClose}>
      <div className="file-viewer" onClick={(e) => e.stopPropagation()}>
        <div className="file-viewer-header">
          <strong>{file.file_path}</strong>
          <button className="ghost" onClick={onClose}>Close</button>
        </div>
        <div className="file-viewer-body">
          {lines.map((line, index) => {
            const info = getChunkInfo(index);
            return (
              <div
                key={index}
                className={`file-viewer-line ${info.inChunk ? 'chunk-highlight' : ''}`}
                style={info.inChunk ? { background: info.color } : undefined}
                title={info.inChunk ? `Chunk ${info.chunkIndex + 1}` : undefined}
              >
                <span className="line-number">{index + 1}</span>
                <span className="line-content">{line || ' '}</span>
              </div>
            );
          })}
        </div>
        {fileChunks.length > 0 && (
          <div className="file-viewer-footer">
            <div className="chunk-legend">
              {fileChunks.map((chunk, i) => (
                <span key={i} className="chunk-legend-item" style={{ background: chunkColors[i % chunkColors.length] }}>
                  Chunk {i + 1}: {(chunk.filtered_line_indices?.length ?? 0) > 0 ? chunk.filtered_line_indices!.length : chunk.end_idx - chunk.start_idx + 1} lines
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return createPortal(viewer, document.body);
}

export function App() {
  const [packages, setPackages] = useState<DataPackageResponse[]>([]);
  const [profiles, setProfiles] = useState<ProfileManifestResponse[]>([]);
  const [selectedPackageId, setSelectedPackageId] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [chunkResult, setChunkResult] = useState<ChunkRequestResponse | null>(null);
  const [hasChunks, setHasChunks] = useState(false);
  const [chunksByFile, setChunksByFile] = useState<ChunkResponse[][]>([]);
  const [viewingFile, setViewingFile] = useState<FileEntryResponse | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [context, setContext] = useState<InitialContext | null>(null);
  const [draft, setDraft] = useState<object | null>(null);
  const [protectedFields, setProtectedFields] = useState<string[]>([]);
  const [patchStatus, setPatchStatus] = useState<PatchTaskStatus | null>(null);
  const [patchProgress, setPatchProgress] = useState<PatchProgress | null>(null);
  const [patchArtifacts, setPatchArtifacts] = useState<PatchArtifacts | null>(null);
  const [tokenUsage, setTokenUsage] = useState<PatchTokenUsage | null>(null);
  const [llmBudget, setLlmBudget] = useState<LlmBudget | null>(null);
  const [ollamaConfig, setOllamaConfig] = useState<OllamaConfig | null>(null);
  const [activeProfileSchema, setActiveProfileSchema] = useState<JsonSchemaDocument | null>(null);
  const [patchReviewState, setPatchReviewState] = useState<PatchReviewState>(emptyReviewState);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(true);
  const [contextEditMode, setContextEditMode] = useState(false);
  const [chunkingDialogOpen, setChunkingDialogOpen] = useState(false);
  const [numChunksPerTurn, setNumChunksPerTurn] = useState(() => {
    try {
      const stored = localStorage.getItem('simone_num_chunks_per_turn');
      const parsed = stored ? parseInt(stored, 10) : NaN;
      return Number.isFinite(parsed) && parsed >= 1 ? parsed : 3;
    } catch {
      return 3;
    }
  });
  const [autoResolve, setAutoResolve] = useState(() => {
    try {
      const stored = localStorage.getItem('simone_auto_resolve');
      return stored === 'true';
    } catch {
      return false;
    }
  });
  const [profileFormOpen, setProfileFormOpen] = useState(false);
  const [profileIdentifier, setProfileIdentifier] = useState('');
  const [profileTargetClass, setProfileTargetClass] = useState('Dataset');
  const [profileSourceMode, setProfileSourceMode] = useState<'url' | 'upload'>('url');
  const [profileSchemaUrl, setProfileSchemaUrl] = useState('');
  const [profileSchemaFile, setProfileSchemaFile] = useState<File | null>(null);
  const [profileVersion, setProfileVersion] = useState('');
  const [profileEnrichableFields, setProfileEnrichableFields] = useState('');
  const datasetUploadInputRef = useRef<HTMLInputElement | null>(null);
  const saveContextTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveDraftTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const selectedPackageIdRef = useRef('');

  const selectedPackage = useMemo(() => packages.find((item) => item.id === selectedPackageId) || null, [packages, selectedPackageId]);
  const selectedProfileManifest = useMemo(() => profiles.find((item) => item.identifier === selectedProfile) || null, [profiles, selectedProfile]);
  const chunkCountByFile = useMemo(() => {
    const counts = new Map<string, number>();
    for (const group of chunksByFile) {
      const filePath = group[0]?.file_path;
      if (filePath) counts.set(filePath, group.length);
    }
    return counts;
  }, [chunksByFile]);
  const isPatching = patchStatus === 'running';
  const hasVisiblePatchArtifacts = Boolean(patchArtifacts && hasPatchArtifacts(patchArtifacts));
  const patchButtonLabel = isPatching ? 'Patching...' : hasVisiblePatchArtifacts ? 'Resume patching from checkpoint' : 'Start new patching';
  const progressBatchNo = patchProgress?.batch_no ?? 0;
  const progressTotalBatches = patchProgress?.total_batches ?? 0;
  const progressPercent = progressTotalBatches > 0 ? Math.min(100, Math.round((progressBatchNo / progressTotalBatches) * 100)) : 0;
  const extractionProgressPercent = patchProgress?.total_chunks
    ? Math.min(100, Math.round((patchProgress.processed_chunks / patchProgress.total_chunks) * 100))
    : 0;
  const extractionProgressLabel = patchProgress
    ? `${formatExtractionStage(patchProgress.stage)}${patchProgress.total_chunks ? ` - ${patchProgress.processed_chunks}/${patchProgress.total_chunks} chunks` : ''}`
    : '';
  const hasPersistedExtractionState = Boolean(
    patchProgress?.interim_context
    || patchProgress?.chunk_results?.some((chunk) => chunk.status === 'completed' || chunk.extraction_context),
  );
  const extractionCanResume = Boolean(
    selectedPackageId
    && selectedProfile
    && !busy
    && !isPatching
    && (
      patchStatus === 'cancelled'
      || patchStatus === 'crashed'
      || (patchStatus === 'unknown' && hasPersistedExtractionState)
    ),
  );
  const reviewItems = useMemo(() => buildReviewItems(patchArtifacts, patchReviewState), [patchArtifacts, patchReviewState]);
  const unresolvedReviewItems = reviewItems.filter((item) => !item.resolved);
  const patchMarkers = unresolvedReviewItems;
  const reviewMarkers = unresolvedReviewItems.filter((marker) => marker.status === 'needs_review' || marker.status === 'unmapped');
  const autoResolutionActive = patchProgress?.resolution_active === true || (isPatching && autoResolve);
  const manualReviewActionsDisabled = autoResolutionActive;

  async function refresh() {
    setBusy('load');
    try {
      const [nextPackages, nextProfiles, nextBudget, nextOllamaConfig] = await Promise.all([listDataPackages(), listProfiles(), getLlmBudget(), getOllamaConfig()]);
      setLlmBudget(nextBudget);
      setOllamaConfig(nextOllamaConfig);
      const storedPackageId = readStoredSelectedPackageId();
      setPackages(nextPackages);
      setProfiles(nextProfiles);
      setSelectedPackageId((current) => {
        const nextPackageId = [current, storedPackageId].find((id) => nextPackages.some((item) => item.id === id)) || nextPackages[0]?.id || '';
        persistSelectedPackageId(nextPackageId);
        return nextPackageId;
      });
      setSelectedProfile((current) => current || nextProfiles[0]?.identifier || '');
      setMessage(nextPackages.length ? 'Select a package or continue the workflow.' : 'Upload a dataset archive to begin.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not load workspace.');
    } finally {
      setBusy(null);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    selectedPackageIdRef.current = selectedPackageId;
  }, [selectedPackageId]);

  useEffect(() => {
    if (!selectedProfile) {
      setActiveProfileSchema(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const schema = await getProfileJsonSchema(selectedProfile);
        if (!cancelled) setActiveProfileSchema(schema);
      } catch {
        if (!cancelled) setActiveProfileSchema(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedProfile]);

  useEffect(() => {
    try {
      localStorage.setItem('simone_num_chunks_per_turn', String(numChunksPerTurn));
    } catch {
      // ignore storage errors
    }
  }, [numChunksPerTurn]);

  useEffect(() => {
    try {
      localStorage.setItem('simone_auto_resolve', String(autoResolve));
    } catch {
      // ignore storage errors
    }
  }, [autoResolve]);

  useEffect(() => {
    return () => {
      if (saveContextTimeoutRef.current) clearTimeout(saveContextTimeoutRef.current);
      if (saveDraftTimeoutRef.current) clearTimeout(saveDraftTimeoutRef.current);
    };
  }, []);

  function resetPackageWorkflowState() {
    if (saveContextTimeoutRef.current) {
      clearTimeout(saveContextTimeoutRef.current);
      saveContextTimeoutRef.current = null;
    }
    if (saveDraftTimeoutRef.current) {
      clearTimeout(saveDraftTimeoutRef.current);
      saveDraftTimeoutRef.current = null;
    }
    setChunkResult(null);
    setHasChunks(false);
    setChunksByFile([]);
    setViewingFile(null);
    setFileContent(null);
    setContext(null);
    setContextEditMode(false);
    setDraft(null);
    setProtectedFields([]);
    setPatchStatus(null);
    setPatchProgress(null);
    setPatchArtifacts(null);
    setTokenUsage(null);
    setPatchReviewState(emptyReviewState);
    setChunkingDialogOpen(false);
    setBusy(null);
  }

  function handlePackageSelection(nextPackageId: string) {
    selectedPackageIdRef.current = nextPackageId;
    persistSelectedPackageId(nextPackageId);
    resetPackageWorkflowState();
    setSelectedPackageId(nextPackageId);
  }

  function resetProfileForm() {
    setProfileIdentifier('');
    setProfileTargetClass('Dataset');
    setProfileSourceMode('url');
    setProfileSchemaUrl('');
    setProfileSchemaFile(null);
    setProfileVersion('');
    setProfileEnrichableFields('');
  }

  async function onRegisterProfile() {
    const identifier = profileIdentifier.trim();
    const targetClass = profileTargetClass.trim() || 'Dataset';
    const schemaUrl = profileSchemaUrl.trim();
    const version = profileVersion.trim();
    const enrichableFields = profileEnrichableFields
      .split(',')
      .map((field) => field.trim())
      .filter(Boolean);

    if (!identifier) {
      setMessage('Profile identifier is required.');
      return;
    }
    if (profileSourceMode === 'url' && !schemaUrl) {
      setMessage('Schema URL is required for URL registration.');
      return;
    }
    if (profileSourceMode === 'upload' && !profileSchemaFile) {
      setMessage('Schema file is required for upload registration.');
      return;
    }

    setBusy('profile');
    try {
      const registered = await registerProfile({
        identifier,
        target_class: targetClass,
        schema_url: profileSourceMode === 'url' ? schemaUrl : undefined,
        schema_file: profileSourceMode === 'upload' ? profileSchemaFile ?? undefined : undefined,
        version: version || undefined,
        enrichable_fields: enrichableFields,
      });
      const nextProfiles = await listProfiles();
      setProfiles(nextProfiles);
      setSelectedProfile(registered.identifier);
      setProfileFormOpen(false);
      resetProfileForm();
      setMessage(`Profile ${registered.identifier} registered.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Profile registration failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onDeleteProfile() {
    if (!selectedProfile) {
      setMessage('Select a profile to remove.');
      return;
    }
    const confirmed = window.confirm(`Remove profile "${selectedProfile}"? This deletes the registered profile artifacts.`);
    if (!confirmed) return;

    setBusy('profile-delete');
    try {
      await deleteProfile(selectedProfile);
      const nextProfiles = await listProfiles();
      setProfiles(nextProfiles);
      setSelectedProfile(nextProfiles[0]?.identifier ?? '');
      setMessage(`Profile ${selectedProfile} removed.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Profile removal failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onDeleteDataPackage() {
    if (!selectedPackageId) {
      setMessage('Select a dataset to remove.');
      return;
    }
    const pkg = packages.find((p) => p.id === selectedPackageId);
    const confirmed = window.confirm(`Remove dataset "${pkg?.file_name ?? selectedPackageId}"? This deletes the uploaded dataset and all associated chunks.`);
    if (!confirmed) return;

    setBusy('dataset-delete');
    try {
      await deleteDataPackage(selectedPackageId);
      const nextPackages = await listDataPackages();
      const nextPackageId = nextPackages[0]?.id ?? '';
      setPackages(nextPackages);
      persistSelectedPackageId(nextPackageId);
      setSelectedPackageId(nextPackageId);
      setChunkResult(null);
      setChunksByFile([]);
      setContext(null);
      setDraft(null);
      setPatchArtifacts(null);
      setPatchReviewState(emptyReviewState);
      setMessage(`Dataset ${pkg?.file_name ?? selectedPackageId} removed.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Dataset removal failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onUpload(file?: File) {
    if (!file) return;
    setBusy('upload');
    try {
      const uploaded = await uploadDataPackage(file);
      const nextPackages = await listDataPackages();
      setPackages(nextPackages);
      persistSelectedPackageId(uploaded.id);
      setSelectedPackageId(uploaded.id);
      setChunkResult(null);
      setChunksByFile([]);
      setContext(null);
      setDraft(null);
      setPatchArtifacts(null);
      setPatchReviewState(emptyReviewState);
      setMessage('Dataset uploaded. Create chunks next.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Upload failed.');
    } finally {
      if (datasetUploadInputRef.current) {
        datasetUploadInputRef.current.value = '';
      }
      setBusy(null);
    }
  }

  async function onChunk(params?: { replace_existing_chunks: boolean; buffer_window_size: number; semantic_chunking_threshold: number; protected_line_indices: Record<string, number[]>; text_quality_config: TextQualityConfig }) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('chunk');
    try {
      const result = await chunkDataPackage({
        id: packageId,
        replace_existing_chunks: params?.replace_existing_chunks ?? false,
        buffer_window_size: params?.buffer_window_size,
        semantic_chunking_threshold: params?.semantic_chunking_threshold,
        protected_line_indices: params?.protected_line_indices,
        text_quality_config: params?.text_quality_config,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      setChunkResult(result);
      setChunksByFile(result.chunks);
      setHasChunks(result.status === 'completed');
      setMessage(result.status === 'completed' ? 'Chunks are ready.' : 'Chunking is running...');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Chunking failed.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function pollChunkProgress() {
    if (!selectedPackageId || !chunkResult) return;
    const packageId = selectedPackageId;
    try {
      const [status, chunks] = await Promise.all([
        getChunkStatus(packageId),
        getDataPackageChunks(packageId),
      ]);
      if (selectedPackageIdRef.current !== packageId) return;
      setHasChunks(status.has_chunks);
      setChunksByFile(chunks);
      if (status.has_chunks) {
        setChunkResult((prev) => prev ? { ...prev, status: 'completed', chunks } : prev);
        setMessage(`Chunking completed — ${chunks.flat().length} chunks created.`);
      } else {
        setChunkResult((prev) => prev ? { ...prev, status: status.status, chunks } : prev);
        const chunkCount = chunks.flat().length;
        if (chunkCount > 0) {
          setMessage(`Chunking in progress — ${chunkCount} chunks created so far...`);
        }
      }
    } catch {
      // ignore polling errors
    }
  }

  async function onViewFile(file: FileEntryResponse) {
    if (!selectedPackageId) return;
    setBusy('load');
    try {
      const [content, chunks] = await Promise.all([
        getFileEntryContent(selectedPackageId, file.file_path),
        hasChunks ? getDataPackageChunks(selectedPackageId) : Promise.resolve([]),
      ]);
      setViewingFile(file);
      setFileContent(content.content);
      setChunksByFile(chunks);
      setMessage(`Viewing ${file.file_path}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to load file content.');
    } finally {
      setBusy(null);
    }
  }

  function closeFileViewer() {
    setViewingFile(null);
    setFileContent(null);
  }

  async function onContext(options: { resume?: boolean } = {}) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      if (!selectedProfile) {
        setMessage('Select a profile before running extraction.');
        return;
      }
      setContext(null);
      setContextEditMode(false);
      setDraft(null);
      setPatchArtifacts(null);
      setPatchProgress(null);
      const response = await runExtraction({
        data_package_id: packageId,
        profile_identifier: selectedProfile,
        resume: options.resume,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      const nextContext = response.result?.extraction_context || response.progress?.interim_context;
      if (nextContext) {
        setContext(initialContextFromExtractionContext(nextContext));
        setContextEditMode(response.status !== 'running');
      }
      if (response.result) setDraft(response.result.document);
      setPatchStatus(response.status);
      setPatchProgress(response.progress ? { ...response.progress } : null);
      setTokenUsage(await getTokenUsage(packageId));
      setMessage(response.status === 'running' ? (options.resume ? 'Extraction resumed.' : 'Extraction is running.') : 'Extraction completed.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Context extraction failed.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function refreshOllamaConfig() {
    setBusy('ollama');
    try {
      const [nextConfig, nextBudget] = await Promise.all([getOllamaConfig(), getLlmBudget()]);
      setOllamaConfig(nextConfig);
      setLlmBudget(nextBudget);
      setMessage('Ollama runtime settings refreshed.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not refresh Ollama settings.');
    } finally {
      setBusy(null);
    }
  }

  async function applyOllamaRuntimeConfig(values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_batch_size: number;
    embedding_num_gpu: number;
  }) {
    setBusy('ollama');
    try {
      const updated = await updateOllamaRuntimeConfig(values);
      const nextBudget = await getLlmBudget();
      setOllamaConfig(updated);
      setLlmBudget(nextBudget);
      setMessage('Ollama runtime settings updated for future calls.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not update Ollama runtime settings.');
    } finally {
      setBusy(null);
    }
  }

  async function pullOllamaModelFromUi(model: string) {
    setBusy('ollama');
    try {
      await pullOllamaModel(model);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage(`Pull requested for ${model} on the configured Ollama host.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not pull Ollama model.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  async function removeOllamaModelFromUi(model: string) {
    setBusy('ollama');
    try {
      await removeOllamaModel(model);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage(`Removed ${model} from the configured Ollama host.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not remove Ollama model.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  async function runOllamaSwitchTest(values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_num_gpu: number;
  }) {
    setBusy('ollama');
    try {
      const result = await runOllamaPerformanceTest(values);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage('Ollama switch test completed.');
      return result;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not run Ollama switch test.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  function onContextChange(update: (current: InitialContext) => InitialContext) {
    if (!context || !selectedPackageId) return;
    const packageId = selectedPackageId;
    const updated = update(context);
    setContext(updated);
    if (saveContextTimeoutRef.current) clearTimeout(saveContextTimeoutRef.current);
    saveContextTimeoutRef.current = setTimeout(async () => {
      try {
        await saveInitialContext(packageId, updated);
        if (selectedPackageIdRef.current === packageId) setMessage('Extraction context saved.');
      } catch (error) {
        if (selectedPackageIdRef.current === packageId) {
          setMessage(error instanceof Error ? error.message : 'Failed to save extraction context.');
        }
      }
    }, 800);
  }

  async function onDraft() {
    if (!selectedPackageId || !selectedProfile) return;
    const isReplacingDraft = Boolean(draft);
    if (isReplacingDraft) {
      const confirmed = window.confirm(
        'Re-create the initial draft?\n\nThis removes the current draft, locked fields, patch artifacts, and review progress for this dataset.',
      );
      if (!confirmed) return;
    }
    if (saveDraftTimeoutRef.current) {
      clearTimeout(saveDraftTimeoutRef.current);
      saveDraftTimeoutRef.current = null;
    }
    setBusy('draft');
    try {
      const result = await extractInitialDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setDraft(result);
      setPatchArtifacts(null);
      setPatchStatus(null);
      setPatchProgress(null);
      setPatchReviewState(emptyReviewState);
      setProtectedFields([]);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage(isReplacingDraft ? 'Initial profile draft re-created. Previous draft progress was removed.' : 'Initial profile draft created.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Draft creation failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onDraftChange(updated: Record<string, unknown>) {
    setDraft(updated);
    if (!selectedPackageId) return;
    if (saveDraftTimeoutRef.current) clearTimeout(saveDraftTimeoutRef.current);
    saveDraftTimeoutRef.current = setTimeout(async () => {
      try {
        await saveDraft(selectedPackageId, updated);
        setMessage('Draft saved.');
      } catch (error) {
        setMessage(error instanceof Error ? error.message : 'Failed to save draft.');
      }
    }, 800);
  }

  async function onPatch() {
    if (!selectedPackageId || !selectedProfile) return;
    setBusy('patch');
    try {
      setPatchProgress(null);
      const result = await patchDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile, num_chunks_per_turn: numChunksPerTurn, auto_resolve: autoResolve });
      setDraft(result.draft);
      setPatchStatus(result.status);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage(
        result.status === 'completed'
          ? 'Draft patching completed. Use Show/refresh artifacts to load the latest artifacts and review items.'
          : 'Draft patching is running. Use Show/refresh artifacts to load the latest artifacts and review items when needed.',
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Patch step failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onShowPatchArtifacts() {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    try {
      const [{ status, progress }, artifacts, reviewState] = await Promise.all([
        getPatchProgress(packageId),
        getPatchArtifacts(packageId),
        getPatchReviewState(packageId),
      ]);
      if (selectedPackageIdRef.current !== packageId) return;
      const completedResult = status === 'completed'
        ? await getExtractionResult(packageId)
        : null;
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress || null);
      if (completedResult) {
        setContext(initialContextFromExtractionContext(completedResult.extraction_context));
        setDraft(completedResult.document);
        setContextEditMode(true);
      } else if (progress?.interim_context) {
        setContext(initialContextFromExtractionContext(progress.interim_context));
        setContextEditMode(false);
      }
      setPatchArtifacts(artifacts);
      setPatchReviewState(reviewState);
      const nextTokenUsage = completedResult?.token_usage ?? await getTokenUsage(packageId);
      setTokenUsage(nextTokenUsage);
      if (status === 'running') {
        setMessage('Extraction is running.');
      } else {
        setMessage(hasPatchArtifacts(artifacts) ? 'Loaded existing patch artifacts.' : 'No existing patch artifacts found.');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to load patch artifacts.');
    }
  }

  async function persistPatchReviewState(nextState: PatchReviewState, successMessage: string) {
    if (!selectedPackageId) return;
    const previous = patchReviewState;
    setPatchReviewState(nextState);
    try {
      const saved = await savePatchReviewState(selectedPackageId, nextState);
      setPatchReviewState(saved);
      setMessage(successMessage);
    } catch (error) {
      setPatchReviewState(previous);
      setMessage(error instanceof Error ? error.message : 'Failed to save review state.');
    }
  }

  async function onApplyPatch(itemId: string, editedValue: unknown) {
    if (!selectedPackageId || !draft) return;
    if (autoResolutionActive) {
      setMessage('Auto resolver is running. Manual review controls are disabled until it finishes.');
      return;
    }
    const item = reviewItems.find((ri) => ri.id === itemId);
    if (!item) return;

    const previousDraft = draft;
    const updatedDraft = setValueAtPath(draft as JsonObject, item.path, editedValue as JsonValue);
    setDraft(updatedDraft as Record<string, unknown>);

    const previousReviewState = patchReviewState;
    const nextReviewState: PatchReviewState = {
      ...patchReviewState,
      resolved_item_ids: [...patchReviewState.resolved_item_ids, itemId],
      resolved_at: { ...patchReviewState.resolved_at, [itemId]: new Date().toISOString() },
    };
    setPatchReviewState(nextReviewState);

    try {
      await Promise.all([
        saveDraft(selectedPackageId, updatedDraft as Record<string, unknown>),
        savePatchReviewState(selectedPackageId, nextReviewState),
      ]);
      setMessage('Patch applied and marked resolved.');
    } catch (error) {
      setDraft(previousDraft);
      setPatchReviewState(previousReviewState);
      setMessage(error instanceof Error ? error.message : 'Failed to apply patch.');
    }
  }

  async function onResolveReviewItem(itemId: string) {
    if (autoResolutionActive) {
      setMessage('Auto resolver is running. Manual review controls are disabled until it finishes.');
      return;
    }
    if (patchReviewState.resolved_item_ids.includes(itemId)) return;
    await persistPatchReviewState(
      {
        ...patchReviewState,
        resolved_item_ids: [...patchReviewState.resolved_item_ids, itemId],
        resolved_at: { ...patchReviewState.resolved_at, [itemId]: new Date().toISOString() },
      },
      'Review item marked resolved.',
    );
  }

  async function onSaveProtectedFields(fields: string[]) {
    if (!selectedPackageId) return;
    const previous = protectedFields;
    setProtectedFields(fields);
    try {
      await apiSetProtectedFields(selectedPackageId, fields);
      setMessage('Protected fields updated.');
    } catch (error) {
      setProtectedFields(previous);
      setMessage(error instanceof Error ? error.message : 'Failed to save protected fields.');
    }
  }

  useEffect(() => {
    resetPackageWorkflowState();
    if (!selectedPackageId) {
      setMessage('Select a package or upload a dataset archive to begin.');
      return;
    }
    const packageId = selectedPackageId;
    setBusy('load');
    void (async () => {
      try {
        const [ctx, draftResult, fields, { status, progress }, artifacts, reviewState, chunkStatus, usage] = await Promise.all([
          getExistingInitialContext(packageId),
          getExistingInitialDraft(packageId),
          getProtectedFields(packageId),
          getPatchProgress(packageId),
          getPatchArtifacts(packageId),
          getPatchReviewState(packageId),
          getChunkStatus(packageId),
          getTokenUsage(packageId),
        ]);
        const chunks = chunkStatus.status !== 'unknown' || chunkStatus.has_chunks
          ? await getDataPackageChunks(packageId)
          : [];
        if (selectedPackageIdRef.current !== packageId) return;
        const progressContext = progress?.interim_context
          ? initialContextFromExtractionContext(progress.interim_context)
          : null;
        if (ctx || progressContext) {
          setContext(ctx ?? progressContext);
          setContextEditMode(status !== 'running');
        }
        if (draftResult) setDraft(draftResult);
        setProtectedFields(fields);
        setPatchStatus(status);
        setPatchProgress(progress || null);
        setTokenUsage(usage);
        if (status === 'completed' || status === 'crashed' || status === 'cancelled' || hasPatchArtifacts(artifacts)) setPatchArtifacts(artifacts);
        setPatchReviewState(reviewState);
        setHasChunks(chunkStatus.has_chunks);
        setChunksByFile(chunks);
        if (chunkStatus.status !== 'unknown' || chunks.flat().length > 0) {
          setChunkResult({ status: chunkStatus.status, chunks });
        }
        setMessage('Workflow state loaded.');
      } catch (error) {
        if (selectedPackageIdRef.current !== packageId) return;
        setMessage(error instanceof Error ? error.message : 'Failed to load workflow state.');
      } finally {
        if (selectedPackageIdRef.current === packageId) setBusy(null);
      }
    })();
  }, [selectedPackageId]);

  useEffect(() => {
    if (!chunkResult || chunkResult.status === 'completed' || chunkResult.status === 'cancelled' || chunkResult.status === 'crashed') return;
    const interval = setInterval(() => void pollChunkProgress(), 3000);
    return () => clearInterval(interval);
  }, [chunkResult, selectedPackageId]);

  useEffect(() => {
    if (!selectedPackageId || patchStatus !== 'running') return;
    const interval = setInterval(() => void onShowPatchArtifacts(), 5000);
    return () => clearInterval(interval);
  }, [patchStatus, selectedPackageId]);

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-main">
          <p className="eyebrow">Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)</p>
          <h1><span>---&gt; Data archive in</span><span>FAIR data out ---&gt;</span></h1>
          <p className="intro">A restrained workflow for extracting dataset metadata, grounding it in a registered DCAT-AP profile, and preparing later vocabulary-backed enrichment.</p>
        </div>
        <aside className="status-card">
          <span>Workspace</span>
          <strong>{busy ? 'Working' : 'Ready'}</strong>
          <p>{message}</p>
        </aside>
      </section>

      <section className={railCollapsed ? 'layout rail-collapsed' : 'layout'}>
        <aside className="rail">
          <button className="rail-toggle" onClick={() => setRailCollapsed(!railCollapsed)} title={railCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
            {railCollapsed ? '->' : '<-'}
          </button>
          {!railCollapsed && (
            <>
              <div className="panel compact">
                <div className="panel-heading">
                  <span>Datasets</span>
                  <button onClick={() => datasetUploadInputRef.current?.click()} disabled={!!busy}>
                    {busy === 'upload' ? 'Uploading...' : 'Upload ZIP'}
                  </button>
                </div>
                <input
                  ref={datasetUploadInputRef}
                  className="hidden-file-input"
                  type="file"
                  accept=".zip"
                  onChange={(event) => void onUpload(event.target.files?.[0])}
                />
                <select value={selectedPackageId} onChange={(event) => handlePackageSelection(event.target.value)}>
                  <option value="">No package selected</option>
                  {packages.map((item) => <option key={item.id} value={item.id}>{item.file_name}</option>)}
                </select>
                <button
                  className="ghost dataset-remove-button"
                  onClick={() => void onDeleteDataPackage()}
                  disabled={!selectedPackageId || !!busy}
                >
                  {busy === 'dataset-delete' ? 'Removing...' : 'Remove selected'}
                </button>
              </div>
              <div className="panel compact">
                <div className="panel-heading">
                  <span>Profile</span>
                  <button onClick={() => setProfileFormOpen((open) => !open)} disabled={!!busy}>
                    {profileFormOpen ? 'Close' : 'Register'}
                  </button>
                </div>
                <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value)}>
                  <option value="">No profile selected</option>
                  {profiles.map((profile) => <option key={profile.identifier} value={profile.identifier}>{profile.identifier}</option>)}
                </select>
                <button
                  className="ghost profile-remove-button"
                  onClick={() => void onDeleteProfile()}
                  disabled={!selectedProfile || !!busy}
                >
                  {busy === 'profile-delete' ? 'Removing...' : 'Remove selected'}
                </button>
                {profileFormOpen && (
                  <div className="profile-register-form">
                    <label>
                      <span>Identifier</span>
                      <input value={profileIdentifier} onChange={(event) => setProfileIdentifier(event.target.value)} placeholder="my-profile" />
                    </label>
                    <label>
                      <span>Target class</span>
                      <input value={profileTargetClass} onChange={(event) => setProfileTargetClass(event.target.value)} placeholder="Dataset" />
                    </label>
                    <div className="profile-source-toggle">
                      <button
                        className={profileSourceMode === 'url' ? '' : 'ghost'}
                        onClick={() => setProfileSourceMode('url')}
                        type="button"
                      >
                        URL
                      </button>
                      <button
                        className={profileSourceMode === 'upload' ? '' : 'ghost'}
                        onClick={() => setProfileSourceMode('upload')}
                        type="button"
                      >
                        Upload
                      </button>
                    </div>
                    {profileSourceMode === 'url' ? (
                      <label>
                        <span>Schema URL</span>
                        <input value={profileSchemaUrl} onChange={(event) => setProfileSchemaUrl(event.target.value)} placeholder="https://..." />
                      </label>
                    ) : (
                      <label>
                        <span>Schema file</span>
                        <input type="file" accept=".yaml,.yml,.json" onChange={(event) => setProfileSchemaFile(event.target.files?.[0] ?? null)} />
                      </label>
                    )}
                    <label>
                      <span>Version</span>
                      <input value={profileVersion} onChange={(event) => setProfileVersion(event.target.value)} placeholder="optional" />
                    </label>
                    <label>
                      <span>Enrichable fields</span>
                      <input value={profileEnrichableFields} onChange={(event) => setProfileEnrichableFields(event.target.value)} placeholder="field_a, field_b" />
                    </label>
                    <button onClick={() => void onRegisterProfile()} disabled={busy === 'profile'}>
                      {busy === 'profile' ? 'Registering...' : 'Register profile'}
                    </button>
                  </div>
                )}
              </div>
              <OllamaSettingsPanel
                config={ollamaConfig}
                budget={llmBudget}
                tokenUsage={tokenUsage}
                patchTokenUsage={patchProgress?.token_usage}
                busy={busy === 'ollama'}
                onApply={(values) => void applyOllamaRuntimeConfig(values)}
                onRefresh={() => void refreshOllamaConfig()}
                onPullModel={(model) => pullOllamaModelFromUi(model)}
                onRemoveModel={(model) => removeOllamaModelFromUi(model)}
                onRunPerformanceTest={(values) => runOllamaSwitchTest(values)}
              />
              <VocabularyPanel onError={setMessage} />
            </>
          )}
        </aside>

        <section className="workflow">
          <StepPanel
            number="01"
            title="Upload dataset and create chunks"
            description="The archive is stored as a data package. Chunking prepares the package for later patch and enrichment stages."
            active
          >
              <div className="actions">
                <button onClick={() => setChunkingDialogOpen(true)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking...' : 'Configure Chunking'}</button>
              </div>
              {selectedPackage && (
                <div className="file-list">
                  {selectedPackage.files.map((file) => {
                    const chunkCount = chunkCountByFile.get(file.file_path) ?? 0;
                    return (
                      <div key={file.file_path} className="file-row" onClick={() => void onViewFile(file)} title="Click to view file content">
                        <span>{file.file_path}</span>
                        <div className="file-meta">
                          {chunkCount > 0 && <span className="chunk-badge">{chunkCount} chunk{chunkCount === 1 ? '' : 's'}</span>}
                          <small>{formatBytes(file.byte_size)}</small>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              {chunkResult && <p className="muted">Chunk status: <strong>{chunkResult.status}</strong> - {chunkResult.chunks.flat().length} chunks visible</p>}
              {viewingFile && fileContent !== null && (
                <FileViewer
                  file={viewingFile}
                  content={fileContent}
                  chunksByFile={chunksByFile}
                  onClose={closeFileViewer}
                />
              )}
              <ChunkingDialog
                isOpen={chunkingDialogOpen}
                packageId={selectedPackageId}
                dataPackage={selectedPackage}
                chunksByFile={chunksByFile}
                onClose={() => setChunkingDialogOpen(false)}
                onSubmit={(params) => {
                  setChunkingDialogOpen(false);
                  void onChunk(params);
                }}
              />
          </StepPanel>

          <StepPanel
            number="02"
            title="Extraction context overview"
            description="Rank files, extract structured context from each chunk, and track extraction results as they are produced."
          >
              <div className="actions">
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !!busy || isPatching}>
                  {isPatching ? 'Extraction running...' : busy === 'context' ? 'Extracting...' : hasPersistedExtractionState || context ? 'Re-extract context overview' : 'Extract context overview'}
                </button>
                {extractionCanResume && (
                  <button className="ghost" onClick={() => void onContext({ resume: true })} disabled={!selectedPackageId || !!busy || isPatching}>
                    Resume extraction
                  </button>
                )}
              </div>
              {patchStatus === 'running' && patchProgress && (
                <div className="patch-progress context-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{formatExtractionStage(patchStatus)}</strong></span>
                    {extractionProgressLabel && <span>{extractionProgressLabel}</span>}
                  </div>
                  {patchProgress.total_chunks > 0 && (
                    <div className="patch-progress-track" aria-hidden="true"><div style={{ width: `${extractionProgressPercent}%` }} /></div>
                  )}
                </div>
              )}
              <ExtractionContextOverview
                rankedFiles={patchProgress?.ranked_files ?? []}
                chunkResults={patchProgress?.chunk_results ?? []}
                currentChunk={patchProgress?.current_chunk ?? null}
                chunksByFile={chunksByFile}
                packageFiles={selectedPackage?.files ?? []}
                progress={patchProgress}
                status={patchStatus}
              />
              <TokenUsageSummary
                tokenUsage={tokenUsage}
                averageUnit="operation"
                heading="Extraction context token usage"
                agentKeys={['file_ranking', 'chunk_extraction', 'chunk_extraction_repair', 'quantity_vocab_selection', 'qualitative_vocab_selection', 'profile_projection']}
                budget={llmBudget}
              />
              <p className="context-window-advice">
                If extraction reaches or overuses the context window, rerun chunking with smaller chunks before extracting again. Lower the semantic chunking threshold in the chunking configuration to reduce chunk sizes.
              </p>
          </StepPanel>

          <StepPanel
            number="03"
            title="Draft workspace"
            description="Create the initial profile draft, edit and lock fields, then patch the draft with chunk evidence while reviewing issues as they appear."
            actions={draft && (
              <button className="ghost draft-refresh-button" onClick={() => void onShowPatchArtifacts()} disabled={!selectedPackageId || busy === 'load'}>Refresh</button>
            )}
          >
              <div className={draft ? 'draft-actions' : 'actions'}>
                {!draft ? (
                  <button onClick={() => void onDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Drafting...' : 'Create new draft'}</button>
                ) : (
                  <>
                    <button onClick={() => void onPatch()} disabled={!selectedProfile || !!busy || isPatching}>{patchButtonLabel}</button>
                    <div className="draft-settings">
                      <label className="patch-config-row" htmlFor="num-chunks-per-turn">
                        <span>Chunks per turn</span>
                        <input
                          id="num-chunks-per-turn"
                          type="number"
                          min={1}
                          value={numChunksPerTurn}
                          onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            setNumChunksPerTurn(Number.isFinite(val) && val >= 1 ? val : 1);
                          }}
                          disabled={isPatching}
                          title="Number of chunks to include in each patch agent call. Higher values process more content per turn but increase token usage."
                        />
                      </label>
                      <label className="patch-config-row checkbox" htmlFor="auto-resolve">
                        <input
                          id="auto-resolve"
                          type="checkbox"
                          checked={autoResolve}
                          onChange={(e) => setAutoResolve(e.target.checked)}
                          disabled={isPatching}
                          title="Automatically send unresolved review items to the resolve agent as patch artifacts are produced."
                        />
                        <span>Auto-resolve review items</span>
                      </label>
                    </div>
                    <button className="ghost draft-recreate-button" onClick={() => void onDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Re-creating...' : 'Re-create draft'}</button>
                  </>
                )}
              </div>
              {draft && patchStatus && (
                <div className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{patchStatus}</strong></span>
                    {progressTotalBatches > 0 && <span>Patching batch {progressBatchNo} of {progressTotalBatches}</span>}
                  </div>
                  <div className="patch-progress-track" aria-hidden="true"><div style={{ width: `${progressPercent}%` }} /></div>
                  <TokenUsageSummary tokenUsage={patchProgress?.token_usage} averageUnit="patch" budget={llmBudget} />
                  <ResolutionLogList entries={patchProgress?.resolution_log ?? []} />
                </div>
              )}
              {draft && !autoResolutionActive && reviewMarkers.length > 0 && (
                <div className="review-strip">
                  <strong>{reviewMarkers.length} unresolved review item{reviewMarkers.length === 1 ? '' : 's'}</strong>
                  <div>{reviewMarkers.slice(0, 8).map((marker, index) => <span key={`${marker.path}-${index}`}>{marker.path}</span>)}</div>
                </div>
              )}
              {draft && (
                <JsonEditor
                  value={draft as Record<string, unknown>}
                  onChange={(updated) => onDraftChange(updated)}
                  protectedPaths={protectedFields}
                  onProtectedPathsChange={(paths) => void onSaveProtectedFields(paths)}
                  patchMarkers={patchMarkers}
                  onApplyPatch={(itemId, value) => void onApplyPatch(itemId, value)}
                  onResolvePatch={(itemId) => void onResolveReviewItem(itemId)}
                  reviewActionsDisabled={manualReviewActionsDisabled}
                  schema={activeProfileSchema}
                  targetClass={selectedProfileManifest?.target_class}
                />
              )}
          </StepPanel>
        </section>
      </section>
    </main>
  );
}
