import { type FormEvent, type PointerEvent, type ReactNode, type WheelEvent, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { chunkDataPackage, deleteDataPackage, getChunkStatus, getDataPackageChunks, getFileEntryContent, listDataPackages, uploadDataPackage } from './api/datasources';
import {
  applyCurationFieldAction,
  getExistingInitialContext,
  getExistingCuratedDocument,
  getExistingGeneratedFinalDraft,
  getExtractionResult,
  getInitialContextProgress,
  getPatchProgress,
  getTokenUsage,
  initialContextFromExtractionContext,
  pauseExtraction,
  rerunAllVocabQueries,
  rerunVocabQuery,
  runExtraction,
  runInitialContext,
  runVocabularyGrounding,
  saveCuratedDocument,
  updateVocabQueryConfig,
} from './api/extraction';
import { deleteProfile, exportProfileDocumentJsonLd, getProfileJsonSchema, listProfiles, registerProfile } from './api/profiles';
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
import { listVocabularies } from './api/semantic';
import { JsonEditor, type JsonObject, type JsonPatchMarker, type JsonSchemaDocument } from './components/JsonEditor';
import { ChunkingDialog } from './components/ChunkingDialog';
import { VocabularyPanel } from './components/VocabularyPanel';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryResponse, InitialContext, ProfileManifestResponse, TextQualityConfig, VocabQueryResult } from './api/types';
import type {
  ExtractionChunkRef,
  ExtractionChunkResult,
  ExtractionVocabQueryConfig,
  ExtractionVocabQueryRecord,
  PatchProgress,
  PatchTaskStatus,
  PatchTokenUsage,
  PatchTokenUsageEntry,
} from './api/extraction';

type BusyKey = 'upload' | 'initial-context' | 'chunk' | 'context' | 'pause' | 'draft' | 'patch' | 'load' | 'profile' | 'profile-delete' | 'dataset-delete' | 'ollama';
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

function jsonPointerToEditorPath(path: string): string {
  if (!path || path === '/') return '';
  return path
    .replace(/^\//, '')
    .split('/')
    .map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'))
    .join('.');
}

function downloadJsonFile(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function sourceContextJsonPointer(query: ExtractionVocabQueryRecord): string | null {
  const path = query.source_context?.json_path;
  return typeof path === 'string' && path.startsWith('/') ? path : null;
}

function vocabResourceTitle(result: Record<string, unknown>, uri: string): string | null {
  const coerced = coerceVocabQueryResult(result);
  const resource = coerced?.resources[uri];
  const properties = resource?.properties ?? {};
  for (const key of ['skos:prefLabel', 'rdfs:label', 'title', 'label', 'name']) {
    const value = properties[key];
    if (typeof value === 'string' && value.trim()) return value;
    if (Array.isArray(value)) {
      const first = value.find((item): item is string => typeof item === 'string' && item.trim().length > 0);
      if (first) return first;
    }
  }
  return null;
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
      objectKind: String(trace.object_kind || 'unknown'),
      sourceText: typeof trace.source_text === 'string' ? trace.source_text : '',
      object: asRecord(trace.extracted_object),
    }))
    .filter((trace) => trace.object);
}

function extractionContextHasStructuredItems(context?: Record<string, unknown> | null): boolean {
  return extractionContextTraces(context).some((trace) => trace.object);
}

function extractionContextTraceLabel(trace: { objectKind: string; object: Record<string, unknown> | null }) {
  const label = trace.object ? extractionContextItemTitle(trace.object, 'Extracted object') : 'Extracted object';
  return `${trace.objectKind}: ${label}`;
}

function ExtractionContextResultView({ context }: { context?: Record<string, unknown> | null }) {
  const [selectedTrace, setSelectedTrace] = useState<ReturnType<typeof extractionContextTraces>[number] | null>(null);
  if (!context) return <p className="muted">No extraction result is available for this chunk yet.</p>;

  const traces = extractionContextTraces(context);
  const sections = [
    { key: 'Resource', label: 'Resources' },
    { key: 'Method', label: 'Methods' },
    { key: 'DataGeneratingActivity', label: 'Activities' },
    { key: 'EvaluatedEntity', label: 'Entities' },
    { key: 'AgenticEntity', label: 'Agents' },
  ].map((section) => ({
    ...section,
    traces: traces.filter((trace) => trace.objectKind === section.key && trace.object),
  })).filter((section) => section.traces.length > 0);

  if (!extractionContextHasStructuredItems(context)) {
    return <p className="muted">The model call completed, but this chunk did not yield structured extraction context items.</p>;
  }

  return (
    <div className="extraction-context-results">
      {sections.map((section) => (
        <section key={section.key} className="extraction-result-section">
          <span>{section.label}</span>
          <div>
            {section.traces.map((trace, index) => {
              const item = trace.object as Record<string, unknown>;
              const keywords = extractionContextKeywords(item);
              return (
                <button
                  className="extraction-result-card"
                  key={`${section.key}-${index}`}
                  type="button"
                  onClick={() => setSelectedTrace(trace)}
                >
                  <strong>{extractionContextItemTitle(item, `${section.label} ${index + 1}`)}</strong>
                  {extractionContextItemDescription(item) && <span className="extraction-result-description">{extractionContextItemDescription(item)}</span>}
                  {keywords.length > 0 && (
                    <span className="extraction-result-keywords">
                      {keywords.map((keyword) => <span key={keyword}>{keyword}</span>)}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </section>
      ))}
      {selectedTrace && <ExtractionObjectModal trace={selectedTrace} onClose={() => setSelectedTrace(null)} />}
    </div>
  );
}

function ExtractionObjectModal({
  trace,
  onClose,
}: {
  trace: ReturnType<typeof extractionContextTraces>[number];
  onClose: () => void;
}) {
  const itemTitle = trace.object ? extractionContextItemTitle(trace.object, 'Extracted object') : 'Extracted object';
  return createPortal(
    <div className="vocab-dialog-overlay" onClick={onClose}>
      <div className="vocab-dialog extraction-object-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="vocab-dialog-header">
          <div>
            <span>{trace.objectKind}</span>
            <strong>{itemTitle}</strong>
          </div>
          <button className="ghost" onClick={onClose}>Close</button>
        </div>
        <div className="vocab-dialog-body extraction-object-dialog-body">
          <section>
            <span>Extracted object</span>
            <ExtractionFieldList value={trace.object ?? {}} />
          </section>
          <section>
            <span>Trace</span>
            <ExtractionFieldList value={{ object_kind: trace.objectKind, source_text: trace.sourceText }} />
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function ExtractionFieldList({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value);
  if (!entries.length) return <p className="muted">No fields.</p>;
  return (
    <dl className="extraction-field-list">
      {entries.map(([key, fieldValue]) => (
        <div key={key}>
          <dt>{formatExtractionFieldLabel(key)}</dt>
          <dd><ExtractionFieldValue value={fieldValue} /></dd>
        </div>
      ))}
    </dl>
  );
}

function ExtractionFieldValue({ value }: { value: unknown }) {
  if (Array.isArray(value)) {
    if (!value.length) return <span className="muted">None</span>;
    return (
      <div className="extraction-field-array">
        {value.map((item, index) => (
          <section key={index}>
            <span>{index + 1}</span>
            {asRecord(item)
              ? <ExtractionFieldList value={asRecord(item) as Record<string, unknown>} />
              : <ExtractionFieldValue value={item} />}
          </section>
        ))}
      </div>
    );
  }
  const record = asRecord(value);
  if (record) return <ExtractionFieldList value={record} />;
  if (value === null || value === undefined || value === '') return <span className="muted">-</span>;
  if (typeof value === 'boolean') return <span>{value ? 'true' : 'false'}</span>;
  return <span>{String(value)}</span>;
}

function formatExtractionFieldLabel(value: string): string {
  return value
    .replace(/^has_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
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
  const [selectedTrace, setSelectedTrace] = useState<ReturnType<typeof extractionContextTraces>[number] | null>(null);
  const traces = extractionContextTraces(context);
  const segments = buildTraceSegments(content, traces);
  const matchedTraces = new Set(segments.flatMap((segment) => segment.traces));
  const unmatchedTraces = traces.filter((trace) => trace.sourceText && !matchedTraces.has(trace));
  return (
    <>
      {createPortal(
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
                    <span
                      className={`chunk-trace-highlight ${segment.tooltipBelow ? 'below' : ''}`}
                      key={index}
                      role="button"
                      tabIndex={0}
                      onClick={() => setSelectedTrace(segment.traces[0])}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          setSelectedTrace(segment.traces[0]);
                        }
                      }}
                    >
                      {segment.text}
                      <span className="chunk-trace-tooltip">
                        {segment.traces.map((trace, traceIndex) => (
                          <span key={`${trace.objectKind}-${traceIndex}`}>
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
                      <button
                        className="chunk-unmatched-trace"
                        key={`${trace.objectKind}-${index}`}
                        type="button"
                        onClick={() => setSelectedTrace(trace)}
                      >
                        <strong>{extractionContextTraceLabel(trace)}</strong>
                        <small>{trace.sourceText}</small>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body,
      )}
      {selectedTrace && <ExtractionObjectModal trace={selectedTrace} onClose={() => setSelectedTrace(null)} />}
    </>
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
  chunkResults,
  currentChunk,
  chunksByFile,
  packageFiles,
  progress,
  status,
  budget,
  tokenUsageSummary,
}: {
  chunkResults: ExtractionChunkResult[];
  currentChunk?: ExtractionChunkRef | null;
  chunksByFile: ChunkResponse[][];
  packageFiles: FileEntryResponse[];
  progress?: PatchProgress | null;
  status?: PatchTaskStatus | null;
  budget?: LlmBudget | null;
  tokenUsageSummary?: ReactNode;
}) {
  const [traceChunk, setTraceChunk] = useState<{ chunk: ExtractionChunkResult; content: string } | null>(null);
  const resultByKey = new Map(chunkResults.map((chunk) => [chunkResultKey(chunk), chunk]));
  const packageFileByPath = new Map(packageFiles.map((file) => [file.file_path, file]));
  const chunkGroupsByPath = new Map(
    chunksByFile
      .map((group): [string, ChunkResponse[]] => [group[0]?.file_path ?? '', group])
      .filter(([filePath]) => Boolean(filePath)),
  );
  const fallbackFilePaths = [
    ...chunksByFile.map((group) => group[0]?.file_path).filter((filePath): filePath is string => Boolean(filePath)),
    ...chunkResults.map((chunk) => chunk.file_path),
  ];
  const filePaths = Array.from(new Set(fallbackFilePaths));
  const totalChunks = progress?.total_chunks || chunkResults.length || chunksByFile.flat().length;
  const completedChunks = chunkResults.filter((chunk) => chunk.status === 'completed').length;
  const runningChunks = chunkResults.filter((chunk) => chunk.status === 'running').length;
  const failedChunks = chunkResults.filter((chunk) => chunk.status === 'failed').length;

  if (!filePaths.length) {
    return (
      <div className="extraction-overview-empty">
        <strong>No chunk extraction context yet.</strong>
        <p>Create chunks and run chunk extraction after initial file understanding.</p>
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

      {tokenUsageSummary}

      <div className="ranked-file-list">
        {filePaths.map((filePath, fileIndex) => {
          const rank = fileIndex + 1;
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
                  const emptyExtractionResult = chunk.status === 'completed' && !extractionContextHasStructuredItems(chunk.extraction_context);
                  const statusClass = running ? 'running' : chunk.status === 'completed' ? 'completed' : chunk.status === 'failed' ? 'failed' : 'pending';
                  const maxContextLength = budget?.max_context_length ?? 0;
                  const reachedTokenLimit = chunk.status === 'completed'
                    && typeof chunk.context_tokens === 'number'
                    && maxContextLength > 0
                    && chunk.context_tokens >= maxContextLength;
                  const nearTokenLimit = chunk.status === 'completed'
                    && typeof chunk.context_tokens === 'number'
                    && maxContextLength > 0
                    && chunk.context_tokens >= maxContextLength * 0.9
                    && !reachedTokenLimit;
                  const sourceChunk = (chunkGroupsByPath.get(chunk.file_path) ?? []).find((item) => chunkResultKey(item) === chunkResultKey(chunk));
                  const chunkText = sourceChunk?.content ?? '';
                  return (
                    <details className={`extraction-chunk-card ${statusClass} ${emptyExtractionResult ? 'empty' : ''} ${nearTokenLimit ? 'token-warning' : ''} ${reachedTokenLimit ? 'token-danger' : ''}`} key={chunkResultKey(chunk)} open={running ? true : undefined}>
                      <summary>
                        <div>
                          <strong>Chunk {chunk.chunk_index + 1}</strong>
                          <small>Lines {chunk.start_idx}-{chunk.end_idx}</small>
                        </div>
                        <span className={`chunk-status ${emptyExtractionResult ? 'empty' : ''}`}>
                          {running && <i aria-hidden="true" />}
                          {running ? 'extracting' : emptyExtractionResult ? 'empty' : chunk.status}
                        </span>
                      </summary>
                      {chunk.status === 'failed' && chunk.error && <p className="warning">{chunk.error}</p>}
                      {(chunk.status === 'completed' || chunkText) && (
                        <div className="chunk-call-meta">
                          {chunk.status === 'completed' && chunk.context_tokens ? (
                            <span
                              className={reachedTokenLimit ? 'danger' : nearTokenLimit ? 'warning' : undefined}
                              title={
                                reachedTokenLimit && budget?.max_context_length
                                  ? `Uses 100% or more of the ${formatTokenCount(budget.max_context_length)} token limit.`
                                  : nearTokenLimit && budget?.max_context_length
                                    ? `Uses at least 90% of the ${formatTokenCount(budget.max_context_length)} token limit.`
                                    : undefined
                              }
                            >
                              {formatTokenCount(chunk.context_tokens)} context tokens{reachedTokenLimit ? ' - limit reached' : nearTokenLimit ? ' - near limit' : ''}
                            </span>
                          ) : null}
                          {chunk.status === 'completed' && chunk.response_duration_ms ? <span className="response-generation">{formatDuration(chunk.response_duration_ms)} response generation</span> : null}
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

function InitialFileUnderstandingPanel({
  progress,
  status,
  tokenUsageSummary,
}: {
  progress?: PatchProgress | null;
  status?: PatchTaskStatus | null;
  tokenUsageSummary?: ReactNode;
}) {
  const rankedCount = progress?.ranked_files?.length ?? 0;
  const summaryCount = progress?.initial_file_summaries?.length ?? 0;
  const hasArtifacts = Boolean(
    rankedCount
    || summaryCount
    || progress?.initial_file_summary_status
    || progress?.initial_extraction_overview_status
    || progress?.initial_extraction_overview,
  );

  if (!hasArtifacts) {
    return (
      <div className="extraction-overview-empty">
        <strong>No initial file understanding yet.</strong>
        <p>Run initial file understanding after uploading a dataset archive.</p>
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
          <span>Current stage</span>
          <strong>{formatExtractionStage(progress?.stage || 'not started')}</strong>
        </div>
        <div>
          <span>Ranked files</span>
          <strong>{rankedCount}</strong>
        </div>
        <div>
          <span>Summaries</span>
          <strong>{summaryCount} files</strong>
        </div>
      </div>

      {progress?.ranked_files?.length ? (
        <details className="initial-overview-panel">
          <summary>
            <div>
              <span>Ranked files</span>
              <strong>{progress.ranked_files.length} ranked paths</strong>
            </div>
          </summary>
          <JsonDetails title="File ranking" value={progress.ranked_files} />
        </details>
      ) : null}

      {((progress?.initial_file_summaries?.length ?? 0) > 0 || progress?.initial_file_summary_status) ? (
        <details className="initial-overview-panel" open>
          <summary>
            <div>
              <span>Ranked file summaries</span>
              <strong>{formatExtractionStage(progress?.initial_file_summary_status || 'not available')}</strong>
            </div>
          </summary>
          {(progress?.initial_file_summaries?.length ?? 0) > 0 ? (
            <JsonDetails title="Per-file extraction guidance" value={progress?.initial_file_summaries ?? []} />
          ) : (
            <p className="muted">No ranked file summaries are available for this run.</p>
          )}
        </details>
      ) : null}

      {(progress?.initial_extraction_overview || progress?.initial_extraction_overview_status) ? (
        <details className="initial-overview-panel" open>
          <summary>
            <div>
              <span>Initial overview</span>
              <strong>{formatExtractionStage(progress?.initial_extraction_overview_status || 'not available')}</strong>
            </div>
          </summary>
          {progress?.initial_extraction_overview ? (
            <JsonDetails title="Run-level extraction guidance" value={progress.initial_extraction_overview} />
          ) : (
            <p className="muted">No overview guidance is available for this run.</p>
          )}
        </details>
      ) : null}

      {tokenUsageSummary}
    </div>
  );
}

const QUANTITATIVE_ATTRIBUTE_VOCABULARIES = [
  { label: 'Quantity kinds', identifier: 'http://qudt.org/vocab/quantitykind' },
  { label: 'Units', identifier: 'http://qudt.org/vocab/unit' },
];

type VocabQueryNumericField = {
  key: keyof ExtractionVocabQueryConfig;
  label: string;
  step?: string;
  min?: string;
};

const QUALITATIVE_VOCAB_QUERY_FIELDS: VocabQueryNumericField[] = [
  { key: 'vector_top_k', label: 'Vector top K' },
  { key: 'fulltext_top_k', label: 'Full-text top K' },
  { key: 'seed_top_k', label: 'Seed top K' },
  { key: 'max_hops', label: 'Max hops', min: '0' },
  { key: 'max_statements_per_seed', label: 'Statements / seed' },
  { key: 'vector_weight', label: 'Vector weight', step: '0.1', min: '0.1' },
  { key: 'fulltext_weight', label: 'Full-text weight', step: '0.1', min: '0.1' },
  { key: 'rrf_k', label: 'RRF K' },
];

const QUANTITATIVE_VOCAB_QUERY_FIELDS: VocabQueryNumericField[] = [
  { key: 'quantitative_vector_top_k', label: 'Vector top K' },
  { key: 'quantitative_fulltext_top_k', label: 'Full-text top K' },
  { key: 'quantitative_seed_top_k', label: 'Seed top K' },
  { key: 'quantitative_max_hops', label: 'Max hops', min: '0' },
  { key: 'quantitative_max_statements_per_seed', label: 'Statements / seed' },
  { key: 'quantitative_vector_weight', label: 'Vector weight', step: '0.1', min: '0.1' },
  { key: 'quantitative_fulltext_weight', label: 'Full-text weight', step: '0.1', min: '0.1' },
  { key: 'quantitative_rrf_k', label: 'RRF K' },
];

const VOCAB_QUERY_NUMERIC_FIELDS = [
  ...QUALITATIVE_VOCAB_QUERY_FIELDS,
  ...QUANTITATIVE_VOCAB_QUERY_FIELDS,
];

const QUANTITATIVE_VOCAB_QUERY_DEFAULTS = {
  quantitative_vector_top_k: 12,
  quantitative_fulltext_top_k: 12,
  quantitative_seed_top_k: 6,
  quantitative_max_hops: 0,
  quantitative_max_statements_per_seed: 12,
  quantitative_traversal_direction: 'undirected',
  quantitative_vector_weight: 1,
  quantitative_fulltext_weight: 1,
  quantitative_rrf_k: 60,
};

type VocabQueryConfigMode = 'qualitative' | 'quantitative';

function normalizeVocabQueryConfig(config: ExtractionVocabQueryConfig): ExtractionVocabQueryConfig {
  return {
    ...QUANTITATIVE_VOCAB_QUERY_DEFAULTS,
    ...config,
  };
}

function VocabQueryConfigPanel({
  config,
  disabled,
  onApply,
  onRerunAll,
}: {
  config: ExtractionVocabQueryConfig;
  disabled?: boolean;
  onApply?: (config: ExtractionVocabQueryConfig) => Promise<void> | void;
  onRerunAll?: () => void;
}) {
  const [draft, setDraft] = useState(normalizeVocabQueryConfig(config));
  const [vocabOptions, setVocabOptions] = useState(config.qualitative_vocab_identifiers);
  const [availableVocabularies, setAvailableVocabularies] = useState<string[]>([]);
  const [selectedVocabIdentifier, setSelectedVocabIdentifier] = useState('');
  const [vocabListMessage, setVocabListMessage] = useState('');
  const [modalMode, setModalMode] = useState<VocabQueryConfigMode | null>(null);
  const [applyMessage, setApplyMessage] = useState('');
  const prevConfigRef = useRef(normalizeVocabQueryConfig(config));
  useEffect(() => {
    const normalized = normalizeVocabQueryConfig(config);
    const prev = prevConfigRef.current;
    const changed = (
      prev.qualitative_vocab_identifiers.length !== normalized.qualitative_vocab_identifiers.length
      || prev.qualitative_vocab_identifiers.some((v, i) => v !== normalized.qualitative_vocab_identifiers[i])
      || prev.vector_top_k !== normalized.vector_top_k
      || prev.fulltext_top_k !== normalized.fulltext_top_k
      || prev.seed_top_k !== normalized.seed_top_k
      || prev.max_hops !== normalized.max_hops
      || prev.max_statements_per_seed !== normalized.max_statements_per_seed
      || prev.traversal_direction !== normalized.traversal_direction
      || prev.vector_weight !== normalized.vector_weight
      || prev.fulltext_weight !== normalized.fulltext_weight
      || prev.rrf_k !== normalized.rrf_k
      || prev.quantitative_vector_top_k !== normalized.quantitative_vector_top_k
      || prev.quantitative_fulltext_top_k !== normalized.quantitative_fulltext_top_k
      || prev.quantitative_seed_top_k !== normalized.quantitative_seed_top_k
      || prev.quantitative_max_hops !== normalized.quantitative_max_hops
      || prev.quantitative_max_statements_per_seed !== normalized.quantitative_max_statements_per_seed
      || prev.quantitative_traversal_direction !== normalized.quantitative_traversal_direction
      || prev.quantitative_vector_weight !== normalized.quantitative_vector_weight
      || prev.quantitative_fulltext_weight !== normalized.quantitative_fulltext_weight
      || prev.quantitative_rrf_k !== normalized.quantitative_rrf_k
    );
    if (changed) {
      setDraft(normalized);
      setVocabOptions((current) => Array.from(new Set([...current, ...normalized.qualitative_vocab_identifiers])));
    }
    prevConfigRef.current = normalized;
  }, [config]);
  useEffect(() => {
    setApplyMessage('');
  }, [modalMode]);
  useEffect(() => {
    if (modalMode !== 'qualitative') return;
    let cancelled = false;
    setVocabListMessage('Loading vocabularies...');
    void (async () => {
      try {
        const next = await listVocabularies();
        if (cancelled) return;
        setAvailableVocabularies(next);
        setVocabOptions((current) => Array.from(new Set([...current, ...next, ...draft.qualitative_vocab_identifiers])));
        setSelectedVocabIdentifier((current) => current || next.find((identifier) => !draft.qualitative_vocab_identifiers.includes(identifier)) || '');
        setVocabListMessage('');
      } catch (error) {
        if (!cancelled) setVocabListMessage(error instanceof Error ? error.message : 'Could not load vocabularies.');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [modalMode]);
  const setNumber = (key: keyof ExtractionVocabQueryConfig, value: string) => {
    const parsed = Number(value);
    setDraft((current) => ({ ...current, [key]: Number.isFinite(parsed) ? parsed : 0 }));
  };
  const validateDraft = () => {
    for (const field of VOCAB_QUERY_NUMERIC_FIELDS) {
      const value = draft[field.key];
      const min = Number(field.min ?? '1');
      if (typeof value !== 'number' || !Number.isFinite(value) || value < min) {
        return `${field.label} must be ${field.min ?? '1'} or greater.`;
      }
      if ((field.step ?? '1') === '1' && !Number.isInteger(value)) {
        return `${field.label} must be a whole number.`;
      }
    }
    return '';
  };
  const applyDraft = async () => {
    if (!onApply) return;
    const validationError = validateDraft();
    if (validationError) {
      setApplyMessage(validationError);
      return;
    }
    setApplyMessage('');
    try {
      await onApply(draft);
      prevConfigRef.current = normalizeVocabQueryConfig(draft);
      setModalMode(null);
    } catch (error) {
      setApplyMessage(error instanceof Error ? error.message : 'Failed to update vocabulary query configuration.');
    }
  };
  const toggleVocab = (identifier: string, checked: boolean) => {
    setDraft((current) => ({
      ...current,
      qualitative_vocab_identifiers: checked
        ? Array.from(new Set([...current.qualitative_vocab_identifiers, identifier]))
        : current.qualitative_vocab_identifiers.filter((item) => item !== identifier),
    }));
  };
  const addVocabOption = () => {
    const identifier = selectedVocabIdentifier.trim();
    if (!identifier) return;
    setVocabOptions((current) => Array.from(new Set([...current, identifier])));
    setDraft((current) => ({
      ...current,
      qualitative_vocab_identifiers: Array.from(new Set([...current.qualitative_vocab_identifiers, identifier])),
    }));
    setSelectedVocabIdentifier('');
  };
  const addableVocabularies = availableVocabularies.filter((identifier) => !draft.qualitative_vocab_identifiers.includes(identifier));
  const renderNumberFields = (fields: VocabQueryNumericField[]) => fields.map((field) => (
    <label key={field.key}>
      <span>{field.label}</span>
      <input
        type="number"
        step={field.step ?? '1'}
        min={field.min ?? '1'}
        value={String(draft[field.key] ?? '')}
        onChange={(event) => setNumber(field.key, event.target.value)}
      />
    </label>
  ));
  const qualitativeControls = (
    <section className="vocab-query-config-section">
      <div>
        <span>Qualitative attribute queries</span>
        <strong>Vocabulary terms for descriptive attributes</strong>
      </div>
      <div className="vocab-query-config-grid">
        <section className="vocab-query-vocab-list">
          <span>Qualitative vocabularies</span>
          <div>
            {vocabOptions.map((identifier) => (
              <label key={identifier}>
                <input
                  type="checkbox"
                  checked={draft.qualitative_vocab_identifiers.includes(identifier)}
                  onChange={(event) => toggleVocab(identifier, event.target.checked)}
                />
                <span>{identifier}</span>
              </label>
            ))}
          </div>
          <div className="vocab-query-add-vocab">
            <select value={selectedVocabIdentifier} onChange={(event) => setSelectedVocabIdentifier(event.target.value)}>
              <option value="">Select vocabulary to add</option>
              {addableVocabularies.map((identifier) => (
                <option key={identifier} value={identifier}>{identifier}</option>
              ))}
            </select>
            <button className="ghost small" type="button" disabled={!selectedVocabIdentifier} onClick={addVocabOption}>Add</button>
          </div>
          {vocabListMessage && <p className="muted vocab-query-config-message">{vocabListMessage}</p>}
        </section>
        {renderNumberFields(QUALITATIVE_VOCAB_QUERY_FIELDS)}
        <label>
          <span>Traversal</span>
          <select
            value={draft.traversal_direction}
            onChange={(event) => setDraft((current) => ({ ...current, traversal_direction: event.target.value }))}
          >
            <option value="undirected">Undirected</option>
            <option value="outgoing">Outgoing</option>
            <option value="incoming">Incoming</option>
          </select>
        </label>
      </div>
    </section>
  );
  const quantitativeControls = (
    <section className="vocab-query-config-section">
      <div>
        <span>Quantitative attribute queries</span>
        <strong>Quantity-kind and unit candidate retrieval</strong>
      </div>
      <div className="vocab-query-config-grid">
        <section className="vocab-query-vocab-list vocab-query-fixed-vocabs">
          <span>Quantitative vocabularies</span>
          <div>
            {QUANTITATIVE_ATTRIBUTE_VOCABULARIES.map((vocabulary) => (
              <label key={vocabulary.identifier}>
                <input type="checkbox" checked readOnly />
                <span>{vocabulary.label}: {vocabulary.identifier}</span>
              </label>
            ))}
          </div>
        </section>
        {renderNumberFields(QUANTITATIVE_VOCAB_QUERY_FIELDS)}
        <label>
          <span>Traversal</span>
          <select
            value={draft.quantitative_traversal_direction}
            onChange={(event) => setDraft((current) => ({ ...current, quantitative_traversal_direction: event.target.value }))}
          >
            <option value="undirected">Undirected</option>
            <option value="outgoing">Outgoing</option>
            <option value="incoming">Incoming</option>
          </select>
        </label>
      </div>
    </section>
  );
  return (
    <>
      <section className="vocab-query-config-panel compact">
        <div>
          <span>Vocabulary query configuration</span>
          <strong>Run-wide parameters</strong>
        </div>
        <div>
          <button className="ghost small" type="button" disabled={disabled} onClick={() => setModalMode('qualitative')}>Configure qualitative</button>
          <button className="ghost small" type="button" disabled={disabled} onClick={() => setModalMode('quantitative')}>Configure quantitative</button>
          <button className="small" type="button" disabled={disabled || !onRerunAll} onClick={() => onRerunAll?.()}>Rerun vocabulary queries</button>
        </div>
      </section>
      {modalMode && createPortal(
        <div className="vocab-dialog-overlay" onClick={() => setModalMode(null)}>
          <div className="vocab-dialog vocab-query-config-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <div>
                <span>{modalMode === 'qualitative' ? 'Qualitative query configuration' : 'Quantitative query configuration'}</span>
                <strong>{modalMode === 'qualitative' ? 'Descriptive attribute parameters' : 'Quantity-kind and unit parameters'}</strong>
              </div>
              <button className="ghost" type="button" onClick={() => setModalMode(null)}>Close</button>
            </div>
            <div className="vocab-dialog-body vocab-query-config-body">
              <div className="vocab-query-config-sections">
                {modalMode === 'qualitative' ? qualitativeControls : quantitativeControls}
              </div>
              <div className="vocab-query-config-actions">
                <button
                  className="ghost"
                  type="button"
                  disabled={disabled || !onApply}
                  onClick={() => void applyDraft()}
                >
                  Apply
                </button>
                <button type="button" disabled={disabled || !onRerunAll} onClick={() => onRerunAll?.()}>Rerun vocabulary queries</button>
              </div>
              {applyMessage && <p className="warning vocab-query-config-message">{applyMessage}</p>}
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

function VocabQueryTraceList({
  queries,
  onRerun,
  onSelectCandidate,
  onMarkUnresolved,
}: {
  queries: ExtractionVocabQueryRecord[];
  onRerun?: (queryId: string) => void;
  onSelectCandidate?: (query: ExtractionVocabQueryRecord, uri: string, title?: string | null) => void;
  onMarkUnresolved?: (query: ExtractionVocabQueryRecord) => void;
}) {
  if (!queries.length) return null;
  const completed = queries.filter((query) => query.status === 'completed').length;
  const failed = queries.filter((query) => query.status === 'failed').length;
  const running = queries.filter((query) => query.status === 'running').length;
  const quantitativeQueries = queries.filter((query) => query.kind === 'quantity_kind' || query.kind === 'unit');
  const qualitativeQueries = queries.filter((query) => query.kind === 'qualitative_attribute');
  const objectGroundingQueries = queries.filter((query) => query.kind === 'object_grounding');
  const otherQueries = queries.filter((query) => !quantitativeQueries.includes(query) && !qualitativeQueries.includes(query) && !objectGroundingQueries.includes(query));
  const queryGroups = [
    { key: 'object-grounding', title: 'Object grounding', description: 'voc4cat term candidates', queries: objectGroundingQueries },
    { key: 'quantitative', title: 'Quantitative queries', description: 'Quantity kinds and units', queries: quantitativeQueries },
    { key: 'qualitative', title: 'Qualitative queries', description: 'Descriptive attribute terms', queries: qualitativeQueries },
    ...(otherQueries.length ? [{ key: 'other', title: 'Other queries', description: 'Additional vocabulary lookups', queries: otherQueries }] : []),
  ];
  const queryStatusSummary = (groupQueries: ExtractionVocabQueryRecord[]) => {
    const groupCompleted = groupQueries.filter((query) => query.status === 'completed').length;
    const groupRunning = groupQueries.filter((query) => query.status === 'running').length;
    const groupFailed = groupQueries.filter((query) => query.status === 'failed').length;
    return `${groupCompleted}/${groupQueries.length} completed${groupRunning ? `, ${groupRunning} running` : ''}${groupFailed ? `, ${groupFailed} failed` : ''}`;
  };
  const renderQuery = (query: ExtractionVocabQueryRecord) => {
    const jsonPath = sourceContextJsonPointer(query);
    const candidateSeeds = query.result ? coerceVocabQueryResult(query.result)?.seeds.slice(0, 5) ?? [] : [];
    const canCurateField = Boolean(jsonPath && query.kind.startsWith('profile_'));
    return (
    <details className={`chunk-vocab-query ${query.status}`} key={query.query_id}>
      <summary>
        <div>
          <strong>{formatExtractionStage(query.kind)}</strong>
          <small>{query.source_value} · {query.vocabulary_identifier} · {query.rdf_type}</small>
        </div>
        <span>{query.status}</span>
      </summary>
      <div className="chunk-vocab-query-body">
        <div className="chunk-call-meta">
          {query.duration_ms ? <span>{formatDuration(query.duration_ms)} query time</span> : null}
          {query.result ? <span>{Object.keys(asRecord(query.result.resources) ?? {}).length} resources</span> : null}
          {jsonPath ? <span>{jsonPath}</span> : null}
          <button className="small ghost" type="button" disabled={!onRerun} onClick={() => onRerun?.(query.query_id)}>Rerun query</button>
        </div>
        {canCurateField ? (
          <div className="vocab-curation-actions">
            {candidateSeeds.map((seed) => {
              const title = query.result ? vocabResourceTitle(query.result, seed.uri) : null;
              return (
                <button
                  className="small ghost"
                  type="button"
                  key={seed.uri}
                  disabled={!onSelectCandidate}
                  onClick={() => onSelectCandidate?.(query, seed.uri, title)}
                  title={seed.uri}
                >
                  {title || seed.uri}
                </button>
              );
            })}
            <button
              className="small ghost"
              type="button"
              disabled={!onMarkUnresolved}
              onClick={() => onMarkUnresolved?.(query)}
            >
              Mark unresolved
            </button>
          </div>
        ) : null}
        {query.error && <p className="warning">{query.error}</p>}
        <JsonDetails title="Query input" value={query.query} />
        <JsonDetails title="Source context" value={query.source_context} />
        {query.result && <VocabQueryGraphPanel result={query.result} />}
        <JsonDetails title="Full query result" value={query.result ?? null} />
      </div>
    </details>
  );
  return (
    <section className="chunk-vocab-query-section">
      <div className="chunk-vocab-query-heading">
        <span>Vocabulary queries</span>
        <strong>{completed}/{queries.length} completed{running ? `, ${running} running` : ''}{failed ? `, ${failed} failed` : ''}</strong>
      </div>
      <div className="chunk-vocab-query-groups">
        {queryGroups.map((group) => (
          <section className={`chunk-vocab-query-group ${group.key}`} key={group.key}>
            <div className="chunk-vocab-query-group-heading">
              <div>
                <span>{group.description}</span>
                <strong>{group.title}</strong>
              </div>
              <small>{queryStatusSummary(group.queries)}</small>
            </div>
            {group.queries.length ? (
              <div className="chunk-vocab-query-list">
                {group.queries.map(renderQuery)}
              </div>
            ) : <p className="muted chunk-vocab-query-empty">No queries in this group.</p>}
          </section>
        ))}
      </div>
      {false && (
        <div className="chunk-vocab-query-list">
          {queries.map((query) => (
          <details className={`chunk-vocab-query ${query.status}`} key={query.query_id}>
            <summary>
              <div>
                <strong>{formatExtractionStage(query.kind)}</strong>
                <small>{query.source_value} · {query.vocabulary_identifier} · {query.rdf_type}</small>
              </div>
              <span>{query.status}</span>
            </summary>
            <div className="chunk-vocab-query-body">
              <div className="chunk-call-meta">
                {query.duration_ms ? <span>{formatDuration(query.duration_ms)} query time</span> : null}
                {query.result ? <span>{Object.keys(asRecord(query.result.resources) ?? {}).length} resources</span> : null}
                <button className="small ghost" type="button" disabled={!onRerun} onClick={() => onRerun?.(query.query_id)}>Rerun query</button>
              </div>
              {query.error && <p className="warning">{query.error}</p>}
              <JsonDetails title="Query input" value={query.query} />
              <JsonDetails title="Source context" value={query.source_context} />
              {query.result && <VocabQueryGraphPanel result={query.result} />}
              <JsonDetails title="Full query result" value={query.result ?? null} />
            </div>
          </details>
          ))}
        </div>
      )}
    </section>
  );
}

function VocabQueryTraceModal({
  title,
  queries,
  onClose,
  onRerun,
}: {
  title: string;
  queries: ExtractionVocabQueryRecord[];
  onClose: () => void;
  onRerun?: (queryId: string) => void;
}) {
  const completed = queries.filter((query) => query.status === 'completed').length;
  return createPortal(
    <div className="vocab-dialog-overlay" onClick={onClose}>
      <div className="vocab-dialog chunk-vocab-query-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="vocab-dialog-header">
          <div>
            <span>Vocabulary queries</span>
            <strong>{title} - {completed}/{queries.length} completed</strong>
          </div>
          <button className="ghost" type="button" onClick={onClose}>Close</button>
        </div>
        <div className="vocab-dialog-body chunk-vocab-query-dialog-body">
          <VocabQueryTraceList queries={queries} onRerun={onRerun} />
        </div>
      </div>
    </div>,
    document.body,
    );
  };
}

function DraftGroundingPanel({
  progress,
  disabled,
  onUpdateVocabQueryConfig,
  onRunGrounding,
  onRerunAllVocabQueries,
  onRerunVocabQuery,
  onSelectCandidate,
  onMarkUnresolved,
}: {
  progress?: PatchProgress | null;
  disabled?: boolean;
  onUpdateVocabQueryConfig?: (config: ExtractionVocabQueryConfig) => void;
  onRunGrounding?: () => void;
  onRerunAllVocabQueries?: () => void;
  onRerunVocabQuery?: (queryId: string) => void;
  onSelectCandidate?: (query: ExtractionVocabQueryRecord, uri: string, title?: string | null) => void;
  onMarkUnresolved?: (query: ExtractionVocabQueryRecord) => void;
}) {
  const queries = progress?.vocab_queries ?? [];
  const completed = queries.filter((query) => query.status === 'completed').length;
  const profileFields = queries.filter((query) => query.kind.startsWith('profile_'));
  return (
    <details className="draft-grounding-panel">
      <summary>
        <div>
          <span>Advanced grounding</span>
          <strong>{queries.length ? `${completed}/${queries.length} vocabulary queries completed` : 'Profile-path vocabulary grounding'}</strong>
        </div>
      </summary>
      <div className="draft-grounding-body">
        <div className="chunk-call-meta">
          <button className="small" type="button" disabled={disabled || !onRunGrounding} onClick={() => onRunGrounding?.()}>
            Run vocabulary grounding
          </button>
          <button className="small ghost" type="button" disabled={disabled || !queries.length || !onRerunAllVocabQueries} onClick={() => onRerunAllVocabQueries?.()}>
            Rerun vocabulary queries
          </button>
        </div>
        {progress?.vocab_query_config ? (
          <VocabQueryConfigPanel
            config={progress.vocab_query_config}
            disabled={disabled}
            onApply={onUpdateVocabQueryConfig}
            onRerunAll={onRerunAllVocabQueries}
          />
        ) : null}
        {queries.length ? (
          <VocabQueryTraceList
            queries={queries}
            onRerun={onRerunVocabQuery}
            onSelectCandidate={onSelectCandidate}
            onMarkUnresolved={onMarkUnresolved}
          />
        ) : (
          <p className="muted">No profile vocabulary queries have been generated yet.</p>
        )}
        {profileFields.length && profileFields.length !== queries.length ? (
          <p className="muted">{profileFields.length} query{profileFields.length === 1 ? '' : 'ies'} target profile fields.</p>
        ) : null}
      </div>
    </details>
  );
}

type VocabGraphNode = {
  uri: string;
  resource: VocabQueryResult['resources'][string] | null;
  seed: VocabQueryResult['seeds'][number] | null;
  x: number;
  y: number;
};

type VocabGraphEdge = {
  id: string;
  source: string;
  target: string;
  predicate: string;
};

const vocabGraphBaseViewBox = { x: 0, y: 0, width: 900, height: 440 };

function VocabQueryGraphPanel({ result }: { result: Record<string, unknown> }) {
  const vocabResult = useMemo(() => coerceVocabQueryResult(result), [result]);
  const graph = useMemo(() => vocabResult ? buildVocabGraph(vocabResult) : null, [vocabResult]);
  const [selectedUri, setSelectedUri] = useState(defaultVocabGraphSelection(graph));
  const [viewBox, setViewBox] = useState(vocabGraphBaseViewBox);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragStartRef = useRef<{ clientX: number; clientY: number; viewBox: typeof vocabGraphBaseViewBox; moved: boolean; targetUri: string | null } | null>(null);

  useEffect(() => {
    setSelectedUri((current) => graph?.nodes.some((node) => node.uri === current) ? current : defaultVocabGraphSelection(graph));
    setViewBox(vocabGraphBaseViewBox);
  }, [graph]);

  if (!graph?.nodes.length) return null;

  const selectedNode = graph.nodes.find((node) => node.uri === selectedUri) ?? graph.nodes[0];

  function zoomAt(clientX: number, clientY: number, scale: number) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    setViewBox((current) => {
      const nextWidth = Math.min(vocabGraphBaseViewBox.width / 0.7, Math.max(vocabGraphBaseViewBox.width / 2.2, current.width / scale));
      const nextHeight = Math.min(vocabGraphBaseViewBox.height / 0.7, Math.max(vocabGraphBaseViewBox.height / 2.2, current.height / scale));
      const pointerX = current.x + ((clientX - rect.left) / rect.width) * current.width;
      const pointerY = current.y + ((clientY - rect.top) / rect.height) * current.height;
      const ratioX = (pointerX - current.x) / current.width;
      const ratioY = (pointerY - current.y) / current.height;
      return {
        x: pointerX - ratioX * nextWidth,
        y: pointerY - ratioY * nextHeight,
        width: nextWidth,
        height: nextHeight,
      };
    });
  }

  function onGraphWheel(event: WheelEvent<Element>) {
    if (!event.ctrlKey) return;
    event.preventDefault();
    zoomAt(event.clientX, event.clientY, event.deltaY < 0 ? 1.16 : 1 / 1.16);
  }

  function onGraphPointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) return;
    const target = event.target instanceof Element ? event.target.closest<SVGGElement>('.vocab-graph-node') : null;
    dragStartRef.current = { clientX: event.clientX, clientY: event.clientY, viewBox, moved: false, targetUri: target?.dataset.uri ?? null };
    setDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onGraphPointerMove(event: PointerEvent<SVGSVGElement>) {
    const drag = dragStartRef.current;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!drag || !rect) return;
    if ((event.buttons & 1) !== 1) {
      dragStartRef.current = null;
      setDragging(false);
      return;
    }
    const deltaX = event.clientX - drag.clientX;
    const deltaY = event.clientY - drag.clientY;
    if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) drag.moved = true;
    setViewBox({
      ...drag.viewBox,
      x: drag.viewBox.x - (deltaX / rect.width) * drag.viewBox.width,
      y: drag.viewBox.y - (deltaY / rect.height) * drag.viewBox.height,
    });
  }

  function onGraphPointerUp(event: PointerEvent<SVGSVGElement>) {
    const drag = dragStartRef.current;
    if (drag && !drag.moved && drag.targetUri) {
      setSelectedUri(drag.targetUri);
    }
    dragStartRef.current = null;
    setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function zoomAtCenter(scale: number) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    zoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, scale);
  }

  return (
    <details className="vocab-graph-details" open>
      <summary>Node graph</summary>
      <div className="vocab-graph-panel">
        <div className={`vocab-graph-canvas ${dragging ? 'dragging' : ''}`} role="img" aria-label="Vocabulary query graph" onWheel={onGraphWheel}>
          <svg
            ref={svgRef}
            viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
            onPointerDown={onGraphPointerDown}
            onPointerMove={onGraphPointerMove}
            onPointerUp={onGraphPointerUp}
            onPointerCancel={onGraphPointerUp}
          >
            <defs>
              <marker id="vocab-graph-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" />
              </marker>
            </defs>
            {graph.edges.map((edge) => {
              const source = graph.nodeMap.get(edge.source);
              const target = graph.nodeMap.get(edge.target);
              if (!source || !target) return null;
              const edgePath = directedEdgePath(source, target);
              const midX = (edgePath.x1 + edgePath.x2) / 2;
              const midY = (edgePath.y1 + edgePath.y2) / 2;
              const showLabel = edge.source === selectedNode.uri || edge.target === selectedNode.uri;
              return (
                <g key={edge.id} className="vocab-graph-edge">
                  <line x1={edgePath.x1} y1={edgePath.y1} x2={edgePath.x2} y2={edgePath.y2} />
                  {showLabel && <text x={midX} y={midY}>{compactPredicate(edge.predicate)}</text>}
                </g>
              );
            })}
            {graph.nodes.map((node) => {
              const selected = node.uri === selectedNode.uri;
              const radius = vocabGraphNodeRadius(node);
              const labelLines = vocabNodeLabelLines(node, radius);
              return (
                <g
                  key={node.uri}
                  className={`vocab-graph-node ${node.seed ? 'seed' : ''} ${selected ? 'selected' : ''}`}
                  data-uri={node.uri}
                  transform={`translate(${node.x} ${node.y})`}
                  onClick={() => setSelectedUri(node.uri)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setSelectedUri(node.uri);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <circle r={radius} />
                  <text className={labelLines.length > 1 ? 'multiline' : ''}>
                    {labelLines.map((line, index) => (
                      <tspan key={`${line}-${index}`} x="0" dy={index === 0 ? labelDy(labelLines.length) : '1.05em'}>{line}</tspan>
                    ))}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="vocab-graph-controls" aria-label="Graph controls">
            <button className="ghost small" type="button" onClick={() => zoomAtCenter(1.18)}>+</button>
            <button className="ghost small" type="button" onClick={() => zoomAtCenter(1 / 1.18)}>-</button>
            <button className="ghost small" type="button" onClick={() => setViewBox(vocabGraphBaseViewBox)}>Fit</button>
          </div>
        </div>
        <aside className="vocab-graph-inspector">
          <div className="vocab-graph-inspector-heading">
            <span>Node properties</span>
            <strong>{vocabNodeTitle(selectedNode)}</strong>
          </div>
          <div className="vocab-graph-labels">
            {selectedNode.seed && <span>Seed</span>}
            {(selectedNode.resource?.rdf_types ?? []).map((rdfType) => <span key={rdfType}>{compactPredicate(rdfType)}</span>)}
          </div>
          <dl>
            <div>
              <dt>uri</dt>
              <dd>{selectedNode.uri}</dd>
            </div>
            {selectedNode.seed && (
              <>
                <div>
                  <dt>rrf score</dt>
                  <dd>{formatScore(selectedNode.seed.rrf_score)}</dd>
                </div>
                <div>
                  <dt>vector rank</dt>
                  <dd>{selectedNode.seed.vector_rank ?? '-'}</dd>
                </div>
                <div>
                  <dt>fulltext rank</dt>
                  <dd>{selectedNode.seed.fulltext_rank ?? '-'}</dd>
                </div>
              </>
            )}
            {Object.entries(selectedNode.resource?.properties ?? {}).map(([key, value]) => (
              <div key={key}>
                <dt>{compactPredicate(key)}</dt>
                <dd>{propertyPreview(value)}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </div>
    </details>
  );
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="json-details">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}

function buildVocabGraph(result: VocabQueryResult) {
  const seedMap = new Map(result.seeds.map((seed) => [seed.uri, seed]));
  const edgeUris = new Set<string>();
  const edges: VocabGraphEdge[] = result.graph_statements.map((statement, index) => {
    edgeUris.add(statement.subject_uri);
    edgeUris.add(statement.object_uri);
    return {
      id: `${statement.subject_uri}-${statement.predicate}-${statement.object_uri}-${index}`,
      source: statement.subject_uri,
      target: statement.object_uri,
      predicate: statement.predicate,
    };
  });
  const uris = Array.from(new Set([
    ...result.seeds.map((seed) => seed.uri),
    ...Object.keys(result.resources),
    ...edgeUris,
  ]));
  const seedUris = new Set(result.seeds.map((seed) => seed.uri));
  const centerX = 450;
  const centerY = 220;
  const seedCount = Math.max(1, result.seeds.length);
  const nodes: VocabGraphNode[] = uris.map((uri, index) => {
    const seedIndex = result.seeds.findIndex((seed) => seed.uri === uri);
    const isSeed = seedIndex >= 0;
    const nonSeedIndex = Math.max(0, index - result.seeds.length);
    const angle = isSeed
      ? ((Math.PI * 2) / seedCount) * seedIndex - Math.PI / 2
      : ((Math.PI * 2) / Math.max(1, uris.length - result.seeds.length)) * nonSeedIndex - Math.PI / 2;
    const radius = isSeed ? (seedCount === 1 ? 0 : 86) : 132 + (nonSeedIndex % 3) * 52;
    return {
      uri,
      resource: result.resources[uri] ?? null,
      seed: seedMap.get(uri) ?? null,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  }).sort((left, right) => Number(seedUris.has(left.uri)) - Number(seedUris.has(right.uri)));
  const nodeMap = new Map(nodes.map((node) => [node.uri, node]));
  return { nodes, edges, nodeMap };
}

function defaultVocabGraphSelection(graph: ReturnType<typeof buildVocabGraph> | null): string {
  return graph?.nodes.find((node) => node.seed)?.uri ?? graph?.nodes[0]?.uri ?? '';
}

function vocabGraphNodeRadius(node: VocabGraphNode): number {
  return node.seed ? 22 : 17;
}

function directedEdgePath(source: VocabGraphNode, target: VocabGraphNode) {
  const deltaX = target.x - source.x;
  const deltaY = target.y - source.y;
  const length = Math.hypot(deltaX, deltaY) || 1;
  const unitX = deltaX / length;
  const unitY = deltaY / length;
  const sourceRadius = vocabGraphNodeRadius(source) + 2;
  const targetRadius = vocabGraphNodeRadius(target) + 7;
  return {
    x1: source.x + unitX * sourceRadius,
    y1: source.y + unitY * sourceRadius,
    x2: target.x - unitX * targetRadius,
    y2: target.y - unitY * targetRadius,
  };
}

function coerceVocabQueryResult(value: Record<string, unknown>): VocabQueryResult | null {
  const resources = asRecord(value.resources);
  if (!resources) return null;
  const seeds = asRecordArray(value.seeds).map((seed) => ({
    uri: String(seed.uri || ''),
    rdf_type: String(seed.rdf_type || ''),
    rrf_score: typeof seed.rrf_score === 'number' ? seed.rrf_score : 0,
    vector_score: typeof seed.vector_score === 'number' ? seed.vector_score : null,
    vector_rank: typeof seed.vector_rank === 'number' ? seed.vector_rank : null,
    fulltext_score: typeof seed.fulltext_score === 'number' ? seed.fulltext_score : null,
    fulltext_rank: typeof seed.fulltext_rank === 'number' ? seed.fulltext_rank : null,
  })).filter((seed) => seed.uri);
  const graphStatements = asRecordArray(value.graph_statements).map((statement) => ({
    subject_uri: String(statement.subject_uri || ''),
    predicate: String(statement.predicate || ''),
    object_uri: String(statement.object_uri || ''),
  })).filter((statement) => statement.subject_uri && statement.object_uri);
  const typedResources = Object.fromEntries(Object.entries(resources).map(([uri, resourceValue]) => {
    const resource = asRecord(resourceValue);
    const properties = asRecord(resource?.properties) ?? {};
    return [uri, {
      uri: String(resource?.uri || uri),
      rdf_types: Array.isArray(resource?.rdf_types) ? resource.rdf_types.map(String) : [],
      properties,
    }];
  }));
  return {
    identifier: String(value.identifier || ''),
    rdf_type: String(value.rdf_type || ''),
    seeds,
    graph_statements: graphStatements,
    resources: typedResources,
  };
}

function vocabNodeLabelLines(node: VocabGraphNode, radius: number): string[] {
  const title = vocabNodeTitle(node);
  const words = title.split(/\s+/).filter(Boolean);
  const maxChars = radius >= 22 ? 9 : 7;
  if (!words.length) return [trimNodeLabel(compactPredicate(node.uri), maxChars)];

  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars) {
      current = next;
      continue;
    }
    if (current) lines.push(current);
    current = word;
    if (lines.length === 2) break;
  }
  if (current && lines.length < 2) lines.push(current);
  return lines.slice(0, 2).map((line) => trimNodeLabel(line, maxChars));
}

function trimNodeLabel(value: string, maxChars: number): string {
  return value.length > maxChars ? `${value.slice(0, Math.max(1, maxChars - 2))}..` : value;
}

function labelDy(lineCount: number): string {
  return lineCount > 1 ? '-0.3em' : '0.32em';
}

function vocabNodeTitle(node: VocabGraphNode): string {
  const label = findVocabPropertyLabel(node.resource);
  return label || compactPredicate(node.uri);
}

function findVocabPropertyLabel(resource: VocabQueryResult['resources'][string] | null): string | null {
  if (!resource) return null;
  const labelKeys = ['preflabel', 'label', 'title', 'name'];
  for (const [key, value] of Object.entries(resource.properties)) {
    const normalized = key.toLowerCase().replace(/[_:-]/g, '');
    if (!labelKeys.some((labelKey) => normalized.endsWith(labelKey))) continue;
    if (typeof value === 'string' && value.trim()) return value;
    if (Array.isArray(value)) {
      const first = value.find((item) => typeof item === 'string' && item.trim());
      if (typeof first === 'string') return first;
    }
  }
  return null;
}

function compactPredicate(value: string): string {
  const hash = value.lastIndexOf('#');
  const slash = value.lastIndexOf('/');
  const colon = value.lastIndexOf(':');
  const index = Math.max(hash, slash, colon);
  return index >= 0 ? value.slice(index + 1) || value : value;
}

function propertyPreview(value: unknown): string {
  if (Array.isArray(value)) {
    const preview = value.slice(0, 4).map(propertyPreview).join(', ');
    return value.length > 4 ? `${preview}, ... ${value.length - 4} more` : preview || '-';
  }
  if (value && typeof value === 'object') return JSON.stringify(value);
  if (value === null || value === undefined || value === '') return '-';
  const text = String(value);
  return text.length > 360 ? `${text.slice(0, 360)}...` : text;
}

function formatScore(value?: number | null): string {
  return typeof value === 'number' ? value.toFixed(4) : '-';
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
};

function formatTokenCount(value?: number): string {
  const numberValue = typeof value === 'number' && Number.isFinite(value) ? value : 0;
  return Math.round(numberValue).toLocaleString();
}

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
                  <small>{averageInput && runtime && averageInput > runtime.input_token_budget * 0.8 ? 'Lower context usage before starting the next run.' : 'Tune chunking settings to reduce per-call context size.'}</small>
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

function ChunkInspectionPanel({
  dataPackage,
  chunksByFile,
  onViewFile,
}: {
  dataPackage?: DataPackageResponse | null;
  chunksByFile: ChunkResponse[][];
  onViewFile: (file: FileEntryResponse) => void;
}) {
  const fileByPath = new Map((dataPackage?.files ?? []).map((file) => [file.file_path, file]));
  const chunkGroups = chunksByFile.filter((group) => group.length > 0);

  if (!chunkGroups.length) {
    return null;
  }

  return (
    <div className="file-list chunk-file-list">
      {chunkGroups.map((group, fileIndex) => {
        const filePath = group[0]?.file_path ?? `file-${fileIndex + 1}`;
        const file = fileByPath.get(filePath);
        const totalIncludedLines = group.reduce((sum, chunk) => sum + chunkLineCount(chunk), 0);
        const chunkLabel = `${group.length} chunk${group.length === 1 ? '' : 's'}`;
        const lineLabel = `${totalIncludedLines} included line${totalIncludedLines === 1 ? '' : 's'}`;
        return (
          <div
            className={`file-row ${file ? '' : 'disabled'}`}
            key={filePath}
            onClick={file ? () => onViewFile(file) : undefined}
            title={file ? 'Click to view file content' : undefined}
          >
            <span>{filePath}</span>
            <div className="file-meta">
              <span className="chunk-badge">{chunkLabel}</span>
              <small>{lineLabel}</small>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ChunkingStatusPanel({
  dataPackage,
  chunkResult,
  chunksByFile,
  busy,
}: {
  dataPackage?: DataPackageResponse | null;
  chunkResult?: ChunkRequestResponse | null;
  chunksByFile: ChunkResponse[][];
  busy: BusyKey | null;
}) {
  const chunkGroups = chunksByFile.filter((group) => group.length > 0);
  const visibleChunks = chunkGroups.flat().length;
  const fileCount = dataPackage?.files.length ?? 0;
  const chunkedFiles = chunkGroups.length;
  const includedLines = chunkGroups
    .flat()
    .reduce((sum, chunk) => sum + chunkLineCount(chunk), 0);
  const status = busy === 'chunk'
    ? 'starting'
    : chunkResult?.status ?? (visibleChunks > 0 ? 'completed' : 'unknown');
  const isRunning = status === 'running' || status === 'starting';
  const statusLabel = status === 'unknown' ? 'not started' : formatExtractionStage(status);
  const statusDetail = isRunning
    ? 'Chunking is active. Saved chunks may stay at 0 until a file finishes.'
    : status === 'completed'
      ? 'Chunking completed and saved chunks are available for inspection.'
      : status === 'crashed'
        ? 'Chunking crashed before producing an inspectable chunk list.'
        : status === 'cancelled'
          ? 'Chunking was cancelled before completion.'
          : 'No chunking run is active for this dataset.';
  const embeddingInput = isRunning
    ? 'Currently embedding retained text-line windows with the configured buffer size.'
    : status === 'completed'
      ? 'Embedding already ran over retained text-line windows; saved chunks are shown below.'
      : 'Nothing is being embedded right now.';
  const pipelineDetail = 'During semantic chunking, SIMONE filters noisy lines, combines each retained line with neighboring retained lines, embeds those windows, then uses adjacent embedding distances to choose chunk breakpoints.';

  return (
    <div className={`chunking-status-panel ${isRunning ? 'running' : status}`}>
      <div className="chunking-status-grid">
        <div>
          <span>Status</span>
          <strong>{statusLabel}</strong>
        </div>
        <div>
          <span>Visible chunks</span>
          <strong>{visibleChunks}</strong>
        </div>
        <div>
          <span>Files covered</span>
          <strong>{chunkedFiles}/{fileCount || 'unknown'}</strong>
        </div>
        <div>
          <span>Included lines</span>
          <strong>{includedLines}</strong>
        </div>
      </div>
      <div className="chunking-status-copy">
        <strong>{statusDetail}</strong>
        <p>{embeddingInput}</p>
        <small>{pipelineDetail}</small>
      </div>
    </div>
  );
}

function chunkLineCount(chunk: ChunkResponse): number {
  if (chunk.filtered_line_indices?.length) return chunk.filtered_line_indices.length;
  return Math.max(0, chunk.end_idx - chunk.start_idx + 1);
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
  const [generatedFinalDraft, setGeneratedFinalDraft] = useState<Record<string, unknown> | null>(null);
  const [curatedDocument, setCuratedDocument] = useState<Record<string, unknown> | null>(null);
  const [patchStatus, setPatchStatus] = useState<PatchTaskStatus | null>(null);
  const [patchProgress, setPatchProgress] = useState<PatchProgress | null>(null);
  const [tokenUsage, setTokenUsage] = useState<PatchTokenUsage | null>(null);
  const [llmBudget, setLlmBudget] = useState<LlmBudget | null>(null);
  const [ollamaConfig, setOllamaConfig] = useState<OllamaConfig | null>(null);
  const [activeProfileSchema, setActiveProfileSchema] = useState<JsonSchemaDocument | null>(null);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(true);
  const [chunkingDialogOpen, setChunkingDialogOpen] = useState(false);
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
  const saveCuratedDocumentTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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
  const isInitialContextStage = Boolean(
    patchProgress?.stage?.startsWith('initial_')
    || patchProgress?.stage === 'file_ranking',
  );
  const isInitialContextRunning = patchStatus === 'running' && isInitialContextStage;
  const isPatching = patchStatus === 'running' && !isInitialContextStage;
  const extractionLimitReachedChunkCount = useMemo(() => {
    const maxContextLength = llmBudget?.max_context_length ?? 0;
    if (maxContextLength <= 0) return 0;
    return (patchProgress?.chunk_results ?? []).filter((chunk) => (
      chunk.status === 'completed'
      && typeof chunk.context_tokens === 'number'
      && chunk.context_tokens >= maxContextLength
    )).length;
  }, [llmBudget?.max_context_length, patchProgress?.chunk_results]);
  const extractionTokenUsageNotes = useMemo(() => (
    extractionLimitReachedChunkCount > 0
      ? {
        chunk_extraction: {
          tone: 'danger' as const,
          message: `${extractionLimitReachedChunkCount} chunk${extractionLimitReachedChunkCount === 1 ? '' : 's'} reached 100% or more of the context token limit.`,
        },
      }
      : undefined
  ), [extractionLimitReachedChunkCount]);
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
  const hasInitialContextArtifacts = Boolean(
    patchProgress?.initial_file_summary_status
    && patchProgress?.initial_extraction_overview_status,
  );
  const extractionCanResume = Boolean(
    selectedPackageId
    && selectedProfile
    && hasInitialContextArtifacts
    && !busy
    && !isPatching
    && (
      patchStatus === 'cancelled'
      || patchStatus === 'crashed'
      || (patchStatus === 'unknown' && hasPersistedExtractionState)
    ),
  );
  const projectionLedger = patchProgress?.projection_ledger ?? [];
  const hasProfileArtifacts = Boolean(generatedFinalDraft || curatedDocument);
  const projectedObjects = projectionLedger.filter((record) => record.status === 'projected').length;
  const notProjectedObjects = projectionLedger.filter((record) => record.status === 'not_projected' || record.status === 'ambiguous').length;
  const editRequiredObjects = projectionLedger.filter((record) => record.status === 'user_edit_required').length;
  const validationErrorCount = patchProgress?.validation?.errors?.length ?? 0;
  const fieldIssueCount = (patchProgress?.field_completion_ledger ?? []).filter((record) => record.issue_categories.length > 0).length;
  const curationMarkers = useMemo<JsonPatchMarker[]>(() => {
    const fieldMarkers = (patchProgress?.field_completion_ledger ?? []).flatMap((record) => {
      const markers: JsonPatchMarker[] = [];
      if (record.validation_status === 'missing') {
        markers.push({
          id: `missing:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'missing',
          label: 'Missing',
          detail: record.edit_needed_reason || 'Field needs a value.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      } else if (record.validation_status === 'invalid') {
        markers.push({
          id: `invalid:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'invalid',
          label: 'Invalid',
          detail: record.edit_needed_reason || 'Field does not satisfy the profile schema.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      }
      if (record.issue_categories.includes('non_enriched')) {
        markers.push({
          id: `non-enriched:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'non_enriched',
          label: 'Non-enriched',
          detail: record.edit_needed_reason || 'No vocabulary term was selected.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      }
      if (record.enrichment_status === 'intentionally_unresolved') {
        markers.push({
          id: `unresolved:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'intentionally_unresolved',
          label: 'Unresolved',
          detail: 'Marked intentionally unresolved.',
          evidence: record.source_evidence,
        });
      }
      return markers;
    });
    const curationMarkers = (patchProgress?.curation_ledger ?? [])
      .filter((record) => record.status !== 'unchanged')
      .map((record): JsonPatchMarker => ({
        id: `curation:${record.status}:${record.json_path}`,
        path: jsonPointerToEditorPath(record.json_path),
        status: record.status === 'user_removed' ? 'user_removed' : record.status === 'user_selected_vocab_term' ? 'user_selected_vocab_term' : 'user_edited',
        label: record.status.replace(/_/g, ' '),
        detail: record.reason || undefined,
        evidence: record.source_evidence,
      }));
    return [...fieldMarkers, ...curationMarkers];
  }, [patchProgress?.field_completion_ledger, patchProgress?.curation_ledger]);

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
    return () => {
      if (saveContextTimeoutRef.current) clearTimeout(saveContextTimeoutRef.current);
      if (saveCuratedDocumentTimeoutRef.current) clearTimeout(saveCuratedDocumentTimeoutRef.current);
    };
  }, []);

  function resetPackageWorkflowState() {
    if (saveContextTimeoutRef.current) {
      clearTimeout(saveContextTimeoutRef.current);
      saveContextTimeoutRef.current = null;
    }
    if (saveCuratedDocumentTimeoutRef.current) {
      clearTimeout(saveCuratedDocumentTimeoutRef.current);
      saveCuratedDocumentTimeoutRef.current = null;
    }
    setChunkResult(null);
    setHasChunks(false);
    setChunksByFile([]);
    setViewingFile(null);
    setFileContent(null);
    setContext(null);
    setGeneratedFinalDraft(null);
    setCuratedDocument(null);
    setPatchStatus(null);
    setPatchProgress(null);
    setTokenUsage(null);
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
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
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
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
      setMessage('Dataset uploaded. Run initial file understanding next.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Upload failed.');
    } finally {
      if (datasetUploadInputRef.current) {
        datasetUploadInputRef.current.value = '';
      }
      setBusy(null);
    }
  }

  async function onChunk(params?: { replace_existing_chunks: boolean; buffer_window_size: number; semantic_chunking_threshold: number; protected_line_indices: Record<string, number[]>; text_quality_config: TextQualityConfig; embedding_num_gpu?: number }) {
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
        embedding_num_gpu: params?.embedding_num_gpu,
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

  async function onInitialContext(forceRerun = false) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('initial-context');
    try {
      if (forceRerun) {
        setContext(null);
        setGeneratedFinalDraft(null);
        setCuratedDocument(null);
      }
      const response = await runInitialContext({
        data_package_id: packageId,
        force_rerun: forceRerun,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(response.status);
      setPatchProgress(response.progress ? { ...response.progress } : null);
      setTokenUsage(await getTokenUsage(packageId));
      setMessage(
        response.status === 'running'
          ? 'Initial file understanding is running.'
          : 'Initial file understanding is complete.',
      );
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Initial file understanding failed.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
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
      if (!hasInitialContextArtifacts) {
        setMessage('Run initial file understanding before chunk extraction.');
        return;
      }
      setContext(null);
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
      setPatchProgress(null);
      const response = await runExtraction({
        data_package_id: packageId,
        profile_identifier: selectedProfile,
        resume: options.resume,
        target_stage: 'context',
      });
      if (selectedPackageIdRef.current !== packageId) return;
      const nextContext = response.result?.machine_extraction_context || response.progress?.interim_context;
      if (nextContext) {
        setContext(initialContextFromExtractionContext(nextContext));
      }
      if (response.result) {
        setGeneratedFinalDraft(response.result.generated_final_draft);
        setCuratedDocument(response.result.curated_document ?? response.result.generated_final_draft);
      } else {
        setGeneratedFinalDraft(response.progress?.generated_final_draft ?? null);
        setCuratedDocument(response.progress?.curated_document ?? response.progress?.generated_final_draft ?? null);
      }
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

  async function onPauseExtraction() {
    if (!selectedPackageId || patchStatus !== 'running') return;
    const packageId = selectedPackageId;
    setBusy('pause');
    try {
      const { status, progress } = await pauseExtraction(packageId);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : null);
      if (progress?.interim_context) {
        setContext(initialContextFromExtractionContext(progress.interim_context));
      }
      setTokenUsage(await getTokenUsage(packageId));
      setMessage(status === 'cancelled' ? 'Extraction paused. Resume extraction to continue from saved chunks.' : 'Extraction is not running.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to pause extraction.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onUpdateVocabConfig(config: ExtractionVocabQueryConfig) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      const { progress } = await updateVocabQueryConfig(packageId, config);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setMessage('Vocabulary query configuration updated.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to update vocabulary query configuration.');
      throw error;
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onRerunVocabularyQueries(queryId?: string) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      const result = queryId
        ? await rerunVocabQuery(packageId, queryId)
        : await rerunAllVocabQueries(packageId);
      if (selectedPackageIdRef.current !== packageId) return;
      setGeneratedFinalDraft(result.generated_final_draft);
      setCuratedDocument(result.curated_document ?? result.generated_final_draft);
      setContext(initialContextFromExtractionContext(result.machine_extraction_context));
      setTokenUsage(result.token_usage);
      const { status, progress } = await getPatchProgress(packageId);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setMessage(queryId ? 'Vocabulary query rerun completed.' : 'Vocabulary queries rerun completed.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to rerun vocabulary queries.');
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

  async function onGenerateDraft() {
    if (!selectedPackageId || !selectedProfile) return;
    const isReplacingGeneratedDraft = Boolean(generatedFinalDraft);
    if (isReplacingGeneratedDraft) {
      const confirmed = window.confirm(
        'Rebuild the generated final draft?\n\nThis updates the machine artifact. Existing curated edits remain separate.',
      );
      if (!confirmed) return;
    }
    if (saveCuratedDocumentTimeoutRef.current) {
      clearTimeout(saveCuratedDocumentTimeoutRef.current);
      saveCuratedDocumentTimeoutRef.current = null;
    }
    setBusy('draft');
    try {
      const response = await runExtraction({
        data_package_id: selectedPackageId,
        profile_identifier: selectedProfile,
        resume: true,
        target_stage: 'profile',
      });
      const nextGenerated = response.result?.generated_final_draft ?? response.progress?.generated_final_draft ?? null;
      setGeneratedFinalDraft(nextGenerated);
      setCuratedDocument(response.result?.curated_document ?? response.progress?.curated_document ?? curatedDocument ?? nextGenerated);
      setPatchStatus(response.status);
      setPatchProgress(response.progress ? { ...response.progress } : null);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage(isReplacingGeneratedDraft ? 'Generated final draft rebuilt.' : 'Generated final draft construction started.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Generated final draft construction failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onCuratedDocumentChange(updated: Record<string, unknown>) {
    setCuratedDocument(updated);
    if (!selectedPackageId) return;
    if (saveCuratedDocumentTimeoutRef.current) clearTimeout(saveCuratedDocumentTimeoutRef.current);
    saveCuratedDocumentTimeoutRef.current = setTimeout(async () => {
      try {
        const saved = await saveCuratedDocument(selectedPackageId, updated, selectedProfile);
        setCuratedDocument(saved);
        setPatchProgress((current) => current ? { ...current, curated_document: saved as Record<string, unknown> } : current);
        setMessage('Curated document saved.');
      } catch (error) {
        setMessage(error instanceof Error ? error.message : 'Failed to save curated document.');
      }
    }, 800);
  }

  async function onGrounding() {
    if (!selectedPackageId || !selectedProfile) return;
    setBusy('patch');
    try {
      const result = await runVocabularyGrounding({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setCuratedDocument(result.curated_document);
      setPatchStatus(result.status);
      const { progress } = await getPatchProgress(selectedPackageId);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage(
        result.status === 'completed'
          ? 'Vocabulary grounding completed and final profile document was saved.'
          : 'Vocabulary grounding is running.',
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Vocabulary grounding failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onSelectVocabularyCandidate(query: ExtractionVocabQueryRecord, uri: string, title?: string | null) {
    if (!selectedPackageId) return;
    const jsonPath = sourceContextJsonPointer(query);
    if (!jsonPath) return;
    setBusy('patch');
    try {
      const { status, progress } = await applyCurationFieldAction({
        data_package_id: selectedPackageId,
        action: 'select_vocab_term',
        json_path: jsonPath,
        selected_uri: uri,
        selected_title: title ?? null,
        vocabulary_identifier: query.vocabulary_identifier,
      });
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      if (progress?.curated_document) setCuratedDocument(progress.curated_document);
      setMessage('Vocabulary term selected for curated document.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to select vocabulary term.');
    } finally {
      setBusy(null);
    }
  }

  async function onMarkVocabularyUnresolved(query: ExtractionVocabQueryRecord) {
    if (!selectedPackageId) return;
    const jsonPath = sourceContextJsonPointer(query);
    if (!jsonPath) return;
    setBusy('patch');
    try {
      const { status, progress } = await applyCurationFieldAction({
        data_package_id: selectedPackageId,
        action: 'mark_unresolved',
        json_path: jsonPath,
        vocabulary_identifier: query.vocabulary_identifier,
      });
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      if (progress?.curated_document) setCuratedDocument(progress.curated_document);
      setMessage('Field marked intentionally unresolved.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to mark field unresolved.');
    } finally {
      setBusy(null);
    }
  }

  async function onExportCuratedJson() {
    if (!selectedPackageId || !curatedDocument) return;
    downloadJsonFile(`${selectedPackageId}-curated-document.json`, curatedDocument);
    setMessage('Curated JSON exported.');
  }

  async function onExportCuratedJsonLd() {
    if (!selectedPackageId || !selectedProfile || !curatedDocument) return;
    setBusy('patch');
    try {
      const exported = await exportProfileDocumentJsonLd(selectedProfile, curatedDocument);
      downloadJsonFile(`${selectedPackageId}-curated-document.jsonld`, exported.document);
      setMessage(`Curated JSON-LD exported with ${exported.triple_count} triples.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Curated JSON-LD export failed.');
    } finally {
      setBusy(null);
    }
  }

  async function refreshExtractionProgress() {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    try {
      const initialProgress = await getInitialContextProgress(packageId);
      const extractionProgress = await getPatchProgress(packageId);
      const status = extractionProgress.status === 'unknown'
        ? initialProgress.status
        : extractionProgress.status;
      const progress = extractionProgress.progress ?? initialProgress.progress;
      if (selectedPackageIdRef.current !== packageId) return;
      const completedResult = status === 'completed'
        ? await getExtractionResult(packageId)
        : null;
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress || null);
      if (completedResult) {
        setContext(initialContextFromExtractionContext(completedResult.machine_extraction_context));
        setGeneratedFinalDraft(completedResult.generated_final_draft);
        setCuratedDocument(completedResult.curated_document ?? completedResult.generated_final_draft);
      } else if (progress?.interim_context) {
        setContext(initialContextFromExtractionContext(progress.interim_context));
      }
      if (!completedResult) {
        setGeneratedFinalDraft(progress?.generated_final_draft ?? null);
        setCuratedDocument(progress?.curated_document ?? progress?.generated_final_draft ?? null);
      }
      const nextTokenUsage = completedResult?.token_usage ?? await getTokenUsage(packageId);
      setTokenUsage(nextTokenUsage);
      if (status === 'running') {
        setMessage('Workflow stage is running.');
      } else {
        setMessage('Workflow progress refreshed.');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to refresh workflow progress.');
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
        const [ctx, generatedResult, curatedResult, initialRun, extractionRun, chunkStatus, usage] = await Promise.all([
          getExistingInitialContext(packageId),
          getExistingGeneratedFinalDraft(packageId),
          getExistingCuratedDocument(packageId),
          getInitialContextProgress(packageId),
          getPatchProgress(packageId),
          getChunkStatus(packageId),
          getTokenUsage(packageId),
        ]);
        const status = extractionRun.status === 'unknown' ? initialRun.status : extractionRun.status;
        const progress = extractionRun.progress ?? initialRun.progress;
        const chunks = chunkStatus.status !== 'unknown' || chunkStatus.has_chunks
          ? await getDataPackageChunks(packageId)
          : [];
        if (selectedPackageIdRef.current !== packageId) return;
        const progressContext = progress?.interim_context
          ? initialContextFromExtractionContext(progress.interim_context)
          : null;
        if (ctx || progressContext) {
          setContext(ctx ?? progressContext);
        }
        setGeneratedFinalDraft(generatedResult ?? progress?.generated_final_draft ?? null);
        setCuratedDocument(curatedResult ?? progress?.curated_document ?? progress?.generated_final_draft ?? null);
        setPatchStatus(status);
        setPatchProgress(progress || null);
        setTokenUsage(usage);
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
    const interval = setInterval(() => void refreshExtractionProgress(), 5000);
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
            title="Upload dataset"
            description="The archive is stored as a data package. File contents can be inspected before running any extraction stage."
            active
          >
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
              {viewingFile && fileContent !== null && (
                <FileViewer
                  file={viewingFile}
                  content={fileContent}
                  chunksByFile={chunksByFile}
                  onClose={closeFileViewer}
                />
              )}
          </StepPanel>

          <StepPanel
            number="02"
            title="Initial file understanding"
            description="Rank files, summarize the top files, and build a profile-independent run overview before chunking."
          >
              <div className="actions">
                <button onClick={() => void onInitialContext(Boolean(hasInitialContextArtifacts))} disabled={!selectedPackageId || !!busy || isInitialContextRunning || isPatching}>
                  {busy === 'initial-context' ? 'Running...' : hasInitialContextArtifacts ? 'Rerun initial overview' : 'Run initial overview'}
                </button>
              </div>
              <InitialFileUnderstandingPanel
                progress={patchProgress}
                status={patchStatus}
                tokenUsageSummary={(
                  <TokenUsageSummary
                    tokenUsage={tokenUsage}
                    averageUnit="request"
                    heading="Initial file understanding token usage"
                    agentKeys={['file_ranking', 'initial_file_summary', 'initial_extraction_overview', 'initial_extraction_overview_fallback']}
                    budget={llmBudget}
                  />
                )}
              />
          </StepPanel>

          <StepPanel
            number="03"
            title="Create chunks"
            description="Chunking prepares file text for later one-shot extraction calls."
          >
              <div className="actions">
                <button onClick={() => setChunkingDialogOpen(true)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking...' : 'Configure chunking'}</button>
              </div>
              <ChunkingStatusPanel
                dataPackage={selectedPackage}
                chunkResult={chunkResult}
                chunksByFile={chunksByFile}
                busy={busy}
              />
              <ChunkInspectionPanel
                dataPackage={selectedPackage}
                chunksByFile={chunksByFile}
                onViewFile={(file) => void onViewFile(file)}
              />
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
            number="04"
            title="Chunk extraction context"
            description="Extract structured machine context from chunks using the initial overview and each file summary as orientation."
          >
              <div className="actions">
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !selectedProfile || !hasChunks || !hasInitialContextArtifacts || !!busy || isPatching}>
                  {isPatching ? 'Extraction running...' : busy === 'context' ? 'Extracting...' : hasPersistedExtractionState || context ? 'Re-extract chunk context' : 'Extract chunk context'}
                </button>
                {isPatching && (
                  <button className="ghost" onClick={() => void onPauseExtraction()} disabled={!selectedPackageId || busy === 'pause'}>
                    {busy === 'pause' ? 'Pausing...' : 'Pause extraction'}
                  </button>
                )}
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
                chunkResults={patchProgress?.chunk_results ?? []}
                currentChunk={patchProgress?.current_chunk ?? null}
                chunksByFile={chunksByFile}
                packageFiles={selectedPackage?.files ?? []}
                progress={patchProgress}
                status={patchStatus}
                budget={llmBudget}
                tokenUsageSummary={(
                  <TokenUsageSummary
                    tokenUsage={tokenUsage}
                    averageUnit="operation"
                    heading="Extraction context token usage"
                    agentKeys={['chunk_extraction', 'chunk_extraction_repair', 'quantity_vocab_selection', 'qualitative_vocab_selection', 'profile_projection']}
                    budget={llmBudget}
                    notesByAgent={extractionTokenUsageNotes}
                  />
                )}
              />
              <p className="context-window-advice">
                If extraction reaches or overuses the context window, rerun chunking with smaller chunks before extracting again. Lower the semantic chunking threshold in the chunking configuration to reduce chunk sizes.
              </p>
          </StepPanel>

          <StepPanel
            number="05"
            title="Generated and curated profile"
            description="Build the machine-generated final draft, inspect projection issues, and curate the separate user document."
            actions={hasProfileArtifacts && (
              <button className="ghost draft-refresh-button" onClick={() => void refreshExtractionProgress()} disabled={!selectedPackageId || busy === 'load'}>Refresh</button>
            )}
          >
              <div className={hasProfileArtifacts ? 'draft-actions' : 'actions'}>
                {!generatedFinalDraft ? (
                  <button onClick={() => void onGenerateDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy || !context}>{busy === 'draft' ? 'Building generated draft...' : 'Build generated final draft'}</button>
                ) : (
                  <>
                    <button className="ghost draft-recreate-button" onClick={() => void onGenerateDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Rebuilding...' : 'Rebuild generated draft'}</button>
                    <button className="ghost" onClick={() => void onExportCuratedJson()} disabled={!curatedDocument || !!busy}>Export curated JSON</button>
                    <button className="ghost" onClick={() => void onExportCuratedJsonLd()} disabled={!curatedDocument || !selectedProfile || !!busy}>Export curated JSON-LD</button>
                  </>
                )}
              </div>
              {hasProfileArtifacts && (
                <section className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Projection and validation</span>
                    <span>
                      {projectedObjects} projected, {notProjectedObjects} unresolved, {editRequiredObjects} edit needed
                    </span>
                  </div>
                  <div className="patch-progress-summary">
                    <span>Draft quality: {formatExtractionStage(patchProgress?.draft_quality_state || 'not run')}</span>
                    <span>Validation: {formatExtractionStage(patchProgress?.validation?.status || 'not run')}</span>
                    <span>{validationErrorCount} validation issue{validationErrorCount === 1 ? '' : 's'}</span>
                    <span>{fieldIssueCount} field issue{fieldIssueCount === 1 ? '' : 's'}</span>
                  </div>
                </section>
              )}
              {hasProfileArtifacts && patchStatus === 'running' && (
                <div className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{formatExtractionStage(patchProgress?.stage || patchStatus)}</strong></span>
                  </div>
                </div>
              )}
              {hasProfileArtifacts && (
                <DraftGroundingPanel
                  progress={patchProgress}
                  disabled={!selectedPackageId || !!busy || isPatching}
                  onUpdateVocabQueryConfig={(config) => void onUpdateVocabConfig(config)}
                  onRunGrounding={() => void onGrounding()}
                  onRerunAllVocabQueries={() => void onRerunVocabularyQueries()}
                  onRerunVocabQuery={(queryId) => void onRerunVocabularyQueries(queryId)}
                  onSelectCandidate={(query, uri, title) => void onSelectVocabularyCandidate(query, uri, title)}
                  onMarkUnresolved={(query) => void onMarkVocabularyUnresolved(query)}
                />
              )}
              {generatedFinalDraft && (
                <JsonDetails title="Generated final draft (machine artifact)" value={generatedFinalDraft} />
              )}
              {curatedDocument && (
                <JsonEditor
                  value={curatedDocument as Record<string, unknown>}
                  onChange={(updated) => onCuratedDocumentChange(updated)}
                  patchMarkers={curationMarkers}
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
