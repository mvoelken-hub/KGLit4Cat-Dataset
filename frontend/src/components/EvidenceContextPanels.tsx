import { type ReactNode, type SyntheticEvent, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import type { ChunkResponse, FileEntryResponse } from '../api/types';
import type { EvidenceCandidate, EvidenceRoute, ExtractionChunkRef, ExtractionChunkResult, RoutedEvidenceContext, WorkflowProgress, WorkflowTaskStatus } from '../api/extraction';
import type { LlmBudget } from '../api/system';
import { formatBytes, formatDuration, formatExtractionStage, formatTokenCount } from '../lib/format';
import { asRecord } from '../lib/records';
import { JsonDetails } from './JsonDetails';
function chunkResultKey(chunk: Pick<ExtractionChunkResult, 'file_path' | 'start_idx' | 'end_idx'>): string {
  return `${chunk.file_path}:${chunk.start_idx}:${chunk.end_idx}`;
}

type ExtractionTrace = {
  objectKind: string;
  sourceText: string;
  object: EvidenceCandidate;
  route: EvidenceRoute;
};

function extractionContextItemTitle(item: EvidenceCandidate, fallback: string): string {
  return item.claim || item.candidate_id || fallback;
}

function extractionContextItemDescription(item: EvidenceCandidate): string | null {
  return item.evidence_text || null;
}

function extractionContextTraces(context?: RoutedEvidenceContext | null): ExtractionTrace[] {
  if (!context) return [];
  return [
    ...context.portable_evidence.map((candidate) => evidenceCandidateTrace(candidate, 'portable_evidence')),
    ...context.contextual_evidence.map((candidate) => evidenceCandidateTrace(candidate, 'contextual_evidence')),
    ...context.rejected_evidence.map((record) => evidenceCandidateTrace(record.candidate, 'rejected_evidence')),
  ].filter((trace): trace is ExtractionTrace => Boolean(trace));
}

function evidenceCandidateTrace(candidate: EvidenceCandidate, route: EvidenceRoute): ExtractionTrace | null {
  if (!candidate.evidence_text.trim()) return null;
  return {
    objectKind: candidate.category,
    sourceText: candidate.evidence_text,
    object: candidate,
    route,
  };
}

function extractionTraceFields(trace: ExtractionTrace): Record<string, unknown> {
  return {
    route: trace.route,
    candidate_id: trace.object.candidate_id,
    category: trace.object.category,
    role: trace.object.role,
    claim: trace.object.claim,
    evidence_text: trace.object.evidence_text,
    source_context: trace.object.source_context,
    uncertainty: trace.object.uncertainty,
    file_path: trace.object.file_path,
    start_idx: trace.object.start_idx,
    end_idx: trace.object.end_idx,
    evidence_match_score: trace.object.evidence_match_score,
  };
}

function evidenceCandidatePreviewFields(candidate: EvidenceCandidate): Array<[string, string | number]> {
  return [
    ['role', candidate.role],
    ['category', candidate.category],
    ['route_score', candidate.evidence_match_score],
    ['file_path', candidate.file_path],
    ['line_span', `${candidate.start_idx}-${candidate.end_idx}`],
    ['candidate_id', candidate.candidate_id],
  ];
}

function extractionContextHasStructuredItems(context?: RoutedEvidenceContext | null): boolean {
  return extractionContextTraces(context).length > 0;
}

function evidenceContextNotes(context?: RoutedEvidenceContext | null) {
  return context?.portable_evidence ?? [];
}

function evidenceContextRouteCount(context: RoutedEvidenceContext | null | undefined, key: keyof Pick<RoutedEvidenceContext, 'portable_evidence' | 'contextual_evidence' | 'rejected_evidence'>): number {
  return context?.[key].length ?? 0;
}

function evidenceContextHasNotes(context?: RoutedEvidenceContext | null): boolean {
  return evidenceContextNotes(context).length > 0;
}

export function evidenceContextNoteCount(context?: RoutedEvidenceContext | null): number {
  return evidenceContextNotes(context).length;
}

function evidenceContextForChunk(context: RoutedEvidenceContext | null | undefined, chunk: { file_path: string; start_idx: number; end_idx: number }): RoutedEvidenceContext | null {
  if (!context) return null;
  const inChunk = (item: EvidenceCandidate) => (
    item.file_path === chunk.file_path
    && item.start_idx <= chunk.end_idx
    && item.end_idx >= chunk.start_idx
  );
  const portable = context.portable_evidence.filter(inChunk);
  const contextual = context.contextual_evidence.filter(inChunk);
  const rejected = context.rejected_evidence.filter((record) => inChunk(record.candidate));
  return portable.length || contextual.length || rejected.length
    ? { ...context, portable_evidence: portable, contextual_evidence: contextual, rejected_evidence: rejected, assessments: [] }
    : null;
}


function extractionContextTraceLabel(trace: ExtractionTrace) {
  const label = extractionContextItemTitle(trace.object, 'Evidence note');
  return `${trace.objectKind}: ${label}`;
}

function EvidenceContextResultView({ context }: { context?: RoutedEvidenceContext | null }) {
  if (!context) return <p className="muted">No extraction result is available for this chunk yet.</p>;
  const portableCount = evidenceContextRouteCount(context, 'portable_evidence');
  const contextualCount = evidenceContextRouteCount(context, 'contextual_evidence');
  const rejectedCount = evidenceContextRouteCount(context, 'rejected_evidence');
  return (
    <div className="extraction-context-results">
      {!evidenceContextHasNotes(context) && (
        <p className="muted">The model call completed, but this chunk did not yield portable evidence.</p>
      )}
      <p className="muted">
        Portable {portableCount} · Contextual {contextualCount} · Rejected {rejectedCount}
      </p>
      <JsonDetails title="Evidence context JSON" value={context} />
    </div>
  );
}

function ExtractionObjectModal({
  trace,
  onClose,
}: {
  trace: ExtractionTrace;
  onClose: () => void;
}) {
  const itemTitle = extractionContextItemTitle(trace.object, 'Evidence note');
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
            <span>Evidence note</span>
            <ExtractionFieldList value={extractionTraceFields(trace)} />
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function EvidenceTracePopover({
  traces,
  x,
  y,
}: {
  traces: ExtractionTrace[];
  x: number;
  y: number;
}) {
  const left = Math.min(Math.max(12, x), Math.max(12, window.innerWidth - 392));
  const top = Math.min(Math.max(12, y), Math.max(12, window.innerHeight - 260));
  return createPortal(
    <div className="evidence-trace-popover" style={{ left, top }} role="tooltip">
      {traces.length > 1 ? (
        <>
          <strong>{traces.length} extracted objects</strong>
          <div className="evidence-trace-popover-list">
            {traces.map((trace, index) => (
              <span key={`${trace.objectKind}-${index}`}>{extractionContextTraceLabel(trace)}</span>
            ))}
          </div>
        </>
      ) : traces.map((trace, index) => {
        const description = extractionContextItemDescription(trace.object);
        const fields = evidenceCandidatePreviewFields(trace.object).filter(([, value]) => String(value).trim()).slice(0, 4);
        return (
          <div className="evidence-trace-popover-object" key={`${trace.objectKind}-${index}`}>
            <span>{trace.objectKind}</span>
            <strong>{extractionContextItemTitle(trace.object, 'Evidence note')}</strong>
            {description ? <p>{description}</p> : null}
            {fields.length ? (
              <dl>
                {fields.map(([key, value]) => (
                  <div key={key}>
                    <dt>{formatExtractionFieldLabel(key)}</dt>
                    <dd>{String(value)}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
          </div>
        );
      })}
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
  context?: RoutedEvidenceContext | null;
  onClose: () => void;
}) {
  const [activePanel, setActivePanel] = useState<'chunk' | 'unmatched'>('chunk');
  const [selectedTrace, setSelectedTrace] = useState<ExtractionTrace | null>(null);
  const [hoveredTraces, setHoveredTraces] = useState<{ traces: ExtractionTrace[]; x: number; y: number } | null>(null);
  const traces = extractionContextTraces(context);
  const segments = buildTraceSegments(content, traces);
  const matchedTraces = new Set(segments.flatMap((segment) => segment.traces));
  const unmatchedTraces = traces.filter((trace) => trace.sourceText && !matchedTraces.has(trace));
  const showTracePopover = (event: SyntheticEvent<HTMLElement>, traceList: ExtractionTrace[]) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setHoveredTraces({ traces: traceList, x: rect.left, y: rect.bottom + 8 });
  };
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
                      onMouseEnter={(event) => showTracePopover(event, segment.traces)}
                      onMouseLeave={() => setHoveredTraces(null)}
                      onFocus={(event) => showTracePopover(event, segment.traces)}
                      onBlur={() => setHoveredTraces(null)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          setSelectedTrace(segment.traces[0]);
                        }
                      }}
                    >
                      {segment.text}
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
                        onMouseEnter={(event) => showTracePopover(event, [trace])}
                        onMouseLeave={() => setHoveredTraces(null)}
                        onFocus={(event) => showTracePopover(event, [trace])}
                        onBlur={() => setHoveredTraces(null)}
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
      {hoveredTraces && <EvidenceTracePopover {...hoveredTraces} />}
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

export function EvidenceContextOverview({
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
  progress?: WorkflowProgress | null;
  status?: WorkflowTaskStatus | null;
  budget?: LlmBudget | null;
  tokenUsageSummary?: ReactNode;
}) {
  const [traceChunk, setTraceChunk] = useState<{ chunk: ExtractionChunkResult; content: string } | null>(null);
  type ChunkCard = ExtractionChunkResult & { content?: string };
  const displayedChunkResults = useMemo<ChunkCard[]>(() => {
    if (chunkResults.some((chunk) => chunk.evidence_context)) return chunkResults;
    const mergedContext = progress?.interim_evidence_context;
    if (!mergedContext) return chunkResults;
    return chunksByFile.flatMap((group) => group.flatMap((chunk, index) => {
      const evidenceContext = evidenceContextForChunk(mergedContext, chunk);
      return evidenceContext ? [{
        chunk_index: index,
        file_path: chunk.file_path,
        start_idx: chunk.start_idx,
        end_idx: chunk.end_idx,
        status: 'completed' as const,
        evidence_context: evidenceContext,
      }] : [];
    }));
  }, [chunkResults, chunksByFile, progress?.interim_evidence_context]);
  const resultByKey = new Map(displayedChunkResults.map((chunk) => [chunkResultKey(chunk), chunk]));
  const packageFileByPath = new Map(packageFiles.map((file) => [file.file_path, file]));
  const chunkGroupsByPath = new Map(
    chunksByFile
      .map((group): [string, ChunkResponse[]] => [group[0]?.file_path ?? '', group])
      .filter(([filePath]) => Boolean(filePath)),
  );
  const fallbackFilePaths = [
    ...chunksByFile.map((group) => group[0]?.file_path).filter((filePath): filePath is string => Boolean(filePath)),
    ...displayedChunkResults.map((chunk) => chunk.file_path),
  ];
  const fallbackFileOrder = new Map(fallbackFilePaths.map((filePath, index) => [filePath, index]));
  const rankByFilePath = new Map((progress?.ranked_files ?? []).map((file) => [file.file_path, file.rank]));
  const filePaths = Array.from(new Set(fallbackFilePaths)).sort((left, right) => {
    const leftRank = rankByFilePath.get(left);
    const rightRank = rankByFilePath.get(right);
    if (leftRank !== undefined || rightRank !== undefined) {
      return (leftRank ?? Number.MAX_SAFE_INTEGER) - (rightRank ?? Number.MAX_SAFE_INTEGER);
    }
    return (fallbackFileOrder.get(left) ?? 0) - (fallbackFileOrder.get(right) ?? 0);
  });
  const totalChunks = progress?.total_chunks || displayedChunkResults.length || chunksByFile.flat().length;
  const completedChunks = displayedChunkResults.filter((chunk) => chunk.status === 'completed').length;
  const skippedChunks = displayedChunkResults.filter((chunk) => chunk.status === 'skipped').length;
  const processedChunks = completedChunks + skippedChunks;
  const runningChunks = displayedChunkResults.filter((chunk) => chunk.status === 'running').length;
  const repairPendingChunks = displayedChunkResults.filter((chunk) => chunk.status === 'repair_pending').length;
  const failedChunks = displayedChunkResults.filter((chunk) => chunk.status === 'failed').length;
  const queueLabel = runningChunks
    ? `${runningChunks} running`
    : repairPendingChunks
      ? `${repairPendingChunks} queued for repair`
      : failedChunks
        ? `${failedChunks} failed`
        : `${Math.max(0, (totalChunks || 0) - processedChunks)} pending`;

  if (!filePaths.length) {
    return (
      <div className="extraction-overview-empty">
        <strong>No chunk evidence yet.</strong>
        <p>Create chunks and run evidence extraction after initial file understanding.</p>
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
          <strong>{processedChunks}/{totalChunks || 0} processed</strong>
        </div>
        <div>
          <span>Current stage</span>
          <strong>{formatExtractionStage(progress?.stage || 'not started')}</strong>
        </div>
        <div>
          <span>Queue</span>
          <strong>{queueLabel}</strong>
        </div>
      </div>

      {tokenUsageSummary}

      <div className="ranked-file-list">
        {filePaths.map((filePath, fileIndex) => {
          const rank = rankByFilePath.get(filePath) ?? fileIndex + 1;
          const file = packageFileByPath.get(filePath);
          const chunks = chunkGroupsByPath.get(filePath) ?? [];
          const persistedChunks = displayedChunkResults.filter((chunk) => chunk.file_path === filePath);
          const chunkCards = chunks.length
            ? chunks.map((chunk, index) => {
              const result = resultByKey.get(chunkResultKey(chunk)) as ChunkCard | undefined;
              const fallbackChunk: ChunkCard = {
                chunk_index: index,
                file_path: chunk.file_path,
                start_idx: chunk.start_idx,
                end_idx: chunk.end_idx,
                status: 'pending',
                evidence_context: null,
                content: chunk.content,
              };
              return result ?? fallbackChunk;
            })
            : persistedChunks;
          const hasRunningChunk = chunkCards.some((chunk) => chunk.status === 'running' || chunk.status === 'repair_pending' || (currentChunk && chunkResultKey(currentChunk) === chunkResultKey(chunk)));

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
                  const repairPending = chunk.status === 'repair_pending';
                  const emptyExtractionResult = chunk.status === 'completed' && !evidenceContextHasNotes(chunk.evidence_context);
                  const statusClass = running ? 'running' : repairPending ? 'repair-pending' : chunk.status === 'completed' ? 'completed' : chunk.status === 'skipped' ? 'skipped' : chunk.status === 'failed' ? 'failed' : 'pending';
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
                  const chunkText = ('content' in chunk && typeof chunk.content === 'string' ? chunk.content : sourceChunk?.content) ?? '';
                  return (
                    <details className={`extraction-chunk-card ${statusClass} ${emptyExtractionResult ? 'empty' : ''} ${nearTokenLimit ? 'token-warning' : ''} ${reachedTokenLimit ? 'token-danger' : ''}`} key={chunkResultKey(chunk)} open={running ? true : undefined}>
                      <summary>
                        <div>
                          <strong>Chunk {chunk.chunk_index + 1}</strong>
                          <small>Lines {chunk.start_idx}-{chunk.end_idx}</small>
                        </div>
                        <span className={`chunk-status ${emptyExtractionResult ? 'empty' : ''}`}>
                          {running && <i aria-hidden="true" />}
                          {running ? 'extracting' : repairPending ? 'queued for repair' : emptyExtractionResult ? 'empty' : chunk.status}
                        </span>
                      </summary>
                      {repairPending && <p className="muted">{chunk.error || 'Queued for repair after first-pass extraction.'}</p>}
                      {chunk.status === 'failed' && chunk.error && <p className="warning">{chunk.error}</p>}
                      {chunk.status === 'skipped' && chunk.skip_reason && <p className="muted">Skipped: {formatExtractionStage(chunk.skip_reason)}</p>}
                      {(chunk.status === 'completed' || chunk.status === 'skipped' || chunkText) && (
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
                      <EvidenceContextResultView context={chunk.evidence_context} />
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
          context={traceChunk.chunk.evidence_context}
          onClose={() => setTraceChunk(null)}
        />
      )}
    </div>
  );
}


