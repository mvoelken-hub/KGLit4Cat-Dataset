import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { createPortal } from 'react-dom';
import { getFileEntryContent } from '../api/datasources';
import type { ChunkResponse, DataPackageResponse, TextQualityConfig } from '../api/types';

export interface ChunkingDialogProps {
  isOpen: boolean;
  packageId: string;
  dataPackage: DataPackageResponse | null;
  chunksByFile: ChunkResponse[][];
  onClose: () => void;
  onSubmit: (params: {
    replace_existing_chunks: boolean;
    buffer_window_size: number;
    semantic_chunking_threshold: number;
    protected_line_indices: Record<string, number[]>;
    text_quality_config: TextQualityConfig;
    embedding_num_gpu?: number;
  }) => void;
}

const chunkColors = [
  'rgba(99, 154, 0, 0.22)',
  'rgba(0, 120, 180, 0.18)',
  'rgba(180, 90, 0, 0.18)',
  'rgba(140, 60, 180, 0.18)',
  'rgba(200, 50, 80, 0.18)',
  'rgba(0, 160, 140, 0.18)',
];

const configTooltips = {
  bufferWindowSize: 'Embeds each kept line together with this many kept neighbor lines before and after it. Higher values smooth local differences and usually create fewer, broader chunks; lower values react to sharper line-to-line changes and can create more granular chunks. A significant side effect of increasing the buffer size is a significant increase in the number of tokens processed by the embedding API.',
  semanticThreshold: 'Percentile cutoff for semantic distance between adjacent embedded line windows. Lower values mark more breakpoints and usually make smaller chunks; higher values keep only the strongest topic shifts and usually make larger chunks.',
  embeddingGpu: 'Per-run Ollama num_gpu override for the embedding model used during semantic chunking. Runtime default uses the current model setting, auto GPU asks Ollama to place all layers automatically, and CPU only keeps chunking embeddings off the GPU.',
  symbolThreshold: 'Symbol-heavy lines receive a quality penalty when their symbol ratio is at or above this value. Lower values are stricter and drop more notation-heavy lines; higher values keep more lines with punctuation, formulas, or metadata keys.',
  digitThreshold: 'Digit-heavy lines receive a quality penalty when their digit ratio is at or above this value. Lower values drop more numeric lines; higher values keep more measurements, identifiers, and tables.',
  keepThreshold: 'Minimum quality score for a line to be included in chunking. Lower values keep more borderline lines and noise; higher values keep fewer, cleaner lines but may remove useful evidence.',
  structuredBonus: 'Score bonus for short structured text such as key/value metadata, headings, bullets, or JSON/YAML-like lines. Higher values keep more structured metadata lines; lower values make them easier to filter out.',
  protectedLines: 'Selected lines are forced into the chunk input even if the text-quality filter would normally drop them. Use this for headers, identifiers, or metadata lines that look noisy but are semantically important.',
};

const defaultTextQualityConfig: Required<TextQualityConfig> = {
  symbol_ratio_threshold: 0.45,
  digit_ratio_threshold: 0.45,
  keep_score_threshold: 0.3,
  structured_text_bonus: 0.7,
};

function ConfigLabel({ children, tooltip }: { children: string; tooltip: string }) {
  return (
    <span className="config-label">
      <span>{children}</span>
      <span className="config-tooltip" tabIndex={0} aria-label={tooltip} data-tooltip={tooltip}>?</span>
    </span>
  );
}

function getChunkInfo(fileChunks: ChunkResponse[], lineIndex: number) {
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
}

type TextQualityDecision = {
  kind: 'keep' | 'drop';
  score: number;
  reason: string;
};

const positiveQualityReasons = new Set([
  'has_words',
  'enough_letters',
  'has_spacing',
  'reasonable_word_length',
  'structured_text',
]);

const negativeQualityReasons = new Set([
  'empty',
  'too_short',
  'many_control_chars',
  'repeated_char_noise',
  'numeric_array',
  'too_many_digits',
  'too_many_symbols',
  'dense_no_spaces',
  'long_dense_token',
  'hex_like',
  'base64_like',
  'high_entropy_dense_text',
]);

const wordRe = /\p{L}[\p{L}'-]{1,}/gu;
const longDenseTokenRe = /\S{80,}/;
const mostlyHexRe = /^[0-9a-fA-F\s:,-]{40,}$/;
const mostlyNumericRe = /^\s*[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?:[\s,;]+[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?){5,}\s*$/;
const base64ishRe = /^[A-Za-z0-9+/=_-]{80,}$/;
const repeatedCharRe = /(.)\1{20,}/;
const controlCharRe = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/gu;
const structuredTextRes = [
  /^\s*[-*\u00e2\u20ac\u00a2]\s+\S+/,
  /^\s*#{1,6}\s+\S+/,
  /^\s*[A-Za-z0-9_#.\-$ ]{2,80}\s*[:=]\s*\S+/,
  /^\s*"?[A-Za-z0-9_#.\-$ ]+"?\s*:\s*/,
];

function entropy(value: string) {
  const counts = new Map<string, number>();
  for (const char of value) counts.set(char, (counts.get(char) ?? 0) + 1);
  const length = Array.from(value).length || 1;
  let total = 0;
  for (const count of counts.values()) {
    const probability = count / length;
    total -= probability * Math.log2(probability);
  }
  return total;
}

function ratio(value: string, predicate: (char: string) => boolean) {
  const chars = Array.from(value);
  if (chars.length === 0) return 0;
  return chars.filter(predicate).length / chars.length;
}

function looksLikeStructuredText(value: string) {
  if (value.length > 300) return false;
  return structuredTextRes.some((pattern) => pattern.test(value));
}

function classifyTextLine(line: string, config: Required<TextQualityConfig>): TextQualityDecision {
  const value = line.trim();
  if (!value) return { kind: 'drop', score: 0, reason: 'empty' };
  if (value.length < 2) return { kind: 'drop', score: 0, reason: 'too_short' };

  const controlChars = value.match(controlCharRe)?.length ?? 0;
  if (controlChars / value.length > 0.02) return { kind: 'drop', score: 0, reason: 'many_control_chars' };
  if (repeatedCharRe.test(value)) return { kind: 'drop', score: 0.05, reason: 'repeated_char_noise' };
  if (mostlyNumericRe.test(value)) return { kind: 'drop', score: 0.05, reason: 'numeric_array' };

  const words = Array.from(value.matchAll(wordRe), (match) => match[0]);
  const wordLengths = words.map((word) => word.length);
  const avgWordLen = wordLengths.length ? wordLengths.reduce((sum, len) => sum + len, 0) / wordLengths.length : 0;
  const features = {
    alphaRatio: ratio(value, (char) => /\p{L}/u.test(char)),
    digitRatio: ratio(value, (char) => /\p{N}/u.test(char)),
    spaceRatio: ratio(value, (char) => /\s/u.test(char)),
    symbolRatio: ratio(value, (char) => !/[\p{L}\p{N}\s]/u.test(char)),
    wordCount: words.length,
    avgWordLen,
    entropy: entropy(value),
  };

  let score = 0;
  const reasons: string[] = [];

  if (features.wordCount >= 3) {
    score += 0.35;
    reasons.push('has_words');
  }
  if (features.alphaRatio >= 0.25) {
    score += 0.20;
    reasons.push('enough_letters');
  }
  if (features.spaceRatio >= 0.08) {
    score += 0.15;
    reasons.push('has_spacing');
  }
  if (features.avgWordLen >= 3) {
    score += 0.10;
    reasons.push('reasonable_word_length');
  }
  if (looksLikeStructuredText(value)) {
    score += config.structured_text_bonus;
    reasons.push('structured_text');
  }

  if (features.digitRatio >= config.digit_ratio_threshold) {
    score -= 0.25;
    reasons.push('too_many_digits');
  }
  if (features.symbolRatio >= config.symbol_ratio_threshold) {
    score -= 0.25;
    reasons.push('too_many_symbols');
  }
  if (features.spaceRatio < 0.02 && value.length > 60) {
    score -= 0.25;
    reasons.push('dense_no_spaces');
  }
  if (longDenseTokenRe.test(value)) {
    score -= 0.30;
    reasons.push('long_dense_token');
  }
  if (mostlyHexRe.test(value)) {
    score -= 0.35;
    reasons.push('hex_like');
  }
  if (base64ishRe.test(value) && features.spaceRatio === 0) {
    score -= 0.30;
    reasons.push('base64_like');
  }
  if (features.entropy >= 4.5 && features.spaceRatio < 0.05) {
    score -= 0.20;
    reasons.push('high_entropy_dense_text');
  }

  const boundedScore = Math.max(0, Math.min(1, score));
  return {
    kind: boundedScore >= config.keep_score_threshold ? 'keep' : 'drop',
    score: boundedScore,
    reason: reasons.join(', '),
  };
}

function QualityTooltip({ decision }: { decision: TextQualityDecision }) {
  const reasons = decision.reason ? decision.reason.split(', ') : [];
  return (
    <span className="quality-tooltip" role="tooltip">
      <span className={`quality-tooltip-kind ${decision.kind}`}>{decision.kind.toUpperCase()}</span>
      <span className="quality-tooltip-score">score {decision.score.toFixed(2)}</span>
      {reasons.length > 0 && (
        <span className="quality-tooltip-reasons">
          {reasons.map((reason) => {
            const tone = positiveQualityReasons.has(reason) ? 'positive' : negativeQualityReasons.has(reason) ? 'negative' : 'neutral';
            return <span key={reason} className={`quality-reason ${tone}`}>{reason}</span>;
          })}
        </span>
      )}
    </span>
  );
}

export function ChunkingDialog({ isOpen, packageId, dataPackage, chunksByFile, onClose, onSubmit }: ChunkingDialogProps) {
  const [bufferWindowSize, setBufferWindowSize] = useState(1);
  const [semanticThreshold, setSemanticThreshold] = useState(95);
  const [embeddingGpuMode, setEmbeddingGpuMode] = useState<'default' | 'auto' | 'cpu'>('default');

  const [draftTextQualityConfig, setDraftTextQualityConfig] = useState<Required<TextQualityConfig>>(defaultTextQualityConfig);
  const [textQualityConfig, setTextQualityConfig] = useState<Required<TextQualityConfig>>(defaultTextQualityConfig);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [protectedLines, setProtectedLines] = useState<Record<string, Set<number>>>({});
  const viewerRef = useRef<HTMLDivElement>(null);
  const mouseDownLine = useRef<number | null>(null);

  const selectedFile = useMemo(
    () => dataPackage?.files.find((f) => f.file_path === selectedFilePath) || null,
    [dataPackage, selectedFilePath]
  );

  const fileChunks = useMemo(
    () => chunksByFile.find((group) => group[0]?.file_path === selectedFilePath) ?? [],
    [chunksByFile, selectedFilePath]
  );

  const lines = useMemo(() => fileContent?.split('\n') ?? [], [fileContent]);

  const qualityDecisions = useMemo(
    () => lines.map((line) => classifyTextLine(line, textQualityConfig)),
    [lines, textQualityConfig]
  );

  const filteredLineCount = useMemo(
    () => qualityDecisions.filter((decision) => decision.kind === 'drop').length,
    [qualityDecisions]
  );

  const hasPendingTextQualityChanges = useMemo(
    () => (
      draftTextQualityConfig.symbol_ratio_threshold !== textQualityConfig.symbol_ratio_threshold ||
      draftTextQualityConfig.digit_ratio_threshold !== textQualityConfig.digit_ratio_threshold ||
      draftTextQualityConfig.keep_score_threshold !== textQualityConfig.keep_score_threshold ||
      draftTextQualityConfig.structured_text_bonus !== textQualityConfig.structured_text_bonus
    ),
    [draftTextQualityConfig, textQualityConfig]
  );

  useEffect(() => {
    if (!isOpen) return;
    setBufferWindowSize(1);
    setSemanticThreshold(95);
    setEmbeddingGpuMode('default');
    setDraftTextQualityConfig(defaultTextQualityConfig);
    setTextQualityConfig(defaultTextQualityConfig);
    setShowAdvanced(false);
    setSelectedFilePath(null);
    setFileContent(null);
    setProtectedLines({});
    mouseDownLine.current = null;
  }, [isOpen]);

  useEffect(() => {
    if (!selectedFilePath || !packageId) {
      setFileContent(null);
      return;
    }
    let cancelled = false;
    getFileEntryContent(packageId, selectedFilePath).then((res) => {
      if (!cancelled) setFileContent(res.content);
    });
    return () => { cancelled = true; };
  }, [selectedFilePath, packageId]);

  const setLineProtected = (filePath: string, lineIndex: number, add: boolean) => {
    setProtectedLines((prev) => {
      const next = { ...prev };
      const set = new Set(next[filePath] ?? []);
      if (add) set.add(lineIndex);
      else set.delete(lineIndex);
      next[filePath] = set;
      return next;
    });
  };

  const handleLineMouseDown = (lineIndex: number) => {
    mouseDownLine.current = lineIndex;
  };

  const handleViewerMouseUp = (filePath: string) => {
    const selection = window.getSelection();
    const downIdx = mouseDownLine.current;
    mouseDownLine.current = null;

    if (!selection || !viewerRef.current) return;

    if (selection.isCollapsed && downIdx !== null) {
      // Single click (no drag) — toggle that line
      const isProtected = protectedLines[filePath]?.has(downIdx) ?? false;
      setLineProtected(filePath, downIdx, !isProtected);
      return;
    }

    if (!selection.isCollapsed) {
      // Text was selected — protect every line touched by the selection
      const range = selection.getRangeAt(0);
      const lineEls = viewerRef.current.querySelectorAll('[data-line-index]');
      lineEls.forEach((el) => {
        if (range.intersectsNode(el)) {
          const idx = Number(el.getAttribute('data-line-index'));
          if (!Number.isNaN(idx)) setLineProtected(filePath, idx, true);
        }
      });
      selection.removeAllRanges();
    }
  };

  const updateDraftTextQualityConfig = (key: keyof TextQualityConfig, value: number) => {
    setDraftTextQualityConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleTextQualitySubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setTextQualityConfig(draftTextQualityConfig);
  };

  const handleSubmit = () => {
    const protected_line_indices: Record<string, number[]> = {};
    for (const [path, set] of Object.entries(protectedLines)) {
      const arr = Array.from(set).sort((a, b) => a - b);
      if (arr.length) protected_line_indices[path] = arr;
    }
    onSubmit({
      replace_existing_chunks: true,
      buffer_window_size: bufferWindowSize,
      semantic_chunking_threshold: semanticThreshold,
      protected_line_indices,
      text_quality_config: textQualityConfig,
      embedding_num_gpu: embeddingGpuMode === 'default' ? undefined : embeddingGpuMode === 'auto' ? -1 : 0,
    });
  };

  if (!isOpen) return null;

  const dialog = (
    <div className="chunking-dialog-overlay" onClick={onClose}>
      <div className="chunking-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="chunking-dialog-header">
          <h3>Configure chunking</h3>
          <button className="ghost" onClick={onClose}>Close</button>
        </div>

        <div className="chunking-dialog-body">
          <div className="chunking-form">
            <label className="form-row">
              <ConfigLabel tooltip={configTooltips.bufferWindowSize}>Buffer window size</ConfigLabel>
              <input type="number" min={0} max={10} value={bufferWindowSize} onChange={(e) => setBufferWindowSize(Number(e.target.value))} />
            </label>
            <label className="form-row">
              <ConfigLabel tooltip={configTooltips.semanticThreshold}>Semantic threshold (%)</ConfigLabel>
              <input type="number" min={0} max={100} value={semanticThreshold} onChange={(e) => setSemanticThreshold(Number(e.target.value))} />
            </label>
            <label className="form-row wide-control">
              <ConfigLabel tooltip={configTooltips.embeddingGpu}>Embedding GPU</ConfigLabel>
              <select value={embeddingGpuMode} onChange={(e) => setEmbeddingGpuMode(e.target.value as 'default' | 'auto' | 'cpu')}>
                <option value="default">Runtime default</option>
                <option value="auto">Auto GPU</option>
                <option value="cpu">CPU only</option>
              </select>
            </label>

            <button className="ghost small" onClick={() => setShowAdvanced((s) => !s)}>
              {showAdvanced ? 'Hide advanced text-quality settings' : 'Show advanced text-quality settings'}
            </button>

            {showAdvanced && (
              <form className="advanced-panel" onSubmit={handleTextQualitySubmit}>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.symbolThreshold}>Symbol ratio threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={draftTextQualityConfig.symbol_ratio_threshold} onChange={(e) => updateDraftTextQualityConfig('symbol_ratio_threshold', Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.digitThreshold}>Digit ratio threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={draftTextQualityConfig.digit_ratio_threshold} onChange={(e) => updateDraftTextQualityConfig('digit_ratio_threshold', Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.keepThreshold}>KEEP score threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={draftTextQualityConfig.keep_score_threshold} onChange={(e) => updateDraftTextQualityConfig('keep_score_threshold', Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.structuredBonus}>Structured-text bonus</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={draftTextQualityConfig.structured_text_bonus} onChange={(e) => updateDraftTextQualityConfig('structured_text_bonus', Number(e.target.value))} />
                </label>
                <button className="ghost small" type="submit" disabled={!hasPendingTextQualityChanges}>Apply text-quality settings</button>
              </form>
            )}
          </div>

          <div className="chunking-file-panel">
            <div className="file-picker">
              <span className="panel-label">
                <ConfigLabel tooltip={configTooltips.protectedLines}>Select a file to protect lines</ConfigLabel>
              </span>
              <select value={selectedFilePath ?? ''} onChange={(e) => setSelectedFilePath(e.target.value || null)}>
                <option value="">No file selected</option>
                {dataPackage?.files.map((f) => (
                  <option key={f.file_path} value={f.file_path}>{f.file_path}</option>
                ))}
              </select>
              {selectedFile && (
                <p className="muted">
                  {filteredLineCount} filtered out -{' '}
                  {lines.length} lines · {protectedLines[selectedFile.file_path]?.size ?? 0} protected
                </p>
              )}
            </div>

            {selectedFile && fileContent !== null && (
              <div
                ref={viewerRef}
                className="file-viewer-body dialog-viewer"
                onMouseUp={() => handleViewerMouseUp(selectedFile.file_path)}
              >
                {lines.map((line, index) => {
                  const info = getChunkInfo(fileChunks, index);
                  const isProtected = protectedLines[selectedFile.file_path]?.has(index) ?? false;
                  const decision = qualityDecisions[index];
                  const isFiltered = decision?.kind === 'drop';
                  const titleParts = [
                    info.inChunk ? `Chunk ${info.chunkIndex + 1}` : null,
                    isProtected ? 'Protected: will be kept despite filtering' : null,
                  ].filter(Boolean);
                  return (
                    <div
                      key={index}
                      data-line-index={index}
                      className={`file-viewer-line ${info.inChunk ? 'chunk-highlight' : ''} ${isFiltered ? 'line-quality-drop' : ''} ${isProtected ? 'line-protected' : ''}`}
                      style={info.inChunk ? { background: info.color } : undefined}
                      title={titleParts.join(' | ')}
                      onMouseDown={() => handleLineMouseDown(index)}
                    >
                      <span className="line-number">{index + 1}</span>
                      <span className="line-content">{line || ' '}</span>
                      {isFiltered && <span className="quality-badge">DROP</span>}
                      {decision && <QualityTooltip decision={decision} />}
                      {isProtected && <span className="protect-badge">★</span>}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="chunking-dialog-footer">
          <button onClick={handleSubmit}>Start chunking</button>
        </div>
      </div>
    </div>
  );

  return createPortal(dialog, document.body);
}
