import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { getFileEntryContent } from '../api/datasources';
import type { ChunkResponse, DataPackageResponse, FileEntryResponse, TextQualityConfig } from '../api/types';

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
  symbolThreshold: 'Symbol-heavy lines receive a quality penalty when their symbol ratio is at or above this value. Lower values are stricter and drop more notation-heavy lines; higher values keep more lines with punctuation, formulas, or metadata keys.',
  digitThreshold: 'Digit-heavy lines receive a quality penalty when their digit ratio is at or above this value. Lower values drop more numeric lines; higher values keep more measurements, identifiers, and tables.',
  keepThreshold: 'Minimum quality score for a line to be included in chunking. Lower values keep more borderline lines and noise; higher values keep fewer, cleaner lines but may remove useful evidence.',
  structuredBonus: 'Score bonus for short structured text such as key/value metadata, headings, bullets, or JSON/YAML-like lines. Higher values keep more structured metadata lines; lower values make them easier to filter out.',
  protectedLines: 'Selected lines are forced into the chunk input even if the text-quality filter would normally drop them. Use this for headers, identifiers, or metadata lines that look noisy but are semantically important.',
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

export function ChunkingDialog({ isOpen, packageId, dataPackage, chunksByFile, onClose, onSubmit }: ChunkingDialogProps) {
  const [bufferWindowSize, setBufferWindowSize] = useState(1);
  const [semanticThreshold, setSemanticThreshold] = useState(95);

  const [symbolThreshold, setSymbolThreshold] = useState(0.45);
  const [digitThreshold, setDigitThreshold] = useState(0.45);
  const [keepThreshold, setKeepThreshold] = useState(0.3);
  const [structuredBonus, setStructuredBonus] = useState(0.7);
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

  useEffect(() => {
    if (!isOpen) return;
    setBufferWindowSize(1);
    setSemanticThreshold(95);
    setSymbolThreshold(0.45);
    setDigitThreshold(0.45);
    setKeepThreshold(0.3);
    setStructuredBonus(0.7);
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
      text_quality_config: {
        symbol_ratio_threshold: symbolThreshold,
        digit_ratio_threshold: digitThreshold,
        keep_score_threshold: keepThreshold,
        structured_text_bonus: structuredBonus,
      },
    });
  };

  if (!isOpen) return null;

  const lines = fileContent?.split('\n') ?? [];

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

            <button className="ghost small" onClick={() => setShowAdvanced((s) => !s)}>
              {showAdvanced ? 'Hide advanced text-quality settings' : 'Show advanced text-quality settings'}
            </button>

            {showAdvanced && (
              <div className="advanced-panel">
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.symbolThreshold}>Symbol ratio threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={symbolThreshold} onChange={(e) => setSymbolThreshold(Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.digitThreshold}>Digit ratio threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={digitThreshold} onChange={(e) => setDigitThreshold(Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.keepThreshold}>KEEP score threshold</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={keepThreshold} onChange={(e) => setKeepThreshold(Number(e.target.value))} />
                </label>
                <label className="form-row">
                  <ConfigLabel tooltip={configTooltips.structuredBonus}>Structured-text bonus</ConfigLabel>
                  <input type="number" min={0} max={1} step={0.05} value={structuredBonus} onChange={(e) => setStructuredBonus(Number(e.target.value))} />
                </label>
              </div>
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
                  return (
                    <div
                      key={index}
                      data-line-index={index}
                      className={`file-viewer-line ${info.inChunk ? 'chunk-highlight' : ''} ${isProtected ? 'line-protected' : ''}`}
                      style={info.inChunk ? { background: info.color } : undefined}
                      title={info.inChunk ? `Chunk ${info.chunkIndex + 1}` : undefined}
                      onMouseDown={() => handleLineMouseDown(index)}
                    >
                      <span className="line-number">{index + 1}</span>
                      <span className="line-content">{line || ' '}</span>
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
