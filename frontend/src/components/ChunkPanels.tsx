import { createPortal } from 'react-dom';
import type { ChunkingStrategy } from '../api/datasources';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryResponse } from '../api/types';
import { formatExtractionStage } from '../lib/format';

const chunkColors = [
  'rgba(99, 154, 0, 0.22)',
  'rgba(0, 120, 180, 0.18)',
  'rgba(180, 90, 0, 0.18)',
  'rgba(140, 60, 180, 0.18)',
  'rgba(200, 50, 80, 0.18)',
  'rgba(0, 160, 140, 0.18)',
];

const imageFileExtensions = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']);

export function FileViewer({ file, content, chunksByFile, onClose }: {
  file: FileEntryResponse;
  content: string;
  chunksByFile: ChunkResponse[][];
  onClose: () => void;
}) {
  const lines = content.split('\n');
  const fileChunks = chunksByFile.find((group) => group[0]?.file_path === file.file_path) ?? [];

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

  return createPortal((
    <div className="file-viewer-overlay" onClick={onClose}>
      <div className="file-viewer" onClick={(event) => event.stopPropagation()}>
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
                  Chunk {i + 1}: {chunkLineCount(chunk)} lines
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  ), document.body);
}

export function ChunkInspectionPanel({
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
    if (!dataPackage?.files.length) return null;
    return (
      <div className="file-list chunk-file-list">
        {dataPackage.files.map((file) => (
          <div
            className="file-row"
            key={file.file_path}
            onClick={() => onViewFile(file)}
            title="Click to view file content"
          >
            <span>{file.file_path}</span>
            <div className="file-meta">
              <small>{formatBytes(file.byte_size)}</small>
            </div>
          </div>
        ))}
      </div>
    );
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

export function ChunkingStatusPanel({
  dataPackage,
  chunkResult,
  chunksByFile,
  chunkViewStrategy,
  onChunkViewStrategyChange,
  busy,
}: {
  dataPackage?: DataPackageResponse | null;
  chunkResult?: ChunkRequestResponse | null;
  chunksByFile: ChunkResponse[][];
  chunkViewStrategy: ChunkingStrategy;
  onChunkViewStrategyChange: (strategy: ChunkingStrategy) => void;
  busy: string | null;
}) {
  const chunkGroups = chunksByFile.filter((group) => group.length > 0);
  const visibleChunks = chunkGroups.flat().length;
  const fileCount = dataPackage?.files.filter(isChunkableFile).length ?? 0;
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
      <label className="form-row chunk-view-strategy">
        <span>Chunking strategy</span>
        <select value={chunkViewStrategy} onChange={(event) => onChunkViewStrategyChange(event.target.value as ChunkingStrategy)}>
          <option value="semantic">Semantic</option>
          <option value="fixed_tokens">Fixed tokens</option>
        </select>
      </label>
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

function isChunkableFile(file: FileEntryResponse): boolean {
  return !imageFileExtensions.has(file.file_extension.toLowerCase());
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
