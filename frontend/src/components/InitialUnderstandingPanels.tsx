import {
  type PointerEvent,
  type ReactNode,
  type WheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import type {
  ExtractionOverview,
  ExtractionOverviewNode,
  WorkflowProgress,
  WorkflowTaskStatus,
} from '../api/extraction';
import { formatExtractionStage, labelDy } from '../lib/format';
import { JsonDetails } from './JsonDetails';

type InitialOverviewGraphNode = ExtractionOverviewNode & {
  x: number;
  y: number;
};

const overviewGraphBaseViewBox = { x: 0, y: 0, width: 900, height: 440 };

function InitialOverviewGraphPanel({ overview }: { overview: ExtractionOverview }) {
  const graph = useMemo(() => buildInitialOverviewGraph(overview), [overview]);
  const [selectedId, setSelectedId] = useState(graph.nodes[0]?.node_id ?? '');
  const [viewBox, setViewBox] = useState(overviewGraphBaseViewBox);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragStartRef = useRef<{ clientX: number; clientY: number; viewBox: typeof overviewGraphBaseViewBox; moved: boolean; targetId: string | null } | null>(null);

  useEffect(() => {
    setSelectedId((current) => graph.nodes.some((node) => node.node_id === current) ? current : graph.nodes[0]?.node_id ?? '');
    setViewBox(overviewGraphBaseViewBox);
  }, [graph]);

  if (!graph.nodes.length) {
    return <p className="muted">No overview graph nodes are available for this run.</p>;
  }

  const selectedNode = graph.nodes.find((node) => node.node_id === selectedId) ?? graph.nodes[0];
  const selectedEdges = graph.edges.filter((edge) => edge.source === selectedNode.node_id || edge.target === selectedNode.node_id);

  function zoomAt(clientX: number, clientY: number, scale: number) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    setViewBox((current) => {
      const nextWidth = Math.min(overviewGraphBaseViewBox.width / 0.7, Math.max(overviewGraphBaseViewBox.width / 2.2, current.width / scale));
      const nextHeight = Math.min(overviewGraphBaseViewBox.height / 0.7, Math.max(overviewGraphBaseViewBox.height / 2.2, current.height / scale));
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
    const target = event.target instanceof Element ? event.target.closest<SVGGElement>('.overview-graph-node') : null;
    dragStartRef.current = { clientX: event.clientX, clientY: event.clientY, viewBox, moved: false, targetId: target?.dataset.nodeId ?? null };
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
    if (drag && !drag.moved && drag.targetId) {
      setSelectedId(drag.targetId);
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
    <div className="overview-graph-panel">
      <div className={`overview-graph-canvas ${dragging ? 'dragging' : ''}`} role="img" aria-label="Initial overview file graph" onWheel={onGraphWheel}>
        <svg
          ref={svgRef}
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          onPointerDown={onGraphPointerDown}
          onPointerMove={onGraphPointerMove}
          onPointerUp={onGraphPointerUp}
          onPointerCancel={onGraphPointerUp}
        >
          <defs>
            <marker id="overview-graph-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" />
            </marker>
          </defs>
          {graph.edges.map((edge) => {
            const source = graph.nodeMap.get(edge.source);
            const target = graph.nodeMap.get(edge.target);
            if (!source || !target) return null;
            const edgePath = overviewEdgePath(source, target);
            const midX = (edgePath.x1 + edgePath.x2) / 2;
            const midY = (edgePath.y1 + edgePath.y2) / 2;
            const showLabel = edge.source === selectedNode.node_id || edge.target === selectedNode.node_id;
            return (
              <g key={edge.edge_id || `${edge.source}-${edge.relation}-${edge.target}`} className="overview-graph-edge">
                <line x1={edgePath.x1} y1={edgePath.y1} x2={edgePath.x2} y2={edgePath.y2} />
                {showLabel && <text x={midX} y={midY}>{formatExtractionStage(edge.relation)}</text>}
              </g>
            );
          })}
          {graph.nodes.map((node) => {
            const selected = node.node_id === selectedNode.node_id;
            const radius = overviewGraphNodeRadius(node);
            const labelLines = overviewNodeLabelLines(node, radius);
            return (
              <g
                key={node.node_id}
                className={`overview-graph-node ${node.kind} ${selected ? 'selected' : ''}`}
                data-node-id={node.node_id}
                transform={`translate(${node.x} ${node.y})`}
                onClick={() => setSelectedId(node.node_id)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    setSelectedId(node.node_id);
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
        <div className="overview-graph-controls" aria-label="Graph controls">
          <button className="ghost small" type="button" onClick={() => zoomAtCenter(1.18)}>+</button>
          <button className="ghost small" type="button" onClick={() => zoomAtCenter(1 / 1.18)}>-</button>
          <button className="ghost small" type="button" onClick={() => setViewBox(overviewGraphBaseViewBox)}>Fit</button>
        </div>
      </div>
      <aside className="overview-graph-inspector">
        <div className="overview-graph-inspector-heading">
          <span>{formatExtractionStage(selectedNode.kind)}</span>
          <strong>{selectedNode.label || selectedNode.node_id}</strong>
        </div>
        <dl>
          <div>
            <dt>node</dt>
            <dd>{selectedNode.node_id}</dd>
          </div>
          {selectedNode.file_path && (
            <div>
              <dt>file</dt>
              <dd>{selectedNode.file_path}</dd>
            </div>
          )}
          {selectedNode.rank && (
            <div>
              <dt>rank</dt>
              <dd>{selectedNode.rank}</dd>
            </div>
          )}
          {selectedNode.summary && (
            <div>
              <dt>summary</dt>
              <dd>{selectedNode.summary}</dd>
            </div>
          )}
          {selectedEdges.slice(0, 8).map((edge) => (
            <div key={edge.edge_id || `${edge.source}-${edge.relation}-${edge.target}`}>
              <dt>{formatExtractionStage(edge.relation)}</dt>
              <dd>{edge.source === selectedNode.node_id ? edge.target : edge.source}{edge.note ? ` - ${edge.note}` : ''}</dd>
            </div>
          ))}
        </dl>
        {(overview.uncertainties ?? []).length > 0 && (
          <div className="overview-graph-uncertainties">
            <span>Uncertainties</span>
            {(overview.uncertainties ?? []).slice(0, 5).map((uncertainty, index) => (
              <p key={`${uncertainty}-${index}`}>{uncertainty}</p>
            ))}
          </div>
        )}
      </aside>
    </div>
  );
}

function buildInitialOverviewGraph(overview: ExtractionOverview) {
  const overviewNodes = overview.nodes ?? [];
  const nodes = overviewNodes.map((node, index) => {
    const angle = ((Math.PI * 2) / Math.max(1, overviewNodes.length)) * index - Math.PI / 2;
    const radius = node.kind === 'package' ? 0 : 116 + (index % 4) * 32;
    return {
      ...node,
      x: 450 + Math.cos(angle) * radius,
      y: 220 + Math.sin(angle) * radius,
    };
  });
  const nodeIds = new Set(nodes.map((node) => node.node_id));
  const edges = (overview.edges ?? []).filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  const nodeMap = new Map(nodes.map((node) => [node.node_id, node]));
  return { nodes, edges, nodeMap };
}

function overviewGraphNodeRadius(node: InitialOverviewGraphNode): number {
  if (node.kind === 'package') return 25;
  if (node.kind === 'group' || node.kind === 'directory') return 21;
  return 17;
}

function overviewEdgePath(source: InitialOverviewGraphNode, target: InitialOverviewGraphNode) {
  const deltaX = target.x - source.x;
  const deltaY = target.y - source.y;
  const length = Math.hypot(deltaX, deltaY) || 1;
  const unitX = deltaX / length;
  const unitY = deltaY / length;
  const sourceRadius = overviewGraphNodeRadius(source) + 2;
  const targetRadius = overviewGraphNodeRadius(target) + 7;
  return {
    x1: source.x + unitX * sourceRadius,
    y1: source.y + unitY * sourceRadius,
    x2: target.x - unitX * targetRadius,
    y2: target.y - unitY * targetRadius,
  };
}

function overviewNodeLabelLines(node: InitialOverviewGraphNode, radius: number): string[] {
  const label = node.label || node.file_path || node.node_id;
  const words = label.replace(/^file:/, '').split(/[\/\s_-]+/).filter(Boolean);
  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length > Math.max(8, radius * 0.65) && current) {
      lines.push(current);
      current = word;
    } else {
      current = next;
    }
    if (lines.length >= 2) break;
  }
  if (current && lines.length < 3) lines.push(current);
  return lines.length ? lines.slice(0, 3) : [node.kind];
}

export function InitialFileUnderstandingPanel({
  progress,
  status,
  tokenUsageSummary,
}: {
  progress?: WorkflowProgress | null;
  status?: WorkflowTaskStatus | null;
  tokenUsageSummary?: ReactNode;
}) {
  const rankedCount = progress?.ranked_files?.length ?? 0;
  const summaryCount = progress?.initial_file_summaries?.length ?? 0;
  const summaryProgress = progress?.initial_file_summary_progress ?? null;
  const summaryProgressTotal = summaryProgress?.total_files ?? summaryCount;
  const summaryProgressProcessed = summaryProgress?.processed_files ?? summaryCount;
  const summaryProgressPercent = summaryProgressTotal
    ? Math.min(100, Math.round((summaryProgressProcessed / summaryProgressTotal) * 100))
    : 0;
  const hasArtifacts = Boolean(
    rankedCount
    || summaryCount
    || summaryProgress
    || progress?.initial_file_summary_status
    || progress?.initial_extraction_overview_status
    || progress?.initial_extraction_overview
    || progress?.initial_extraction_overview_diagnostic
    || progress?.dataset_summary,
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
          <strong>
            {summaryProgressTotal
              ? `${summaryProgressProcessed}/${summaryProgressTotal} files`
              : `${summaryCount} files`}
          </strong>
        </div>
      </div>

      {summaryProgress ? (
        <div className="patch-progress summary-progress">
          <div className="patch-progress-header">
            <span>File summary progress</span>
            <strong>{summaryProgressProcessed}/{summaryProgressTotal || 0}</strong>
          </div>
          <div className="patch-progress-track" aria-hidden="true"><div style={{ width: `${summaryProgressPercent}%` }} /></div>
          <div className="patch-progress-summary">
            <span>{summaryProgress.summarized_files} summarized</span>
            <span>{summaryProgress.skipped_files} skipped</span>
            <span>{summaryProgress.failed_files} failed</span>
          </div>
          {summaryProgress.current_file_path ? (
            <p className="muted">Current file: <code>{summaryProgress.current_file_path}</code></p>
          ) : null}
        </div>
      ) : null}

      {progress?.dataset_summary ? (
        <details className="initial-overview-panel" open>
          <summary>
            <div>
              <span>Dataset summary</span>
              <strong>Initial overview artifact</strong>
            </div>
          </summary>
          <div className="summary-box">{progress.dataset_summary}</div>
        </details>
      ) : null}

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
              <span>File summaries</span>
              <strong>{formatExtractionStage(progress?.initial_file_summary_status || 'not available')}</strong>
            </div>
          </summary>
          {(progress?.initial_file_summaries?.length ?? 0) > 0 ? (
            <JsonDetails title="Per-file extraction guidance" value={progress?.initial_file_summaries ?? []} />
          ) : (
            <p className="muted">No file summaries are available for this run.</p>
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
            <>
              <InitialOverviewGraphPanel overview={progress.initial_extraction_overview} />
              <JsonDetails title="Overview graph JSON" value={progress.initial_extraction_overview} />
            </>
          ) : (
            <p className="muted">No overview guidance is available for this run.</p>
          )}
          {progress?.initial_extraction_overview_diagnostic ? (
            <JsonDetails
              title="Overview prompt diagnostic"
              value={progress.initial_extraction_overview_diagnostic}
            />
          ) : null}
        </details>
      ) : null}

      {tokenUsageSummary}
    </div>
  );
}

