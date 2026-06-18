import { useState, type ReactNode } from 'react';
import type { ChunkingStrategy } from '../api/datasources';

export type WorkflowBranch = {
  strategy: ChunkingStrategy;
  chatModel: string | null;
};

export function WorkflowBranchBar({ branch }: { branch: WorkflowBranch }) {
  return (
    <div className="workflow-branch-bar">
      <div>
        <span>Current branch</span>
        <strong>{formatChunkingStrategy(branch.strategy)} / {branch.chatModel ?? 'Runtime default'}</strong>
      </div>
      <small>Artifacts, progress, and token usage</small>
    </div>
  );
}

export function StepPanel({
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

function formatChunkingStrategy(strategy: ChunkingStrategy): string {
  return strategy === 'fixed_tokens' ? 'Fixed tokens' : 'Semantic';
}
