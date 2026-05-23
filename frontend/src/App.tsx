import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { chunkDataPackage, deleteDataPackage, getChunkStatus, getDataPackageChunks, getFileEntryContent, listDataPackages, uploadDataPackage } from './api/datasources';
import {
  extractInitialContext,
  extractInitialDraft,
  getExistingInitialContext,
  getExistingInitialDraft,
  getPatchArtifacts,
  getPatchProgress,
  getPatchReviewState,
  getProtectedFields,
  getTokenUsage,
  patchDraft,
  saveDraft,
  savePatchReviewState,
  setProtectedFields as apiSetProtectedFields,
} from './api/extraction';
import { deleteProfile, getProfileJsonSchema, listProfiles, registerProfile } from './api/profiles';
import { getLlmBudget, type LlmBudget } from './api/system';
import { JsonEditor, type JsonObject, type JsonPatchMarker, type JsonSchemaDocument, type JsonValue, setValueAtPath } from './components/JsonEditor';
import { ChunkingDialog } from './components/ChunkingDialog';
import { VocabularyPanel } from './components/VocabularyPanel';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryResponse, InitialContext, ProfileManifestResponse, TextQualityConfig } from './api/types';
import type { PatchArtifacts, PatchProgress, PatchReviewState, PatchTaskStatus, PatchTokenUsage, PatchTokenUsageEntry } from './api/extraction';

type BusyKey = 'upload' | 'chunk' | 'context' | 'draft' | 'patch' | 'load' | 'profile' | 'profile-delete' | 'dataset-delete';
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

function Field({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="field">
      <span>{label}</span>
      <strong>{value || '-'}</strong>
    </div>
  );
}

function contextTechnique(context: InitialContext): string | null | undefined {
  return context.activities?.find((activity) => activity.technique)?.technique;
}

function contextAgentLabel(context: InitialContext): string | null | undefined {
  if (context.agents?.length) {
    return context.agents
      .map((agent) => agent.model ? `${agent.name} (${agent.model})` : agent.name)
      .join(', ');
  }
  return null;
}

function contextEntityLabel(context: InitialContext): string | null {
  const labels = context.entities?.length
    ? context.entities.map((entity) => entity.identifier || entity.label)
    : [];
  return labels.length ? labels.join(', ') : null;
}

function contextActivityLabel(context: InitialContext): string | null {
  const labels = context.activities
    ?.map((activity) => activity.label || activity.technique)
    .filter(Boolean) as string[] | undefined;
  return labels?.length ? labels.join(', ') : null;
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
  initial_context: 'Initial context',
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
      const order = ['initial_context', 'patch_discovery', 'schema_patch_writer', 'schema_repair', 'patch_extraction', 'patch_quality', 'auto_resolve'];
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
            <span>{formatTokenCount(usageAverage(row.usage, averageUnit, 'total'))} avg total / {averageUnit === 'patch' ? 'patch' : 'run'}</span>
            <small>
              {formatTokenCount(requestAverage(row.usage, 'input'))} input / {formatTokenCount(requestAverage(row.usage, 'output'))} output avg per model call
              {' - '}
              {formatTokenCount(requestAverage(row.usage, 'total'))} total avg per model call
              {' - '}
              {formatTokenCount(row.usage.requests)} model call{row.usage.requests === 1 ? '' : 's'}
            </small>
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
  const [activeProfileSchema, setActiveProfileSchema] = useState<JsonSchemaDocument | null>(null);
  const [patchReviewState, setPatchReviewState] = useState<PatchReviewState>(emptyReviewState);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(true);
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
  const reviewItems = useMemo(() => buildReviewItems(patchArtifacts, patchReviewState), [patchArtifacts, patchReviewState]);
  const unresolvedReviewItems = reviewItems.filter((item) => !item.resolved);
  const patchMarkers = unresolvedReviewItems;
  const reviewMarkers = unresolvedReviewItems.filter((marker) => marker.status === 'needs_review' || marker.status === 'unmapped');
  const autoResolutionActive = patchProgress?.resolution_active === true || (isPatching && autoResolve);
  const manualReviewActionsDisabled = autoResolutionActive;

  async function refresh() {
    setBusy('load');
    try {
      const [nextPackages, nextProfiles, nextBudget] = await Promise.all([listDataPackages(), listProfiles(), getLlmBudget()]);
      setLlmBudget(nextBudget);
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

  function resetPackageWorkflowState() {
    setChunkResult(null);
    setHasChunks(false);
    setChunksByFile([]);
    setViewingFile(null);
    setFileContent(null);
    setContext(null);
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

  async function onContext() {
    if (!selectedPackageId) return;
    setBusy('context');
    try {
      const result = await extractInitialContext({ data_package_id: selectedPackageId });
      setContext(result);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage('Initial context extracted.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Context extraction failed.');
    } finally {
      setBusy(null);
    }
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
    try {
      const [{ status, progress }, artifacts, reviewState] = await Promise.all([
        getPatchProgress(selectedPackageId),
        getPatchArtifacts(selectedPackageId),
        getPatchReviewState(selectedPackageId),
      ]);
      setPatchStatus(status);
      setPatchProgress(progress || null);
      setPatchArtifacts(artifacts);
      setPatchReviewState(reviewState);
      setTokenUsage(await getTokenUsage(selectedPackageId));
      setMessage(hasPatchArtifacts(artifacts) ? 'Loaded existing patch artifacts.' : 'No existing patch artifacts found.');
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
        if (ctx) setContext(ctx);
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
    const interval = setInterval(() => void onShowPatchArtifacts(), 30000);
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
              <VocabularyPanel onError={setMessage} />
            </>
          )}
        </aside>

        <section className="workflow">
          <article className="step-card active">
            <div className="step-index">01</div>
            <div className="step-body">
              <h2>Upload dataset and create chunks</h2>
              <p>The archive is stored as a data package. Chunking prepares the package for later patch and enrichment stages.</p>
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
            </div>
          </article>

          <article className="step-card">
            <div className="step-index">02</div>
            <div className="step-body">
              <h2>Determine initial context</h2>
              <p>Extract high-level context, likely metadata sources, keywords, file relationships, and evidence from the package.</p>
              <div className="actions">
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !!busy}>{busy === 'context' ? 'Extracting...' : context ? 'Re-extract and remove old context' : 'Extract new context'}</button>
              </div>
              {context && (
                <div className="context-grid">
                  <Field label="Dataset" value={context.dataset_title} />
                  <Field label="Technique" value={contextTechnique(context)} />
                  <Field label="Agents" value={contextAgentLabel(context)} />
                  <Field label="Entities" value={contextEntityLabel(context)} />
                  <Field label="Activities" value={contextActivityLabel(context)} />
                  <Field label="Model" value={context.agents?.find((agent) => agent.model)?.model} />
                  <div className="summary-box">{context.dataset_description || context.summary}</div>
                  <div className="chips">{context.keywords.map((keyword) => <span key={keyword}>{keyword}</span>)}</div>
                </div>
              )}
              <TokenUsageSummary tokenUsage={tokenUsage} averageUnit="operation" heading="Extraction token usage" agentKeys={['initial_context']} budget={llmBudget} />
            </div>
          </article>

          <article className="step-card">
            <div className="step-index">03</div>
            <div className="step-body">
              <div className="draft-workspace-heading">
                <h2>Draft workspace</h2>
                {draft && (
                  <button className="ghost draft-refresh-button" onClick={() => void onShowPatchArtifacts()} disabled={!selectedPackageId || busy === 'load'}>Refresh</button>
                )}
              </div>
              <p>Create the initial profile draft, edit and lock fields, then patch the draft with chunk evidence while reviewing issues as they appear.</p>
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
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
