import { useEffect, useMemo, useRef, useState } from 'react';
import { chunkDataPackage, getChunkStatus, listDataPackages, uploadDataPackage } from './api/datasources';
import {
  extractInitialContext,
  extractInitialDraft,
  getExistingInitialContext,
  getExistingInitialDraft,
  getPatchArtifacts,
  getPatchProgress,
  getPatchReviewState,
  getProtectedFields,
  patchDraft,
  resolvePatchReview,
  saveDraft,
  savePatchReviewState,
  setProtectedFields as apiSetProtectedFields,
} from './api/extraction';
import { listProfiles } from './api/profiles';
import { JsonEditor, type JsonObject, type JsonPatchMarker, type JsonValue, setValueAtPath, getValueAtPath, extractPatchInnerValue, PatchValueEditor } from './components/JsonEditor';
import type { ChunkRequestResponse, DataPackageResponse, InitialContext, ProfileManifestResponse } from './api/types';
import type { PatchArtifacts, PatchProgress, PatchReviewResolutionItem, PatchReviewState, PatchTaskStatus } from './api/extraction';

type BusyKey = 'upload' | 'chunk' | 'context' | 'draft' | 'patch' | 'resolve' | 'load';
type ReviewTab = 'matched' | 'unmapped' | 'resolved';
type ReviewItem = JsonPatchMarker & { kind: 'matched' | 'unmapped'; targetPath?: string; fact?: string; reason?: string; outcome?: string; resolutionNote?: string };

const TERMINAL_PATCH_STATUSES = new Set<PatchTaskStatus>(['unknown', 'completed', 'cancelled', 'crashed']);
const emptyReviewState: PatchReviewState = {
  resolved_item_ids: [],
  unmapped_assignments: {},
  resolution_notes: {},
  resolved_at: {},
};

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

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(asRecord).filter((item): item is Record<string, unknown> => Boolean(item)) : [];
}

function topLevelFields(draft: object | null): string[] {
  return draft && typeof draft === 'object' ? Object.keys(draft).sort() : [];
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

function reviewOutcomeLabel(outcome?: string): string {
  if (outcome === 'included') return 'Included';
  if (outcome === 'already_present') return 'Already present';
  if (outcome === 'excluded') return 'Excluded';
  if (outcome === 'unresolved') return 'Unresolved';
  return 'Resolved';
}

function reviewActionLabel(outcome?: string): string {
  if (outcome === 'included') return 'Accepted into draft';
  if (outcome === 'already_present') return 'Already present in draft';
  if (outcome === 'excluded') return 'Kept out of draft';
  if (outcome === 'unresolved') return 'Still needs review';
  return 'Marked resolved';
}

function resolutionTargetPath(item: ReviewItem): string {
  const noteTarget = item.resolutionNote?.match(/Target: ([^.]+(?:\.[^.]+)*)\./)?.[1];
  if (noteTarget) return noteTarget;
  if (item.targetPath) return item.targetPath;
  return item.path === 'Unassigned' ? '' : item.path;
}

function formatDraftValue(value: unknown): string {
  if (value === undefined) return 'No value found at this path.';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function resolutionSummaryMessage(resolvedCount: number, unresolvedCount: number, decisions: { outcome: string }[]): string {
  const included = decisions.filter((decision) => decision.outcome === 'included' || decision.outcome === 'already_present').length;
  const excluded = decisions.filter((decision) => decision.outcome === 'excluded').length;
  if (decisions.length > 0) {
    const parts = [
      `Included ${included}`,
      `excluded ${excluded}`,
      `${unresolvedCount} still need review`,
    ];
    return parts.join(', ') + '.';
  }
  if (unresolvedCount > 0) {
    return `Review agent resolved ${resolvedCount} item${resolvedCount === 1 ? '' : 's'}; ${unresolvedCount} still need manual review.`;
  }
  return `Review agent resolved ${resolvedCount} item${resolvedCount === 1 ? '' : 's'}.`;
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
        resolved: resolvedIds.has(id),
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

function toResolutionItem(item: ReviewItem): PatchReviewResolutionItem {
  const resolutionItem: PatchReviewResolutionItem = {
    id: item.id,
    kind: item.kind,
    path: item.targetPath ?? item.path,
    detail: item.detail,
    issues: item.issues || [],
    evidence: item.evidence || [],
    fact: item.fact,
    reason: item.reason,
    confidence: item.confidence,
    file_name: item.fileName,
  };
  if (item.patch && typeof item.patch === 'object' && !Array.isArray(item.patch)) {
    resolutionItem.patch = item.patch as Record<string, unknown>;
  }
  return resolutionItem;
}

function ReviewItemList({ items, draft, onResolve, onApplyPatch }: { items: ReviewItem[]; draft?: object | null; onResolve: (itemId: string) => void; onApplyPatch?: (itemId: string, value: unknown) => void }) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editedValue, setEditedValue] = useState<unknown>(null);

  if (!items.length) return <p className="muted">No review items in this category.</p>;

  const startEditing = (item: ReviewItem) => {
    const innerValue = extractPatchInnerValue(item.patch);
    setEditingId(item.id);
    setEditedValue(innerValue);
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditedValue(null);
  };

  return (
    <ul className="artifact-list">
      {items.map((item) => {
        const targetPath = resolutionTargetPath(item);
        const draftValue = draft && targetPath ? getValueAtPath(draft as JsonObject, targetPath) : undefined;
        const shouldShowDraftValue = item.resolved && item.outcome !== 'excluded' && Boolean(targetPath);
        return (
        <li key={item.id} className="review-item">
          <div className="review-item-heading">
            <strong>{item.path}</strong>
            <span className={`patch-marker-badge ${item.status}`}>{item.resolved ? reviewOutcomeLabel(item.outcome) : item.label}</span>
          </div>
          {item.detail && <p>{item.detail}</p>}
          {item.resolved && (
            <div className="review-resolution">
              <div>
                <span>Action</span>
                <strong>{reviewActionLabel(item.outcome)}</strong>
              </div>
              {targetPath && (
                <div>
                  <span>Draft location</span>
                  <code>{targetPath}</code>
                </div>
              )}
              {item.resolutionNote && (
                <div>
                  <span>Reason</span>
                  <p>{item.resolutionNote.replace(/^(included|already_present|excluded|unresolved):\s*/, '')}</p>
                </div>
              )}
              {shouldShowDraftValue && (
                <div>
                  <span>Current draft value</span>
                  <pre className="review-item-value">{formatDraftValue(draftValue)}</pre>
                </div>
              )}
            </div>
          )}
          {item.issues && item.issues.length > 0 && (
            <div className="review-item-section">
              <span>Issues</span>
              <ul>{item.issues.map((issue, index) => <li key={`${item.id}-issue-${index}`}>{issue}</li>)}</ul>
            </div>
          )}
          {item.evidence && item.evidence.length > 0 && (
            <div className="review-item-section">
              <span>Evidence</span>
              <ul>{item.evidence.map((evidence, index) => <li key={`${item.id}-evidence-${index}`}>{evidence}</li>)}</ul>
            </div>
          )}
          {item.patch !== undefined && (
            <div className="review-item-section">
              <span>Proposed change</span>
              {editingId === item.id ? (
                <>
                  <PatchValueEditor value={editedValue} onChange={setEditedValue} />
                  <div className="patch-edit-actions">
                    <button className="ghost" onClick={cancelEditing}>Cancel</button>
                    <button onClick={() => { onApplyPatch?.(item.id, editedValue); cancelEditing(); }}>Apply patch</button>
                  </div>
                </>
              ) : (
                <>
                  <pre className="review-item-patch">{JSON.stringify(item.patch, null, 2)}</pre>
                  {!item.resolved && onApplyPatch && (
                    <button className="ghost" onClick={() => startEditing(item)}>Edit patch</button>
                  )}
                </>
              )}
            </div>
          )}
          {!item.resolved && <button className="ghost" onClick={() => onResolve(item.id)}>Mark resolved</button>}
        </li>
        );
      })}
    </ul>
  );
}

function UnmappedFactList({ facts, fields, reviewState, onAssign }: {
  facts: Record<string, unknown>[];
  fields: string[];
  reviewState: PatchReviewState;
  onAssign: (key: string, field: string) => void;
}) {
  if (!facts.length) return <p className="muted">No unmapped facts yet.</p>;
  return (
    <ul className="artifact-list">
      {facts.map((fact, index) => {
        const key = unmappedReviewItemId(fact);
        return (
          <li key={`${key}-${index}`} className="unmapped-fact">
            <strong>{String(fact.fact || 'Unmapped fact')}</strong>
            <p>{String(fact.reason || 'No mapping reason provided.')}</p>
            {Boolean(fact.source_hint) && <small>{String(fact.source_hint)}</small>}
            <select value={reviewState.unmapped_assignments[key] || ''} onChange={(event) => onAssign(key, event.target.value)}>
              <option value="">Select matching field</option>
              {fields.map((field) => <option key={field} value={field}>{field}</option>)}
            </select>
          </li>
        );
      })}
    </ul>
  );
}

export function App() {
  const [packages, setPackages] = useState<DataPackageResponse[]>([]);
  const [profiles, setProfiles] = useState<ProfileManifestResponse[]>([]);
  const [selectedPackageId, setSelectedPackageId] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [chunkResult, setChunkResult] = useState<ChunkRequestResponse | null>(null);
  const [hasChunks, setHasChunks] = useState(false);
  const [context, setContext] = useState<InitialContext | null>(null);
  const [draft, setDraft] = useState<object | null>(null);
  const [protectedFields, setProtectedFields] = useState<string[]>([]);
  const [patchStatus, setPatchStatus] = useState<PatchTaskStatus | null>(null);
  const [patchProgress, setPatchProgress] = useState<PatchProgress | null>(null);
  const [patchArtifacts, setPatchArtifacts] = useState<PatchArtifacts | null>(null);
  const [reviewTab, setReviewTab] = useState<ReviewTab>('matched');
  const [patchReviewState, setPatchReviewState] = useState<PatchReviewState>(emptyReviewState);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(false);
  const saveDraftTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const selectedPackage = useMemo(() => packages.find((item) => item.id === selectedPackageId) || null, [packages, selectedPackageId]);
  const isPatching = patchStatus === 'running';
  const hasVisiblePatchArtifacts = Boolean(patchArtifacts && hasPatchArtifacts(patchArtifacts));
  const patchButtonLabel = isPatching ? 'Patching...' : hasVisiblePatchArtifacts ? 'Resume patching from checkpoint' : 'Start new patching';
  const progressBatchNo = patchProgress?.batch_no ?? 0;
  const progressTotalBatches = patchProgress?.total_batches ?? 0;
  const progressPercent = progressTotalBatches > 0 ? Math.min(100, Math.round((progressBatchNo / progressTotalBatches) * 100)) : 0;
  const draftFields = useMemo(() => topLevelFields(draft), [draft]);
  const reviewItems = useMemo(() => buildReviewItems(patchArtifacts, patchReviewState), [patchArtifacts, patchReviewState]);
  const unresolvedReviewItems = reviewItems.filter((item) => !item.resolved);
  const patchMarkers = unresolvedReviewItems;
  const reviewMarkers = unresolvedReviewItems.filter((marker) => marker.status === 'needs_review' || marker.status === 'unmapped');
  const matchedReviewItems = reviewItems.filter((item) => item.kind === 'matched' && !item.resolved);
  const unmappedReviewFacts = patchArtifacts?.unmapped_facts ?? [];
  const resolvedReviewItems = reviewItems.filter((item) => item.resolved);

  async function refresh() {
    setBusy('load');
    try {
      const [nextPackages, nextProfiles] = await Promise.all([listDataPackages(), listProfiles()]);
      setPackages(nextPackages);
      setProfiles(nextProfiles);
      setSelectedPackageId((current) => current || nextPackages[0]?.id || '');
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

  async function onUpload(file?: File) {
    if (!file) return;
    setBusy('upload');
    try {
      const uploaded = await uploadDataPackage(file);
      const nextPackages = await listDataPackages();
      setPackages(nextPackages);
      setSelectedPackageId(uploaded.id);
      setChunkResult(null);
      setContext(null);
      setDraft(null);
      setPatchArtifacts(null);
      setPatchReviewState(emptyReviewState);
      setMessage('Dataset uploaded. Create chunks next.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Upload failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onChunk(replace = false) {
    if (!selectedPackageId) return;
    setBusy('chunk');
    try {
      const result = await chunkDataPackage({ id: selectedPackageId, replace_existing_chunks: replace });
      setChunkResult(result);
      setMessage(result.status === 'completed' ? 'Chunks are ready.' : 'Chunking is running. Run this step again to refresh status.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Chunking failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onContext() {
    if (!selectedPackageId) return;
    setBusy('context');
    try {
      const result = await extractInitialContext({ data_package_id: selectedPackageId });
      setContext(result);
      setMessage('Initial context extracted.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Context extraction failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onDraft() {
    if (!selectedPackageId || !selectedProfile) return;
    setBusy('draft');
    try {
      const result = await extractInitialDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setDraft(result);
      setPatchArtifacts(null);
      setPatchStatus(null);
      setPatchProgress(null);
      setPatchReviewState(emptyReviewState);
      setProtectedFields([]);
      setMessage('Initial profile draft created.');
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
      const result = await patchDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setDraft(result.draft);
      setPatchStatus(result.status);
      const [artifacts, reviewState] = await Promise.all([getPatchArtifacts(selectedPackageId), getPatchReviewState(selectedPackageId)]);
      setPatchArtifacts(artifacts);
      setPatchReviewState(reviewState);
      if (result.status === 'completed' && !hasPatchArtifacts(artifacts)) {
        setPatchStatus('unknown');
        setMessage('No patch artifacts found. Start patching to create a new checkpoint.');
      } else {
        setMessage(result.status === 'completed' ? 'Draft patching completed.' : 'Draft patching is running. You can keep editing and reviewing.');
        void pollPatchProgress();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Patch step failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onShowPatchArtifacts() {
    if (!selectedPackageId) return;
    try {
      const [artifacts, reviewState] = await Promise.all([getPatchArtifacts(selectedPackageId), getPatchReviewState(selectedPackageId)]);
      setPatchArtifacts(artifacts);
      setPatchReviewState(reviewState);
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

  async function onAssignUnmappedFact(itemId: string, field: string) {
    const unmapped_assignments = { ...patchReviewState.unmapped_assignments };
    if (field) unmapped_assignments[itemId] = field;
    else delete unmapped_assignments[itemId];
    await persistPatchReviewState({ ...patchReviewState, unmapped_assignments }, field ? 'Unmapped fact assigned to field.' : 'Unmapped fact assignment removed.');
  }

  async function onDelegateReviewResolution() {
    if (!selectedPackageId || !selectedProfile || !unresolvedReviewItems.length) return;
    setBusy('resolve');
    try {
      const result = await resolvePatchReview({
        data_package_id: selectedPackageId,
        profile_identifier: selectedProfile,
        review_items: unresolvedReviewItems.map(toResolutionItem),
      });
      setDraft(result.draft);
      setPatchReviewState(result.review_state);
      if (result.validation_errors.length > 0) {
        setMessage('Review agent produced schema issues. No review items were resolved.');
      } else {
        setMessage(resolutionSummaryMessage(result.resolved_count, result.unresolved_item_ids.length, result.resolution_decisions || []));
      }
      void onShowPatchArtifacts();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Review resolution agent failed.');
    } finally {
      setBusy(null);
    }
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

  async function pollPatchProgress() {
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
    } catch {
      // ignore polling errors
    }
  }

  useEffect(() => {
    if (!selectedPackageId) return;
    setBusy('load');
    void (async () => {
      try {
        const [ctx, draftResult, fields, { status, progress }, artifacts, reviewState, chunkStatus] = await Promise.all([
          getExistingInitialContext(selectedPackageId),
          getExistingInitialDraft(selectedPackageId),
          getProtectedFields(selectedPackageId),
          getPatchProgress(selectedPackageId),
          getPatchArtifacts(selectedPackageId),
          getPatchReviewState(selectedPackageId),
          getChunkStatus(selectedPackageId),
        ]);
        if (ctx) setContext(ctx);
        if (draftResult) setDraft(draftResult);
        setProtectedFields(fields);
        setPatchStatus(status);
        setPatchProgress(progress || null);
        if (status === 'completed' || status === 'crashed' || status === 'cancelled' || hasPatchArtifacts(artifacts)) setPatchArtifacts(artifacts);
        setPatchReviewState(reviewState);
        setHasChunks(chunkStatus.has_chunks);
        setMessage('Workflow state loaded.');
      } catch (error) {
        setMessage(error instanceof Error ? error.message : 'Failed to load workflow state.');
      } finally {
        setBusy(null);
      }
    })();
  }, [selectedPackageId]);

  useEffect(() => {
    if (!patchStatus || TERMINAL_PATCH_STATUSES.has(patchStatus)) return;
    const interval = setInterval(() => void pollPatchProgress(), 2000);
    return () => clearInterval(interval);
  }, [patchStatus, selectedPackageId]);

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">SIMONE - DCAT metadata extraction</p>
          <h1>Dataset in. Profile draft out.</h1>
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
              <label className="upload-box">
                <input type="file" accept=".zip" onChange={(event) => void onUpload(event.target.files?.[0])} />
                <span>Upload dataset ZIP</span>
                <strong>{busy === 'upload' ? 'Uploading...' : 'Choose archive'}</strong>
              </label>
              <div className="panel compact">
                <div className="panel-heading">
                  <span>Packages</span>
                  <button onClick={() => void refresh()} disabled={!!busy}>Refresh</button>
                </div>
                <select value={selectedPackageId} onChange={(event) => setSelectedPackageId(event.target.value)}>
                  <option value="">No package selected</option>
                  {packages.map((item) => <option key={item.id} value={item.id}>{item.file_name}</option>)}
                </select>
              </div>
              <div className="panel compact">
                <div className="panel-heading"><span>Profile</span></div>
                <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value)}>
                  <option value="">No profile selected</option>
                  {profiles.map((profile) => <option key={profile.identifier} value={profile.identifier}>{profile.identifier}</option>)}
                </select>
              </div>
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
                <button onClick={() => void onChunk(hasChunks)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking...' : hasChunks ? 'Re-create and remove old chunks' : 'Create new chunks'}</button>
              </div>
              {selectedPackage && (
                <div className="file-list">
                  {selectedPackage.files.slice(0, 8).map((file) => (
                    <div key={file.file_path} className="file-row">
                      <span>{file.file_path}</span>
                      <small>{formatBytes(file.byte_size)}</small>
                    </div>
                  ))}
                </div>
              )}
              {chunkResult && <p className="muted">Chunk status: <strong>{chunkResult.status}</strong> - {chunkResult.chunks.flat().length} chunks visible</p>}
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
                  <Field label="Technique" value={context.analytical_technique} />
                  <Field label="Device" value={context.device_name} />
                  <Field label="Model" value={context.device_model} />
                  <div className="summary-box">{context.summary}</div>
                  <div className="chips">{context.keywords.map((keyword) => <span key={keyword}>{keyword}</span>)}</div>
                </div>
              )}
            </div>
          </article>

          <article className="step-card">
            <div className="step-index">03</div>
            <div className="step-body">
              <h2>Draft workspace</h2>
              <p>Create the initial profile draft, edit and lock fields, then patch the draft with chunk evidence while reviewing issues as they appear.</p>
              <div className="actions">
                <button onClick={() => void onDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Drafting...' : draft ? 'Re-create and remove old draft' : 'Create new draft'}</button>
                <button onClick={() => void onPatch()} disabled={!draft || !selectedProfile || !!busy || isPatching}>{patchButtonLabel}</button>
                <button className="ghost" onClick={() => void onShowPatchArtifacts()} disabled={!selectedPackageId || busy === 'load'}>Show/refresh artifacts</button>
              </div>
              {patchStatus && (
                <div className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{patchStatus}</strong></span>
                    {progressTotalBatches > 0 && <span>Patching batch {progressBatchNo} of {progressTotalBatches}</span>}
                  </div>
                  <div className="patch-progress-track" aria-hidden="true"><div style={{ width: `${progressPercent}%` }} /></div>
                  {patchProgress && (
                    <div className="patch-progress-summary">
                      {patchProgress.file_name && <span>Current patch <strong>{patchProgress.file_name}</strong></span>}
                      <span>{patchProgress.accepted_fields?.length || 0} accepted field changes</span>
                      <span>{patchProgress.total_candidates || 0} candidates reviewed</span>
                      {(patchProgress.validation_errors?.length || 0) > 0 && <span className="warning">Schema review required</span>}
                    </div>
                  )}
                </div>
              )}
              {reviewMarkers.length > 0 && (
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
                />
              )}
              {patchArtifacts && (
                <div className="artifact-panel">
                  <h3>Review</h3>
                  <div className="review-toolbar">
                    <span>{unresolvedReviewItems.length} unresolved item{unresolvedReviewItems.length === 1 ? '' : 's'}</span>
                    <button onClick={() => void onDelegateReviewResolution()} disabled={!unresolvedReviewItems.length || !!busy}>
                      {busy === 'resolve' ? 'Agent resolving...' : 'Delegate remaining to agent'}
                    </button>
                  </div>
                  <div className="artifact-tabs" role="tablist" aria-label="Patch review">
                    <button className={reviewTab === 'matched' ? 'active' : ''} onClick={() => setReviewTab('matched')}>Matched issues ({matchedReviewItems.length})</button>
                    <button className={reviewTab === 'unmapped' ? 'active' : ''} onClick={() => setReviewTab('unmapped')}>Unmapped ({unmappedReviewFacts.length})</button>
                    <button className={reviewTab === 'resolved' ? 'active' : ''} onClick={() => setReviewTab('resolved')}>Resolved ({resolvedReviewItems.length})</button>
                  </div>
                  {reviewTab === 'matched' && <ReviewItemList items={matchedReviewItems} draft={draft} onResolve={(itemId) => void onResolveReviewItem(itemId)} onApplyPatch={(itemId, value) => void onApplyPatch(itemId, value)} />}
                  {reviewTab === 'unmapped' && (
                    <UnmappedFactList
                      facts={unmappedReviewFacts}
                      fields={draftFields}
                      reviewState={patchReviewState}
                      onAssign={(key, field) => void onAssignUnmappedFact(key, field)}
                    />
                  )}
                  {reviewTab === 'resolved' && <ReviewItemList items={resolvedReviewItems} draft={draft} onResolve={(itemId) => void onResolveReviewItem(itemId)} />}
                </div>
              )}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
