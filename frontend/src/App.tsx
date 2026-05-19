import { useEffect, useMemo, useRef, useState } from 'react';
import { chunkDataPackage, getChunkStatus, listDataPackages, uploadDataPackage } from './api/datasources';
import { extractInitialContext, extractInitialDraft, getExistingInitialContext, getExistingInitialDraft, patchDraft, getProtectedFields, setProtectedFields as apiSetProtectedFields, getPatchProgress, getPatchArtifacts, saveDraft } from './api/extraction';
import { listProfiles } from './api/profiles';
import { JsonEditor } from './components/JsonEditor';
import type { ChunkRequestResponse, DataPackageResponse, InitialContext, ProfileManifestResponse } from './api/types';
import type { PatchArtifacts, PatchProgress, PatchTaskStatus } from './api/extraction';

type BusyKey = 'upload' | 'chunk' | 'context' | 'draft' | 'patch' | 'load' | 'loadContext' | 'loadDraft';
type ArtifactTab = 'patches' | 'quality_reports' | 'unmapped_facts';

const TERMINAL_PATCH_STATUSES = new Set<PatchTaskStatus>(['unknown', 'completed', 'cancelled', 'crashed']);

function formatBytes(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

function Field({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="field">
      <span>{label}</span>
      <strong>{value || '—'}</strong>
    </div>
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
  const [artifactTab, setArtifactTab] = useState<ArtifactTab>('patches');
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(false);

  const selectedPackage = useMemo(
    () => packages.find((item) => item.id === selectedPackageId) || null,
    [packages, selectedPackageId],
  );
  const isPatching = patchStatus === 'running';
  const progressBatchNo = patchProgress?.batch_no ?? 0;
  const progressTotalBatches = patchProgress?.total_batches ?? 0;
  const progressPercent = progressTotalBatches > 0
    ? Math.min(100, Math.round((progressBatchNo / progressTotalBatches) * 100))
    : 0;
  const visibleArtifacts = patchArtifacts?.[artifactTab] ?? [];

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

  async function onLoadContext() {
    if (!selectedPackageId) return;
    setBusy('loadContext');
    try {
      const result = await getExistingInitialContext(selectedPackageId);
      if (result) {
        setContext(result);
        setMessage('Loaded existing initial context.');
      } else {
        setMessage('No existing initial context found for this package.');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to load existing context.');
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
      setProtectedFields([]);
      setMessage('Initial profile draft created.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Draft creation failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onLoadDraft() {
    if (!selectedPackageId) return;
    setBusy('loadDraft');
    try {
      const result = await getExistingInitialDraft(selectedPackageId);
      if (result) {
        setDraft(result);
        setMessage('Loaded existing initial draft.');
      } else {
        setMessage('No existing initial draft found for this package.');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to load existing draft.');
    } finally {
      setBusy(null);
    }
  }

  const saveDraftTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function onDraftChange(updated: Record<string, unknown>) {
    setDraft(updated);
    if (!selectedPackageId) return;
    if (saveDraftTimeoutRef.current) {
      clearTimeout(saveDraftTimeoutRef.current);
    }
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
      setPatchArtifacts(null);
      setPatchProgress(null);
      const result = await patchDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setDraft(result.draft);
      setPatchStatus(result.status);
      setMessage(result.status === 'completed' ? 'Draft patching completed.' : 'Draft patching is running. Repeat to refresh current draft.');
      if (result.status === 'running') {
        void pollPatchProgress();
      } else if (result.status === 'completed' || result.status === 'crashed' || result.status === 'cancelled') {
        const artifacts = await getPatchArtifacts(selectedPackageId);
        setPatchArtifacts(artifacts);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Patch step failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onLoadProtectedFields() {
    if (!selectedPackageId) return;
    try {
      const fields = await getProtectedFields(selectedPackageId);
      setProtectedFields(fields);
    } catch {
      // ignore
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
      const { status, progress } = await getPatchProgress(selectedPackageId);
      setPatchStatus(status);
      setPatchProgress(progress || null);
      if (status === 'completed' || status === 'crashed' || status === 'cancelled') {
        const artifacts = await getPatchArtifacts(selectedPackageId);
        setPatchArtifacts(artifacts);
      }
    } catch {
      // ignore polling errors
    }
  }

  useEffect(() => {
    if (!selectedPackageId) return;
    setBusy('load');
    void (async () => {
      try {
        const [ctx, draftResult, fields, { status, progress }, artifacts, chunkStatus] = await Promise.all([
          getExistingInitialContext(selectedPackageId),
          getExistingInitialDraft(selectedPackageId),
          getProtectedFields(selectedPackageId),
          getPatchProgress(selectedPackageId),
          getPatchArtifacts(selectedPackageId),
          getChunkStatus(selectedPackageId),
        ]);
        if (ctx) setContext(ctx);
        if (draftResult) setDraft(draftResult);
        setProtectedFields(fields);
        setPatchStatus(status);
        setPatchProgress(progress || null);
        if (status === 'completed' || status === 'crashed' || status === 'cancelled') {
          setPatchArtifacts(artifacts);
        }
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
          <p className="eyebrow">SIMONE · DCAT metadata extraction</p>
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
          <button
            className="rail-toggle"
            onClick={() => setRailCollapsed(!railCollapsed)}
            title={railCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {railCollapsed ? '→' : '←'}
          </button>
          {!railCollapsed && (
            <>
              <label className="upload-box">
                <input type="file" accept=".zip" onChange={(event) => void onUpload(event.target.files?.[0])} />
                <span>Upload dataset ZIP</span>
                <strong>{busy === 'upload' ? 'Uploading…' : 'Choose archive'}</strong>
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
                <button onClick={() => void onChunk(hasChunks)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking…' : hasChunks ? 'Re-create and remove old chunks' : 'Create new chunks'}</button>
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
              {chunkResult && <p className="muted">Chunk status: <strong>{chunkResult.status}</strong> · {chunkResult.chunks.flat().length} chunks visible</p>}
            </div>
          </article>

          <article className="step-card">
            <div className="step-index">02</div>
            <div className="step-body">
              <h2>Determine initial context</h2>
              <p>Extract high-level context, likely metadata sources, keywords, file relationships, and evidence from the package.</p>
              <div className="actions">
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !!busy}>{busy === 'context' ? 'Extracting…' : context ? 'Re-extract and remove old context' : 'Extract new context'}</button>
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
              <h2>Create DCAT profile draft</h2>
              <p>Generate the first schema-conforming dataset object, then validate it against the selected registered profile.</p>
              <div className="actions">
                <button onClick={() => void onDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Drafting…' : draft ? 'Re-create and remove old draft' : 'Create new draft'}</button>
              </div>
              {draft && (
                <JsonEditor
                  value={draft as Record<string, unknown>}
                  onChange={(updated) => onDraftChange(updated)}
                  protectedPaths={protectedFields}
                  onProtectedPathsChange={(paths) => void onSaveProtectedFields(paths)}
                />
              )}
            </div>
          </article>

          <article className="step-card">
            <div className="step-index">04</div>
            <div className="step-body">
              <h2>Patch draft with content chunks</h2>
              <p>Run the patch agent against chunked content. Review artifacts and apply human-in-the-loop decisions before accepting changes.</p>
              <div className="actions">
                <button onClick={() => void onPatch()} disabled={!draft || !selectedProfile || !!busy || isPatching}>{isPatching ? 'Patching…' : patchStatus || patchArtifacts ? 'Re-start and remove old patching' : 'Start new patching'}</button>
              </div>
              {patchStatus && (
                <div className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{patchStatus}</strong></span>
                    {progressTotalBatches > 0 && (
                      <span>Patching batch {progressBatchNo} of {progressTotalBatches}</span>
                    )}
                  </div>
                  <div className="patch-progress-track" aria-hidden="true">
                    <div style={{ width: `${progressPercent}%` }} />
                  </div>
                  {patchProgress && (
                    <pre>{JSON.stringify(patchProgress, null, 2)}</pre>
                  )}
                </div>
              )}
              {patchArtifacts && (
                <div className="artifact-panel">
                  <h3>Patch artifacts</h3>
                  <div className="artifact-tabs" role="tablist" aria-label="Patch artifacts">
                    <button className={artifactTab === 'patches' ? 'active' : ''} onClick={() => setArtifactTab('patches')}>Patches ({patchArtifacts.patches.length})</button>
                    <button className={artifactTab === 'quality_reports' ? 'active' : ''} onClick={() => setArtifactTab('quality_reports')}>Quality ({patchArtifacts.quality_reports.length})</button>
                    <button className={artifactTab === 'unmapped_facts' ? 'active' : ''} onClick={() => setArtifactTab('unmapped_facts')}>Unmapped ({patchArtifacts.unmapped_facts.length})</button>
                  </div>
                  {visibleArtifacts.length === 0 ? (
                    <p className="muted">No artifacts in this category yet.</p>
                  ) : (
                    <ul>
                      {visibleArtifacts.map((artifact, i) => (
                        <li key={`${artifact.file_name || artifactTab}-${i}`}>
                          <strong>{String(artifact.file_name || `Artifact ${i + 1}`)}</strong>
                          <pre>{JSON.stringify(artifact, null, 2)}</pre>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
