import { useEffect, useMemo, useState } from 'react';
import { chunkDataPackage, listDataPackages, uploadDataPackage } from './api/datasources';
import { extractInitialContext, extractInitialDraft, patchDraft } from './api/extraction';
import { listProfiles, validateProfileDocument } from './api/profiles';
import type { ChunkRequestResponse, DataPackageResponse, InitialContext, ProfileManifestResponse } from './api/types';

type BusyKey = 'upload' | 'chunk' | 'context' | 'draft' | 'patch' | 'validate' | 'load';

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

function JsonPanel({ value }: { value: unknown }) {
  return <pre className="json-panel">{JSON.stringify(value, null, 2)}</pre>;
}

export function App() {
  const [packages, setPackages] = useState<DataPackageResponse[]>([]);
  const [profiles, setProfiles] = useState<ProfileManifestResponse[]>([]);
  const [selectedPackageId, setSelectedPackageId] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [chunkResult, setChunkResult] = useState<ChunkRequestResponse | null>(null);
  const [context, setContext] = useState<InitialContext | null>(null);
  const [draft, setDraft] = useState<object | null>(null);
  const [validation, setValidation] = useState<{ valid: boolean; errors: Array<{ path: string; message: string }> } | null>(null);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');

  const selectedPackage = useMemo(
    () => packages.find((item) => item.id === selectedPackageId) || null,
    [packages, selectedPackageId],
  );

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
      setValidation(null);
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
      setValidation(null);
      setMessage('Initial profile draft created.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Draft creation failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onPatch() {
    if (!selectedPackageId || !selectedProfile) return;
    setBusy('patch');
    try {
      const result = await patchDraft({ data_package_id: selectedPackageId, profile_identifier: selectedProfile });
      setDraft(result.draft);
      setMessage(result.status === 'completed' ? 'Draft patching completed.' : 'Draft patching is running. Repeat to refresh current draft.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Patch step failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onValidate() {
    if (!draft || !selectedProfile) return;
    setBusy('validate');
    try {
      const result = await validateProfileDocument(selectedProfile, draft);
      setValidation(result);
      setMessage(result.valid ? 'Draft validates against the selected profile.' : 'Draft has validation issues.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Validation failed.');
    } finally {
      setBusy(null);
    }
  }

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

      <section className="layout">
        <aside className="rail">
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
        </aside>

        <section className="workflow">
          <article className="step-card active">
            <div className="step-index">01</div>
            <div className="step-body">
              <h2>Upload dataset and create chunks</h2>
              <p>The archive is stored as a data package. Chunking prepares the package for later patch and enrichment stages.</p>
              <div className="actions">
                <button onClick={() => void onChunk(false)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking…' : 'Create / refresh chunks'}</button>
                <button className="ghost" onClick={() => void onChunk(true)} disabled={!selectedPackageId || !!busy}>Replace chunks</button>
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
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !!busy}>{busy === 'context' ? 'Extracting…' : 'Extract context'}</button>
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
                <button onClick={() => void onDraft()} disabled={!selectedPackageId || !selectedProfile || !!busy}>{busy === 'draft' ? 'Drafting…' : 'Create draft'}</button>
                <button className="ghost" onClick={() => void onValidate()} disabled={!draft || !selectedProfile || !!busy}>Validate</button>
                <button className="ghost" onClick={() => void onPatch()} disabled={!draft || !selectedProfile || !!busy}>Start patching</button>
              </div>
              {validation && <p className={validation.valid ? 'ok' : 'warning'}>{validation.valid ? 'Valid profile document.' : `${validation.errors.length} validation issue(s).`}</p>}
              {draft && <JsonPanel value={draft} />}
            </div>
          </article>
        </section>
      </section>
    </main>
  );
}
