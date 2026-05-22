import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { checkVocabularyEmbeddings, deleteVocabulary, getVocabulary, importVocabulary, listVocabularies } from '../api/semantic';
import type { VocabEmbeddingStatus, VocabSchemeInfo } from '../api/types';

type VocabEmbeddingStatusState = VocabEmbeddingStatus & { status: 'ready' | 'pending' | 'running' | 'unknown' | 'error'; message?: string };

function DetailField({ label, value }: { label: string; value?: string | number | null }) {
  return (
    <div className="field">
      <span>{label}</span>
      <strong>{value || '-'}</strong>
    </div>
  );
}

function VocabularyDetails({ details }: { details: VocabSchemeInfo }) {
  const [expandedTermSchemes, setExpandedTermSchemes] = useState<Set<string>>(new Set());

  return (
    <div className="vocab-detail-panel">
      <div className="vocab-detail-grid">
        <DetailField label="Identifier" value={details.identifier} />
        <DetailField label="Source" value={details.source} />
        <DetailField label="Format" value={details.rdf_format} />
        <DetailField label="Triples" value={details.num_triples} />
      </div>
      {details.description && (
        <>
          <p className="muted">Description</p>
          <pre className="json-panel">{details.description}</pre>
        </>
      )}
      {details.vocab_term_schemes.length > 0 && (
        <>
          <p className="muted">Term schemes</p>
          <ul className="vocab-term-scheme-list">
            {details.vocab_term_schemes.map((scheme) => {
              const isExpanded = expandedTermSchemes.has(scheme.rdf_type);
              return (
                <li key={scheme.rdf_type} className="vocab-term-scheme">
                  <div className="vocab-term-scheme-heading">
                    <strong>{scheme.rdf_type}</strong>
                    <span className="chunk-badge">{scheme.count} term{scheme.count === 1 ? '' : 's'}</span>
                    <button
                      className="ghost small"
                      onClick={() => {
                        setExpandedTermSchemes((prev) => {
                          const next = new Set(prev);
                          if (next.has(scheme.rdf_type)) next.delete(scheme.rdf_type);
                          else next.add(scheme.rdf_type);
                          return next;
                        });
                      }}
                    >
                      {isExpanded ? 'Collapse' : 'Expand'}
                    </button>
                  </div>
                  {isExpanded && (
                    <div className="vocab-term-scheme-details">
                      {scheme.properties.length > 0 && (
                        <div>
                          <span>Properties</span>
                          <div className="chips">
                            {scheme.properties.map((prop) => <span key={prop}>{prop}</span>)}
                          </div>
                        </div>
                      )}
                      {scheme.applicable_relationships.length > 0 && (
                        <div>
                          <span>Relationships</span>
                          <div className="chips">
                            {scheme.applicable_relationships.map((rel) => <span key={rel}>{rel}</span>)}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

export function VocabularyPanel({ onError }: { onError: (message: string) => void }) {
  const [vocabularies, setVocabularies] = useState<string[]>([]);
  const [selectedVocabulary, setSelectedVocabulary] = useState('');
  const [vocabDetails, setVocabDetails] = useState<VocabSchemeInfo | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [managementOpen, setManagementOpen] = useState(false);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [managementBusy, setManagementBusy] = useState<'import' | 'delete' | null>(null);
  const [newIdentifier, setNewIdentifier] = useState('');
  const [newRdfSource, setNewRdfSource] = useState('');
  const [deleteIdentifier, setDeleteIdentifier] = useState('');
  const [deleteConfirmation, setDeleteConfirmation] = useState('');
  const [managementMessage, setManagementMessage] = useState('');
  const [embeddingStatuses, setEmbeddingStatuses] = useState<Record<string, VocabEmbeddingStatusState>>({});
  const [loadingEmbeddingStatuses, setLoadingEmbeddingStatuses] = useState(false);

  function normalizeEmbeddingStatus(status: VocabEmbeddingStatus): VocabEmbeddingStatusState {
    if (status.pending_updates === 0) return { ...status, status: 'ready' };
    if (status.task_status === 'running') return { ...status, status: 'running' };
    if (status.pending_updates > 0) return { ...status, status: 'pending' };
    return { ...status, status: 'unknown' };
  }

  async function refreshVocabularies() {
    const nextVocabularies = await listVocabularies();
    setVocabularies(nextVocabularies);
    setDeleteIdentifier((current) => nextVocabularies.includes(current) ? current : '');
    return nextVocabularies;
  }

  async function refreshEmbeddingStatuses(identifiers: string[]) {
    if (!identifiers.length) {
      setEmbeddingStatuses({});
      return;
    }

    setLoadingEmbeddingStatuses(true);
    const results = await Promise.all(identifiers.map(async (identifier) => {
      try {
        const status = await checkVocabularyEmbeddings(identifier);
        return [identifier, normalizeEmbeddingStatus(status)] as const;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Embedding status check failed.';
        return [identifier, { pending_updates: -1, task_status: 'unknown', status: 'error', message } satisfies VocabEmbeddingStatusState] as const;
      }
    }));
    setEmbeddingStatuses(Object.fromEntries(results));
    setLoadingEmbeddingStatuses(false);
  }

  useEffect(() => {
    let cancelled = false;
    setLoadingList(true);
    void (async () => {
      try {
        const nextVocabularies = await listVocabularies();
        if (!cancelled) {
          setVocabularies(nextVocabularies);
          setDeleteIdentifier((current) => nextVocabularies.includes(current) ? current : '');
        }
      } catch (error) {
        if (!cancelled) onError(error instanceof Error ? error.message : 'Could not load vocabularies.');
      } finally {
        if (!cancelled) setLoadingList(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [onError]);

  useEffect(() => {
    if (!dialogOpen || !selectedVocabulary) {
      setVocabDetails(null);
      return;
    }

    let cancelled = false;
    setLoadingDetails(true);
    void (async () => {
      try {
        const details = await getVocabulary(selectedVocabulary);
        if (!cancelled) setVocabDetails(details);
      } catch (error) {
        if (!cancelled) onError(error instanceof Error ? error.message : 'Failed to load vocabulary details.');
      } finally {
        if (!cancelled) setLoadingDetails(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [dialogOpen, onError, selectedVocabulary]);

  function openDialog(identifier: string) {
    setSelectedVocabulary(identifier);
    setDialogOpen(true);
  }

  function closeDialog() {
    setDialogOpen(false);
    setSelectedVocabulary('');
    setVocabDetails(null);
  }

  function openManagement() {
    setManagementOpen(true);
    setManagementMessage('');
    setDeleteIdentifier((current) => current || vocabularies[0] || '');
    setDeleteConfirmation('');
    void refreshEmbeddingStatuses(vocabularies);
  }

  function closeManagement() {
    if (managementBusy) return;
    setManagementOpen(false);
    setManagementMessage('');
    setDeleteConfirmation('');
  }

  async function onImportVocabulary() {
    const identifier = newIdentifier.trim();
    const rdfSource = newRdfSource.trim();
    if (!identifier || !rdfSource) {
      setManagementMessage('Identifier and RDF source URL are required.');
      return;
    }

    setManagementBusy('import');
    setManagementMessage('');
    try {
      await importVocabulary({ identifier, rdfSource });
      const importedEmbeddingStatus = await checkVocabularyEmbeddings(identifier);
      setEmbeddingStatuses((current) => ({
        ...current,
        [identifier]: normalizeEmbeddingStatus(importedEmbeddingStatus),
      }));
      const nextVocabularies = await refreshVocabularies();
      await refreshEmbeddingStatuses(nextVocabularies);
      setNewIdentifier('');
      setNewRdfSource('');
      setManagementMessage(`Vocabulary ${identifier} added. Embedding generation started.`);
    } catch (error) {
      setManagementMessage(error instanceof Error ? error.message : 'Vocabulary import failed.');
    } finally {
      setManagementBusy(null);
    }
  }

  async function onDeleteVocabulary() {
    if (!deleteIdentifier || deleteConfirmation !== deleteIdentifier) {
      setManagementMessage('Type the exact vocabulary identifier before removing it.');
      return;
    }

    setManagementBusy('delete');
    setManagementMessage('');
    try {
      await deleteVocabulary(deleteIdentifier);
      const removedIdentifier = deleteIdentifier;
      const nextVocabularies = await refreshVocabularies();
      await refreshEmbeddingStatuses(nextVocabularies);
      if (selectedVocabulary === deleteIdentifier) closeDialog();
      setDeleteConfirmation('');
      setManagementMessage(`Vocabulary ${removedIdentifier} removed.`);
    } catch (error) {
      setManagementMessage(error instanceof Error ? error.message : 'Vocabulary removal failed.');
    } finally {
      setManagementBusy(null);
    }
  }

  const canDeleteVocabulary = Boolean(deleteIdentifier) && deleteConfirmation === deleteIdentifier && !managementBusy;

  function embeddingBadgeText(status?: VocabEmbeddingStatusState): string {
    if (!status) return loadingEmbeddingStatuses ? 'Checking' : 'Unknown';
    if (status.status === 'ready') return 'Embedded';
    if (status.status === 'running') return `${status.pending_updates} pending`;
    if (status.status === 'pending') return `${status.pending_updates} pending`;
    if (status.status === 'error') return 'Check failed';
    return 'Unknown';
  }

  function embeddingBadgeTitle(status?: VocabEmbeddingStatusState): string {
    if (!status) return loadingEmbeddingStatuses ? 'Checking embedding status.' : 'Embedding status has not been checked.';
    if (status.status === 'ready') return 'All vocabulary resources are embedded and can be queried.';
    if (status.status === 'error') return status.message || 'Embedding status check failed.';
    return `Task status: ${status.task_status}. Pending updates: ${status.pending_updates}.`;
  }

  return (
    <>
      <div className="panel compact">
        <div className="panel-heading">
          <span>Vocabularies</span>
          <button onClick={openManagement}>Manage</button>
        </div>
        {loadingList && <p className="muted compact-empty">Loading vocabularies...</p>}
        {!loadingList && vocabularies.length > 0 && (
          <div className="vocab-list">
            {vocabularies.map((vocab) => (
              <button
                key={vocab}
                className="vocab-row"
                onClick={() => openDialog(vocab)}
                title={vocab}
              >
                <span>{vocab}</span>
              </button>
            ))}
          </div>
        )}
        {!loadingList && vocabularies.length === 0 && (
          <p className="muted compact-empty">No vocabularies found.</p>
        )}
      </div>
      {dialogOpen && createPortal((
        <div className="vocab-dialog-overlay" onClick={closeDialog}>
          <div className="vocab-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <strong>{selectedVocabulary}</strong>
              <button className="ghost" onClick={closeDialog}>Close</button>
            </div>
            <div className="vocab-dialog-body">
              {loadingDetails && <p className="muted">Loading vocabulary details...</p>}
              {vocabDetails && <VocabularyDetails details={vocabDetails} />}
            </div>
          </div>
        </div>
      ), document.body)}
      {managementOpen && createPortal((
        <div className="vocab-dialog-overlay" onClick={closeManagement}>
          <div className="vocab-dialog vocab-management-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <strong>Vocabulary Management</strong>
              <button className="ghost" onClick={closeManagement} disabled={Boolean(managementBusy)}>Close</button>
            </div>
            <div className="vocab-dialog-body vocab-management-body">
              <section className="vocab-management-section">
                <h3>Add Vocabulary</h3>
                <label>
                  <span>Identifier</span>
                  <input
                    value={newIdentifier}
                    onChange={(event) => setNewIdentifier(event.target.value)}
                    placeholder="https://example.org/my-vocabulary"
                    disabled={Boolean(managementBusy)}
                  />
                </label>
                <label>
                  <span>RDF source URL</span>
                  <input
                    value={newRdfSource}
                    onChange={(event) => setNewRdfSource(event.target.value)}
                    placeholder="https://example.org/my-vocabulary.ttl"
                    disabled={Boolean(managementBusy)}
                  />
                </label>
                <button onClick={() => void onImportVocabulary()} disabled={Boolean(managementBusy)}>
                  {managementBusy === 'import' ? 'Adding...' : 'Add vocabulary'}
                </button>
              </section>

              <section className="vocab-management-section">
                <div className="vocab-management-heading-row">
                  <h3>Embedding Status</h3>
                  <button className="ghost small" onClick={() => void refreshEmbeddingStatuses(vocabularies)} disabled={loadingEmbeddingStatuses || Boolean(managementBusy)}>
                    {loadingEmbeddingStatuses ? 'Checking...' : 'Refresh'}
                  </button>
                </div>
                {vocabularies.length > 0 ? (
                  <ul className="vocab-embedding-status-list">
                    {vocabularies.map((vocab) => {
                      const status = embeddingStatuses[vocab];
                      return (
                        <li key={vocab}>
                          <span>{vocab}</span>
                          <strong className={`vocab-embedding-badge ${status?.status ?? (loadingEmbeddingStatuses ? 'checking' : 'unknown')}`} title={embeddingBadgeTitle(status)}>
                            {embeddingBadgeText(status)}
                          </strong>
                        </li>
                      );
                    })}
                  </ul>
                ) : (
                  <p className="muted compact-empty">No vocabularies found.</p>
                )}
              </section>

              <section className="vocab-management-section danger">
                <h3>Remove Vocabulary</h3>
                <label>
                  <span>Vocabulary</span>
                  <select
                    value={deleteIdentifier}
                    onChange={(event) => {
                      setDeleteIdentifier(event.target.value);
                      setDeleteConfirmation('');
                    }}
                    disabled={Boolean(managementBusy) || vocabularies.length === 0}
                  >
                    <option value="">Select vocabulary</option>
                    {vocabularies.map((vocab) => <option key={vocab} value={vocab}>{vocab}</option>)}
                  </select>
                </label>
                <label>
                  <span>Type identifier to confirm</span>
                  <input
                    value={deleteConfirmation}
                    onChange={(event) => setDeleteConfirmation(event.target.value)}
                    placeholder={deleteIdentifier || 'Select a vocabulary first'}
                    disabled={Boolean(managementBusy) || !deleteIdentifier}
                  />
                </label>
                <button className="danger-button" onClick={() => void onDeleteVocabulary()} disabled={!canDeleteVocabulary}>
                  {managementBusy === 'delete' ? 'Removing...' : 'Remove vocabulary'}
                </button>
              </section>

              {managementMessage && <p className="muted vocab-management-message">{managementMessage}</p>}
            </div>
          </div>
        </div>
      ), document.body)}
    </>
  );
}
