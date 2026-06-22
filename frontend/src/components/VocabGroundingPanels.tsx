import {
  type PointerEvent,
  type ReactNode,
  type SyntheticEvent,
  type WheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { createPortal } from 'react-dom';
import { listVocabularies } from '../api/semantic';
import type { VocabQueryResult } from '../api/types';
import type {
  ExtractionVocabQueryConfig,
  ExtractionVocabQueryRecord,
  WorkflowProgress,
  WorkflowTaskStatus,
  ProjectionLedgerRecord,
  RequirementReport,
  RequirementReportItem,
} from '../api/extraction';
import { formatDuration, formatExtractionStage, labelDy } from '../lib/format';
import { asRecord, asRecordArray } from '../lib/records';
import { JsonDetails } from './JsonDetails';
import { evidenceContextNoteCount } from './EvidenceContextPanels';
export function sourceContextJsonPointer(query: ExtractionVocabQueryRecord): string | null {
  const path = query.source_context?.json_path;
  return typeof path === 'string' && path.startsWith('/') ? path : null;
}

function vocabResourceTitle(result: Record<string, unknown>, uri: string): string | null {
  const coerced = coerceVocabQueryResult(result);
  const resource = coerced?.resources[uri];
  const properties = resource?.properties ?? {};
  for (const key of ['skos:prefLabel', 'rdfs:label', 'title', 'label', 'name']) {
    const value = properties[key];
    if (typeof value === 'string' && value.trim()) return value;
    if (Array.isArray(value)) {
      const first = value.find((item): item is string => typeof item === 'string' && item.trim().length > 0);
      if (first) return first;
    }
  }
  return null;
}


function projectionEvidencePreview(record: ProjectionLedgerRecord): string {
  const value = record.source_evidence || record.reason || record.error || record.object_identifier;
  return value.length > 180 ? `${value.slice(0, 177)}...` : value;
}

function projectionFailureMessage(record: ProjectionLedgerRecord): string {
  return record.error || record.reason || 'No profile path was selected for this evidence note.';
}

function draftClassInstances(document?: Record<string, unknown> | null): Array<{ className: string; count: number; labels: string[] }> {
  if (!document) return [];
  const groups: Array<{ className: string; values: Record<string, unknown>[] }> = [
    { className: 'Dataset', values: [document] },
    { className: 'DataGeneratingActivity', values: asRecordArray(document.was_generated_by) },
    { className: 'Distribution', values: asRecordArray(document.dataset_distribution) },
    { className: 'EvaluatedEntity', values: asRecordArray(document.is_about_entity) },
    { className: 'EvaluatedActivity', values: asRecordArray(document.is_about_activity) },
    { className: 'Agent', values: asRecordArray(document.creator) },
  ];
  const nestedAgenticEntities = groups[1].values.flatMap((activity) => asRecordArray(activity.carried_out_by));
  groups.push({ className: 'AgenticEntity', values: nestedAgenticEntities });
  return groups
    .filter((group) => group.values.length > 0)
    .map((group) => ({
      className: group.className,
      count: group.values.length,
      labels: group.values.slice(0, 3).map((value) => metadataObjectLabel(value)),
    }));
}

function metadataObjectLabel(value: Record<string, unknown>): string {
  const title = value.title;
  const name = value.name;
  const id = value.id;
  if (Array.isArray(title) && title.length) return String(title[0]);
  if (typeof title === 'string' && title.trim()) return title;
  if (Array.isArray(name) && name.length) return String(name[0]);
  if (typeof name === 'string' && name.trim()) return name;
  if (typeof id === 'string' && id.trim()) return id.split(/[/:#]/).filter(Boolean).pop() || id;
  return 'Untitled';
}

function requirementStatusSummary(report?: RequirementReport | null) {
  const requirements = report?.semantic_requirements ?? [];
  return requirements.reduce((acc, requirement) => {
    acc[requirement.status] = (acc[requirement.status] ?? 0) + 1;
    return acc;
  }, {} as Record<string, number>);
}

function requirementScoreLabel(requirement: RequirementReportItem) {
  return `${(requirement.quality * requirement.weight).toFixed(2)} / ${requirement.weight.toFixed(2)}`;
}

function RequirementReviewPopover({
  requirement,
  x,
  y,
}: {
  requirement: RequirementReportItem;
  x: number;
  y: number;
}) {
  const left = Math.min(Math.max(12, x), Math.max(12, window.innerWidth - 392));
  const top = Math.min(Math.max(12, y), Math.max(12, window.innerHeight - 260));
  const evidenceCount = (requirement.selected_evidence?.length ?? 0) + (requirement.context_window?.length ?? 0);
  return createPortal(
    <div className="requirement-review-popover" style={{ left, top }} role="tooltip">
      <span>{formatExtractionStage(requirement.status)} · {requirementScoreLabel(requirement)}</span>
      <strong>{requirement.label}</strong>
      {requirement.rationale ? <p>{requirement.rationale}</p> : null}
      <dl>
        <div><dt>Evidence</dt><dd>{evidenceCount}</dd></div>
        <div><dt>Targets</dt><dd>{requirement.target_paths?.length ?? 0}</dd></div>
        <div><dt>Patch</dt><dd>{formatExtractionStage(requirement.patch?.status ?? 'not_attempted')}</dd></div>
      </dl>
    </div>,
    document.body,
  );
}

function RequirementReviewDialog({
  requirement,
  onClose,
}: {
  requirement: RequirementReportItem;
  onClose: () => void;
}) {
  return createPortal(
    <div className="vocab-dialog-overlay" onClick={onClose}>
      <div className="vocab-dialog requirement-review-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="vocab-dialog-header">
          <div>
            <span>{formatExtractionStage(requirement.status)} · {requirementScoreLabel(requirement)}</span>
            <strong>{requirement.label}</strong>
          </div>
          <button className="ghost" type="button" onClick={onClose}>Close</button>
        </div>
        <div className="vocab-dialog-body requirement-review-body">
          {requirement.rationale ? <p>{requirement.rationale}</p> : null}
          <JsonDetails title="Requirement review artifact" value={requirement} />
        </div>
      </div>
    </div>,
    document.body,
  );
}

function formatPercentScore(value: number | undefined | null): string {
  if (value == null || !Number.isFinite(value)) return '0%';
  return `${Math.round(value * 100)}%`;
}


const QUANTITATIVE_ATTRIBUTE_VOCABULARIES = [
  { label: 'Quantity kinds', identifier: 'http://qudt.org/vocab/quantitykind' },
  { label: 'Units', identifier: 'http://qudt.org/vocab/unit' },
];

type VocabQueryNumericField = {
  key: keyof ExtractionVocabQueryConfig;
  label: string;
  step?: string;
  min?: string;
};

const QUALITATIVE_VOCAB_QUERY_FIELDS: VocabQueryNumericField[] = [
  { key: 'vector_top_k', label: 'Vector top K' },
  { key: 'fulltext_top_k', label: 'Full-text top K' },
  { key: 'seed_top_k', label: 'Seed top K' },
  { key: 'max_hops', label: 'Max hops', min: '0' },
  { key: 'max_statements_per_seed', label: 'Statements / seed' },
  { key: 'vector_weight', label: 'Vector weight', step: '0.1', min: '0.1' },
  { key: 'fulltext_weight', label: 'Full-text weight', step: '0.1', min: '0.1' },
  { key: 'rrf_k', label: 'RRF K' },
];

const QUANTITATIVE_VOCAB_QUERY_FIELDS: VocabQueryNumericField[] = [
  { key: 'quantitative_vector_top_k', label: 'Vector top K' },
  { key: 'quantitative_fulltext_top_k', label: 'Full-text top K' },
  { key: 'quantitative_seed_top_k', label: 'Seed top K' },
  { key: 'quantitative_max_hops', label: 'Max hops', min: '0' },
  { key: 'quantitative_max_statements_per_seed', label: 'Statements / seed' },
  { key: 'quantitative_vector_weight', label: 'Vector weight', step: '0.1', min: '0.1' },
  { key: 'quantitative_fulltext_weight', label: 'Full-text weight', step: '0.1', min: '0.1' },
  { key: 'quantitative_rrf_k', label: 'RRF K' },
];

const VOCAB_QUERY_NUMERIC_FIELDS = [
  ...QUALITATIVE_VOCAB_QUERY_FIELDS,
  ...QUANTITATIVE_VOCAB_QUERY_FIELDS,
];

const QUANTITATIVE_VOCAB_QUERY_DEFAULTS = {
  quantitative_vector_top_k: 12,
  quantitative_fulltext_top_k: 12,
  quantitative_seed_top_k: 6,
  quantitative_max_hops: 0,
  quantitative_max_statements_per_seed: 12,
  quantitative_traversal_direction: 'undirected',
  quantitative_vector_weight: 1,
  quantitative_fulltext_weight: 1,
  quantitative_rrf_k: 60,
};

type VocabQueryConfigMode = 'qualitative' | 'quantitative';

function normalizeVocabQueryConfig(config: ExtractionVocabQueryConfig): ExtractionVocabQueryConfig {
  return {
    ...QUANTITATIVE_VOCAB_QUERY_DEFAULTS,
    ...config,
  };
}

function VocabQueryConfigPanel({
  config,
  disabled,
  onApply,
  onRerunAll,
}: {
  config: ExtractionVocabQueryConfig;
  disabled?: boolean;
  onApply?: (config: ExtractionVocabQueryConfig) => Promise<void> | void;
  onRerunAll?: () => void;
}) {
  const [draft, setDraft] = useState(normalizeVocabQueryConfig(config));
  const [vocabOptions, setVocabOptions] = useState(config.qualitative_vocab_identifiers);
  const [availableVocabularies, setAvailableVocabularies] = useState<string[]>([]);
  const [selectedVocabIdentifier, setSelectedVocabIdentifier] = useState('');
  const [vocabListMessage, setVocabListMessage] = useState('');
  const [modalMode, setModalMode] = useState<VocabQueryConfigMode | null>(null);
  const [applyMessage, setApplyMessage] = useState('');
  const prevConfigRef = useRef(normalizeVocabQueryConfig(config));
  useEffect(() => {
    const normalized = normalizeVocabQueryConfig(config);
    const prev = prevConfigRef.current;
    const changed = (
      prev.qualitative_vocab_identifiers.length !== normalized.qualitative_vocab_identifiers.length
      || prev.qualitative_vocab_identifiers.some((v, i) => v !== normalized.qualitative_vocab_identifiers[i])
      || prev.vector_top_k !== normalized.vector_top_k
      || prev.fulltext_top_k !== normalized.fulltext_top_k
      || prev.seed_top_k !== normalized.seed_top_k
      || prev.max_hops !== normalized.max_hops
      || prev.max_statements_per_seed !== normalized.max_statements_per_seed
      || prev.traversal_direction !== normalized.traversal_direction
      || prev.vector_weight !== normalized.vector_weight
      || prev.fulltext_weight !== normalized.fulltext_weight
      || prev.rrf_k !== normalized.rrf_k
      || prev.quantitative_vector_top_k !== normalized.quantitative_vector_top_k
      || prev.quantitative_fulltext_top_k !== normalized.quantitative_fulltext_top_k
      || prev.quantitative_seed_top_k !== normalized.quantitative_seed_top_k
      || prev.quantitative_max_hops !== normalized.quantitative_max_hops
      || prev.quantitative_max_statements_per_seed !== normalized.quantitative_max_statements_per_seed
      || prev.quantitative_traversal_direction !== normalized.quantitative_traversal_direction
      || prev.quantitative_vector_weight !== normalized.quantitative_vector_weight
      || prev.quantitative_fulltext_weight !== normalized.quantitative_fulltext_weight
      || prev.quantitative_rrf_k !== normalized.quantitative_rrf_k
    );
    if (changed) {
      setDraft(normalized);
      setVocabOptions((current) => Array.from(new Set([...current, ...normalized.qualitative_vocab_identifiers])));
    }
    prevConfigRef.current = normalized;
  }, [config]);
  useEffect(() => {
    setApplyMessage('');
  }, [modalMode]);
  useEffect(() => {
    if (modalMode !== 'qualitative') return;
    let cancelled = false;
    setVocabListMessage('Loading vocabularies...');
    void (async () => {
      try {
        const next = await listVocabularies();
        if (cancelled) return;
        setAvailableVocabularies(next);
        setVocabOptions((current) => Array.from(new Set([...current, ...next, ...draft.qualitative_vocab_identifiers])));
        setSelectedVocabIdentifier((current) => current || next.find((identifier) => !draft.qualitative_vocab_identifiers.includes(identifier)) || '');
        setVocabListMessage('');
      } catch (error) {
        if (!cancelled) setVocabListMessage(error instanceof Error ? error.message : 'Could not load vocabularies.');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [modalMode]);
  const setNumber = (key: keyof ExtractionVocabQueryConfig, value: string) => {
    const parsed = Number(value);
    setDraft((current) => ({ ...current, [key]: Number.isFinite(parsed) ? parsed : 0 }));
  };
  const validateDraft = () => {
    for (const field of VOCAB_QUERY_NUMERIC_FIELDS) {
      const value = draft[field.key];
      const min = Number(field.min ?? '1');
      if (typeof value !== 'number' || !Number.isFinite(value) || value < min) {
        return `${field.label} must be ${field.min ?? '1'} or greater.`;
      }
      if ((field.step ?? '1') === '1' && !Number.isInteger(value)) {
        return `${field.label} must be a whole number.`;
      }
    }
    return '';
  };
  const applyDraft = async () => {
    if (!onApply) return;
    const validationError = validateDraft();
    if (validationError) {
      setApplyMessage(validationError);
      return;
    }
    setApplyMessage('');
    try {
      await onApply(draft);
      prevConfigRef.current = normalizeVocabQueryConfig(draft);
      setModalMode(null);
    } catch (error) {
      setApplyMessage(error instanceof Error ? error.message : 'Failed to update vocabulary query configuration.');
    }
  };
  const toggleVocab = (identifier: string, checked: boolean) => {
    setDraft((current) => ({
      ...current,
      qualitative_vocab_identifiers: checked
        ? Array.from(new Set([...current.qualitative_vocab_identifiers, identifier]))
        : current.qualitative_vocab_identifiers.filter((item) => item !== identifier),
    }));
  };
  const addVocabOption = () => {
    const identifier = selectedVocabIdentifier.trim();
    if (!identifier) return;
    setVocabOptions((current) => Array.from(new Set([...current, identifier])));
    setDraft((current) => ({
      ...current,
      qualitative_vocab_identifiers: Array.from(new Set([...current.qualitative_vocab_identifiers, identifier])),
    }));
    setSelectedVocabIdentifier('');
  };
  const addableVocabularies = availableVocabularies.filter((identifier) => !draft.qualitative_vocab_identifiers.includes(identifier));
  const renderNumberFields = (fields: VocabQueryNumericField[]) => fields.map((field) => (
    <label key={field.key}>
      <span>{field.label}</span>
      <input
        type="number"
        step={field.step ?? '1'}
        min={field.min ?? '1'}
        value={String(draft[field.key] ?? '')}
        onChange={(event) => setNumber(field.key, event.target.value)}
      />
    </label>
  ));
  const qualitativeControls = (
    <section className="vocab-query-config-section">
      <div>
        <span>Qualitative attribute queries</span>
        <strong>Vocabulary terms for descriptive attributes</strong>
      </div>
      <div className="vocab-query-config-grid">
        <section className="vocab-query-vocab-list">
          <span>Qualitative vocabularies</span>
          <div>
            {vocabOptions.map((identifier) => (
              <label key={identifier}>
                <input
                  type="checkbox"
                  checked={draft.qualitative_vocab_identifiers.includes(identifier)}
                  onChange={(event) => toggleVocab(identifier, event.target.checked)}
                />
                <span>{identifier}</span>
              </label>
            ))}
          </div>
          <div className="vocab-query-add-vocab">
            <select value={selectedVocabIdentifier} onChange={(event) => setSelectedVocabIdentifier(event.target.value)}>
              <option value="">Select vocabulary to add</option>
              {addableVocabularies.map((identifier) => (
                <option key={identifier} value={identifier}>{identifier}</option>
              ))}
            </select>
            <button className="ghost small" type="button" disabled={!selectedVocabIdentifier} onClick={addVocabOption}>Add</button>
          </div>
          {vocabListMessage && <p className="muted vocab-query-config-message">{vocabListMessage}</p>}
        </section>
        {renderNumberFields(QUALITATIVE_VOCAB_QUERY_FIELDS)}
        <label>
          <span>Traversal</span>
          <select
            value={draft.traversal_direction}
            onChange={(event) => setDraft((current) => ({ ...current, traversal_direction: event.target.value }))}
          >
            <option value="undirected">Undirected</option>
            <option value="outgoing">Outgoing</option>
            <option value="incoming">Incoming</option>
          </select>
        </label>
      </div>
    </section>
  );
  const quantitativeControls = (
    <section className="vocab-query-config-section">
      <div>
        <span>Quantitative attribute queries</span>
        <strong>Quantity-kind and unit candidate retrieval</strong>
      </div>
      <div className="vocab-query-config-grid">
        <section className="vocab-query-vocab-list vocab-query-fixed-vocabs">
          <span>Quantitative vocabularies</span>
          <div>
            {QUANTITATIVE_ATTRIBUTE_VOCABULARIES.map((vocabulary) => (
              <label key={vocabulary.identifier}>
                <input type="checkbox" checked readOnly />
                <span>{vocabulary.label}: {vocabulary.identifier}</span>
              </label>
            ))}
          </div>
        </section>
        {renderNumberFields(QUANTITATIVE_VOCAB_QUERY_FIELDS)}
        <label>
          <span>Traversal</span>
          <select
            value={draft.quantitative_traversal_direction}
            onChange={(event) => setDraft((current) => ({ ...current, quantitative_traversal_direction: event.target.value }))}
          >
            <option value="undirected">Undirected</option>
            <option value="outgoing">Outgoing</option>
            <option value="incoming">Incoming</option>
          </select>
        </label>
      </div>
    </section>
  );
  return (
    <>
      <section className="vocab-query-config-panel compact">
        <div>
          <span>Vocabulary query configuration</span>
          <strong>Run-wide parameters</strong>
        </div>
        <div>
          <button className="ghost small" type="button" disabled={disabled} onClick={() => setModalMode('qualitative')}>Configure qualitative</button>
          <button className="ghost small" type="button" disabled={disabled} onClick={() => setModalMode('quantitative')}>Configure quantitative</button>
          <button className="small" type="button" disabled={disabled || !onRerunAll} onClick={() => onRerunAll?.()}>Rerun vocabulary queries</button>
        </div>
      </section>
      {modalMode && createPortal(
        <div className="vocab-dialog-overlay" onClick={() => setModalMode(null)}>
          <div className="vocab-dialog vocab-query-config-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <div>
                <span>{modalMode === 'qualitative' ? 'Qualitative query configuration' : 'Quantitative query configuration'}</span>
                <strong>{modalMode === 'qualitative' ? 'Descriptive attribute parameters' : 'Quantity-kind and unit parameters'}</strong>
              </div>
              <button className="ghost" type="button" onClick={() => setModalMode(null)}>Close</button>
            </div>
            <div className="vocab-dialog-body vocab-query-config-body">
              <div className="vocab-query-config-sections">
                {modalMode === 'qualitative' ? qualitativeControls : quantitativeControls}
              </div>
              <div className="vocab-query-config-actions">
                <button
                  className="ghost"
                  type="button"
                  disabled={disabled || !onApply}
                  onClick={() => void applyDraft()}
                >
                  Apply
                </button>
                <button type="button" disabled={disabled || !onRerunAll} onClick={() => onRerunAll?.()}>Rerun vocabulary queries</button>
              </div>
              {applyMessage && <p className="warning vocab-query-config-message">{applyMessage}</p>}
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

function VocabQueryTraceList({
  queries,
  onRerun,
  onSelectCandidate,
  onMarkUnresolved,
}: {
  queries: ExtractionVocabQueryRecord[];
  onRerun?: (queryId: string) => void;
  onSelectCandidate?: (query: ExtractionVocabQueryRecord, uri: string, title?: string | null) => void;
  onMarkUnresolved?: (query: ExtractionVocabQueryRecord) => void;
}) {
  if (!queries.length) return null;
  const completed = queries.filter((query) => query.status === 'completed').length;
  const failed = queries.filter((query) => query.status === 'failed').length;
  const running = queries.filter((query) => query.status === 'running').length;
  const quantitativeQueries = queries.filter((query) => query.kind === 'quantity_kind' || query.kind === 'unit');
  const qualitativeQueries = queries.filter((query) => query.kind === 'qualitative_attribute');
  const objectGroundingQueries = queries.filter((query) => query.kind === 'object_grounding');
  const otherQueries = queries.filter((query) => !quantitativeQueries.includes(query) && !qualitativeQueries.includes(query) && !objectGroundingQueries.includes(query));
  const queryGroups = [
    { key: 'object-grounding', title: 'Object grounding', description: 'voc4cat term candidates', queries: objectGroundingQueries },
    { key: 'quantitative', title: 'Quantitative queries', description: 'Quantity kinds and units', queries: quantitativeQueries },
    { key: 'qualitative', title: 'Qualitative queries', description: 'Descriptive attribute terms', queries: qualitativeQueries },
    ...(otherQueries.length ? [{ key: 'other', title: 'Other queries', description: 'Additional vocabulary lookups', queries: otherQueries }] : []),
  ];
  const queryStatusSummary = (groupQueries: ExtractionVocabQueryRecord[]) => {
    const groupCompleted = groupQueries.filter((query) => query.status === 'completed').length;
    const groupRunning = groupQueries.filter((query) => query.status === 'running').length;
    const groupFailed = groupQueries.filter((query) => query.status === 'failed').length;
    return `${groupCompleted}/${groupQueries.length} completed${groupRunning ? `, ${groupRunning} running` : ''}${groupFailed ? `, ${groupFailed} failed` : ''}`;
  };
  const renderQuery = (query: ExtractionVocabQueryRecord) => {
    const jsonPath = sourceContextJsonPointer(query);
    const candidateSeeds = query.result ? coerceVocabQueryResult(query.result)?.seeds.slice(0, 5) ?? [] : [];
    const canCurateField = Boolean(jsonPath && query.kind.startsWith('profile_'));
    return (
    <details className={`chunk-vocab-query ${query.status}`} key={query.query_id}>
      <summary>
        <div>
          <strong>{formatExtractionStage(query.kind)}</strong>
          <small>{query.source_value} · {query.vocabulary_identifier} · {query.rdf_type}</small>
        </div>
        <span>{query.status}</span>
      </summary>
      <div className="chunk-vocab-query-body">
        <div className="chunk-call-meta">
          {query.duration_ms ? <span>{formatDuration(query.duration_ms)} query time</span> : null}
          {query.result ? <span>{Object.keys(asRecord(query.result.resources) ?? {}).length} resources</span> : null}
          {jsonPath ? <span>{jsonPath}</span> : null}
          <button className="small ghost" type="button" disabled={!onRerun} onClick={() => onRerun?.(query.query_id)}>Rerun query</button>
        </div>
        {canCurateField ? (
          <div className="vocab-curation-actions">
            {candidateSeeds.map((seed) => {
              const title = query.result ? vocabResourceTitle(query.result, seed.uri) : null;
              return (
                <button
                  className="small ghost"
                  type="button"
                  key={seed.uri}
                  disabled={!onSelectCandidate}
                  onClick={() => onSelectCandidate?.(query, seed.uri, title)}
                  title={seed.uri}
                >
                  {title || seed.uri}
                </button>
              );
            })}
            <button
              className="small ghost"
              type="button"
              disabled={!onMarkUnresolved}
              onClick={() => onMarkUnresolved?.(query)}
            >
              Mark unresolved
            </button>
          </div>
        ) : null}
        {query.error && <p className="warning">{query.error}</p>}
        <JsonDetails title="Query input" value={query.query} />
        <JsonDetails title="Source context" value={query.source_context} />
        {query.result && <VocabQueryGraphPanel result={query.result} />}
        <JsonDetails title="Full query result" value={query.result ?? null} />
      </div>
    </details>
  );
  return (
    <section className="chunk-vocab-query-section">
      <div className="chunk-vocab-query-heading">
        <span>Vocabulary queries</span>
        <strong>{completed}/{queries.length} completed{running ? `, ${running} running` : ''}{failed ? `, ${failed} failed` : ''}</strong>
      </div>
      <div className="chunk-vocab-query-groups">
        {queryGroups.map((group) => (
          <section className={`chunk-vocab-query-group ${group.key}`} key={group.key}>
            <div className="chunk-vocab-query-group-heading">
              <div>
                <span>{group.description}</span>
                <strong>{group.title}</strong>
              </div>
              <small>{queryStatusSummary(group.queries)}</small>
            </div>
            {group.queries.length ? (
              <div className="chunk-vocab-query-list">
                {group.queries.map(renderQuery)}
              </div>
            ) : <p className="muted chunk-vocab-query-empty">No queries in this group.</p>}
          </section>
        ))}
      </div>
      {false && (
        <div className="chunk-vocab-query-list">
          {queries.map((query) => (
          <details className={`chunk-vocab-query ${query.status}`} key={query.query_id}>
            <summary>
              <div>
                <strong>{formatExtractionStage(query.kind)}</strong>
                <small>{query.source_value} · {query.vocabulary_identifier} · {query.rdf_type}</small>
              </div>
              <span>{query.status}</span>
            </summary>
            <div className="chunk-vocab-query-body">
              <div className="chunk-call-meta">
                {query.duration_ms ? <span>{formatDuration(query.duration_ms)} query time</span> : null}
                {query.result ? <span>{Object.keys(asRecord(query.result.resources) ?? {}).length} resources</span> : null}
                <button className="small ghost" type="button" disabled={!onRerun} onClick={() => onRerun?.(query.query_id)}>Rerun query</button>
              </div>
              {query.error && <p className="warning">{query.error}</p>}
              <JsonDetails title="Query input" value={query.query} />
              <JsonDetails title="Source context" value={query.source_context} />
              {query.result && <VocabQueryGraphPanel result={query.result} />}
              <JsonDetails title="Full query result" value={query.result ?? null} />
            </div>
          </details>
          ))}
        </div>
      )}
    </section>
  );
}

function VocabQueryTraceModal({
  title,
  queries,
  onClose,
  onRerun,
}: {
  title: string;
  queries: ExtractionVocabQueryRecord[];
  onClose: () => void;
  onRerun?: (queryId: string) => void;
}) {
  const completed = queries.filter((query) => query.status === 'completed').length;
  return createPortal(
    <div className="vocab-dialog-overlay" onClick={onClose}>
      <div className="vocab-dialog chunk-vocab-query-dialog" onClick={(event) => event.stopPropagation()}>
        <div className="vocab-dialog-header">
          <div>
            <span>Vocabulary queries</span>
            <strong>{title} - {completed}/{queries.length} completed</strong>
          </div>
          <button className="ghost" type="button" onClick={onClose}>Close</button>
        </div>
        <div className="vocab-dialog-body chunk-vocab-query-dialog-body">
          <VocabQueryTraceList queries={queries} onRerun={onRerun} />
        </div>
      </div>
    </div>,
    document.body,
    );
  };
}

export function DraftGroundingPanel({
  progress,
  disabled,
  onUpdateVocabQueryConfig,
  onRunGrounding,
  onRerunAllVocabQueries,
  onRerunVocabQuery,
  onSelectCandidate,
  onMarkUnresolved,
}: {
  progress?: WorkflowProgress | null;
  disabled?: boolean;
  onUpdateVocabQueryConfig?: (config: ExtractionVocabQueryConfig) => void;
  onRunGrounding?: () => void;
  onRerunAllVocabQueries?: () => void;
  onRerunVocabQuery?: (queryId: string) => void;
  onSelectCandidate?: (query: ExtractionVocabQueryRecord, uri: string, title?: string | null) => void;
  onMarkUnresolved?: (query: ExtractionVocabQueryRecord) => void;
}) {
  const queries = progress?.vocab_queries ?? [];
  const completed = queries.filter((query) => query.status === 'completed').length;
  const profileFields = queries.filter((query) => query.kind.startsWith('profile_'));
  return (
    <details className="draft-grounding-panel">
      <summary>
        <div>
          <span>Advanced grounding</span>
          <strong>{queries.length ? `${completed}/${queries.length} vocabulary queries completed` : 'Profile-path vocabulary grounding'}</strong>
        </div>
      </summary>
      <div className="draft-grounding-body">
        <div className="chunk-call-meta">
          <button className="small" type="button" disabled={disabled || !onRunGrounding} onClick={() => onRunGrounding?.()}>
            Run vocabulary grounding
          </button>
          <button className="small ghost" type="button" disabled={disabled || !queries.length || !onRerunAllVocabQueries} onClick={() => onRerunAllVocabQueries?.()}>
            Rerun vocabulary queries
          </button>
        </div>
        {progress?.vocab_query_config ? (
          <VocabQueryConfigPanel
            config={progress.vocab_query_config}
            disabled={disabled}
            onApply={onUpdateVocabQueryConfig}
            onRerunAll={onRerunAllVocabQueries}
          />
        ) : null}
        {queries.length ? (
          <VocabQueryTraceList
            queries={queries}
            onRerun={onRerunVocabQuery}
            onSelectCandidate={onSelectCandidate}
            onMarkUnresolved={onMarkUnresolved}
          />
        ) : (
          <p className="muted">No profile vocabulary queries have been generated yet.</p>
        )}
        {profileFields.length && profileFields.length !== queries.length ? (
          <p className="muted">{profileFields.length} query{profileFields.length === 1 ? '' : 'ies'} target profile fields.</p>
        ) : null}
      </div>
    </details>
  );
}

type VocabGraphNode = {
  uri: string;
  resource: VocabQueryResult['resources'][string] | null;
  seed: VocabQueryResult['seeds'][number] | null;
  x: number;
  y: number;
};

type VocabGraphEdge = {
  id: string;
  source: string;
  target: string;
  predicate: string;
};

const vocabGraphBaseViewBox = { x: 0, y: 0, width: 900, height: 440 };

function VocabQueryGraphPanel({ result }: { result: Record<string, unknown> }) {
  const vocabResult = useMemo(() => coerceVocabQueryResult(result), [result]);
  const graph = useMemo(() => vocabResult ? buildVocabGraph(vocabResult) : null, [vocabResult]);
  const [selectedUri, setSelectedUri] = useState(defaultVocabGraphSelection(graph));
  const [viewBox, setViewBox] = useState(vocabGraphBaseViewBox);
  const [dragging, setDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragStartRef = useRef<{ clientX: number; clientY: number; viewBox: typeof vocabGraphBaseViewBox; moved: boolean; targetUri: string | null } | null>(null);

  useEffect(() => {
    setSelectedUri((current) => graph?.nodes.some((node) => node.uri === current) ? current : defaultVocabGraphSelection(graph));
    setViewBox(vocabGraphBaseViewBox);
  }, [graph]);

  if (!graph?.nodes.length) return null;

  const selectedNode = graph.nodes.find((node) => node.uri === selectedUri) ?? graph.nodes[0];

  function zoomAt(clientX: number, clientY: number, scale: number) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    setViewBox((current) => {
      const nextWidth = Math.min(vocabGraphBaseViewBox.width / 0.7, Math.max(vocabGraphBaseViewBox.width / 2.2, current.width / scale));
      const nextHeight = Math.min(vocabGraphBaseViewBox.height / 0.7, Math.max(vocabGraphBaseViewBox.height / 2.2, current.height / scale));
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
    const target = event.target instanceof Element ? event.target.closest<SVGGElement>('.vocab-graph-node') : null;
    dragStartRef.current = { clientX: event.clientX, clientY: event.clientY, viewBox, moved: false, targetUri: target?.dataset.uri ?? null };
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
    if (drag && !drag.moved && drag.targetUri) {
      setSelectedUri(drag.targetUri);
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
    <details className="vocab-graph-details" open>
      <summary>Node graph</summary>
      <div className="vocab-graph-panel">
        <div className={`vocab-graph-canvas ${dragging ? 'dragging' : ''}`} role="img" aria-label="Vocabulary query graph" onWheel={onGraphWheel}>
          <svg
            ref={svgRef}
            viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
            onPointerDown={onGraphPointerDown}
            onPointerMove={onGraphPointerMove}
            onPointerUp={onGraphPointerUp}
            onPointerCancel={onGraphPointerUp}
          >
            <defs>
              <marker id="vocab-graph-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" />
              </marker>
            </defs>
            {graph.edges.map((edge) => {
              const source = graph.nodeMap.get(edge.source);
              const target = graph.nodeMap.get(edge.target);
              if (!source || !target) return null;
              const edgePath = directedEdgePath(source, target);
              const midX = (edgePath.x1 + edgePath.x2) / 2;
              const midY = (edgePath.y1 + edgePath.y2) / 2;
              const showLabel = edge.source === selectedNode.uri || edge.target === selectedNode.uri;
              return (
                <g key={edge.id} className="vocab-graph-edge">
                  <line x1={edgePath.x1} y1={edgePath.y1} x2={edgePath.x2} y2={edgePath.y2} />
                  {showLabel && <text x={midX} y={midY}>{compactPredicate(edge.predicate)}</text>}
                </g>
              );
            })}
            {graph.nodes.map((node) => {
              const selected = node.uri === selectedNode.uri;
              const radius = vocabGraphNodeRadius(node);
              const labelLines = vocabNodeLabelLines(node, radius);
              return (
                <g
                  key={node.uri}
                  className={`vocab-graph-node ${node.seed ? 'seed' : ''} ${selected ? 'selected' : ''}`}
                  data-uri={node.uri}
                  transform={`translate(${node.x} ${node.y})`}
                  onClick={() => setSelectedUri(node.uri)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setSelectedUri(node.uri);
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
          <div className="vocab-graph-controls" aria-label="Graph controls">
            <button className="ghost small" type="button" onClick={() => zoomAtCenter(1.18)}>+</button>
            <button className="ghost small" type="button" onClick={() => zoomAtCenter(1 / 1.18)}>-</button>
            <button className="ghost small" type="button" onClick={() => setViewBox(vocabGraphBaseViewBox)}>Fit</button>
          </div>
        </div>
        <aside className="vocab-graph-inspector">
          <div className="vocab-graph-inspector-heading">
            <span>Node properties</span>
            <strong>{vocabNodeTitle(selectedNode)}</strong>
          </div>
          <div className="vocab-graph-labels">
            {selectedNode.seed && <span>Seed</span>}
            {(selectedNode.resource?.rdf_types ?? []).map((rdfType) => <span key={rdfType}>{compactPredicate(rdfType)}</span>)}
          </div>
          <dl>
            <div>
              <dt>uri</dt>
              <dd>{selectedNode.uri}</dd>
            </div>
            {selectedNode.seed && (
              <>
                <div>
                  <dt>rrf score</dt>
                  <dd>{formatScore(selectedNode.seed.rrf_score)}</dd>
                </div>
                <div>
                  <dt>vector rank</dt>
                  <dd>{selectedNode.seed.vector_rank ?? '-'}</dd>
                </div>
                <div>
                  <dt>fulltext rank</dt>
                  <dd>{selectedNode.seed.fulltext_rank ?? '-'}</dd>
                </div>
              </>
            )}
            {Object.entries(selectedNode.resource?.properties ?? {}).map(([key, value]) => (
              <div key={key}>
                <dt>{compactPredicate(key)}</dt>
                <dd>{propertyPreview(value)}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </div>
    </details>
  );
}

export function ProjectionWorkflowPanel({
  progress,
  status,
  ledger,
  tokenUsageSummary,
}: {
  progress?: WorkflowProgress | null;
  status?: WorkflowTaskStatus | null;
  ledger: ProjectionLedgerRecord[];
  tokenUsageSummary?: ReactNode;
}) {
  const [hoveredRequirement, setHoveredRequirement] = useState<{ requirement: RequirementReportItem; x: number; y: number } | null>(null);
  const [selectedRequirement, setSelectedRequirement] = useState<RequirementReportItem | null>(null);
  const evidenceNoteTotal = evidenceContextNoteCount(progress?.interim_evidence_context);
  const projectionRunning = (progress?.stage || status) === 'profile_projection';
  const groupLedger = ledger.filter((record) => record.object_kind === 'InstanceProjectionGroup');
  const pendingGroupRecords = groupLedger.filter((record) => record.status === 'pending');
  const runningRecords = ledger.filter((record) => record.status === 'running');
  const completedLedger = ledger.filter((record) => record.status !== 'running' && record.status !== 'pending');
  const completedGroupRecords = groupLedger.filter((record) => record.status !== 'running' && record.status !== 'pending');
  const processedEvidenceIds = new Set(completedLedger.flatMap((record) => record.evidence_note_identifiers ?? []));
  const processedCount = processedEvidenceIds.size || completedLedger.length;
  const totalCount = Math.max(evidenceNoteTotal, processedCount);
  const pendingCount = Math.max(0, totalCount - processedCount);
  const groupTotal = groupLedger.length;
  const groupDone = completedGroupRecords.length;
  const groupPercent = groupTotal ? Math.min(100, Math.round((groupDone / groupTotal) * 100)) : projectionRunning ? 8 : 0;
  const activeRecord = runningRecords[0] ?? pendingGroupRecords[0] ?? null;
  const projectedRecords = ledger.filter((record) => record.status === 'projected');
  const attentionRecords = ledger.filter((record) => record.status !== 'projected');
  const generatedInstances = draftClassInstances(progress?.generated_final_draft);
  const requirementReport = progress?.requirement_report ?? null;
  const requirementSummary = requirementStatusSummary(requirementReport);
  const applicableRequirements = requirementReport?.semantic_requirements.filter((requirement) => requirement.applicable) ?? [];
  const coverageFilled = requirementReport?.coverage.filled_fields ?? requirementReport?.coverage_score ?? 0;
  const coverageTotal = requirementReport?.coverage.total_fields ?? 0;
  const draftSteps = profileDraftSteps(progress, status);
  const activeDraftStep = draftSteps.find((step) => step.status === 'active');
  const completedDraftSteps = draftSteps.filter((step) => step.status === 'completed').length;
  const draftHeadline = activeDraftStep?.title
    ?? (completedDraftSteps === 3
      ? 'Complete'
      : draftSteps.some((step) => ['skipped', 'failed', 'cancelled'].includes(step.status))
        ? 'Incomplete'
        : 'Not started');
  const draftActivity = profileDraftActivity(progress?.stage, activeDraftStep?.title, draftSteps);
  const showRequirementPopover = (event: SyntheticEvent<HTMLElement>, requirement: RequirementReportItem) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setHoveredRequirement({ requirement, x: rect.left, y: rect.bottom + 8 });
  };

  return (
    <section className="projection-workflow-panel">
      <section className="draft-stage-progress" aria-label="Draft creation progress">
        <div className="draft-stage-progress-heading">
          <div>
            <span>Draft creation</span>
            <strong>{draftHeadline}</strong>
          </div>
          <small>{completedDraftSteps}/3 completed</small>
        </div>
        <div className="draft-stage-list">
          {draftSteps.map((step) => (
            <article
              key={step.number}
              className={`draft-stage-item ${step.status}`}
              aria-current={step.status === 'active' ? 'step' : undefined}
            >
              <span className="draft-stage-number">#{step.number}</span>
              <div>
                <strong>{step.title}</strong>
                <small>{step.description}</small>
              </div>
              <span className="draft-stage-status">{formatExtractionStage(step.status)}</span>
            </article>
          ))}
        </div>
        <p className={`draft-stage-activity ${activeDraftStep ? 'active' : ''}`}>{draftActivity}</p>
      </section>
      <div className="projection-workflow-heading">
        <div>
          <span>Projection ledger</span>
          <strong>{formatExtractionStage(progress?.stage || status || 'not started')}</strong>
        </div>
        <div>
          <span>{groupTotal ? `${groupDone}/${groupTotal} groups` : `${processedCount}/${totalCount || 0}`}</span>
          <strong>{groupTotal ? `${Math.max(0, groupTotal - groupDone)} queued` : `${pendingCount} pending`}</strong>
        </div>
      </div>
      <div className={`patch-progress-track ${projectionRunning && !groupDone ? 'indeterminate' : ''}`} aria-hidden="true"><div style={{ width: `${groupPercent}%` }} /></div>
      {projectionRunning && !groupTotal && (
        <p className="projection-sink-note active">
          Projection engine running. Preparing group plan; first update arrives after planner publishes queued groups.
        </p>
      )}
      {activeRecord && (
        <p className="projection-sink-note active">
          {activeRecord.status === 'running' ? 'Building' : 'Next'} {activeRecord.target_class || 'profile'} at <code>{activeRecord.target_path || '/'}</code>
        </p>
      )}
      {groupTotal > 0 && (
        <p className="projection-sink-note compact">
          Evidence coverage: {processedCount}/{totalCount || 0} notes resolved, {pendingCount} pending.
        </p>
      )}
      {generatedInstances.length > 0 && (
        <section className="projection-class-summary" aria-label="Generated DCAT-AP+ class instances">
          <span>Generated DCAT-AP+ class instances</span>
          <div>
            {generatedInstances.map((item) => (
              <article key={item.className}>
                <strong>{item.className}</strong>
                <span>{item.count}</span>
                <small>{item.labels.join(', ')}</small>
              </article>
            ))}
          </div>
        </section>
      )}
      {requirementReport && (
        <section className="requirement-completeness-panel">
          <div className="requirement-score-heading">
            <div>
              <span>Filled fields</span>
              <strong>{coverageTotal ? `${coverageFilled}/${coverageTotal}` : coverageFilled}</strong>
            </div>
            <small>
              Semantic {formatPercentScore(requirementReport.semantic_requirements_score)} - Trace {formatPercentScore(requirementReport.source_trace_score)}
            </small>
          </div>
          <div className="requirement-status-strip">
            <span>{requirementReport.coverage_patches.filter((requirement) => requirement.status === 'fulfilled').length} coverage slots</span>
            <span>{requirementSummary.fulfilled ?? 0} fulfilled</span>
            <span>{requirementSummary.partial ?? 0} partial</span>
            <span>{requirementSummary.missing ?? 0} missing</span>
            <span>{requirementReport.source_trace.used_evidence_count} trace evidence</span>
          </div>
          <div className="requirement-breakdown">
            {applicableRequirements.map((requirement) => (
              <article
                key={requirement.requirement_id}
                className={`requirement-row ${requirement.status}`}
                role="button"
                tabIndex={0}
                onClick={() => setSelectedRequirement(requirement)}
                onMouseEnter={(event) => showRequirementPopover(event, requirement)}
                onMouseLeave={() => setHoveredRequirement(null)}
                onFocus={(event) => showRequirementPopover(event, requirement)}
                onBlur={() => setHoveredRequirement(null)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    setSelectedRequirement(requirement);
                  }
                }}
              >
                <div>
                  <strong>{requirement.label}</strong>
                  <span>{formatExtractionStage(requirement.status)}</span>
                </div>
                <small>
                  {requirementScoreLabel(requirement)}
                  {requirement.patch?.status && requirement.patch.status !== 'not_attempted' ? ` - patch ${formatExtractionStage(requirement.patch.status)}` : ''}
                </small>
              </article>
            ))}
          </div>
        </section>
      )}
      {tokenUsageSummary}
      {attentionRecords.length > 0 && (
        <details className="projection-issues">
          <summary>{attentionRecords.length} projection issue{attentionRecords.length === 1 ? '' : 's'} need review</summary>
          <div>
            {attentionRecords.slice(0, 12).map((record) => (
              <article key={record.object_identifier}>
                <div>
                  <strong>{record.object_identifier}</strong>
                  <span>{formatExtractionStage(record.object_kind)} - {formatExtractionStage(record.status)}{record.target_path ? ` - ${record.target_path}` : ''}</span>
                </div>
                <p>{projectionFailureMessage(record)}</p>
                {record.source_evidence && <small>{projectionEvidencePreview(record)}</small>}
              </article>
            ))}
          </div>
        </details>
      )}
      {hoveredRequirement && <RequirementReviewPopover {...hoveredRequirement} />}
      {selectedRequirement && <RequirementReviewDialog requirement={selectedRequirement} onClose={() => setSelectedRequirement(null)} />}
    </section>
  );
}

type DraftStageStatus = 'pending' | 'active' | 'completed' | 'skipped' | 'failed' | 'cancelled';

type DraftStageItem = {
  number: number;
  title: string;
  description: string;
  status: DraftStageStatus;
};

function profileDraftSteps(
  progress?: WorkflowProgress | null,
  status?: WorkflowTaskStatus | null,
): DraftStageItem[] {
  const stage = progress?.stage ?? '';
  const initialComplete = Boolean(
    progress?.generated_initial_draft
    || progress?.generated_patched_draft
    || progress?.generated_reconstructed_draft
    || (stage === 'profile_draft' && progress?.generated_final_draft),
  );
  const patchingComplete = Boolean(progress?.generated_patched_draft || progress?.generated_reconstructed_draft);
  const reconstructionComplete = Boolean(progress?.generated_reconstructed_draft);
  const activeIndex = stage === 'profile_projection'
    ? 0
    : ['evidence_patching', 'description_mining', 'coverage_scoring'].includes(stage)
      ? 1
      : ['semantic_evaluation', 'semantic_reconstruction', 'semantic_revalidation'].includes(stage)
        ? 2
        : -1;
  const runEnded = stage === 'profile_draft' || status === 'completed';
  const runFailed = status === 'crashed';
  const runCancelled = status === 'cancelled';
  const completed = [initialComplete, patchingComplete, reconstructionComplete];
  const definitions = [
    {
      title: 'Initial draft creation',
      description: 'Create the profile structure and initial DCAT-AP+ class instances.',
    },
    {
      title: 'Evidence patching',
      description: 'Mine grounded facts, score coverage, and patch missing profile fields.',
    },
    {
      title: 'Semantic reconstruction',
      description: 'Evaluate semantics and repair placement, ranges, and duplicate attributes.',
    },
  ];

  return definitions.map((definition, index) => {
    let stepStatus: DraftStageStatus = completed[index] ? 'completed' : 'pending';
    if (!completed[index] && index === activeIndex) {
      stepStatus = runFailed ? 'failed' : runCancelled ? 'cancelled' : 'active';
    } else if (!completed[index] && runEnded) {
      stepStatus = 'skipped';
    }
    return {
      number: index + 1,
      ...definition,
      status: stepStatus,
    };
  });
}

function profileDraftActivity(stage: string | undefined, activeTitle: string | undefined, steps: DraftStageItem[]): string {
  if (stage === 'profile_draft') {
    const completedCount = steps.filter((step) => step.status === 'completed').length;
    return completedCount === 3
      ? 'Draft creation finished. The initial, patched, and reconstructed artifacts are available for inspection.'
      : `Draft creation ended with ${completedCount}/3 substeps complete. Skipped substeps did not produce boundary artifacts.`;
  }
  const detailByStage: Record<string, string> = {
    profile_projection: 'Building the initial profile structure and class instances from accumulated evidence.',
    evidence_patching: 'Initial draft saved. Preparing the evidence-backed coverage pass.',
    description_mining: 'Mining dataset descriptions for additional grounded facts.',
    coverage_scoring: 'Checking profile coverage and applying evidence-backed patches to missing fields.',
    semantic_evaluation: 'Evaluating semantic requirements before reconstructing the draft.',
    semantic_reconstruction: 'Applying semantic placement, range, and coherence repairs.',
    semantic_revalidation: 'Re-evaluating the reconstructed draft and compiling the final requirement report.',
  };
  return detailByStage[stage ?? ''] ?? (activeTitle ? `${activeTitle} is running.` : 'Draft creation has not started.');
}

function buildVocabGraph(result: VocabQueryResult) {
  const seedMap = new Map(result.seeds.map((seed) => [seed.uri, seed]));
  const edgeUris = new Set<string>();
  const edges: VocabGraphEdge[] = result.graph_statements.map((statement, index) => {
    edgeUris.add(statement.subject_uri);
    edgeUris.add(statement.object_uri);
    return {
      id: `${statement.subject_uri}-${statement.predicate}-${statement.object_uri}-${index}`,
      source: statement.subject_uri,
      target: statement.object_uri,
      predicate: statement.predicate,
    };
  });
  const uris = Array.from(new Set([
    ...result.seeds.map((seed) => seed.uri),
    ...Object.keys(result.resources),
    ...edgeUris,
  ]));
  const seedUris = new Set(result.seeds.map((seed) => seed.uri));
  const centerX = 450;
  const centerY = 220;
  const seedCount = Math.max(1, result.seeds.length);
  const nodes: VocabGraphNode[] = uris.map((uri, index) => {
    const seedIndex = result.seeds.findIndex((seed) => seed.uri === uri);
    const isSeed = seedIndex >= 0;
    const nonSeedIndex = Math.max(0, index - result.seeds.length);
    const angle = isSeed
      ? ((Math.PI * 2) / seedCount) * seedIndex - Math.PI / 2
      : ((Math.PI * 2) / Math.max(1, uris.length - result.seeds.length)) * nonSeedIndex - Math.PI / 2;
    const radius = isSeed ? (seedCount === 1 ? 0 : 86) : 132 + (nonSeedIndex % 3) * 52;
    return {
      uri,
      resource: result.resources[uri] ?? null,
      seed: seedMap.get(uri) ?? null,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };
  }).sort((left, right) => Number(seedUris.has(left.uri)) - Number(seedUris.has(right.uri)));
  const nodeMap = new Map(nodes.map((node) => [node.uri, node]));
  return { nodes, edges, nodeMap };
}

function defaultVocabGraphSelection(graph: ReturnType<typeof buildVocabGraph> | null): string {
  return graph?.nodes.find((node) => node.seed)?.uri ?? graph?.nodes[0]?.uri ?? '';
}

function vocabGraphNodeRadius(node: VocabGraphNode): number {
  return node.seed ? 22 : 17;
}

function directedEdgePath(source: VocabGraphNode, target: VocabGraphNode) {
  const deltaX = target.x - source.x;
  const deltaY = target.y - source.y;
  const length = Math.hypot(deltaX, deltaY) || 1;
  const unitX = deltaX / length;
  const unitY = deltaY / length;
  const sourceRadius = vocabGraphNodeRadius(source) + 2;
  const targetRadius = vocabGraphNodeRadius(target) + 7;
  return {
    x1: source.x + unitX * sourceRadius,
    y1: source.y + unitY * sourceRadius,
    x2: target.x - unitX * targetRadius,
    y2: target.y - unitY * targetRadius,
  };
}

function coerceVocabQueryResult(value: Record<string, unknown>): VocabQueryResult | null {
  const resources = asRecord(value.resources);
  if (!resources) return null;
  const seeds = asRecordArray(value.seeds).map((seed) => ({
    uri: String(seed.uri || ''),
    rdf_type: String(seed.rdf_type || ''),
    rrf_score: typeof seed.rrf_score === 'number' ? seed.rrf_score : 0,
    vector_score: typeof seed.vector_score === 'number' ? seed.vector_score : null,
    vector_rank: typeof seed.vector_rank === 'number' ? seed.vector_rank : null,
    fulltext_score: typeof seed.fulltext_score === 'number' ? seed.fulltext_score : null,
    fulltext_rank: typeof seed.fulltext_rank === 'number' ? seed.fulltext_rank : null,
  })).filter((seed) => seed.uri);
  const graphStatements = asRecordArray(value.graph_statements).map((statement) => ({
    subject_uri: String(statement.subject_uri || ''),
    predicate: String(statement.predicate || ''),
    object_uri: String(statement.object_uri || ''),
  })).filter((statement) => statement.subject_uri && statement.object_uri);
  const typedResources = Object.fromEntries(Object.entries(resources).map(([uri, resourceValue]) => {
    const resource = asRecord(resourceValue);
    const properties = asRecord(resource?.properties) ?? {};
    return [uri, {
      uri: String(resource?.uri || uri),
      rdf_types: Array.isArray(resource?.rdf_types) ? resource.rdf_types.map(String) : [],
      properties,
    }];
  }));
  return {
    identifier: String(value.identifier || ''),
    rdf_type: String(value.rdf_type || ''),
    seeds,
    graph_statements: graphStatements,
    resources: typedResources,
  };
}

function vocabNodeLabelLines(node: VocabGraphNode, radius: number): string[] {
  const title = vocabNodeTitle(node);
  const words = title.split(/\s+/).filter(Boolean);
  const maxChars = radius >= 22 ? 9 : 7;
  if (!words.length) return [trimNodeLabel(compactPredicate(node.uri), maxChars)];

  const lines: string[] = [];
  let current = '';
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars) {
      current = next;
      continue;
    }
    if (current) lines.push(current);
    current = word;
    if (lines.length === 2) break;
  }
  if (current && lines.length < 2) lines.push(current);
  return lines.slice(0, 2).map((line) => trimNodeLabel(line, maxChars));
}

function trimNodeLabel(value: string, maxChars: number): string {
  return value.length > maxChars ? `${value.slice(0, Math.max(1, maxChars - 2))}..` : value;
}

function vocabNodeTitle(node: VocabGraphNode): string {
  const label = findVocabPropertyLabel(node.resource);
  return label || compactPredicate(node.uri);
}

function findVocabPropertyLabel(resource: VocabQueryResult['resources'][string] | null): string | null {
  if (!resource) return null;
  const labelKeys = ['preflabel', 'label', 'title', 'name'];
  for (const [key, value] of Object.entries(resource.properties)) {
    const normalized = key.toLowerCase().replace(/[_:-]/g, '');
    if (!labelKeys.some((labelKey) => normalized.endsWith(labelKey))) continue;
    if (typeof value === 'string' && value.trim()) return value;
    if (Array.isArray(value)) {
      const first = value.find((item) => typeof item === 'string' && item.trim());
      if (typeof first === 'string') return first;
    }
  }
  return null;
}

function compactPredicate(value: string): string {
  const hash = value.lastIndexOf('#');
  const slash = value.lastIndexOf('/');
  const colon = value.lastIndexOf(':');
  const index = Math.max(hash, slash, colon);
  return index >= 0 ? value.slice(index + 1) || value : value;
}

function propertyPreview(value: unknown): string {
  if (Array.isArray(value)) {
    const preview = value.slice(0, 4).map(propertyPreview).join(', ');
    return value.length > 4 ? `${preview}, ... ${value.length - 4} more` : preview || '-';
  }
  if (value && typeof value === 'object') return JSON.stringify(value);
  if (value === null || value === undefined || value === '') return '-';
  const text = String(value);
  return text.length > 360 ? `${text.slice(0, 360)}...` : text;
}

function formatScore(value?: number | null): string {
  return typeof value === 'number' ? value.toFixed(4) : '-';
}


