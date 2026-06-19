import { type FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { chunkDataPackage, deleteDataPackage, getChunkStatus, getDataPackageChunks, getFileEntryContent, listDataPackages, uploadDataPackage, type ChunkingStrategy } from './api/datasources';
import {
  applyCurationFieldAction,
  getExtractionResult,
  getInitialContextProgress,
  getPatchProgress,
  getTokenUsage,
  initialContextFromEvidenceContext,
  pauseExtraction,
  rerunAllVocabQueries,
  rerunVocabQuery,
  runExtraction,
  runInitialContext,
  runVocabularyGrounding,
  saveCuratedDocument,
  updateVocabQueryConfig,
} from './api/extraction';
import { deleteProfile, getProfileJsonSchema, listProfiles, registerProfile } from './api/profiles';
import {
  getLlmBudget,
  getOllamaConfig,
  pullOllamaModel,
  removeOllamaModel,
  runOllamaPerformanceTest,
  updateOllamaRuntimeConfig,
  type LlmBudget,
  type OllamaConfig,
  type OllamaPerformanceTest,
} from './api/system';
import { JsonEditor, type JsonPatchMarker, type JsonSchemaDocument } from './components/JsonEditor';
import { ChunkingDialog } from './components/ChunkingDialog';
import { ChunkingStatusPanel, ChunkInspectionPanel, FileViewer } from './components/ChunkPanels';
import { VocabularyPanel } from './components/VocabularyPanel';
import { WorkflowBranchBar, StepPanel, type WorkflowBranch } from './components/WorkflowPanels';
import { JsonDetails } from './components/JsonDetails';
import { InitialFileUnderstandingPanel } from './components/InitialUnderstandingPanels';
import { EvidenceContextOverview, evidenceContextNoteCount } from './components/EvidenceContextPanels';
import { TokenUsageSummary } from './components/TokenUsageSummary';
import { DraftGroundingPanel, ProjectionWorkflowPanel, sourceContextJsonPointer } from './components/VocabGroundingPanels';
import { OllamaSettingsPanel } from './components/OllamaSettingsPanel';
import { formatExtractionStage } from './lib/format';
import { asRecord, asRecordArray } from './lib/records';
import type { ChunkRequestResponse, ChunkResponse, DataPackageResponse, FileEntryResponse, InitialContext, ProfileManifestResponse, TextQualityConfig, VocabQueryResult } from './api/types';
import type {
  ExtractionChunkRef,
  ExtractionChunkResult,
  ExtractionRunResult,
  ExtractionVocabQueryConfig,
  ExtractionVocabQueryRecord,
  ChunkRepairMode,
  PatchProgress,
  PatchTaskStatus,
  PatchTokenUsage,
  PatchTokenUsageEntry,
  ExtractionOverview,
  ExtractionOverviewNode,
  ProjectionLedgerRecord,
  RequirementReport,
} from './api/extraction';

type BusyKey = 'upload' | 'initial-context' | 'chunk' | 'context' | 'pause' | 'draft' | 'patch' | 'load' | 'profile' | 'profile-delete' | 'dataset-delete' | 'ollama';
type WorkflowBranchSnapshot = {
  strategy: ChunkingStrategy;
  chunks: ChunkResponse[][];
  chunkStatus: { has_chunks: boolean; file_count: number; status: ChunkRequestResponse['status'] };
  status: PatchTaskStatus;
  progress: PatchProgress | null;
  initialProgress: PatchProgress | null;
  result: ExtractionRunResult | null;
  tokenUsage: PatchTokenUsage;
  initialStatus: PatchTaskStatus;
};
const selectedPackageStorageKey = 'simone_selected_package_id';
const chunkingStrategyCookieKey = 'simone_chunking_strategy';

function isChunkingStrategy(value: string): value is ChunkingStrategy {
  return value === 'semantic' || value === 'fixed_tokens';
}

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

function readStoredChunkingStrategy(): ChunkingStrategy {
  try {
    const cookie = document.cookie
      .split('; ')
      .find((part) => part.startsWith(`${chunkingStrategyCookieKey}=`));
    const value = cookie ? decodeURIComponent(cookie.split('=').slice(1).join('=')) : '';
    return isChunkingStrategy(value) ? value : 'semantic';
  } catch {
    return 'semantic';
  }
}

function persistChunkingStrategy(strategy: ChunkingStrategy) {
  try {
    document.cookie = `${chunkingStrategyCookieKey}=${encodeURIComponent(strategy)}; path=/; SameSite=Lax`;
  } catch {
    // ignore cookie errors
  }
}

function jsonPointerToEditorPath(path: string): string {
  if (!path || path === '/') return '';
  return path
    .replace(/^\//, '')
    .split('/')
    .map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'))
    .join('.');
}

function EditableContextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="context-field">
      <span>{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function EditableContextTextArea({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="context-field context-field-wide">
      <span>{label}</span>
      <textarea value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function PersistedContextField({
  label,
  value,
  wide = false,
}: {
  label: string;
  value?: string | number | null;
  wide?: boolean;
}) {
  return (
    <div className={`context-field context-field-readonly ${wide ? 'context-field-wide' : ''}`}>
      <span>{label}</span>
      <strong>{value || '-'}</strong>
    </div>
  );
}

function PersistedContextChips({ label, values }: { label: string; values: string[] }) {
  return (
    <div className="context-field context-field-wide context-field-readonly context-chips-field">
      <span>{label}</span>
      {values.length ? (
        <div className="chips">
          {values.map((value) => <span key={value}>{value}</span>)}
        </div>
      ) : (
        <strong>-</strong>
      )}
    </div>
  );
}

function withInitialProgress(branch: PatchProgress | null, initial: PatchProgress | null): PatchProgress | null {
  if (!branch) return initial;
  if (!initial) return branch;
  return {
    ...branch,
    ranked_files: branch.ranked_files?.length ? branch.ranked_files : initial.ranked_files,
    initial_file_summaries: branch.initial_file_summaries?.length ? branch.initial_file_summaries : initial.initial_file_summaries,
    initial_file_summary_progress: branch.initial_file_summary_progress ?? initial.initial_file_summary_progress,
    initial_file_summary_status: branch.initial_file_summary_status ?? initial.initial_file_summary_status,
    initial_extraction_overview: branch.initial_extraction_overview ?? initial.initial_extraction_overview,
    initial_extraction_overview_status: branch.initial_extraction_overview_status ?? initial.initial_extraction_overview_status,
    initial_extraction_overview_diagnostic: branch.initial_extraction_overview_diagnostic ?? initial.initial_extraction_overview_diagnostic,
    dataset_summary: branch.dataset_summary || initial.dataset_summary,
  };
}

function progressWithResult(progress: PatchProgress | null, result: ExtractionRunResult): PatchProgress {
  return {
    ...(progress ?? {
      stage: 'completed',
      processed_chunks: 0,
      total_chunks: 0,
      normalized_quantities: 0,
      normalized_qualitative_attributes: 0,
      warnings: [],
    }),
    stage: 'completed',
    interim_evidence_context: result.machine_evidence_context,
    generated_final_draft: result.generated_final_draft,
    curated_document: result.curated_document ?? result.generated_final_draft,
    generated_initial_draft: result.generated_initial_draft ?? progress?.generated_initial_draft ?? null,
    requirement_report: result.requirement_report ?? progress?.requirement_report ?? null,
    dataset_summary: result.dataset_summary ?? progress?.dataset_summary ?? '',
    draft_quality_state: result.draft_quality_state,
    validation: result.validation,
    curated_validation: result.curated_validation ?? null,
    projection_ledger: result.projection_ledger,
    field_completion_ledger: result.field_completion_ledger,
    curation_ledger: result.curation_ledger,
    warnings: result.warnings,
  };
}

function contextTechnique(context: InitialContext): string | null | undefined {
  return context.activities?.find((activity) => activity.technique)?.technique;
}

function contextActivityLabel(context: InitialContext): string | null {
  const labels = context.activities
    ?.map((activity) => activity.label || activity.technique)
    .filter(Boolean) as string[] | undefined;
  return labels?.length ? labels.join(', ') : null;
}

function contextAgentLabel(context: InitialContext): string | null {
  if (!context.agents?.length) return null;
  const labels = context.agents.map((agent) => agent.model ? `${agent.name} (${agent.model})` : agent.name).filter(Boolean);
  return labels.length ? labels.join(', ') : null;
}

function contextEntityLabel(context: InitialContext): string | null {
  const labels = context.entities?.map((entity) => entity.label).filter(Boolean) ?? [];
  return labels.length ? labels.join(', ') : null;
}

function commaList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function contextEntitiesInput(context: InitialContext): string {
  return (context.entities ?? []).map((entity) => entity.label).join(', ');
}

function updateContextDatasetTitle(context: InitialContext, value: string): InitialContext {
  return { ...context, dataset_title: value.trim() || null };
}

function updateContextTechnique(context: InitialContext, value: string): InitialContext {
  const activities = [...(context.activities ?? [])];
  const first = activities[0] ?? { label: null, technique: null, agent_names: [] };
  activities[0] = { ...first, technique: value.trim() || null };
  return { ...context, activities };
}

function updateContextAgentName(context: InitialContext, value: string): InitialContext {
  const agents = [...(context.agents ?? [])];
  const first = agents[0] ?? { name: '', role: 'unknown' as const };
  agents[0] = { ...first, name: value };
  return { ...context, agents };
}

function updateContextAgentModel(context: InitialContext, value: string): InitialContext {
  const agents = [...(context.agents ?? [])];
  const first = agents[0] ?? { name: '', role: 'unknown' as const };
  agents[0] = { ...first, model: value.trim() || null };
  return { ...context, agents };
}

function updateContextEntities(context: InitialContext, value: string): InitialContext {
  const labels = commaList(value);
  const entities = labels.map((label, index) => {
    const existing = context.entities?.[index];
    return existing ? { ...existing, label } : { label, role: 'unknown' as const };
  });
  return { ...context, entities };
}

function updateContextActivityLabel(context: InitialContext, value: string): InitialContext {
  const activities = [...(context.activities ?? [])];
  const first = activities[0] ?? { label: null, technique: null, agent_names: [] };
  activities[0] = { ...first, label: value.trim() || null };
  return { ...context, activities };
}

function updateContextDescription(context: InitialContext, value: string): InitialContext {
  return { ...context, dataset_description: value.trim() || null };
}

function updateContextKeywords(context: InitialContext, value: string): InitialContext {
  return { ...context, keywords: commaList(value) };
}

export function App() {
  const [packages, setPackages] = useState<DataPackageResponse[]>([]);
  const [profiles, setProfiles] = useState<ProfileManifestResponse[]>([]);
  const [selectedPackageId, setSelectedPackageId] = useState('');
  const [selectedProfile, setSelectedProfile] = useState('');
  const [chunkResult, setChunkResult] = useState<ChunkRequestResponse | null>(null);
  const [chunkViewStrategy, setChunkViewStrategy] = useState<ChunkingStrategy>(readStoredChunkingStrategy);
  const [hasChunks, setHasChunks] = useState(false);
  const [chunksByFile, setChunksByFile] = useState<ChunkResponse[][]>([]);
  const [viewingFile, setViewingFile] = useState<FileEntryResponse | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [context, setContext] = useState<InitialContext | null>(null);
  const [generatedFinalDraft, setGeneratedFinalDraft] = useState<Record<string, unknown> | null>(null);
  const [curatedDocument, setCuratedDocument] = useState<Record<string, unknown> | null>(null);
  const [patchStatus, setPatchStatus] = useState<PatchTaskStatus | null>(null);
  const [initialContextStatus, setInitialContextStatus] = useState<PatchTaskStatus | null>(null);
  const [patchProgress, setPatchProgress] = useState<PatchProgress | null>(null);
  const [tokenUsage, setTokenUsage] = useState<PatchTokenUsage | null>(null);
  const [llmBudget, setLlmBudget] = useState<LlmBudget | null>(null);
  const [ollamaConfig, setOllamaConfig] = useState<OllamaConfig | null>(null);
  const [activeProfileSchema, setActiveProfileSchema] = useState<JsonSchemaDocument | null>(null);
  const [busy, setBusy] = useState<BusyKey | null>('load');
  const [message, setMessage] = useState('Loading workspace.');
  const [railCollapsed, setRailCollapsed] = useState(true);
  const [chunkingDialogOpen, setChunkingDialogOpen] = useState(false);
  const [chunkRepairMode, setChunkRepairMode] = useState<ChunkRepairMode>('deferred');
  const [curatedEditorOpen, setCuratedEditorOpen] = useState(false);
  const [profileFormOpen, setProfileFormOpen] = useState(false);
  const [profileIdentifier, setProfileIdentifier] = useState('');
  const [profileTargetClass, setProfileTargetClass] = useState('Dataset');
  const [profileSourceMode, setProfileSourceMode] = useState<'url' | 'upload'>('url');
  const [profileSchemaUrl, setProfileSchemaUrl] = useState('');
  const [profileSchemaFile, setProfileSchemaFile] = useState<File | null>(null);
  const [profileVersion, setProfileVersion] = useState('');
  const [profileEnrichableFields, setProfileEnrichableFields] = useState('');
  const datasetUploadInputRef = useRef<HTMLInputElement | null>(null);
  const saveContextTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveCuratedDocumentTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const selectedPackageIdRef = useRef('');
  const workflowChatModel = ollamaConfig?.runtime.chat_model ?? null;
  const currentWorkflowBranch = useMemo(
    () => ({ strategy: chunkViewStrategy, chatModel: workflowChatModel }),
    [chunkViewStrategy, workflowChatModel],
  );

  const selectedPackage = useMemo(() => packages.find((item) => item.id === selectedPackageId) || null, [packages, selectedPackageId]);
  const selectedProfileManifest = useMemo(() => profiles.find((item) => item.identifier === selectedProfile) || null, [profiles, selectedProfile]);
  const isInitialContextStage = Boolean(
    patchProgress?.stage?.startsWith('initial_')
    || patchProgress?.stage === 'file_ranking',
  );
  const isInitialContextRunning = initialContextStatus === 'running';
  const isPatching = patchStatus === 'running' && initialContextStatus !== 'running' && !isInitialContextStage;
  const extractionLimitReachedChunkCount = useMemo(() => {
    const maxContextLength = llmBudget?.max_context_length ?? 0;
    if (maxContextLength <= 0) return 0;
    return (patchProgress?.chunk_results ?? []).filter((chunk) => (
      chunk.status === 'completed'
      && typeof chunk.context_tokens === 'number'
      && chunk.context_tokens >= maxContextLength
    )).length;
  }, [llmBudget?.max_context_length, patchProgress?.chunk_results]);
  const extractionTokenUsageNotes = useMemo(() => (
    extractionLimitReachedChunkCount > 0
      ? {
        chunk_extraction: {
          tone: 'danger' as const,
          message: `${extractionLimitReachedChunkCount} chunk${extractionLimitReachedChunkCount === 1 ? '' : 's'} reached 100% or more of the context token limit.`,
        },
      }
      : undefined
  ), [extractionLimitReachedChunkCount]);
  const extractionProgressPercent = patchProgress?.total_chunks
    ? Math.min(100, Math.round((patchProgress.processed_chunks / patchProgress.total_chunks) * 100))
    : 0;
  const extractionProgressLabel = patchProgress
    ? `${formatExtractionStage(patchProgress.stage)}${patchProgress.total_chunks ? ` - ${patchProgress.processed_chunks}/${patchProgress.total_chunks} chunks` : ''}`
    : '';
  const hasPersistedExtractionState = Boolean(
    patchProgress?.interim_evidence_context
    || patchProgress?.chunk_results?.some((chunk) => chunk.status === 'completed' || chunk.status === 'skipped' || chunk.evidence_context),
  );
  const hasInitialContextArtifacts = Boolean(
    patchProgress?.initial_file_summary_status
    && patchProgress?.initial_extraction_overview_status,
  );
  const extractionCanResume = Boolean(
    selectedPackageId
    && hasInitialContextArtifacts
    && !busy
    && !isPatching
    && (
      patchStatus === 'cancelled'
      || patchStatus === 'crashed'
      || (patchStatus === 'unknown' && hasPersistedExtractionState)
    ),
  );
  const projectionLedger = patchProgress?.projection_ledger ?? [];
  const hasProfileArtifacts = Boolean(generatedFinalDraft || curatedDocument);
  const isProfileBuildRunning = isPatching && (
    patchProgress?.stage === 'profile_projection'
    || patchProgress?.stage === 'profile_draft'
    || Boolean(patchProgress?.generated_final_draft)
  );
  const projectionEvidenceNoteTotal = evidenceContextNoteCount(patchProgress?.interim_evidence_context);
  const projectionHasPendingNotes = projectionEvidenceNoteTotal > projectionLedger.length;
  const projectionCanContinue = Boolean(
    selectedPackageId
    && selectedProfile
    && hasPersistedExtractionState
    && !busy
    && !isPatching
    && (
      projectionHasPendingNotes
      || patchProgress?.stage === 'profile_draft'
      || patchProgress?.stage === 'profile_projection'
    )
  );
  const projectedObjects = projectionLedger.filter((record) => record.status === 'projected').length;
  const notProjectedObjects = projectionLedger.filter((record) => record.status === 'not_projected' || record.status === 'ambiguous').length;
  const editRequiredObjects = projectionLedger.filter((record) => record.status === 'user_edit_required').length;
  const validationErrorCount = patchProgress?.validation?.errors?.length ?? 0;
  const fieldIssueCount = (patchProgress?.field_completion_ledger ?? []).filter((record) => record.issue_categories.length > 0).length;
  const curationMarkers = useMemo<JsonPatchMarker[]>(() => {
    const fieldMarkers = (patchProgress?.field_completion_ledger ?? []).flatMap((record) => {
      const markers: JsonPatchMarker[] = [];
      if (record.validation_status === 'missing') {
        markers.push({
          id: `missing:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'missing',
          label: 'Missing',
          detail: record.edit_needed_reason || 'Field needs a value.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      } else if (record.validation_status === 'invalid') {
        markers.push({
          id: `invalid:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'invalid',
          label: 'Invalid',
          detail: record.edit_needed_reason || 'Field does not satisfy the profile schema.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      }
      if (record.issue_categories.includes('non_enriched')) {
        markers.push({
          id: `non-enriched:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'non_enriched',
          label: 'Non-enriched',
          detail: record.edit_needed_reason || 'No vocabulary term was selected.',
          evidence: record.source_evidence,
          issues: record.issue_categories,
        });
      }
      if (record.enrichment_status === 'intentionally_unresolved') {
        markers.push({
          id: `unresolved:${record.json_path}`,
          path: jsonPointerToEditorPath(record.json_path),
          status: 'intentionally_unresolved',
          label: 'Unresolved',
          detail: 'Marked intentionally unresolved.',
          evidence: record.source_evidence,
        });
      }
      return markers;
    });
    const curationMarkers = (patchProgress?.curation_ledger ?? [])
      .filter((record) => record.status !== 'unchanged')
      .map((record): JsonPatchMarker => ({
        id: `curation:${record.status}:${record.json_path}`,
        path: jsonPointerToEditorPath(record.json_path),
        status: record.status === 'user_removed' ? 'user_removed' : record.status === 'user_selected_vocab_term' ? 'user_selected_vocab_term' : 'user_edited',
        label: record.status.replace(/_/g, ' '),
        detail: record.reason || undefined,
        evidence: record.source_evidence,
      }));
    return [...fieldMarkers, ...curationMarkers];
  }, [patchProgress?.field_completion_ledger, patchProgress?.curation_ledger]);

  async function refresh() {
    setBusy('load');
    try {
      const [nextPackages, nextProfiles, nextBudget, nextOllamaConfig] = await Promise.all([listDataPackages(), listProfiles(), getLlmBudget(), getOllamaConfig()]);
      setLlmBudget(nextBudget);
      setOllamaConfig(nextOllamaConfig);
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
    persistChunkingStrategy(chunkViewStrategy);
  }, [chunkViewStrategy]);

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
    return () => {
      if (saveContextTimeoutRef.current) clearTimeout(saveContextTimeoutRef.current);
      if (saveCuratedDocumentTimeoutRef.current) clearTimeout(saveCuratedDocumentTimeoutRef.current);
    };
  }, []);

  function resetPackageWorkflowState() {
    if (saveContextTimeoutRef.current) {
      clearTimeout(saveContextTimeoutRef.current);
      saveContextTimeoutRef.current = null;
    }
    if (saveCuratedDocumentTimeoutRef.current) {
      clearTimeout(saveCuratedDocumentTimeoutRef.current);
      saveCuratedDocumentTimeoutRef.current = null;
    }
    setChunkResult(null);
    setHasChunks(false);
    setChunksByFile([]);
    setViewingFile(null);
    setFileContent(null);
    setContext(null);
    setGeneratedFinalDraft(null);
    setCuratedDocument(null);
    setPatchStatus(null);
    setInitialContextStatus(null);
    setPatchProgress(null);
    setTokenUsage(null);
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
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
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
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
      setMessage('Dataset uploaded. Run initial file understanding next.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Upload failed.');
    } finally {
      if (datasetUploadInputRef.current) {
        datasetUploadInputRef.current.value = '';
      }
      setBusy(null);
    }
  }

  async function onChunk(params?: { replace_existing_chunks: boolean; buffer_window_size: number; semantic_chunking_threshold: number; chunking_strategy: 'semantic' | 'fixed_tokens'; fixed_tokens_per_chunk: number; min_tokens_per_chunk: number; max_tokens_per_chunk: number; protected_line_indices: Record<string, number[]>; text_quality_config: TextQualityConfig; embedding_num_gpu?: number }) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('chunk');
    try {
      const result = await chunkDataPackage({
        id: packageId,
        replace_existing_chunks: params?.replace_existing_chunks ?? false,
        buffer_window_size: params?.buffer_window_size,
        semantic_chunking_threshold: params?.semantic_chunking_threshold,
        chunking_strategy: params?.chunking_strategy,
        fixed_tokens_per_chunk: params?.fixed_tokens_per_chunk,
        min_tokens_per_chunk: params?.min_tokens_per_chunk,
        max_tokens_per_chunk: params?.max_tokens_per_chunk,
        protected_line_indices: params?.protected_line_indices,
        text_quality_config: params?.text_quality_config,
        embedding_num_gpu: params?.embedding_num_gpu,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      setChunkViewStrategy(params?.chunking_strategy ?? 'semantic');
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
        getChunkStatus(packageId, chunkViewStrategy),
        getDataPackageChunks(packageId, chunkViewStrategy),
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
        hasChunks ? getDataPackageChunks(selectedPackageId, chunkViewStrategy) : Promise.resolve([]),
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

  async function loadWorkflowBranch(packageId: string, strategy: ChunkingStrategy, chatModel: string | null): Promise<WorkflowBranchSnapshot> {
    const [initialRun, extractionRun, chunkStatus, chunks, result, tokenUsage] = await Promise.all([
      getInitialContextProgress(packageId),
      getPatchProgress(packageId, strategy, chatModel),
      getChunkStatus(packageId, strategy),
      getDataPackageChunks(packageId, strategy),
      getExtractionResult(packageId, strategy, chatModel),
      getTokenUsage(packageId, strategy, chatModel),
    ]);
    return {
      strategy,
      chunks,
      chunkStatus,
      initialStatus: initialRun.status,
      status: result ? 'completed' : extractionRun.status,
      progress: extractionRun.progress ?? null,
      initialProgress: initialRun.progress ?? null,
      result,
      tokenUsage: result?.token_usage ?? tokenUsage,
    };
  }

  function applyWorkflowBranchSnapshot(snapshot: WorkflowBranchSnapshot) {
    setHasChunks(snapshot.chunkStatus.has_chunks);
    setChunksByFile(snapshot.chunks);
    setChunkResult({ status: snapshot.chunkStatus.status, chunks: snapshot.chunks });
    setTokenUsage(snapshot.tokenUsage);
    setInitialContextStatus(snapshot.initialStatus);

    const mergedProgress = withInitialProgress(snapshot.progress, snapshot.initialProgress);
    if (snapshot.result) {
      const resultProgress = progressWithResult(mergedProgress, snapshot.result);
      setContext(initialContextFromEvidenceContext(snapshot.result.machine_evidence_context));
      setGeneratedFinalDraft(snapshot.result.generated_final_draft);
      setCuratedDocument(snapshot.result.curated_document ?? snapshot.result.generated_final_draft);
      setPatchStatus('completed');
      setPatchProgress(resultProgress);
      return;
    }

    setContext(snapshot.progress?.interim_evidence_context ? initialContextFromEvidenceContext(snapshot.progress.interim_evidence_context) : null);
    setGeneratedFinalDraft(snapshot.progress?.generated_final_draft ?? null);
    setCuratedDocument(snapshot.progress?.curated_document ?? snapshot.progress?.generated_final_draft ?? null);
    setPatchStatus(snapshot.status);
    setPatchProgress(mergedProgress);
  }

  async function onInitialContext(forceRerun = false) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('initial-context');
    try {
      if (forceRerun) {
        setContext(null);
        setGeneratedFinalDraft(null);
        setCuratedDocument(null);
      }
      const response = await runInitialContext({
        data_package_id: packageId,
        force_rerun: forceRerun,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(response.status);
      setInitialContextStatus(response.status);
      setPatchProgress(response.progress ? { ...response.progress } : null);
      setTokenUsage(await getTokenUsage(packageId, chunkViewStrategy, workflowChatModel));
      setMessage(
        response.status === 'running'
          ? 'Initial file understanding is running.'
          : 'Initial file understanding is complete.',
      );
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Initial file understanding failed.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onContext(options: { resume?: boolean } = {}) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      if (!hasInitialContextArtifacts) {
        setMessage('Run initial file understanding before chunk extraction.');
        return;
      }
      setContext(null);
      setGeneratedFinalDraft(null);
      setCuratedDocument(null);
      setPatchProgress(null);
      const response = await runExtraction({
        data_package_id: packageId,
        resume: options.resume,
        target_stage: 'context',
        chunking_strategy: chunkViewStrategy,
        chat_model: workflowChatModel,
        chunk_repair_mode: chunkRepairMode,
      });
      if (selectedPackageIdRef.current !== packageId) return;
      const nextContext = response.result?.machine_evidence_context || response.progress?.interim_evidence_context;
      if (nextContext) {
        setContext(initialContextFromEvidenceContext(nextContext));
      }
      if (response.result) {
        setGeneratedFinalDraft(response.result.generated_final_draft);
        setCuratedDocument(response.result.curated_document ?? response.result.generated_final_draft);
      } else {
        setGeneratedFinalDraft(response.progress?.generated_final_draft ?? null);
        setCuratedDocument(response.progress?.curated_document ?? response.progress?.generated_final_draft ?? null);
      }
      setPatchStatus(response.status);
      setPatchProgress(response.progress ? { ...response.progress } : null);
      setTokenUsage(await getTokenUsage(packageId, chunkViewStrategy, workflowChatModel));
      setMessage(response.status === 'running' ? (options.resume ? 'Extraction resumed.' : 'Extraction is running.') : 'Extraction completed.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Context extraction failed.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onPauseExtraction() {
    if (!selectedPackageId || patchStatus !== 'running') return;
    const packageId = selectedPackageId;
    setBusy('pause');
    try {
      const { status, progress } = await pauseExtraction(packageId);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : null);
      if (progress?.interim_evidence_context) {
        setContext(initialContextFromEvidenceContext(progress.interim_evidence_context));
      }
      setTokenUsage(await getTokenUsage(packageId, chunkViewStrategy, workflowChatModel));
      setMessage(status === 'cancelled' ? 'Extraction paused. Resume extraction to continue from saved chunks.' : 'Extraction is not running.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to pause extraction.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onUpdateVocabConfig(config: ExtractionVocabQueryConfig) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      const { progress } = await updateVocabQueryConfig(packageId, config);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setMessage('Vocabulary query configuration updated.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to update vocabulary query configuration.');
      throw error;
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function onRerunVocabularyQueries(queryId?: string) {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    setBusy('context');
    try {
      const result = queryId
        ? await rerunVocabQuery(packageId, queryId)
        : await rerunAllVocabQueries(packageId);
      if (selectedPackageIdRef.current !== packageId) return;
      setGeneratedFinalDraft(result.generated_final_draft);
      setCuratedDocument(result.curated_document ?? result.generated_final_draft);
      setContext(initialContextFromEvidenceContext(result.machine_evidence_context));
      setTokenUsage(result.token_usage);
      const { status, progress } = await getPatchProgress(packageId, chunkViewStrategy, workflowChatModel);
      if (selectedPackageIdRef.current !== packageId) return;
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setMessage(queryId ? 'Vocabulary query rerun completed.' : 'Vocabulary queries rerun completed.');
    } catch (error) {
      if (selectedPackageIdRef.current !== packageId) return;
      setMessage(error instanceof Error ? error.message : 'Failed to rerun vocabulary queries.');
    } finally {
      if (selectedPackageIdRef.current === packageId) setBusy(null);
    }
  }

  async function refreshOllamaConfig() {
    setBusy('ollama');
    try {
      const [nextConfig, nextBudget] = await Promise.all([getOllamaConfig(), getLlmBudget()]);
      setOllamaConfig(nextConfig);
      setLlmBudget(nextBudget);
      setMessage('Ollama runtime settings refreshed.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not refresh Ollama settings.');
    } finally {
      setBusy(null);
    }
  }

  async function applyOllamaRuntimeConfig(values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_batch_size: number;
    embedding_num_gpu: number;
  }) {
    setBusy('ollama');
    try {
      const updated = await updateOllamaRuntimeConfig(values);
      const nextBudget = await getLlmBudget();
      setOllamaConfig(updated);
      setLlmBudget(nextBudget);
      setMessage('Ollama runtime settings updated for future calls.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not update Ollama runtime settings.');
    } finally {
      setBusy(null);
    }
  }

  async function pullOllamaModelFromUi(model: string) {
    setBusy('ollama');
    try {
      await pullOllamaModel(model);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage(`Pull requested for ${model} on the configured Ollama host.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not pull Ollama model.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  async function removeOllamaModelFromUi(model: string) {
    setBusy('ollama');
    try {
      await removeOllamaModel(model);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage(`Removed ${model} from the configured Ollama host.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not remove Ollama model.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  async function runOllamaSwitchTest(values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_num_gpu: number;
  }) {
    setBusy('ollama');
    try {
      const result = await runOllamaPerformanceTest(values);
      const nextConfig = await getOllamaConfig();
      setOllamaConfig(nextConfig);
      setMessage('Ollama switch test completed.');
      return result;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not run Ollama switch test.');
      throw error;
    } finally {
      setBusy(null);
    }
  }

  async function onGenerateDraft(options: { mode?: 'continue' | 'rebuild' | 'build' } = {}) {
    if (!selectedPackageId || !selectedProfile) return;
    const mode = options.mode ?? (generatedFinalDraft ? 'rebuild' : 'build');
    const isReplacingGeneratedDraft = mode === 'rebuild' && Boolean(generatedFinalDraft);
    if (isReplacingGeneratedDraft) {
      const confirmed = window.confirm(
        'Rebuild the generated final draft?\n\nThis updates the machine artifact. Existing curated edits remain separate.',
      );
      if (!confirmed) return;
    }
    if (saveCuratedDocumentTimeoutRef.current) {
      clearTimeout(saveCuratedDocumentTimeoutRef.current);
      saveCuratedDocumentTimeoutRef.current = null;
    }
    if (isReplacingGeneratedDraft) {
      setGeneratedFinalDraft(null);
      setPatchProgress((current) => current ? {
        ...current,
        stage: 'profile_projection',
        generated_final_draft: null,
        requirement_report: null,
        draft_quality_state: null,
        validation: { status: 'not_run', errors: [], warnings: [] },
        curated_validation: null,
        projection_ledger: [],
        initial_draft_scaffold: {},
        field_completion_ledger: [],
        vocab_queries: [],
        warnings: [],
      } : current);
    }
    setBusy('draft');
    try {
      const response = await runExtraction({
        data_package_id: selectedPackageId,
        profile_identifier: selectedProfile,
        resume: true,
        force_profile_rebuild: isReplacingGeneratedDraft,
        target_stage: 'profile',
        chunking_strategy: chunkViewStrategy,
        chat_model: workflowChatModel,
      });
      const nextGenerated = response.result?.generated_final_draft ?? response.progress?.generated_final_draft ?? null;
      setGeneratedFinalDraft(nextGenerated);
      setCuratedDocument(response.result?.curated_document ?? response.progress?.curated_document ?? curatedDocument ?? nextGenerated);
      setPatchStatus(response.status);
      setPatchProgress(response.progress ? {
        ...response.progress,
        requirement_report: response.result?.requirement_report ?? response.progress.requirement_report ?? null,
        generated_final_draft: response.result?.generated_final_draft ?? response.progress.generated_final_draft ?? null,
        ...(isReplacingGeneratedDraft && response.status === 'running' ? {
          generated_final_draft: null,
          requirement_report: null,
          draft_quality_state: null,
          validation: { status: 'not_run', errors: [], warnings: [] },
          curated_validation: null,
          projection_ledger: [],
          initial_draft_scaffold: {},
          field_completion_ledger: [],
          vocab_queries: [],
        } : {}),
      } : null);
      setTokenUsage(await getTokenUsage(selectedPackageId, chunkViewStrategy, workflowChatModel));
      setMessage(
        mode === 'continue'
          ? (response.status === 'running' ? 'Projection resumed.' : 'Projection is up to date.')
          : isReplacingGeneratedDraft ? 'Generated final draft rebuilt.' : 'Generated final draft construction started.'
      );
      if (response.status === 'running') {
        await refreshExtractionProgress();
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Generated final draft construction failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onCuratedDocumentChange(updated: Record<string, unknown>) {
    setCuratedDocument(updated);
    if (!selectedPackageId) return;
    if (saveCuratedDocumentTimeoutRef.current) clearTimeout(saveCuratedDocumentTimeoutRef.current);
    saveCuratedDocumentTimeoutRef.current = setTimeout(async () => {
      try {
        const saved = await saveCuratedDocument(selectedPackageId, updated, selectedProfile);
        setCuratedDocument(saved);
        setPatchProgress((current) => current ? { ...current, curated_document: saved as Record<string, unknown> } : current);
        setMessage('Curated document saved.');
      } catch (error) {
        setMessage(error instanceof Error ? error.message : 'Failed to save curated document.');
      }
    }, 800);
  }

  async function onGrounding() {
    if (!selectedPackageId || !selectedProfile) return;
    setBusy('patch');
    try {
      const result = await runVocabularyGrounding({ data_package_id: selectedPackageId, profile_identifier: selectedProfile, chunking_strategy: chunkViewStrategy, chat_model: workflowChatModel });
      setCuratedDocument(result.curated_document);
      setPatchStatus(result.status);
      const { progress } = await getPatchProgress(selectedPackageId, chunkViewStrategy, workflowChatModel);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      setTokenUsage(await getTokenUsage(selectedPackageId, chunkViewStrategy, workflowChatModel));
      setMessage(
        result.status === 'completed'
          ? 'Vocabulary grounding completed and final profile document was saved.'
          : 'Vocabulary grounding is running.',
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Vocabulary grounding failed.');
    } finally {
      setBusy(null);
    }
  }

  async function onSelectVocabularyCandidate(query: ExtractionVocabQueryRecord, uri: string, title?: string | null) {
    if (!selectedPackageId) return;
    const jsonPath = sourceContextJsonPointer(query);
    if (!jsonPath) return;
    setBusy('patch');
    try {
      const { status, progress } = await applyCurationFieldAction({
        data_package_id: selectedPackageId,
        action: 'select_vocab_term',
        json_path: jsonPath,
        selected_uri: uri,
        selected_title: title ?? null,
        vocabulary_identifier: query.vocabulary_identifier,
      });
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      if (progress?.curated_document) setCuratedDocument(progress.curated_document);
      setMessage('Vocabulary term selected for curated document.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to select vocabulary term.');
    } finally {
      setBusy(null);
    }
  }

  async function onMarkVocabularyUnresolved(query: ExtractionVocabQueryRecord) {
    if (!selectedPackageId) return;
    const jsonPath = sourceContextJsonPointer(query);
    if (!jsonPath) return;
    setBusy('patch');
    try {
      const { status, progress } = await applyCurationFieldAction({
        data_package_id: selectedPackageId,
        action: 'mark_unresolved',
        json_path: jsonPath,
        vocabulary_identifier: query.vocabulary_identifier,
      });
      setPatchStatus(status);
      setPatchProgress(progress ? { ...progress } : patchProgress);
      if (progress?.curated_document) setCuratedDocument(progress.curated_document);
      setMessage('Field marked intentionally unresolved.');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to mark field unresolved.');
    } finally {
      setBusy(null);
    }
  }

  async function refreshExtractionProgress() {
    if (!selectedPackageId) return;
    const packageId = selectedPackageId;
    try {
      const snapshot = await loadWorkflowBranch(packageId, chunkViewStrategy, workflowChatModel);
      if (selectedPackageIdRef.current !== packageId) return;
      applyWorkflowBranchSnapshot(snapshot);
      if (snapshot.status === 'running' || snapshot.initialStatus === 'running') {
        setMessage('Workflow stage is running.');
      } else {
        setMessage('Workflow progress refreshed.');
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to refresh workflow progress.');
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
        const snapshot = await loadWorkflowBranch(packageId, chunkViewStrategy, workflowChatModel);
        if (selectedPackageIdRef.current !== packageId) return;
        applyWorkflowBranchSnapshot(snapshot);
        setMessage('Workflow state loaded.');
      } catch (error) {
        if (selectedPackageIdRef.current !== packageId) return;
        setMessage(error instanceof Error ? error.message : 'Failed to load workflow state.');
      } finally {
        if (selectedPackageIdRef.current === packageId) setBusy(null);
      }
    })();
  }, [selectedPackageId, chunkViewStrategy, workflowChatModel]);

  useEffect(() => {
    if (!chunkResult || chunkResult.status === 'completed' || chunkResult.status === 'cancelled' || chunkResult.status === 'crashed') return;
    const interval = setInterval(() => void pollChunkProgress(), 3000);
    return () => clearInterval(interval);
  }, [chunkResult, selectedPackageId]);

  useEffect(() => {
    if (!selectedPackageId || (patchStatus !== 'running' && initialContextStatus !== 'running')) return;
    const interval = setInterval(() => void refreshExtractionProgress(), 5000);
    return () => clearInterval(interval);
  }, [patchStatus, initialContextStatus, selectedPackageId, chunkViewStrategy, workflowChatModel]);

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
              <OllamaSettingsPanel
                config={ollamaConfig}
                budget={llmBudget}
                tokenUsage={tokenUsage}
                patchTokenUsage={patchProgress?.token_usage}
                busy={busy === 'ollama'}
                onApply={(values) => void applyOllamaRuntimeConfig(values)}
                onRefresh={() => void refreshOllamaConfig()}
                onPullModel={(model) => pullOllamaModelFromUi(model)}
                onRemoveModel={(model) => removeOllamaModelFromUi(model)}
                onRunPerformanceTest={(values) => runOllamaSwitchTest(values)}
              />
            </>
          )}
        </aside>

        <aside className="branch-rail">
          <WorkflowBranchBar branch={currentWorkflowBranch} />
        </aside>

        <section className="workflow">
          <StepPanel
            number="01"
            title="Upload dataset and create chunks"
            description="Store the archive, inspect files, and prepare chunk branches before initial file understanding."
            active
          >
              <div className="dataset-upload-panel">
                <input
                  ref={datasetUploadInputRef}
                  className="hidden-file-input"
                  type="file"
                  accept=".zip"
                  onChange={(event) => void onUpload(event.target.files?.[0])}
                />
                <label className="dataset-package-select">
                  <span>Dataset package</span>
                  <select value={selectedPackageId} onChange={(event) => handlePackageSelection(event.target.value)}>
                    <option value="">No package selected</option>
                    {packages.map((item) => <option key={item.id} value={item.id}>{item.file_name}</option>)}
                  </select>
                </label>
                <div className="dataset-upload-actions">
                  <button onClick={() => datasetUploadInputRef.current?.click()} disabled={!!busy}>
                    {busy === 'upload' ? 'Uploading...' : 'Upload ZIP'}
                  </button>
                  <button
                    className="ghost dataset-remove-button"
                    onClick={() => void onDeleteDataPackage()}
                    disabled={!selectedPackageId || !!busy}
                  >
                    {busy === 'dataset-delete' ? 'Removing...' : 'Remove selected'}
                  </button>
                </div>
              </div>
              <div className="actions">
                <button onClick={() => setChunkingDialogOpen(true)} disabled={!selectedPackageId || !!busy}>{busy === 'chunk' ? 'Checking...' : 'Configure chunking'}</button>
              </div>
              <ChunkingStatusPanel
                dataPackage={selectedPackage}
                chunkResult={chunkResult}
                chunksByFile={chunksByFile}
                chunkViewStrategy={chunkViewStrategy}
                onChunkViewStrategyChange={setChunkViewStrategy}
                busy={busy}
              />
              <ChunkInspectionPanel
                dataPackage={selectedPackage}
                chunksByFile={chunksByFile}
                onViewFile={(file) => void onViewFile(file)}
              />
              <ChunkingDialog
                isOpen={chunkingDialogOpen}
                packageId={selectedPackageId}
                dataPackage={selectedPackage}
                chunksByFile={chunksByFile}
                chunkingStrategy={chunkViewStrategy}
                onClose={() => setChunkingDialogOpen(false)}
                onSubmit={(params) => {
                  setChunkingDialogOpen(false);
                  void onChunk(params);
                }}
              />
              {viewingFile && fileContent !== null && (
                <FileViewer
                  file={viewingFile}
                  content={fileContent}
                  chunksByFile={chunksByFile}
                  onClose={closeFileViewer}
                />
              )}
          </StepPanel>

          <StepPanel
            number="02"
            title="Initial file understanding"
            description="Summarize text-extractable files, rank them, and build a profile-independent run overview before chunking."
          >
              <div className="actions">
                <button onClick={() => void onInitialContext(Boolean(hasInitialContextArtifacts))} disabled={!selectedPackageId || !!busy || isInitialContextRunning || isPatching}>
                  {busy === 'initial-context' ? 'Running...' : hasInitialContextArtifacts ? 'Rerun initial overview' : 'Run initial overview'}
                </button>
              </div>
              <InitialFileUnderstandingPanel
                progress={patchProgress}
                status={initialContextStatus}
                tokenUsageSummary={(
                  <TokenUsageSummary
                    tokenUsage={tokenUsage}
                    averageUnit="request"
                    heading="Initial file understanding token usage"
                    agentKeys={['file_ranking', 'initial_file_summary', 'initial_extraction_overview', 'initial_extraction_overview_fallback', 'dataset_summary']}
                    budget={llmBudget}
                  />
                )}
              />
          </StepPanel>

          <StepPanel
            number="03"
            title="Chunk evidence extraction"
            description="Collect validated evidence notes from chunks using the initial overview and each file summary as orientation."
          >
              <div className="actions">
                <button onClick={() => void onContext()} disabled={!selectedPackageId || !hasChunks || !hasInitialContextArtifacts || !!busy || isPatching}>
                  {isPatching ? 'Extraction running...' : busy === 'context' ? 'Extracting...' : hasPersistedExtractionState || context ? 'Re-extract chunk evidence' : 'Extract chunk evidence'}
                </button>
                {isPatching && (
                  <button className="ghost" onClick={() => void onPauseExtraction()} disabled={!selectedPackageId || busy === 'pause'}>
                    {busy === 'pause' ? 'Pausing...' : 'Pause extraction'}
                  </button>
                )}
                {extractionCanResume && (
                  <button className="ghost" onClick={() => void onContext({ resume: true })} disabled={!selectedPackageId || !!busy || isPatching}>
                    Resume extraction
                  </button>
                )}
              </div>
              <div className="advanced-inline-control" aria-label="Chunk repair mode">
                <span>Repair</span>
                <div className="segmented-control">
                  {(['deferred', 'immediate', 'disabled'] as ChunkRepairMode[]).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      className={chunkRepairMode === mode ? 'active' : ''}
                      onClick={() => setChunkRepairMode(mode)}
                      disabled={!!busy || isPatching}
                    >
                      {mode === 'deferred' ? 'Deferred' : mode === 'immediate' ? 'Immediate' : 'Disabled'}
                    </button>
                  ))}
                </div>
              </div>
              {patchStatus === 'running' && patchProgress && (
                <div className="patch-progress context-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{formatExtractionStage(patchStatus)}</strong></span>
                    {extractionProgressLabel && <span>{extractionProgressLabel}</span>}
                  </div>
                  {patchProgress.total_chunks > 0 && (
                    <div className="patch-progress-track" aria-hidden="true"><div style={{ width: `${extractionProgressPercent}%` }} /></div>
                  )}
                </div>
              )}
              <EvidenceContextOverview
                chunkResults={patchProgress?.chunk_results ?? []}
                currentChunk={patchProgress?.current_chunk ?? null}
                chunksByFile={chunksByFile}
                packageFiles={selectedPackage?.files ?? []}
                progress={patchProgress}
                status={patchStatus}
                budget={llmBudget}
                tokenUsageSummary={(
                  <TokenUsageSummary
                    tokenUsage={tokenUsage}
                    averageUnit="operation"
                    heading="Evidence extraction token usage"
                    agentKeys={['chunk_extraction', 'chunk_extraction_repair', 'quantity_vocab_selection', 'qualitative_vocab_selection', 'profile_projection']}
                    budget={llmBudget}
                    notesByAgent={extractionTokenUsageNotes}
                  />
                )}
              />
              <p className="context-window-advice">
                If extraction reaches or overuses the context window, rerun chunking with smaller chunks before extracting again. Lower the semantic chunking threshold in the chunking configuration to reduce chunk sizes.
              </p>
          </StepPanel>

          <StepPanel
            number="04"
            title="Generated and curated profile"
            description="Build the machine-generated final draft, inspect projection issues, and curate the separate user document."
            actions={hasProfileArtifacts && (
              <button className="ghost draft-refresh-button" onClick={() => void refreshExtractionProgress()} disabled={!selectedPackageId || busy === 'load'}>Refresh</button>
            )}
          >
              <div className="profile-selection-panel">
                <div className="panel-heading profile-selection-heading">
                  <span>Profile</span>
                  <button onClick={() => setProfileFormOpen((open) => !open)} disabled={!!busy}>
                    {profileFormOpen ? 'Close' : 'Register'}
                  </button>
                </div>
                <div className="profile-selection-row">
                  <label className="profile-select">
                    <span>Profile document</span>
                    <select value={selectedProfile} onChange={(event) => setSelectedProfile(event.target.value)}>
                      <option value="">No profile selected</option>
                      {profiles.map((profile) => <option key={profile.identifier} value={profile.identifier}>{profile.identifier}</option>)}
                    </select>
                  </label>
                  <button
                    className="ghost profile-remove-button"
                    onClick={() => void onDeleteProfile()}
                    disabled={!selectedProfile || !!busy}
                  >
                    {busy === 'profile-delete' ? 'Removing...' : 'Remove selected'}
                  </button>
                </div>
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
              <div className={hasProfileArtifacts ? 'draft-actions' : 'actions'}>
                {!generatedFinalDraft ? (
                  <button onClick={() => void onGenerateDraft({ mode: 'build' })} disabled={!selectedPackageId || !selectedProfile || !!busy || isPatching || !context}>{busy === 'draft' ? 'Building generated draft...' : isProfileBuildRunning ? 'Building generated draft...' : 'Build generated final draft'}</button>
                ) : (
                  <>
                    {projectionCanContinue && (
                      <button onClick={() => void onGenerateDraft({ mode: 'continue' })} disabled={!projectionCanContinue}>
                        {busy === 'draft' ? 'Continuing...' : 'Continue projection'}
                      </button>
                    )}
                    <button className="ghost" onClick={() => setCuratedEditorOpen(true)} disabled={!curatedDocument}>Edit</button>
                    <button className="ghost draft-recreate-button" onClick={() => void onGenerateDraft({ mode: 'rebuild' })} disabled={!selectedPackageId || !selectedProfile || !!busy || isPatching}>{busy === 'draft' ? 'Rebuilding...' : isProfileBuildRunning ? 'Rebuilding...' : 'Rebuild generated draft'}</button>
                  </>
                )}
              </div>
              {(hasProfileArtifacts || isProfileBuildRunning || projectionLedger.length > 0) && (
                <ProjectionWorkflowPanel
                  progress={patchProgress}
                  status={patchStatus}
                  ledger={projectionLedger}
                  tokenUsageSummary={(
                    <TokenUsageSummary
                      tokenUsage={tokenUsage}
                      averageUnit="operation"
                      heading="Projection token usage"
                      agentKeys={[
                        'dataset_level_projection',
                        'dataset_level_projection_repair',
                        'profile_target_planner',
                        'profile_target_writer',
                        'profile_patch',
                        'profile_projection',
                        'metadata_completeness_evaluator',
                        'metadata_requirement_patcher',
                      ]}
                      budget={llmBudget}
                    />
                  )}
                />
              )}
              {(hasProfileArtifacts || isProfileBuildRunning) && patchStatus === 'running' && (
                <div className="patch-progress">
                  <div className="patch-progress-header">
                    <span>Status: <strong>{formatExtractionStage(patchProgress?.stage || patchStatus)}</strong></span>
                    {!hasProfileArtifacts && <span>Generating draft artifact</span>}
                  </div>
                </div>
              )}
              {generatedFinalDraft && (
                <JsonDetails title="Generated final draft (machine artifact)" value={generatedFinalDraft} />
              )}
          </StepPanel>

          <StepPanel
            number="05"
            title="Grounding and validation"
            description="Review projection validity, run vocabulary grounding, and manage the vocabulary resources used for enrichment."
            actions={hasProfileArtifacts && (
              <button className="ghost draft-refresh-button" onClick={() => void refreshExtractionProgress()} disabled={!selectedPackageId || busy === 'load'}>Refresh</button>
            )}
          >
              {hasProfileArtifacts ? (
                <section className="patch-progress grounding-validation-summary">
                  <div className="patch-progress-header">
                    <span>Projection and validation</span>
                    <span>
                      {projectedObjects} projected, {notProjectedObjects} unresolved, {editRequiredObjects} edit needed
                    </span>
                  </div>
                  <div className="patch-progress-summary">
                    <span>Draft quality: {formatExtractionStage(patchProgress?.draft_quality_state || 'not run')}</span>
                    <span>Validation: {formatExtractionStage(patchProgress?.validation?.status || 'not run')}</span>
                    <span>{validationErrorCount} validation issue{validationErrorCount === 1 ? '' : 's'}</span>
                    <span>{fieldIssueCount} field issue{fieldIssueCount === 1 ? '' : 's'}</span>
                  </div>
                </section>
              ) : (
                <p className="muted">Build the generated profile before reviewing projection validation.</p>
              )}
              {hasProfileArtifacts && (
                <DraftGroundingPanel
                  progress={patchProgress}
                  disabled={!selectedPackageId || !!busy || isPatching}
                  onUpdateVocabQueryConfig={(config) => void onUpdateVocabConfig(config)}
                  onRunGrounding={() => void onGrounding()}
                  onRerunAllVocabQueries={() => void onRerunVocabularyQueries()}
                  onRerunVocabQuery={(queryId) => void onRerunVocabularyQueries(queryId)}
                  onSelectCandidate={(query, uri, title) => void onSelectVocabularyCandidate(query, uri, title)}
                  onMarkUnresolved={(query) => void onMarkVocabularyUnresolved(query)}
                />
              )}
              <div className="workflow-vocabulary-panel">
                <VocabularyPanel onError={setMessage} />
              </div>
          </StepPanel>
        </section>
      </section>
      {curatedEditorOpen && curatedDocument && createPortal(
        <div className="vocab-dialog-overlay" onClick={() => setCuratedEditorOpen(false)}>
          <div className="vocab-dialog curated-editor-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <div>
                <span>Curated document</span>
                <strong>Edit curated JSON</strong>
              </div>
              <button className="ghost" type="button" onClick={() => setCuratedEditorOpen(false)}>Close</button>
            </div>
            <div className="vocab-dialog-body curated-editor-dialog-body">
              <JsonEditor
                value={curatedDocument as Record<string, unknown>}
                onChange={(updated) => onCuratedDocumentChange(updated)}
                patchMarkers={curationMarkers}
                schema={activeProfileSchema}
                targetClass={selectedProfileManifest?.target_class}
              />
            </div>
          </div>
        </div>,
        document.body,
      )}
    </main>
  );
}
