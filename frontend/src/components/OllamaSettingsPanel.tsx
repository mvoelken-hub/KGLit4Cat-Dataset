import { type FormEvent, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import type { PatchTokenUsage } from '../api/extraction';
import type { LlmBudget, OllamaConfig, OllamaPerformanceTest } from '../api/system';
import { formatDuration, formatMemory, formatTokenCount } from '../lib/format';
import { requestAverage } from './TokenUsageSummary';
export function OllamaSettingsPanel({
  config,
  budget,
  tokenUsage,
  patchTokenUsage,
  busy,
  onApply,
  onRefresh,
  onPullModel,
  onRemoveModel,
  onRunPerformanceTest,
}: {
  config: OllamaConfig | null;
  budget: LlmBudget | null;
  tokenUsage: PatchTokenUsage | null;
  patchTokenUsage?: PatchTokenUsage | null;
  busy: boolean;
  onApply: (values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_batch_size: number;
    embedding_num_gpu: number;
  }) => void;
  onRefresh: () => void;
  onPullModel: (model: string) => Promise<void>;
  onRemoveModel: (model: string) => Promise<void>;
  onRunPerformanceTest: (values: {
    chat_model: string;
    embedding_model: string;
    max_context_length: number;
    embedding_num_gpu: number;
  }) => Promise<OllamaPerformanceTest>;
}) {
  const runtime = config?.runtime;
  const combinedUsage = patchTokenUsage?.combined ?? tokenUsage?.combined ?? null;
  const averageInput = combinedUsage ? requestAverage(combinedUsage, 'input') : 0;
  const [chatModel, setChatModel] = useState('');
  const [embeddingModel, setEmbeddingModel] = useState('');
  const [maxContextLength, setMaxContextLength] = useState(8192);
  const [embeddingBatchSize, setEmbeddingBatchSize] = useState(32);
  const [embeddingNumGpu, setEmbeddingNumGpu] = useState(-1);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [pullModelName, setPullModelName] = useState('');
  const [performanceTest, setPerformanceTest] = useState<OllamaPerformanceTest | null>(null);
  const [localMessage, setLocalMessage] = useState('');

  useEffect(() => {
    if (!runtime) return;
    setChatModel(runtime.chat_model);
    setEmbeddingModel(runtime.embedding_model);
    setMaxContextLength(runtime.max_context_length);
    setEmbeddingBatchSize(runtime.embedding_batch_size);
    setEmbeddingNumGpu(runtime.embedding_num_gpu);
  }, [runtime]);

  function submit(event: FormEvent) {
    event.preventDefault();
    onApply({
      chat_model: chatModel.trim(),
      embedding_model: embeddingModel.trim(),
      max_context_length: Math.max(512, maxContextLength),
      embedding_batch_size: Math.max(1, embeddingBatchSize),
      embedding_num_gpu: embeddingNumGpu,
    });
  }

  async function runSwitchTest() {
    if (!runtime) return;
    setLocalMessage('');
    try {
      const result = await onRunPerformanceTest({
        chat_model: chatModel.trim(),
        embedding_model: embeddingModel.trim(),
        max_context_length: Math.max(512, maxContextLength),
        embedding_num_gpu: embeddingNumGpu,
      });
      setPerformanceTest(result);
      setLocalMessage('Switch test completed.');
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Switch test failed.');
    }
  }

  async function pullModel(event: FormEvent) {
    event.preventDefault();
    const model = pullModelName.trim();
    if (!model) return;
    const host = config?.host.base_url ?? 'the configured Ollama host';
    if (!window.confirm(`Pull ${model} on ${host}? This mutates the Ollama host.`)) return;
    setLocalMessage('');
    try {
      await onPullModel(model);
      setPullModelName('');
      setLocalMessage(`Pull requested for ${model}.`);
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Could not pull model.');
    }
  }

  async function removeModel(model: string) {
    const host = config?.host.base_url ?? 'the configured Ollama host';
    if (!window.confirm(`Remove ${model} from ${host}? This mutates the Ollama host.`)) return;
    setLocalMessage('');
    try {
      await onRemoveModel(model);
      setLocalMessage(`Removed ${model}.`);
    } catch (error) {
      setLocalMessage(error instanceof Error ? error.message : 'Could not remove model.');
    }
  }

  const availableModels = config?.models.models ?? [];
  const chatModelOptions = Array.from(new Set([
    chatModel,
    ...availableModels
      .filter((model) => model.kind !== 'embedding')
      .map((model) => model.model || model.name || '')
      .filter(Boolean),
  ])).filter(Boolean);
  const embeddingModelOptions = Array.from(new Set([
    embeddingModel,
    ...availableModels
      .filter((model) => model.kind === 'embedding')
      .map((model) => model.model || model.name || '')
      .filter(Boolean),
  ])).filter(Boolean);
  const selectedChatModelInfo = availableModels.find((model) => (model.model || model.name) === chatModel);
  const selectedChatModelName = chatModel.toLowerCase();
  const selectedChatModelSize = selectedChatModelInfo?.size ?? null;
  const selectedChatModelIsCloud = Boolean(selectedChatModelInfo?.is_cloud) || selectedChatModelName.includes('cloud') || selectedChatModelSize === 0;
  const showPerformanceTest = Boolean(runtime && !selectedChatModelIsCloud);

  useEffect(() => {
    if (selectedChatModelIsCloud) setPerformanceTest(null);
  }, [selectedChatModelIsCloud]);

  return (
    <section className="ollama-panel" aria-label="Ollama runtime settings">
      <div className="ollama-panel-header">
        <div>
          <span>Ollama</span>
          <strong>{config?.host.base_url ?? 'Unavailable'}</strong>
          {config && <small className="ollama-mode-chip">{config.host.is_local ? 'Local' : 'Remote'}</small>}
        </div>
        <button
          className="ghost small"
          type="button"
          onClick={() => setDetailsOpen(true)}
        >
          Details
        </button>
      </div>

      <div className="ollama-summary-grid">
        <div>
          <span>Models</span>
          <strong>{runtime?.chat_model ?? 'No chat model'}</strong>
          <small>{runtime?.embedding_model ?? 'No embedding model'}</small>
        </div>
        <div>
          <span>Context budget</span>
          <strong>{formatTokenCount(budget?.input_token_budget ?? runtime?.input_token_budget)} input tokens</strong>
          <small>{formatTokenCount(budget?.max_context_length ?? runtime?.max_context_length)} total context tokens</small>
        </div>
      </div>

      {detailsOpen && createPortal((
        <div className="vocab-dialog-overlay" onClick={() => setDetailsOpen(false)}>
          <div className="vocab-dialog ollama-dialog" onClick={(event) => event.stopPropagation()}>
            <div className="vocab-dialog-header">
              <strong>Ollama Runtime Settings</strong>
              <div className="ollama-dialog-actions">
                <button className="ghost small" type="button" onClick={onRefresh} disabled={busy}>Refresh</button>
                <button className="ghost" type="button" onClick={() => setDetailsOpen(false)}>Close</button>
              </div>
            </div>
            <div className="vocab-dialog-body ollama-dialog-body">
              <div className="ollama-status-grid">
                <div>
                  <span>Host mode</span>
                  <strong>{config ? (config.host.is_local ? 'Local host' : 'Remote host') : 'Unknown'}</strong>
                  <small>{config?.host.is_local ? config.host.server_settings_note : 'Remote mode shows runtime controls and Ollama API diagnostics only.'}</small>
                </div>
                {config?.host.is_local && (
                  <div>
                    <span>Server memory settings</span>
                    <strong>flash {String(config.host.flash_attention ?? false)} / KV {config.host.kv_cache_type ?? 'unknown'}</strong>
                    <small>Change these on the Ollama host, not from the UI.</small>
                  </div>
                )}
                <div>
                  <span>Context budget</span>
                  <strong>{formatTokenCount(budget?.input_token_budget ?? runtime?.input_token_budget)} input tokens</strong>
                  <small>{formatTokenCount(budget?.max_context_length ?? runtime?.max_context_length)} total context tokens</small>
                </div>
                <div>
                  <span>Recent average input</span>
                  <strong>{formatTokenCount(averageInput)} tokens</strong>
                  <small>{averageInput && runtime && averageInput > runtime.input_token_budget * 0.8 ? 'Lower context usage before starting the next run.' : 'Tune chunking settings to reduce per-call context size.'}</small>
                </div>
                <div>
                  <span>Diagnostics</span>
                  <strong>{config?.diagnostics.status ?? 'unknown'}</strong>
                  <small>{config?.diagnostics.summary ?? 'Run the switch test for residency guidance.'}</small>
                </div>
              </div>

              <form className="ollama-runtime-form" onSubmit={submit}>
                <label>
                  <span>Chat model</span>
                  <select value={chatModel} onChange={(event) => setChatModel(event.target.value)} disabled={!runtime || busy || chatModelOptions.length === 0}>
                    {chatModelOptions.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Embedding model</span>
                  <select value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)} disabled={!runtime || busy || embeddingModelOptions.length === 0}>
                    {embeddingModelOptions.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Max context</span>
                  <input type="number" min={512} step={512} value={maxContextLength} onChange={(event) => setMaxContextLength(parseInt(event.target.value, 10) || 512)} disabled={!runtime || busy} />
                </label>
                <label>
                  <span>Embedding batch</span>
                  <input type="number" min={1} value={embeddingBatchSize} onChange={(event) => setEmbeddingBatchSize(parseInt(event.target.value, 10) || 1)} disabled={!runtime || busy} />
                </label>
                <label>
                  <span>Embedding GPU</span>
                  <select value={embeddingNumGpu} onChange={(event) => setEmbeddingNumGpu(parseInt(event.target.value, 10))} disabled={!runtime || busy}>
                    <option value={-1}>Auto</option>
                    <option value={0}>CPU only</option>
                    <option value={999}>GPU only</option>
                  </select>
                </label>
                <button type="submit" disabled={!runtime || busy}>Apply runtime settings</button>
              </form>

              {showPerformanceTest && (
                <div className="ollama-test-panel">
                  <div className="ollama-section-header">
                    <div>
                      <span>Performance test</span>
                      <strong>Embedding/chat switch test</strong>
                    </div>
                    <button className="ghost small" type="button" onClick={() => void runSwitchTest()} disabled={!runtime || busy}>
                      Run switch test
                    </button>
                  </div>
                  <small>Loads embedding -{'>'} chat -{'>'} embedding -{'>'} chat to detect reload pressure.</small>
                  {performanceTest && (
                    <div className="ollama-test-result">
                      <strong>{performanceTest.diagnostics.summary}</strong>
                      <div className="ollama-test-steps">
                        {performanceTest.steps.map((step) => (
                          <div key={step.key}>
                            <span>{step.label}</span>
                            <strong>{step.success ? formatDuration(step.load_duration_ms) : 'failed'}</strong>
                            <small>{step.success ? `${step.snapshot.models.length} loaded after call` : step.error}</small>
                          </div>
                        ))}
                      </div>
                      {performanceTest.diagnostics.recommendations.length > 0 && (
                        <div className="ollama-recommendations">
                          {performanceTest.diagnostics.recommendations.map((recommendation) => (
                            <small key={recommendation}>{recommendation}</small>
                          ))}
                          <div className="ollama-recommendation-actions">
                            <button className="ghost small" type="button" onClick={() => setEmbeddingNumGpu(0)} disabled={busy}>
                              Set embeddings to CPU
                            </button>
                            <button className="ghost small" type="button" onClick={() => setEmbeddingBatchSize(Math.max(1, Math.floor(embeddingBatchSize / 2)))} disabled={busy}>
                              Halve embed batch
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="ollama-loaded-models">
                <span>Loaded models</span>
                {!config?.running.available && <small>{config?.running.error?.message ?? 'Could not inspect loaded Ollama models.'}</small>}
                {config?.running.available && !config.running.models.length && <small>No models are currently resident.</small>}
                {config?.running.models.map((model, index) => (
                  <small key={model.model || index}>
                    {model.model || 'unknown'} - {model.processor ? `Processor ${model.processor}` : `VRAM ${formatMemory(model.size_vram)} / total ${formatMemory(model.size)}`}
                    {model.processor ? ` - VRAM ${formatMemory(model.size_vram)} / total ${formatMemory(model.size)}` : ''}
                    {model.context_length ? ` - context ${formatTokenCount(model.context_length)}` : ''}
                  </small>
                ))}
                {config?.running.available && (
                  <small>
                    Chat {config.running.chat_model_loaded ? 'resident' : 'not resident'}; embedding {config.running.embedding_model_loaded ? 'resident' : 'not resident'}.
                  </small>
                )}
              </div>

              <div className="ollama-model-overview">
                <div className="ollama-section-header">
                  <div>
                    <span>Available models</span>
                    <strong>{config?.models.available ? `${availableModels.length} installed` : 'Unavailable'}</strong>
                  </div>
                </div>
                {!config?.models.available && <small>{config?.models.error?.message ?? 'Could not inspect installed models.'}</small>}
                <form className="ollama-pull-form" onSubmit={(event) => void pullModel(event)}>
                  <input
                    value={pullModelName}
                    onChange={(event) => setPullModelName(event.target.value)}
                    placeholder="model:tag"
                    disabled={busy}
                  />
                  <button type="submit" disabled={busy || !pullModelName.trim()}>Pull model</button>
                </form>
                {availableModels.length > 0 && (
                  <div className="ollama-model-list">
                    {availableModels.map((model, index) => {
                      const name = model.model || model.name || '';
                      const canUseAsChat = model.kind !== 'embedding';
                      const canUseAsEmbedding = model.kind === 'embedding';
                      return (
                        <div className="ollama-model-row" key={name || index}>
                          <div>
                            <strong>{name || 'unknown'}</strong>
                            <small>
                              {formatMemory(model.size)}
                              {model.details?.parameter_size ? ` - ${model.details.parameter_size}` : ''}
                              {model.details?.quantization_level ? ` - ${model.details.quantization_level}` : ''}
                              {model.kind ? ` - ${model.kind}` : ''}
                              {model.is_cloud ? ' - Cloud' : ''}
                            </small>
                          </div>
                          <div className="ollama-model-actions">
                            {canUseAsChat && (
                              <button className="ghost small" type="button" onClick={() => setChatModel(name)} disabled={!name || busy}>Chat</button>
                            )}
                            {canUseAsEmbedding && (
                              <button className="ghost small" type="button" onClick={() => setEmbeddingModel(name)} disabled={!name || busy}>Embed</button>
                            )}
                            <button className="ghost small danger-button" type="button" onClick={() => void removeModel(name)} disabled={!name || busy}>Remove</button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {localMessage && <p className="muted">{localMessage}</p>}
              <p className="muted">ℹ️ Runtime edits affect future SIMONE calls only and reset when the API restarts.</p>
            </div>
          </div>
        </div>
      ), document.body)}
    </section>
  );
}

