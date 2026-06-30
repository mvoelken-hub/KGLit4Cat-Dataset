# SIMONE Workflow Reliability Assessment — Interim Report

**Date:** 2026-06-24
**Model:** `gemma3:4b-cloud` (Ollama cloud)
**Embedding model:** `qwen3-embedding:0.6b` (local)
**Profile:** `dcat-ap-plus`
**Chunking:** `fixed_tokens` (1024 tokens/chunk) + one `semantic` run for comparison
**Datasets:** NMR/IR/SCR ZIP files from `data/only_NMR_Files/`

---

## 1. Executive Summary

SIMONE workflow runs end-to-end without hard infrastructure failures. However, **output quality is insufficient for production use** with `gemma3:4b-cloud`. Schema validation passes consistently, but profile conformance, field coverage, and evidence grounding fail across all datasets. Cloud model reliability is poor — 503 overloads and structured output failures dominate error logs. A larger model would likely improve structured output compliance and semantic routing, but core architecture issues (empty evidence query ledger, missing requirement report, projection failures) suggest problems beyond model capacity alone.

---

## 2. Datasets Processed

| # | Dataset | Package ID | Strategy | Status | Chunks (processed/total) |
|---|---------|-----------|----------|--------|--------------------------|
| 1 | 1H_NMR-1H_NMR.zip | 75ac48c5 | semantic | completed | ~54/55 |
| 2 | 13C-13C_STM125.zip | b3df299b | fixed_tokens | completed | 46/90 |
| 3 | 13C-Gel-NMR-13C-Gel-NMR(1).zip | b9c1616a | fixed_tokens | completed | 31/48 |
| 4 | 13C-Gel-NMR-13C-Gel-NMR.zip | 5a910011 | fixed_tokens | completed | 27/51 |
| 5 | 13C_Gel-NMR-13C_Gel-NMR(1).zip | 5b4b0684 | fixed_tokens | completed | 0/0* |
| 6 | 13C_Gel-NMR-13C_Gel-NMR.zip | 1c7c2a1d | fixed_tokens | crashed | 0/0 |

*Dataset 5 completed in workflow status but progress endpoint showed 0 chunks — likely stale state. Results endpoint returned valid output with 49/165 coverage.

**6 datasets attempted, 5 completed, 1 crashed.** Remaining 15 datasets still in queue.

---

## 3. Error Susceptibility

### 3.1 Error Frequency

| Error Type | Count | Impact |
|------------|-------|--------|
| Chunk extraction completion failed | 123 | Individual chunks skipped, evidence lost |
| Chunk extraction structured output failed | 90 | JSON generation failures, retry exhaustion |
| Workflow crash (ChunkingRequiredError) | 1 | Entire dataset lost (v1 run, missing replace_existing_chunks) |
| Workflow crash (initial context race) | 1 | Extraction attempted before file summaries ready |
| Model overloaded (503) | 70+ | Cloud model unavailable, chunks skipped after 3 retries |
| **Total ERROR log entries** | **216** | — |

### 3.2 Error Categories

**A. Cloud model capacity (503 Service Unavailable)**
- `gemma3:4b-cloud` returns 503 under sustained load
- Pattern: burst of requests -> model overloaded -> 3 retries exhausted -> chunk skipped
- Dataset 5 (5b4b0684): 70 of 73 warnings are model overload errors
- Affects 29+ unique file/chunk combinations per dataset
- **Root cause:** shared cloud model, no dedicated capacity, rate limiting

**B. Structured output generation failures**
- Model fails to produce valid JSON for evidence extraction prompts
- 90 failures across all datasets
- Repair attempts also fail ("Max retries exceeded while generating structured output")
- `gemma3:4b-cloud` struggles with complex JSON schemas
- **Root cause:** small model capacity, limited instruction following for structured formats

**C. Chunk extraction completion failures (123)**
- Largest error category
- Occurs when both initial generation AND repair attempts fail
- Cascade: structured output fail -> repair fail -> completion fail
- Results in permanent evidence loss for affected chunks

**D. Race condition on force_rerun**
- Dataset 6 (1c7c2a1d) crashed with "Run initial file understanding before chunk extraction"
- `force_rerun=true` clears extraction artifacts including initial file summaries
- Workflow re-runs `initial_context` with `force_rerun=False` -> finds existing (now cleared) state -> skips
- Extraction then fails because summaries are missing
- **Root cause:** workflow logic doesn't force re-run of initial context when clearing artifacts

**E. Configuration gap (replace_existing_chunks)**
- v1 batch run: all datasets timed out because `replace_existing_chunks=false`
- Previous semantic chunks existed, new fixed_tokens chunks never created
- Workflow found existing task (COMPLETED) but wrong strategy -> empty chunk list -> crash
- **Root cause:** API design separates `force_rerun` from `replace_existing_chunks` — user must set both

### 3.3 Error Recovery

| Mechanism | Effectiveness |
|-----------|--------------|
| Retry (3 attempts per chunk) | Low — same model, same capacity, same failure |
| Chunk extraction repair | Low — repair uses same model, often fails again |
| Workflow-level crash handling | None — crashed workflows produce partial or no output |
| Progress endpoint | Unreliable — shows stale stage/chunk counts from previous runs |

### 3.4 Reliability Score

| Aspect | Rating | Notes |
|--------|--------|-------|
| Infrastructure stability | Good | Docker stack, Neo4j, Ollama — no crashes |
| Workflow completion | Moderate | 5/6 completed, 1 crash from race condition |
| Chunk processing | Poor | 30-50% chunks successfully processed; 50-70% failed |
| Error recovery | Poor | No graceful degradation, failed chunks = permanent loss |
| State management | Poor | Stale progress, force_rerun race condition, strategy confusion |

---

## 4. Output Quality and Usefulness

### 4.1 Quality Metrics Summary

| Dataset | schema_valid | profile_conformant | evidence_grounded | coverage | projected/total | fields | warnings |
|---------|-------------|-------------------|-------------------|----------|----------------|--------|----------|
| 1H_NMR (semantic) | True | **False** | True | 43/152 (28%) | 7/42 | 12 | 6 |
| 13C-13C_STM125 | True | **False** | True | 57/188 (30%) | 7/30 | 14 | 56 |
| 13C-Gel-NMR(1) | True | **False** | True | 44/145 (30%) | 7/35 | 12 | 21 |
| 13C-Gel-NMR | True | **False** | **False** | 7/93 (8%) | 0/34 | 4 | 29 |
| 13C_Gel-NMR(1) | True | **False** | True | 49/165 (30%) | 7/10 | — | 73 |

### 4.2 Key Findings

**Schema validation:** Passes consistently. JSON-LD output structurally valid against base schema. This is architectural quality — not model-dependent.

**Profile conformance:** **Fails for ALL datasets.** DCAT-AP+ profile requirements never met. Output contains valid JSON but doesn't conform to target profile. Missing fields include: `access_rights`, `applicable_legislation`, `conforms_to`, `contact_point`, `creator`, `dataset_distribution`, `description` (often empty), `documentation`, `frequency`, `keyword`, `publisher`, `spatial_coverage`, `temporal_coverage`, `theme`, `was_generated_by` (incomplete).

**Field coverage:** Extremely low — **7-30%** of profile fields filled. 70-92% of target fields remain empty. This means most metadata a user would need is missing.

**Evidence grounding:** Mixed. 4/5 datasets have `evidence_grounded=True`, but evidence query ledger is **empty for all datasets** (0 entries). This suggests grounding flag is set at coarse level but individual evidence-to-field traceability is absent.

**Projection success:** Very low. 0-7 projections out of 10-42 total evidence candidates. Most evidence notes have status `not_projected` with reasons like:
- "target_unresolved: LLM router skipped the note"
- "target_unresolved: semantic router selected a target, but no attribute candidate could be parsed"
- "No reason provided"

**Semantic validation:** `None` for all datasets — semantic validation either not configured or not executing.

**Requirement report:** **Empty for all datasets** (0/0 requirements). The requirement report — a key thesis metric — is not populated. This means workflow doesn't assess its own output quality against the 9 DCAT-AP+ requirements.

### 4.3 Extracted Content Assessment

**What works:**
- Dataset `id` and `identifier` correctly extracted
- `was_generated_by` activity with basic structure present
- Some quantitative attributes extracted (reference frequency, pulse width)
- File-level evidence candidates generated (portable evidence)
- Source traceability score = 1.0 for some datasets

**What fails:**
- `description` — empty or missing in most datasets
- `title` — generic ("SIMONE extraction result for {id}") instead of descriptive
- `is_about_entity` / `is_about_activity` — mostly absent
- `creator` — missing
- `modification_date` — present but sometimes wrong (inherited from file metadata, not dataset)
- `carried_out_by` (technical agents) — wrong types (CV IRIs instead of labels, e.g., `voc4cat_0000282` instead of "Bruker NMR")
- `realized_plan` — often wrong or nonsensical (e.g., `AQ` as plan type)
- `has_quantitative_attribute` — values extracted but quantity types wrong (e.g., `RadiantIntensity` for pulse width)
- Vocabulary grounding — `from_CV` fields contain raw IRIs, not resolved labels

### 4.4 Usefulness Verdict

**Current output is not useful as DCAT-AP+ metadata.** A user receiving this output would get:
- Valid JSON structure
- Dataset identifier
- Some instrument parameters (partially correct)
- No description, title, creator, publisher, themes, keywords
- No profile conformance
- No evidence traceability
- No requirement assessment

Output requires **significant manual curation** before it could serve as actual metadata.

---

## 5. Integration into Overall Workflow

### 5.1 API Integration

| Endpoint | Status | Issues |
|----------|--------|--------|
| POST /api/v1/extraction/workflows | Works | `replace_existing_chunks` must be explicitly set |
| GET /workflows/{id}/progress | Unreliable | Stale state from previous runs, stage not updating |
| GET /extraction/results/{id} | Requires params | Must pass `chunking_strategy` and `chat_model` query params or returns empty |
| GET /workflows/{id}/token-usage | Works | Must pass `chunking_strategy` and `chat_model` query params |

### 5.2 Batch Processing Integration

**Critical issues for batch automation:**

1. **Progress endpoint stale state** — When using `force_rerun`, progress endpoint shows previous run's stage/chunk counts until new run catches up. Makes monitoring unreliable.

2. **"crashed" status not detected** — Batch script v2 only checked for "completed" and "failed", missing "crashed" workflows. Script polled indefinitely on crashed dataset.

3. **No `replace_existing_chunks` default** — API defaults to `false`. Must be explicitly set when changing chunking strategy. Undocumented coupling with `force_rerun`.

4. **Results endpoint parameter sensitivity** — Results keyed by `(chunking_strategy, chat_model)` tuple. Omitting query params returns empty results even when data exists.

5. **No batch API** — Each dataset requires individual upload + poll cycle. No queue management, no parallel processing support, no automatic retry of failed datasets.

### 5.3 Frontend Integration

Frontend (React app) works for single interactive runs. For batch evaluation:
- Frontend polls progress endpoint continuously, adding load
- Frontend cannot initiate batch runs
- Results viewer works for individual datasets but no comparative view

### 5.4 Thesis Evaluation Integration

For thesis evaluation chapter, current results provide:
- Token usage metrics per agent (useful for cost analysis)
- Processing time per dataset
- Warning/error counts (reliability metrics)
- Coverage scores (but very low — hard to present positively)
- Requirement report (empty — cannot assess requirement fulfillment)
- Evidence query ledger (empty — cannot assess evidence retrieval quality)
- Profile conformance (always False — cannot show conformance)

**Recommendation:** Thesis evaluation should focus on architectural analysis (what SIMONE attempts, pipeline design, evidence-grounded approach) rather than output quality metrics, which are currently too poor to support claims of production-readiness.

---

## 6. Model Capabilities Assessment

### 6.1 Current Model: `gemma3:4b-cloud`

**Architecture:** 4B parameter model, served via Ollama cloud

**Strengths:**
- Fast response time when available (~10-15s per generation)
- Can follow basic extraction instructions
- Produces valid JSON for simple schemas
- Embedding-adjacent tasks (semantic routing) partially functional

**Weaknesses:**
- **Structured output compliance:** Fails on complex JSON schemas. 90+ structured output failures. Model produces malformed JSON, wrong field types, or hallucinated fields.
- **Instruction following:** Ignores parts of multi-step prompts. Evidence notes often have empty `uncertainty` fields, wrong `category` assignments.
- **Domain knowledge:** Misassigns quantity types (e.g., `RadiantIntensity` for NMR pulse width). Cannot map NMR parameters to appropriate QUDT vocabulary.
- **Vocabulary grounding:** Cannot resolve CV IRIs to labels. Uses raw IRIs where human-readable labels needed.
- **Semantic routing:** Often skips evidence notes ("LLM router skipped the note") or selects targets but cannot parse attribute candidates.
- **Capacity under load:** 503 errors when processing multiple chunks in sequence. Cloud model shared capacity insufficient for sustained batch processing.
- **Context window:** 4096 token context limit. Large JCAMP-DX files exceed context. Chunks truncated, evidence lost.
- **Consistency:** Same input produces different output across retries. Repair attempts often produce different (still wrong) output.

### 6.2 Token Economy

| Dataset | Total Tokens | Requests | Avg Tokens/Request |
|---------|-------------|----------|--------------------|
| 1H_NMR (semantic) | 618,655 | 233 | 2,653 |
| 13C-13C_STM125 (fixed) | 507,122 | 178 | 2,849 |
| 13C-Gel-NMR(1) (fixed) | 419,212 | ~120 | ~3,493 |
| 13C-Gel-NMR (fixed) | 319,507 | ~95 | ~3,363 |

**Per-agent token breakdown (13C-13C_STM125, 507K total):**

| Agent | Input | Output | Requests | % of Total |
|-------|-------|--------|----------|-----------|
| chunk_extraction | 159,906 | 50,254 | 68 | 41% |
| measurement_semantic_router | 55,568 | 2,424 | 27 | 11% |
| evidence_critic | 46,463 | 20,902 | 19 | 13% |
| chunk_extraction_repair | 39,492 | 25,048 | 18 | 13% |
| metadata_completeness_evaluator | 33,600 | 3,445 | 19 | 7% |
| profile_field_vocab_selection | 29,008 | 982 | 7 | 6% |
| metadata_requirement_patcher | 12,159 | 4,104 | 5 | 3% |
| semantic_reconstruction | 7,109 | 816 | 5 | 2% |
| dataset_level_projection | 6,650 | 1,024 | 3 | 2% |
| semantic_reconstruction_synthesis | 7,629 | 539 | 7 | 2% |

**Observations:**
- Chunk extraction dominates token usage (41%)
- Repair attempts consume 13% — significant overhead for failed chunks
- Evidence critic and semantic router are expensive but produce limited value
- Total ~500K tokens per dataset at 4B model scale

### 6.3 Expected Improvements with Larger Model

Running same extractions with larger model (e.g., `gemma3:12b`, `gemma3:27b`, or `qwen2.5:14b`):

| Aspect | Expected Improvement | Confidence |
|--------|---------------------|------------|
| **Structured output compliance** | Significant — larger models follow JSON schemas more reliably. 90 failures -> likely 5-20. | High |
| **Instruction following** | Significant — multi-step prompts handled better. More evidence notes with correct categories. | High |
| **Domain knowledge** | Moderate — NMR-specific quantity types mapped more accurately. But still needs domain-specific fine-tuning for QUDT vocabulary. | Medium |
| **Vocabulary grounding** | Moderate — can resolve CV IRIs to labels with better semantic understanding. | Medium |
| **Semantic routing** | Significant — larger models better at zero-shot classification. Fewer "skipped" notes. | High |
| **Structured output repair** | Significant — repair attempts more likely to succeed. Less cascade failure. | High |
| **503 overload errors** | Depends — if local model, no cloud capacity issues. If cloud, same problem. | N/A |
| **Context window** | Depends on model — some larger models have 8K-32K context. Would reduce truncation. | Medium |
| **Profile conformance** | Moderate — better field filling, but profile conformance also depends on evidence availability, not just model capacity. | Medium |
| **Evidence query ledger** | Unknown — current empty ledger may be architectural bug, not model issue. | Low |
| **Requirement report** | Unknown — same concern as evidence query ledger. | Low |
| **Token cost** | Higher — larger models use more tokens per request. 500K -> 1-2M tokens per dataset. | High |
| **Processing time** | Slower per request, but fewer retries needed. Net effect uncertain. | Medium |

### 6.4 Model Recommendation

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Development/debugging | `gemma3:4b` (local) | Fast iteration, low cost |
| Batch evaluation | `qwen2.5:14b` or `gemma3:12b` (local) | Better structured output, no cloud 503s |
| Production quality target | `gemma3:27b` or `qwen2.5:32b` (local GPU) | Best instruction following, domain knowledge |
| Thesis demonstration | `gemma3:12b` (local) | Balance of quality and feasibility |

**Critical:** Use local model, not cloud. Cloud model 503 errors are unacceptable for batch processing. Local model eliminates capacity issues entirely.

---

## 7. Architecture Observations

### 7.1 Pipeline Design

SIMONE pipeline is well-architected conceptually:
- File ingestion -> text extraction -> chunking -> evidence extraction -> projection -> reconstruction -> validation
- Each stage has clear inputs/outputs
- Evidence-grounded approach (evidence notes with source traceability)
- Semantic reconstruction for missing fields
- Profile-aware projection with vocabulary selection

### 7.2 Implementation Gaps

| Gap | Impact | Severity |
|-----|--------|----------|
| Evidence query ledger always empty | Cannot assess evidence retrieval quality | Critical |
| Requirement report always empty | Cannot assess requirement fulfillment | Critical |
| Progress endpoint stale state | Monitoring unreliable | High |
| `force_rerun` race condition | Dataset crashes on re-run | High |
| `replace_existing_chunks` coupling | Silent failure when switching strategies | High |
| No graceful degradation | Failed chunks = permanent evidence loss | Medium |
| Vocabulary grounding incomplete | Raw IRIs instead of labels in output | Medium |
| Token budgeting uses conservative estimates | Chunks may be wrong size (tokenizer unavailable) | Low |

### 7.3 What Works Well

- Docker containerization — stable, reproducible
- File ingestion — handles JCAMP-DX, ZIP, Bruker raw, PDF, PNG
- Text quality classification — filters noise lines appropriately
- Evidence note structure — portable evidence with source context, match scores
- Schema validation — consistent JSON-LD output structure
- Token tracking — comprehensive per-agent metrics
- Task registry — async task management with status tracking
- Semantic reconstruction — attempts to fix missing fields via targeted LLM calls

---

## 8. Conclusions

### 8.1 Reliability

SIMONE workflow is **architecturally sound but operationally unreliable** at current model capacity. Infrastructure (Docker, Neo4j, Ollama) is stable. Pipeline logic has gaps (empty ledgers, stale progress, race conditions) that need fixing regardless of model choice.

### 8.2 Output Quality

Output quality is **insufficient for production metadata** with `gemma3:4b-cloud`. Schema validity is good, but profile conformance (0%), field coverage (7-30%), and evidence traceability (empty ledger) are unacceptable. A larger model would improve structured output compliance and field filling, but architectural gaps (empty evidence query ledger, empty requirement report) must be fixed first.

### 8.3 Model Capabilities

`gemma3:4b-cloud` is adequate for **prototyping and pipeline validation** but not for **quality extraction**. Key limitations: structured output failures, domain knowledge gaps, cloud capacity issues. A 12-14B local model would address most model-related issues. Architecture-related issues (empty ledgers, race conditions) require code fixes, not model upgrades.

### 8.4 Thesis Implications

For thesis evaluation chapter:
- **Present:** Pipeline architecture, evidence-grounded approach, per-agent token analysis, error taxonomy, processing time metrics
- **Discuss:** Model capacity as independent variable — same pipeline, different models, different quality
- **Acknowledge:** Current results are baseline at minimum viable model; larger models expected to improve coverage and conformance
- **Quantify:** 216 errors across 6 datasets, 0% profile conformance, 7-30% coverage — honest baseline
- **Frame:** SIMONE as research prototype, not production system — value in approach, not in current output quality

---

## 9. Next Steps

1. **Fix architectural gaps:** Evidence query ledger, requirement report, progress endpoint stale state
2. **Re-run with larger local model** (e.g., `gemma3:12b`) for quality comparison
3. **Complete batch run** — 15 datasets remaining
4. **Compare fixed_tokens vs semantic** chunking on same dataset/model
5. **Extract per-agent latency metrics** for performance analysis
6. **Validate against gold standard** references in `data/evaluation/references/`