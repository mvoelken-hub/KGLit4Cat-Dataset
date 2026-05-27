# Replace Pydantic AI with Custom Structured Completion Module

**Status:** Phase 1.5 complete - Phase 2 domain migration pending  
**Created:** 2026-05-24  
**Last updated:** 2026-05-27

## Goal

Remove the `pydantic-ai` dependency entirely. Replace it with a custom lightweight **structured completion** module in `app/ollama/` that calls Ollama's `/api/generate` endpoint (with `format` parameter for structured output), handling output parsing, retry logic, and token usage tracking natively.

**Terminology:** These are *structured completions*, not "agents." In the target architecture, every LLM interaction is a single-turn `/api/generate` call — no tool calling, no multi-turn conversation, no agent loop. The module is a **completion client**, not an agent framework.

**Target state: no tool calling.** The current `initial_context` agent's two tools (`list_dataset_files`, `read_file_content`) will be replaced by a **file ranking completion** + caller-side pre-fetching (see §File Ranking below).

## Why

- **Context pressure:** Pydantic AI routes all calls through `/v1/chat/completions` (OpenAI-compatible endpoint). The `PromptedOutput` mode injects ~1,134 tokens of schema-in-prompt instructions per call; `NativeOutput` is even worse (~1,230 tokens via `json_schema` response_format). Direct `/api/generate` with the `format` parameter costs only ~143 prompt tokens for the same structured output.
- **Timeout risk:** `/v1/chat/completions` with `json_schema` response_format times out (>300s) for large schemas on small models. `/api/generate` with `format` completes in ~8s (warm cache).
- **Unnecessary abstraction:** SIMONE uses a small, fixed subset of Pydantic AI's surface - 7 existing completion factories plus one schema-validation helper, with tool calling only in `initial_context`. No streaming, no conversation history, no graph/agent orchestration, and no output validators beyond schema validation.
- **Direct control:** Custom module eliminates ~1,000 wasted tokens per call and gives precise control over token budget accounting.

## Current Pydantic AI Usage Inventory

### Symbols imported

- `Agent` — `pydantic_ai` — 7 domain completion factories + `agents.py` schema helper
- `RunContext` — `pydantic_ai` — 7 domain completion factories (for `@agent.instructions`/`@agent.tool`) + `agents.py`
- `ModelRetry` — `pydantic_ai` — `agents.py` (validation retry signal)
- `StructuredDict` — `pydantic_ai` — `agents.py` (dynamic schema output type)
- `PromptedOutput` — `pydantic_ai.output` — `agents.py` (schema-in-prompt output mode)
- `OllamaModel` — `pydantic_ai.models.ollama` — `client.py` (model adapter)
- `OllamaProvider` — `pydantic_ai.providers.ollama` — `client.py` (HTTP base URL config)
- `AgentRunError` — `pydantic_ai.exceptions` — `extraction.py` (error mapping in API layer)
- `TestModel` — `pydantic_ai.models.test` — 3 test files (mock agent)

### Completion usage patterns

Target state: all 8 LLM interactions follow the same pattern: build system message from deps, call `/api/generate` with `format=json_schema`, parse JSON response, and validate it either with a Pydantic model or raw JSON Schema.

| Completion | Factory | Output Type | Dynamic Context | Deps Type |
|---|---|---|---|---|
| File Ranking | `create_file_ranking_completion()` **NEW** | `FileRankingResult` **NEW** | Yes (file list + metadata) | `FileRankingDeps` **NEW** |
| Initial Context | `create_initial_context_completion()` → refactor | `InitialContext` | Yes (pre-read files) | `InitialContextDeps` (simplified) |
| Patch Draft | `create_patch_draft_completion()` | `FieldPatchResult` | Yes | `PatchDraftDeps` |
| Patch Discovery | `create_patch_discovery_completion()` | `PatchDiscoveryResult` | Yes | `PatchDiscoveryDeps` |
| Schema Patch | `create_schema_patch_completion()` | `SchemaPatchResult` | Yes | `SchemaPatchDeps` |
| Schema Repair | `create_schema_repair_completion()` | `SchemaPatchResult` | Yes | `SchemaRepairDeps` |
| Patch Quality | `create_patch_quality_completion()` | `PatchQualityReport` | Yes | `PatchQualityDeps` |
| Review Resolution | `create_patch_review_resolution_completion()` | `PatchReviewResolution` | Yes | `PatchReviewResolutionDeps` |

**Target state:** no completion uses tools. Current state: `initial_context` still uses two Pydantic AI tools until Phase 2d replaces them with file ranking plus caller-side prefetch.

### What we need to replicate from `Agent.run()`

1. **System message + dynamic context** — callbacks that receive deps and return context strings
2. **Output parsing** — `format` param with JSON schema; parse JSON + validate against Pydantic model or raw JSON Schema
3. **Output repair retries** — on parse/validation failure, send only the failed response plus error to a repair prompt
4. **Token usage** — `prompt_eval_count` / `eval_count` from Ollama response
5. **Model settings** — `temperature`, `seed` → Ollama `options`

### What we do NOT need from Pydantic AI

- Streaming (`run_stream`)
- Conversation history (`message_history`)
- Tool calling in the final architecture (the current `initial_context` tools are removed in Phase 2d)
- `NativeOutput` (tested: worse than PromptedOutput for self-hosted Ollama)
- `@agent.output_validator` — only used by `create_schema_validated_agent()`, handled in retry loop
- Graph/agent orchestration features
- Dependency injection beyond what a simple dataclass carries
- `/api/chat` endpoint entirely
- `Agent` class with decorators — replaced by `generate_structured()` function

### File Ranking (NEW)

The previous `initial_context` agent used two tools (`list_dataset_files`, `read_file_content`) in a multi-turn loop to discover and read files. This is replaced by:

1. **File ranking completion** — a single `/api/generate` call that receives a file list (paths + extensions + sizes) and returns a ranked list by metadata potential.
   - Input: file paths with extension and size hints
   - Output: `FileRankingResult` — ranked list of file paths with relevance scores and brief reasoning
   - First use is initial context only. Reordering the patching pipeline by rank is deferred to a later optimization.

2. **Caller-side file reading** — the service layer reads top-N files (by ranking) and injects their content directly into the initial context completion's prompt, exactly like how other completions already receive context via dynamic instructions.

This removes the remaining tool-calling path and makes every LLM interaction a single `/api/generate` call in the target architecture.

---

## Architecture

### Module structure

```
app/ollama/
├── __init__.py          # Re-export public API ✅
├── client.py            # OllamaClientWrapper (unchanged except `ollama_client` property + pydantic_ai still present)
├── runtime.py           # Model management, diagnostics (existing, unchanged)
├── completion.py        # ✅ NEW: generate_structured() — single /api/generate call + retry + parse
├── errors.py            # ✅ NEW: CompletionError, ModelRetry
└── usage.py             # ✅ NEW: RunUsage dataclass
```

No `tools.py` in the target architecture — tool calling is removed entirely.  
No `output.py` — schema formatting and output parsing live in `completion.py`.  
No `Agent` class — `generate_structured()` is a plain async function, not a class with decorators.

*(Phase 2 note: `client.py` still builds `self.agent_model` because the domain layer still references it. That will be removed in Phase 2g.)*

### Key design decisions

1. **Function, not class:** `generate_structured()` is a plain async function. No `Agent` class, no decorators, no `@instructions` pattern. The domain layer builds system messages directly from its own deps and passes them as arguments. This eliminates 90% of pydantic-ai's abstraction surface.
2. **Single endpoint:** All completions use `/api/generate` with `format` param. No `/api/chat`, no tool calling.
3. **Pydantic models stay:** Output types remain Pydantic `BaseModel` subclasses where static models already exist. We extract JSON schema from them for the `format` param, then parse the LLM response with `model.model_validate()`.
4. **Raw JSON Schema is validated locally:** Dynamic schema outputs that previously used `StructuredDict` are passed as dict schemas and validated with `jsonschema` inside the retry loop.
5. **Repair on validation failure:** On JSON parse, Pydantic validation, or raw JSON Schema validation failure, do not rerun the original task. Send only the failed response plus validation error to a focused repair prompt. A dedicated `repair_model` can be supplied; otherwise repair uses the same model. Empty model output is not repairable and fails immediately with `EmptyResponseError`.
6. **`MaxRetriesExceeded` is the exhausted-retry error:** `OutputParsingError` describes the underlying parse/validation cause, but callers should catch `MaxRetriesExceeded` when all attempts fail.
7. **Token usage via Ollama response metadata:** `/api/generate` returns `prompt_eval_count` and `eval_count`. We map these to `input_tokens` / `output_tokens`.
8. **Dynamic context lives in the domain layer:** Each domain function builds its own system message from its deps dataclass. No framework dependency for context injection - just string concatenation.
9. **File ranking replaces tool-calling file reading:** New `create_file_ranking_completion()` produces a ranked file list; the service layer pre-reads top-N files and injects content into the initial context completion's prompt.

---

## Implementation Plan

### Phase 1: Build the custom module alongside pydantic-ai (no disruption)

**Branch:** `feature/custom-ollama-agent`

#### 1a. `app/ollama/errors.py` ✅

```python
class CompletionError(Exception):
    """Base error for structured completion failures."""

class ModelRetry(CompletionError):
    """Signal that model output should be retried or repaired."""

class OutputParsingError(CompletionError):
    """JSON decode or schema validation failure."""

class EmptyResponseError(OutputParsingError):
    """Ollama returned no response text to parse or repair."""

class MaxRetriesExceeded(CompletionError):
    """Exhausted output_retries without valid output."""
```

#### 1b. `app/ollama/usage.py` ✅

```python
@dataclass
class RunUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    details: dict[str, int] = field(default_factory=dict)
    
    def merge(self, other: RunUsage) -> RunUsage: ...
```

- Drop-in replacement for `pydantic_ai.result.RunUsage` (subset used: `input_tokens`, `output_tokens`, `requests`, `details`)
- `BudgetedUsage` in `token_budget.py` uses `getattr(self._usage, name)` — works with any object that has these attributes

#### 1c. `app/ollama/completion.py` ✅

Condensed core function shape. The implementation also includes helpers for markdown fence stripping, raw JSON Schema validation, and exhausted-retry error details:

```python
from pydantic import BaseModel
from app.ollama.client import OllamaClientWrapper

async def generate_structured(
    client: OllamaClientWrapper,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[BaseModel] | dict,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: bool = False,
    repair_model: str | None = None,
) -> CompletionResult:
    """Single /api/generate call with format=json_schema, parse + validate + retry.
    
    Args:
        client: OllamaClientWrapper instance
        model: Ollama model name (e.g. "qwen3.5:4b")
        system: System message (static + dynamic context combined)
        prompt: User message (the extraction instruction)
        output_type: Pydantic model class for structured output, or dict for raw JSON Schema validation
        retries: Max repair attempts on parse/validation failure
        temperature: Sampling temperature
        seed: Reproducibility seed
        think: Enable thinking mode (for qwen3.5 etc)
    
    Returns:
        CompletionResult with validated output and token usage.
    
    Raises:
        MaxRetriesExceeded: If retries are exhausted after parse/validation failures
        CompletionError: For Ollama API errors
    """
    # 1. Extract JSON schema from output_type
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        schema = output_type.model_json_schema()
    elif isinstance(output_type, dict):
        schema = output_type
    else:
        raise CompletionError(f"Unsupported output_type: {output_type}")
    
    # 2. Attempt completion with retry loop
    current_prompt = prompt
    current_system = system
    current_model = model
    total_usage = RunUsage()
    
    for attempt in range(retries + 1):
        response = await client.ollama_client.generate(
            model=current_model,
            prompt=current_prompt,
            system=current_system,
            format=schema,
            options={"temperature": temperature, "seed": seed},
            think=think,
        )
        
        total_usage.input_tokens += getattr(response, 'prompt_eval_count', 0) or 0
        total_usage.output_tokens += getattr(response, 'eval_count', 0) or 0
        total_usage.requests += 1
        
        try:
            raw = response.response
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            
            parsed = json.loads(raw)
            
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                output = output_type.model_validate(parsed)
            else:
                output = validate_json_schema(parsed, schema)
            
            return CompletionResult(output=output, usage=total_usage)
        
        except (json.JSONDecodeError, ValidationError, JsonSchemaValidationError) as e:
            if attempt < retries:
                current_system = REPAIR_SYSTEM_PROMPT
                current_prompt = build_repair_prompt(
                    failed_response=raw,
                    error=OutputParsingError(str(e)),
                )
                current_model = repair_model or model
                continue
            raise MaxRetriesExceeded(last_error=OutputParsingError(str(e))) from e
    
    raise MaxRetriesExceeded(last_error=last_error)


@dataclass
class CompletionResult(Generic[T]):
    output: T
    usage: RunUsage
```

This is the key replacement surface. The pydantic-ai `Agent` class, `PromptedOutput`, `StructuredDict`, `RunContext`, decorators, and tool calling disappear as domain call sites migrate to this function.

#### 1d. `client.py` changes ✅ (partial)

- ✅ **Added** `ollama_client` property returning `self.chat_client` (the underlying `AsyncClient`); `generate_structured()` accepts the wrapper and calls through this property
- ⏳ **Deferred** removal of `OllamaModel`/`OllamaProvider` imports and `agent_model` — the domain layer (`extraction_service.py`) still references `self.ollama_client.agent_model`. Will be removed in Phase 2g once all domain code migrates to passing model name strings.
- All other functionality (embeddings, model management, etc.) unchanged

#### 1e. Comprehensive tests ✅

- ✅ `tests/test_ollama_completion.py` — 31 tests covering:
  - Happy path: valid JSON → parsed Pydantic model ✅
  - Happy path: dict output_type → JSON Schema validated raw dict ✅
  - Repair retry on JSON decode failure ✅
  - Repair retry on Pydantic validation failure ✅
  - Optional dedicated `repair_model` for repair attempts ✅
  - `MaxRetriesExceeded` as the single exhausted-retry error ✅
  - Raw JSON Schema validation failures retry and then raise `MaxRetriesExceeded` ✅
  - Invalid raw JSON Schema raises `CompletionError` before calling Ollama ✅
  - Empty response fails immediately without repair ✅
  - Token usage accumulation across retries ✅
  - Markdown fence stripping ✅
  - `think` / `temperature` / `seed` / `num_ctx` / `keep_alive` passthrough ✅
  - `BudgetedUsage` `getattr` compatibility ✅
  - `RunUsage.merge()` and `__add__` ✅
  - `RunUsage.from_ollama_response()` ✅

**Note:** `tests/test_ollama_usage.py` was merged into `test_ollama_completion.py` (single file is sufficient for the small surface). `test_ollama_client.py` still passes (14 tests) — it tests the existing client and still references `agent_model` (expected until Phase 2g).

---

### Pitfalls & Deviations (from actual implementation)

**1d — `client.py`:** Did not remove pydantic-ai imports yet because `extraction_service.py` still reads `self.ollama_client.agent_model`. Removing it now would break the app. The plan now is: keep `_build_agent_model()` and `agent_model` alive through Phase 2, remove them in Phase 2g when the final pydantic-ai cleanup happens.

**1c — `completion.py` signature vs. plan:** Added `num_ctx`, `keep_alive`, and `repair_model` parameters beyond what the original plan specified. They are essential for runtime control (context length), warm-cache integration, and focused repair attempts. Also added `from_ollama_response()` classmethod to `RunUsage` so callers can build usage from responses directly.

**1c — `_strip_markdown_fences()` implementation:** The spec used `split('\n', 1)[1]` which was fragile for edge cases like nested or partial fences. Implemented a regex-based `_MARKDOWN_FENCE_RE` that handles ` ```json`, ` ``` `, stray backticks, and no fences gracefully.

**1e — test file consolidation:** The spec suggested `test_ollama_usage.py` as a separate file, but since `RunUsage` is small, all its tests were folded into `test_ollama_completion.py` under the `RunUsageTests` class. BudgetedUsage compatibility is verified directly in `test_budgeted_usage_getattr_compat`.

**1c — Error propagation:** `Ollama API errors` become `CompletionError`. Parse and validation failures are represented as `OutputParsingError` while retrying. Exhausted retries always raise `MaxRetriesExceeded`, with the final `OutputParsingError` attached as `last_error` and mirrored in `details`.

**1.5 — Raw JSON Schema validation:** The first implementation parsed dict-schema output but did not validate it locally. This is now fixed with `jsonschema` validation inside the retry loop, matching the old `StructuredDict` + `@agent.output_validator` behavior.

**1.5 — Repair retries instead of full regeneration:** Parse/validation retries no longer resend the original extraction prompt. The first call performs the task. Later attempts use a narrow repair system prompt with only the failed response and the validation error, plus the same `format` schema. This keeps overloaded extraction prompts out of the retry path and lets an optional `repair_model` handle correction.

**1.5 — Empty output is terminal:** If Ollama returns no response text, there is nothing to repair. `generate_structured()` raises `EmptyResponseError` immediately so the domain/service layer can decide whether to abort, reschedule, or apply domain-specific fallback behavior.

---

### Phase 1.5: Harden the completion contract ✅

**Goal:** Make the custom completion module safe enough to migrate domain code onto it.

Completed decisions:

- `MaxRetriesExceeded` is the single public exhausted-retry error.
- `OutputParsingError` remains useful as the underlying parse/validation cause and is attached to `MaxRetriesExceeded.last_error`. Empty model output raises `EmptyResponseError` directly because it is not repairable.
- Raw dict `output_type` values are treated as JSON Schemas and validated locally with `jsonschema`.
- Invalid JSON Schemas fail fast with `CompletionError` before any Ollama call.
- Repair prompts include only the failed response plus JSON parse, Pydantic validation, or JSON Schema validation details.
- Tests cover successful dict-schema validation, schema-validation repair, exhausted schema-validation repair, invalid schema rejection, dedicated repair model selection, empty output failing without repair, and last-error details.

Acceptance criteria:

- `uv run pytest tests/test_ollama_completion.py` passes.
- A dict schema with required fields rejects `{}` and repairs using the failed response plus validation error.
- Persistent parse/validation failure raises `MaxRetriesExceeded`, not `OutputParsingError`.
- API-layer migration can map `CompletionError` subclasses to HTTP 502 without knowing the low-level cause.

---

### Phase 2: Migrate domain layer (sequential, one completion at a time)

**Goal:** Every completion call uses `generate_structured()` directly. No pydantic-ai imports remain in `backend/app` or backend tests. Existing behavior and token accounting stay intact.

#### 2a. Shared migration shape

- Do not create a replacement `Agent` abstraction.
- Move each `@agent.instructions` callback into a plain context-builder function that accepts the deps dataclass.
- Keep domain entry points such as `review_patch_semantic_quality()` and `discover_patch_information()` stable where possible.
- Pass `client=ollama_client`, `model=ollama_client.chat_model`, and `num_ctx=ollama_client.max_context_length` from service-layer call sites.
- Consider adding a small `OllamaClientWrapper.generate_structured(...)` convenience method only if call sites become noisy. The core API remains the plain function.
- Keep `validate_json_output_against_schema()` for non-LLM schema validation paths such as initial draft generation. Remove `prompted_json_output()`, `structured_profile_output()`, and `create_schema_validated_agent()` once no tests or call sites need them.

#### 2b. Migrate non-tool completions first

Order matters: start with completions that only transform prompt + deps into a Pydantic model.

1. `patch_quality.py` - `create_patch_quality_agent()` -> direct `generate_structured(..., output_type=PatchQualityReport)`.
2. `review_resolution.py` - `create_patch_review_resolution_agent()` -> direct `generate_structured(..., output_type=PatchReviewResolution)`.
3. `patch_draft.py` discovery path - `create_patch_discovery_agent()` -> direct `generate_structured(..., output_type=PatchDiscoveryResult)`.
4. `patch_draft.py` schema patch path - `create_schema_patch_agent()` -> direct `generate_structured(..., output_type=SchemaPatchResult)`.
5. `patch_draft.py` schema repair path - `create_schema_repair_agent()` -> direct `generate_structured(..., output_type=SchemaPatchResult)`.
6. Legacy `create_patch_draft_agent()` / `extract_field_patch_candidates()` - migrate after the active schema-late path is stable, or remove if no longer used.

Each migration should:

- Keep the public async function return type unchanged.
- Preserve token usage callbacks and `BudgetedUsage` metadata.
- Preserve `temperature=0.0`, `seed=42`, `retries=DEFAULT_OUTPUT_RETRIES`, `keep_alive=-1`, and runtime `num_ctx`.
- Add tests that mock `generate_structured()` and assert the built `system`, `prompt`, `output_type`, retry count, and token usage callback.

#### 2c. Add `file_ranking.py` for initial context only

Initial context is behavior-changing because it removes tool calls. Keep it isolated.

```python
async def extract_file_ranking(
    client: OllamaClientWrapper,
    *,
    model: str,
    file_list: list[FileInfo],
    retries: int = DEFAULT_OUTPUT_RETRIES,
) -> CompletionResult[FileRankingResult]:
    """Rank files by metadata extraction potential."""
    system = FILE_RANKING_SYSTEM_PROMPT
    prompt = build_file_ranking_prompt(file_list)
    return await generate_structured(
        client,
        model=model,
        system=system,
        prompt=prompt,
        output_type=FileRankingResult,
        retries=retries,
    )
```

- `FileRankingResult`: ranked paths with relevance score and short reason.
- The first use is `initial_context` only.
- Do not reorder the patching pipeline by file ranking in this phase. That is a separate optimization with broader behavior impact.

#### 2d. Refactor `initial_context.py`

- Remove `@agent.tool` usage and `files_read` tracking.
- Rank files by metadata potential.
- Caller-side read top-N files using existing data package methods.
- Build one system/context string containing file list, selected file content snippets, package metadata, limits, and extraction rules.
- Call `generate_structured(..., output_type=InitialContext)`.
- Preserve token usage reporting, including estimated input tokens for the final initial-context call. Decide separately whether file-ranking usage is recorded as `file_ranking` or folded into `initial_context`.

#### 2e. API and service wiring

- Replace `from pydantic_ai.exceptions import AgentRunError` with `from app.ollama.errors import CompletionError`.
- Map `CompletionError` and its subclasses to HTTP 502.
- Change extraction service call sites from `model=self.ollama_client.agent_model` to passing the wrapper or direct `client/model/num_ctx` values.
- Preserve existing operation names in token usage output unless a new `file_ranking` bucket is intentionally added.

#### 2f. Tests

- Replace `TestModel`-based tests with `generate_structured()` mocks or fake Ollama clients.
- Update `test_extraction_agents.py`, `test_patch_quality.py`, `test_initial_context_extraction.py`, and `test_ollama_client.py`.
- Add `test_file_ranking.py`.
- Keep regression tests for token usage, retry budgets, protected fields, schema-late patch validation, and initial-context file limits.

#### 2g. Final pydantic-ai cleanup

- Remove `OllamaModel`, `OllamaProvider`, `_build_agent_model()`, and `agent_model` from `client.py`.
- Remove all `pydantic_ai` imports from `backend/app` and backend tests.
- Remove `pydantic-ai-slim[openai]` from dependencies and refresh `uv.lock`.

#### 2h. Integration check

- Run the full backend suite.
- Run a mocked end-to-end extraction path to verify output objects, token usage, and error mapping.
- If a connected Ollama instance is available, run one small real extraction smoke test and compare prompt token usage against the benchmark table.

### Phase 3: Remove pydantic-ai dependency

#### 3a. Remove `pydantic-ai-slim[openai]` from `pyproject.toml`

- Also remove `openai` if it was only pulled in by pydantic-ai's `[openai]` extra
- Keep `ollama`, `pydantic`, `jsonschema`, `httpx` (still needed)

#### 3b. Remove all pydantic-ai imports

```bash
rg "pydantic_ai|pydantic-ai" backend/app backend/tests
```

#### 3c. Clean up unused dependencies

```bash
cd backend
uv remove pydantic-ai-slim
uv sync
uv run pytest
```

#### 3d. Update `app/ollama/__init__.py`

```python
from app.ollama.completion import generate_structured, CompletionResult
from app.ollama.errors import CompletionError, EmptyResponseError, ModelRetry, OutputParsingError, MaxRetriesExceeded
from app.ollama.usage import RunUsage
from app.ollama.client import OllamaClientWrapper
```

---

## Ongoing Maintenance

This document is the living source of truth. Update it as implementation progresses:

- Mark phases with ✅ when complete
- Add pitfalls and deviations as they emerge
- Update the token-overhead table after benchmarking the custom module

## Benchmark Reference (from previous session)

| Approach | Prompt tokens | Valid JSON? | Latency |
|---|---|---|---|
| `/api/generate` bare prompt | 20 | No (raw text) | ~3s |
| `/api/generate` format=schema think=false | **143** | Yes | ~38s cold, ~8s warm |
| `/api/generate` schema-in-text | 1,154 | No (markdown fences) | — |
| `/v1/chat/completions` json_schema | 1,248 | Yes | 95s, timeout risk |
| `/v1/chat/completions` schema-in-system-msg | **timeout** | — | >300s |
| Pydantic AI `NativeOutput` | 3,953 | Partial | 221s |
| Pydantic AI `PromptedOutput` | **timeout** | — | >600s |

**Target:** All completions → `/api/generate` with `format` (~143 tokens each).
