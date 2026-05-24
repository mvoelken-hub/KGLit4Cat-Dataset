# Replace Pydantic AI with Custom Structured Completion Module

**Status:** Planning  
**Created:** 2026-05-24  
**Last updated:** 2026-05-24

## Goal

Remove the `pydantic-ai` dependency entirely. Replace it with a custom lightweight **structured completion** module in `app/ollama/` that calls Ollama's `/api/generate` endpoint (with `format` parameter for structured output), handling output parsing, retry logic, and token usage tracking natively.

**Terminology:** These are *structured completions*, not "agents." Every LLM interaction is a single-turn `/api/generate` call — no tool calling, no multi-turn conversation, no agent loop. The module is a **completion client**, not an agent framework.

**No tool calling.** The previous `initial_context` agent's two tools (`list_dataset_files`, `read_file_content`) are replaced by a **file ranking completion** + caller-side pre-fetching (see §File Ranking below).

## Why

- **Context pressure:** Pydantic AI routes all calls through `/v1/chat/completions` (OpenAI-compatible endpoint). The `PromptedOutput` mode injects ~1,134 tokens of schema-in-prompt instructions per call; `NativeOutput` is even worse (~1,230 tokens via `json_schema` response_format). Direct `/api/generate` with the `format` parameter costs only ~143 prompt tokens for the same structured output.
- **Timeout risk:** `/v1/chat/completions` with `json_schema` response_format times out (>300s) for large schemas on small models. `/api/generate` with `format` completes in ~8s (warm cache).
- **Unnecessary abstraction:** SIMONE uses a small, fixed subset of Pydantic AI's surface — 6 completion factories, 2 with `@agent.tool`, all using `PromptedOutput` or `StructuredDict+PromptedOutput`. No streaming, no conversation history, no multi-turn beyond tool calls, no output validators beyond schema validation.
- **Direct control:** Custom module eliminates ~1,000 wasted tokens per call and gives precise control over token budget accounting.

## Current Pydantic AI Usage Inventory

### Symbols imported

- `Agent` — `pydantic_ai` — 5 agent factories + `agents.py` helper
- `RunContext` — `pydantic_ai` — 5 agent factories (for `@agent.instructions`/`@agent.tool`)
- `ModelRetry` — `pydantic_ai` — `agents.py` (validation retry signal)
- `StructuredDict` — `pydantic_ai` — `agents.py` (dynamic schema output type)
- `PromptedOutput` — `pydantic_ai.output` — `agents.py` (schema-in-prompt output mode)
- `OllamaModel` — `pydantic_ai.models.ollama` — `client.py` (model adapter)
- `OllamaProvider` — `pydantic_ai.providers.ollama` — `client.py` (HTTP base URL config)
- `AgentRunError` — `pydantic_ai.exceptions` — `extraction.py` (error mapping in API layer)
- `TestModel` — `pydantic_ai.models.test` — 3 test files (mock agent)

### Completion usage patterns

All 8 LLM interactions follow the same pattern: build system message from deps, call `/api/generate` with `format=json_schema`, parse JSON response into a Pydantic model.

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

**No completion uses tools.** They are all single-turn `/api/generate` calls with the `format` parameter.

### What we need to replicate from `Agent.run()`

1. **System message + dynamic context** — callbacks that receive deps and return context strings
2. **Output parsing** — `format` param with JSON schema; parse JSON + validate against Pydantic model
3. **Output retries** — re-prompt with error appended on parse/validation failure
4. **Token usage** — `prompt_eval_count` / `eval_count` from Ollama response
5. **Model settings** — `temperature`, `seed` → Ollama `options`

### What we do NOT need from Pydantic AI

- Streaming (`run_stream`)
- Conversation history (`message_history`)
- Tool calling (removed — replaced by file ranking + pre-fetch)
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
   - Ranked output also flows into the **patching pipeline** — files are chunked and processed in rank order

2. **Caller-side file reading** — the service layer reads top-N files (by ranking) and injects their content directly into the initial context completion's prompt, exactly like how other completions already receive context via dynamic instructions.

This eliminates all tool calling and makes every LLM interaction a single `/api/generate` call.

---

## Architecture

### Module structure

```
app/ollama/
├── __init__.py          # Re-export public API
├── client.py            # OllamaClientWrapper (existing, unchanged except removing pydantic_ai)
├── runtime.py           # Model management, diagnostics (existing, unchanged)
├── completion.py        # NEW: generate_structured() — single /api/generate call + retry + parse
├── errors.py            # NEW: CompletionError, ModelRetry
└── usage.py             # NEW: RunUsage dataclass
```

No `tools.py` — tool calling is removed entirely.  
No `output.py` — schema formatting and output parsing live in `completion.py`.  
No `Agent` class — `generate_structured()` is a plain async function, not a class with decorators.

### Key design decisions

1. **Function, not class:** `generate_structured()` is a plain async function. No `Agent` class, no decorators, no `@instructions` pattern. The domain layer builds system messages directly from its own deps and passes them as arguments. This eliminates 90% of pydantic-ai's abstraction surface.
2. **Single endpoint:** All completions use `/api/generate` with `format` param. No `/api/chat`, no tool calling.
3. **Pydantic models stay:** Output types remain Pydantic `BaseModel` subclasses. We extract JSON schema from them for the `format` param, then parse the LLM response with `model.model_validate()`. `StructuredDict` is replaced with plain JSON schema validation.
4. **Retry on validation failure:** On JSON parse or Pydantic validation error, append the error message to the prompt and retry up to N times. Same semantics as pydantic-ai's `ModelRetry`.
5. **Token usage via Ollama response metadata:** `/api/generate` returns `prompt_eval_count` and `eval_count`. We map these to `input_tokens` / `output_tokens`.
6. **Dynamic context lives in the domain layer:** Each domain function builds its own system message from its deps dataclass. No framework dependency for context injection — just string concatenation.
7. **File ranking replaces tool-calling file reading:** New `create_file_ranking_completion()` produces a ranked file list; the service layer pre-reads top-N files and injects content into the initial context completion's prompt.

---

## Implementation Plan

### Phase 1: Build the custom module alongside pydantic-ai (no disruption)

**Branch:** `feature/custom-ollama-agent`

#### 1a. `app/ollama/errors.py`

```python
class CompletionError(Exception):
    """Base error for structured completion failures."""

class ModelRetry(CompletionError):
    """Signal to re-prompt with error message appended. Same semantics as pydantic-ai ModelRetry."""

class OutputParsingError(CompletionError):
    """JSON decode or schema validation failure."""

class MaxRetriesExceeded(CompletionError):
    """Exhausted output_retries without valid output."""
```

#### 1b. `app/ollama/usage.py`

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

#### 1c. `app/ollama/completion.py`

Core function — the entire module in one file:

```python
from ollama import AsyncClient
from pydantic import BaseModel

async def generate_structured(
    client: AsyncClient,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[BaseModel] | dict,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: bool = False,
) -> CompletionResult:
    """Single /api/generate call with format=json_schema, parse + validate + retry.
    
    Args:
        client: ollama.AsyncClient instance
        model: Ollama model name (e.g. "qwen3.5:4b")
        system: System message (static + dynamic context combined)
        prompt: User message (the extraction instruction)
        output_type: Pydantic model class for structured output, or dict for raw JSON schema
        retries: Max re-prompts on parse/validation failure
        temperature: Sampling temperature
        seed: Reproducibility seed
        think: Enable thinking mode (for qwen3.5 etc)
    
    Returns:
        CompletionResult with validated output and token usage.
    
    Raises:
        OutputParsingError: JSON decode or validation failure after all retries
        MaxRetriesExceeded: If retries exhausted
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
    total_usage = RunUsage()
    
    for attempt in range(retries + 1):
        response = await client.generate(
            model=model,
            prompt=current_prompt,
            system=system,
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
                output = parsed
            
            return CompletionResult(output=output, usage=total_usage)
        
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < retries:
                current_prompt = f"{current_prompt}\n\nError: {e}. Please return valid JSON."
                continue
            raise OutputParsingError(str(e)) from e
    
    raise MaxRetriesExceeded()


@dataclass
class CompletionResult(Generic[T]):
    output: T
    usage: RunUsage
```

That's it. The entire pydantic-ai `Agent` class, `PromptedOutput`, `StructuredDict`, `RunContext`, decorators, and tool calling — all replaced by one async function.

#### 1d. `client.py` changes

- Remove `from pydantic_ai.models.ollama import OllamaModel` and `from pydantic_ai.providers.ollama import OllamaProvider`
- Remove `agent_model` property and `_build_agent_model()` method
- Expose `ollama_client: AsyncClient` property for `generate_structured()` to use
- Keep all other functionality (embeddings, model management, etc.) unchanged

#### 1e. Write comprehensive tests

- `tests/test_ollama_completion.py` — test `generate_structured()` with mocked responses
  - Happy path: valid JSON → parsed Pydantic model
  - Happy path: dict output_type → raw dict
  - Retry on JSON decode failure
  - Retry on Pydantic validation failure  
  - `MaxRetriesExceeded` after exhausting retries
  - Token usage accumulation across retries
  - Markdown fence stripping
  - `think` parameter passthrough
- `tests/test_ollama_usage.py` — test `RunUsage` compatibility with `BudgetedUsage`
  - Attribute access (`input_tokens`, `output_tokens`, `requests`)
  - `merge()` accumulation
  - `getattr()` compatibility (used by `BudgetedUsage`)

### Phase 2: Migrate domain layer (sequential, one completion at a time)

**Goal:** Every completion factory rewritten to call `generate_structured()` directly. No pydantic-ai imports remain in domain code. Existing tests pass.

#### 2a. `agents.py` refactoring

- Remove imports of `Agent`, `ModelRetry`, `RunContext`, `StructuredDict`, `PromptedOutput`
- Import `generate_structured`, `CompletionResult`, `CompletionError`, `ModelRetry` from `app.ollama`
- `prompted_json_output()` → helper that returns `(schema, name, description)` tuple for prompt construction
- `structured_profile_output()` → same
- `create_schema_validated_agent()` → becomes a wrapper that calls `generate_structured()` with schema validation in the retry loop
- `validate_json_output_against_schema()` → keep as-is (uses `jsonschema`, not pydantic-ai)

#### 2b. NEW `file_ranking.py`

```python
async def extract_file_ranking(
    client: AsyncClient,
    *,
    model: str,
    file_list: list[FileInfo],
    retries: int = 2,
) -> CompletionResult:
    """Rank files by metadata extraction potential."""
    system = FILE_RANKING_SYSTEM_PROMPT
    prompt = "\n".join(f"- {f.path} ({f.extension}, {f.size_bytes} bytes)" for f in file_list)
    return await generate_structured(
        client, model=model, system=system, prompt=prompt,
        output_type=FileRankingResult, retries=retries,
    )
```

- `FileRankingResult` — ranked file list with relevance scores and reasoning
- `FileRankingDeps` — file list with extensions and sizes (or just pass list directly)

#### 2c. `initial_context.py` refactoring

- **Remove `@agent.tool` decorators** — `list_dataset_files` and `read_file_content` replaced by file ranking + pre-fetch
- Replace `pydantic_ai.Agent` with direct `generate_structured()` call
- Dynamic instructions become plain string concatenation: system = static_prompt + build_context_from_deps(deps)
- `extract_initial_context_from_data_package()` refactored:
  1. Call `extract_file_ranking()` to rank files
  2. Read top-N files using existing service-layer functions
  3. Call `generate_structured()` with pre-read content in the system message
- Remove `files_read` tracking from `InitialContextDeps`

#### 2d. `patch_draft.py`, `patch_quality.py`, `review_resolution.py` refactoring

All follow the same pattern:

```python
async def extract_patch_draft(deps: PatchDraftDeps, client: AsyncClient, model: str) -> FieldPatchResult:
    system = build_patch_extraction_context(deps)
    prompt = "Extract patches based on the provided context."
    result = await generate_structured(
        client, model=model, system=system, prompt=prompt,
        output_type=FieldPatchResult, retries=2,
    )
    return result.output
```

- No more `Agent` class instantiation
- No more `@agent.instructions` decorator — just a function that builds the system string from deps
- `RunContext[Deps]` → function parameter `(deps: PatchDraftDeps)`

#### 2e. `extraction.py` (API layer)

- Replace `from pydantic_ai.exceptions import AgentRunError` with `from app.ollama.errors import CompletionError`
- Map `CompletionError` → HTTP 502 (same behavior as before)

#### 2f. `client.py` final cleanup

- Remove last pydantic-ai references
- `OllamaClientWrapper` exposes `ollama_client: AsyncClient` for `generate_structured()`
- Model name from `OllamaClientWrapper.chat_model` (already a string property)

#### 2g. Update test files

- `test_extraction_agents.py`: Mock `generate_structured()` instead of `TestModel`
- `test_patch_quality.py`: Same
- `test_initial_context_extraction.py`: Same; add tests for file ranking flow
- `test_ollama_client.py`: Remove `OllamaModel`/`OllamaProvider` assertions
- Add `test_file_ranking.py`

#### 2h. Integration test

- Run full end-to-end extraction with a connected Ollama instance (or mock)
- Verify token usage reporting still works via `BudgetedUsage`
- Verify output parsing produces identical Pydantic models

### Phase 3: Remove pydantic-ai dependency

#### 3a. Remove `pydantic-ai-slim[openai]` from `pyproject.toml`

- Also remove `openai` if it was only pulled in by pydantic-ai's `[openai]` extra
- Keep `ollama`, `pydantic`, `jsonschema`, `httpx` (still needed)

#### 3b. Remove all pydantic-ai imports

```bash
grep -rn "pydantic_ai\|pydantic-ai" backend/app/ backend/tests/
```

#### 3c. Clean up unused dependencies

```bash
cd backend && poetry remove pydantic-ai-slim
# Verify: poetry install && poetry run pytest
```

#### 3d. Update `app/ollama/__init__.py`

```python
from app.ollama.completion import generate_structured, CompletionResult
from app.ollama.errors import CompletionError, ModelRetry, OutputParsingError, MaxRetriesExceeded
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