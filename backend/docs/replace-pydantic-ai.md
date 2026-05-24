# Replace Pydantic AI with Custom Ollama Agent Module

**Status:** Planning  
**Created:** 2026-05-24  
**Last updated:** 2026-05-24

## Goal

Remove the `pydantic-ai` dependency entirely. Replace it with a custom lightweight agent module in `app/ollama/` that calls Ollama's `/api/generate` endpoint (with `format` parameter for structured output), handling output parsing, retry logic, context injection, and token usage observation natively.

**No tool calling — all agents use single-turn `/api/generate`.** The previous `initial_context` agent's two tools (`list_dataset_files`, `read_file_content`) are replaced by a new **file ranking agent** + caller-side pre-fetching (see §File Ranking below).

## Why

- **Context pressure:** Pydantic AI routes all calls through `/v1/chat/completions` (OpenAI-compatible endpoint). The `PromptedOutput` mode injects ~1,134 tokens of schema-in-prompt instructions per call; `NativeOutput` is even worse (~1,230 tokens via `json_schema` response_format). Direct `/api/generate` with the `format` parameter costs only ~143 prompt tokens for the same structured output.
- **Timeout risk:** `/v1/chat/completions` with `json_schema` response_format times out (>300s) for large schemas on small models. `/api/generate` with `format` completes in ~8s (warm cache).
- **Unnecessary abstraction:** SIMONE uses a small, fixed subset of Pydantic AI's surface — 6 agents, 2 with `@agent.tool`, all using `PromptedOutput` or `StructuredDict+PromptedOutput`. No streaming, no conversation history, no multi-turn beyond tool calls, no output validators beyond schema validation.
- **Direct control:** Custom module enables per-agent routing (e.g., `/api/generate` for extraction agents without tools, `/api/chat` for the one agent that uses tools) and better token budget accounting.

## Current Pydantic AI Usage Inventory

### Symbols imported

| Symbol | From | Used in |
|--------|------|---------|
| `Agent` | `pydantic_ai` | 5 agent factories + `agents.py` helper |
| `RunContext` | `pydantic_ai` | 5 agent factories (for `@agent.instructions`/`@agent.tool`) |
| `ModelRetry` | `pydantic_ai` | `agents.py` (validation retry signal) |
| `StructuredDict` | `pydantic_ai` | `agents.py` (dynamic schema output type) |
| `PromptedOutput` | `pydantic_ai.output` | `agents.py` (schema-in-prompt output mode) |
| `OllamaModel` | `pydantic_ai.models.ollama` | `client.py` (model adapter) |
| `OllamaProvider` | `pydantic_ai.providers.ollama` | `client.py` (HTTP base URL config) |
| `AgentRunError` | `pydantic_ai.exceptions` | `extraction.py` (error mapping in API layer) |
| `TestModel` | `pydantic_ai.models.test` | 3 test files (mock agent) |

### Agent usage patterns

| Agent | Factory | Output Type | Dynamic Instructions | `deps_type` |
|-------|---------|-------------|---------------------|-------------|
| File Ranking | `create_file_ranking_agent()` **NEW** | `FileRankingResult` **NEW** | Yes (file list + metadata) | `FileRankingDeps` **NEW** |
| Initial Context | `create_initial_context_agent()` → refactor | `PromptedOutput(InitialContext)` → direct | Yes (pre-read files) | `InitialContextDeps` (simplified) |
| Patch Draft | `create_patch_draft_agent()` | `PromptedOutput(FieldPatchResult)` → direct | Yes | `PatchDraftDeps` |
| Patch Discovery | `create_patch_discovery_agent()` | `PromptedOutput(PatchDiscoveryResult)` → direct | Yes | `PatchDiscoveryDeps` |
| Schema Patch | `create_schema_patch_agent()` | `PromptedOutput(SchemaPatchResult)` → direct | Yes | `SchemaPatchDeps` |
| Schema Repair | `create_schema_repair_agent()` | `PromptedOutput(SchemaPatchResult)` → direct | Yes | `SchemaRepairDeps` |
| Patch Quality | `create_patch_quality_agent()` | `PromptedOutput(PatchQualityReport)` → direct | Yes | `PatchQualityDeps` |
| Review Resolution | `create_patch_review_resolution_agent()` | `PromptedOutput(PatchReviewResolution)` → direct | Yes | `PatchReviewResolutionDeps` |
| Schema Validated | `create_schema_validated_agent()` | `StructuredDict+PromptedOutput` → direct | Static or schema-in-text | `deps_type` param |

**No agent uses tools.** All agents are single-turn `/api/generate` calls with `format` parameter.

### What `Agent.run()` provides that we need to replicate

1. **System message injection** — `instructions` param → Ollama `system` field
2. **Dynamic instructions** — callbacks that receive deps and return context strings
3. **Output parsing** — `PromptedOutput` → `/api/generate` `format` param with JSON schema; parse + validate
4. **Output retries** — `output_retries` param; `ModelRetry` re-prompts with error appended
5. **Token usage** — `prompt_eval_count` / `eval_count` from Ollama response
6. **Model settings** — `temperature`, `seed` → Ollama `options`
7. **Error handling** — `AgentRunError` wraps model/output failures

### What we do NOT need from Pydantic AI

- Streaming (`run_stream`)
- Conversation history (`message_history`)
- Tool calling (removed — replaced by file ranking agent + pre-fetch)
- `NativeOutput` (tested: worse than PromptedOutput for self-hosted Ollama)
- `@agent.output_validator` — only used by `create_schema_validated_agent()`, we'll handle directly
- Graph/agent orchestration features
- Dependency injection beyond what a simple dataclass carries
- `/api/chat` endpoint entirely (all agents use `/api/generate`)

### File Ranking (NEW)

The previous `initial_context` agent used two tools (`list_dataset_files`, `read_file_content`) in a multi-turn loop to discover and read files. This is replaced by:

1. **File ranking agent** — a new single-turn `/api/generate` call that receives a file list (paths + extensions + sizes) and returns a ranked list by metadata potential.
   - Input: file paths with extension and size hints
   - Output: `FileRankingResult` — ranked list of file paths with relevance scores and brief reasoning
   - Ranked output also flows into the **patching pipeline** — files are chunked and processed in rank order

2. **Caller-side file reading** — the service layer reads top-N files (by ranking) and injects their content directly into the initial context agent's prompt, exactly like how other agents already receive context via dynamic instructions.

This eliminates all tool calling and makes every agent a single `/api/generate` call.

---

## Architecture

### Module structure

```
app/ollama/
├── __init__.py          # Re-export public API
├── client.py            # OllamaClientWrapper (existing, unchanged except removing pydantic_ai)
├── runtime.py           # Model management, diagnostics (existing, unchanged)
├── agent.py             # NEW: Agent class — single-turn /api/generate only
├── output.py            # NEW: Output parsing, schema formatting, validation
├── retry.py             # NEW: ModelRetry exception, retry loop logic
├── usage.py             # NEW: RunUsage dataclass, token counting
└── errors.py            # NEW: AgentRunError and custom error types
```

No `tools.py` — tool calling is removed entirely.

### Key design decisions

1. **Single endpoint:** All agents use `/api/generate` with `format` param. No `/api/chat`, no tool calling.
2. **Pydantic models stay:** Output types remain Pydantic `BaseModel` subclasses. We extract JSON schema from them for the `format` param, then parse the LLM response with `model.model_validate()`. `StructuredDict` is replaced with plain JSON schema validation.
3. **ModelRetry as exception:** Keep the same pattern — `ModelRetry(message)` raised in validation/retry callbacks triggers a re-prompt with the error message appended to the prompt.
4. **Token usage via Ollama response metadata:** `/api/generate` returns `prompt_eval_count` and `eval_count`. We map these to `input_tokens` / `output_tokens`.
5. **File ranking replaces tool-calling file reading:** New `create_file_ranking_agent()` produces a ranked file list; the service layer pre-reads top-N files and injects content into the initial context agent's prompt.

---

## Implementation Plan

### Phase 1: Build the custom module alongside pydantic-ai (no disruption)

**Branch:** `feature/custom-ollama-agent`

#### 1a. `app/ollama/errors.py`

- `AgentRunError(Exception)` — wraps any error from the agent run loop
- `ModelRetry(Exception)` — signal to re-prompt (same semantics as pydantic-ai)
- `OutputParsingError(AgentRunError)` — JSON decode or schema validation failure
- `MaxRetriesExceeded(AgentRunError)` — exhausted output_retries

#### 1b. `app/ollama/usage.py`

```python
@dataclass
class RunUsage:
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    details: dict[str, int] = field(default_factory=dict)
    
    def merge(self, other: RunUsage) -> RunUsage: ...
```

- Drop-in replacement for `pydantic_ai.result.RunUsage` (subset used: `input_tokens`, `output_tokens`, `requests`, `details`)
- `BudgetedUsage` in `token_budget.py` uses `getattr(self._usage, name)` — works with any object that has these attributes

#### 1c. `app/ollama/output.py`

```python
def format_schema_prompt(output_type, *, name, description, template) -> str:
    """Build the JSON schema instruction string to append to the system prompt."""

def format_json_schema(output_type) -> dict:
    """Extract JSON schema from a Pydantic model or StructuredDict-like schema dict."""

def parse_structured_output(response_text: str, output_type) -> BaseModel | dict:
    """Parse LLM response text into the expected output type.
    - Strip markdown fences if present
    - JSON parse
    - If output_type is a Pydantic model: model_validate()
    - If output_type is dict: return raw dict
    """
```

- `JSON_OUTPUT_TEMPLATE` moved here (or imported from `agents.py`)

#### 1d. `app/ollama/tools.py`

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON schema for parameters
    function: Callable  # The actual Python function

@dataclass
class ToolCall:
    name: str
    arguments: dict

@dataclass  
class ToolResult:
    call: ToolCall
    output: str

def pydantic_model_to_tool_parameters(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model's JSON schema to Ollama tool parameter schema."""

def build_ollama_tools(tool_defs: list[ToolDefinition]) -> list[dict]:
    """Format tool definitions for /api/chat tool_calls parameter."""
```

#### 1e. `app/ollama/agent.py`

```python
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

@dataclass
class AgentConfig:
    model: str
    system_prompt: str | None = None
    output_type: Any = str  # Pydantic model class, dict (for raw JSON schema), or str
    output_name: str | None = None
    output_description: str | None = None
    output_retries: int = 2
    temperature: float = 0.0
    seed: int = 42
    max_context_length: int = 8192
    think: bool = False  # For qwen3.5 etc

class Agent(Generic[DepsT, OutputT]):
    """Single-turn /api/generate agent. No tool calling."""
    
    def __init__(self, config: AgentConfig):
        self._config = config
        self._dynamic_instructions: list[Callable[[DepsT], str]] = []
    
    def instructions(self, fn):
        """Decorator: register dynamic instructions callback. Mirrors @agent.instructions."""
        self._dynamic_instructions.append(fn)
        return fn
    
    async def run(self, prompt: str, *, deps: DepsT = None) -> AgentResult[OutputT]:
        """Execute a single /api/generate call with format param.
        
        1. Build system message from static instructions + dynamic instructions(deps)
        2. Call /api/generate with format=JSON schema
        3. Parse response into output_type
        4. On ModelRetry: append error to prompt, retry up to output_retries
        5. Return AgentResult(output, usage)
        """

@dataclass
class AgentResult(Generic[OutputT]):
    output: OutputT
    usage: RunUsage
```

**`_run_generate` implementation sketch:**

```python
async def _run_generate(self, prompt: str, deps: Any) -> AgentResult:
    """All agents use /api/generate — no tool calling, no chat loop."""
    schema = format_json_schema(self._config.output_type)
    system_msg = self._build_system_message(deps)
    current_prompt = prompt
    
    total_usage = RunUsage()
    
    for attempt in range(self._config.output_retries + 1):
        response = await self._ollama_client.chat_client.generate(
            model=self._config.model,
            prompt=current_prompt,
            system=system_msg,
            format=schema,
            options={
                "temperature": self._config.temperature,
                "seed": self._config.seed,
                "num_ctx": self._config.max_context_length,
            },
            think=self._config.think,
        )
        
        total_usage.input_tokens += getattr(response, 'prompt_eval_count', 0) or 0
        total_usage.output_tokens += getattr(response, 'eval_count', 0) or 0
        total_usage.requests += 1
        
        try:
            output = parse_structured_output(response.response, self._config.output_type)
            return AgentResult(output=output, usage=total_usage)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < self._config.output_retries:
                current_prompt = f"{current_prompt}\n\nError: {e}. Please return valid JSON."
                continue
            raise OutputParsingError(str(e)) from e
    
    raise MaxRetriesExceeded()
```

#### 1f. `client.py` changes

- Remove `from pydantic_ai.models.ollama import OllamaModel` and `from pydantic_ai.providers.ollama import OllamaProvider`
- Remove `agent_model` property and `_build_agent_model()` method
- `OllamaClientWrapper` no longer needs to produce pydantic-ai model objects
- Keep all other functionality (embeddings, model management, etc.) unchanged

#### 1g. Write comprehensive tests for the new module

- `tests/test_ollama_agent.py` — test Agent.run with mocked responses
- Test generate path (no tools)
- Test chat path (with tools)
- Test output parsing (Pydantic model, dict, malformed)
- Test retry logic (ModelRetry in validation, MaxRetriesExceeded)
- Test token usage accounting
- Test dynamic instructions
- Test that `RunUsage` is compatible with `BudgetedUsage`

### Phase 2: Migrate agent factories to use custom Agent (dual path)

**Goal:** Every agent factory has been rewritten to use `app.ollama.agent.Agent` instead of `pydantic_ai.Agent`. Existing tests pass. Tool calling removed; file ranking agent added.

#### 2a. `agents.py` refactoring

- Remove imports of `Agent`, `ModelRetry`, `RunContext`, `StructuredDict`, `PromptedOutput`
- Import from `app.ollama.agent`, `app.ollama.errors`, `app.ollama.output`
- `prompted_json_output()` → becomes a helper that returns output config (schema + name + description) instead of a `PromptedOutput` object
- `structured_profile_output()` → same
- `create_schema_validated_agent()` → uses new `Agent` class with schema validation in retry loop
- `validate_json_output_against_schema()` → keep as-is (uses `jsonschema`, not pydantic-ai)
- `ModelRetry` → import from `app.ollama.errors`

#### 2b. NEW `file_ranking.py`

- Create `FileRankingResult` Pydantic model — ranked file list with relevance scores and reasoning
- Create `FileRankingDeps` dataclass — file list with extensions and sizes
- Create `create_file_ranking_agent()` → `Agent` with `FileRankingResult` output type
- Instructions: rank files by metadata potential (README, instrument exports, tables, etc.)

#### 2c. `initial_context.py` refactoring

- Replace `pydantic_ai.Agent` with `app.ollama.agent.Agent`
- **Remove `@agent.tool` decorators** — `list_dataset_files` and `read_file_content` are replaced by file ranking + pre-fetch
- Replace `RunContext[InitialContextDeps]` — dynamic instructions receive deps directly
- `extract_initial_context_from_data_package()` refactored:
  1. Call `create_file_ranking_agent()` to rank files
  2. Read top-N files using existing `list_initial_context_dataset_files()` / `read_initial_context_file_content()` in caller
  3. Call `create_initial_context_agent()` with pre-read content injected via dynamic instructions
- Remove `files_read` tracking from `InitialContextDeps` (no longer needed)

#### 2d. `patch_draft.py`, `patch_quality.py`, `review_resolution.py` refactoring

- Replace `@agent.instructions` with new `@agent.instructions`
- Replace `RunContext[Deps]` — function signatures become `(deps: DepsT) -> str`
- Same pattern for all agent factories

#### 2e. `extraction.py` (API layer)

- Replace `from pydantic_ai.exceptions import AgentRunError` with `from app.ollama.errors import AgentRunError`
- Error handling stays the same

#### 2f. `client.py` changes

- Remove `from pydantic_ai.models.ollama import OllamaModel` and `from pydantic_ai.providers.ollama import OllamaProvider`
- Remove `_build_agent_model()` and `agent_model` attribute
- Agent factories receive model name string from `OllamaClientWrapper.chat_model` instead of pydantic-ai model object

#### 2g. Update test files

- `test_extraction_agents.py`: Remove `TestModel` import, replace with mock Agent or mock Ollama responses
- `test_patch_quality.py`: Same
- `test_initial_context_extraction.py`: Same; add tests for file ranking flow
- `test_ollama_client.py`: Remove `OllamaModel`/`OllamaProvider` assertions
- Add new test file for `app/ollama/agent.py`
- Add new test file for `file_ranking.py`

#### 2h. Integration test

- Run full end-to-end extraction with a connected Ollama instance (or mock)
- Verify token usage reporting still works
- Verify output parsing produces identical Pydantic models

### Phase 3: Remove pydantic-ai dependency

#### 3a. Remove `pydantic-ai-slim[openai]` from `pyproject.toml`

- Also remove `openai` if it was only pulled in by pydantic-ai's `[openai]` extra
- Keep `ollama`, `pydantic`, `jsonschema`, `httpx` (still needed)

#### 3b. Remove all pydantic-ai imports

- Verify no remaining `from pydantic_ai` or `import pydantic_ai` references
- Search: `grep -rn "pydantic_ai\|pydantic-ai" backend/app/ backend/tests/`

#### 3c. Remove `agent_model` / `_build_agent_model` from `OllamaClientWrapper`

- The service layer no longer needs a pydantic-ai model object
- Agent factories receive the model name string (e.g., `"qwen3.5:4b"`) instead

#### 3d. Clean up unused dependencies

```bash
cd backend && poetry remove pydantic-ai-slim
# Verify: poetry install && poetry run pytest
```

#### 3e. Update `app/ollama/__init__.py`

- Add exports for `Agent`, `AgentResult`, `AgentConfig`, `RunUsage`, `ModelRetry`, `AgentRunError`

---

## Ongoing Maintenance

This document is the living source of truth. Update it as implementation progresses:

- Mark phases with ✅ when complete
- Add pitfalls and deviations as they emerge
- Update the token-overhead table after benchmarking the custom module

## Benchmark Reference (from previous session)

| Approach | Prompt tokens | Valid JSON? | Latency |
|----------|--------------|-------------|---------|
| `/api/generate` bare prompt | 20 | No (raw text) | ~3s |
| `/api/generate` format=schema think=false | **143** | Yes | ~38s cold, ~8s warm |
| `/api/generate` schema-in-text | 1,154 | No (markdown fences) | — |
| `/v1/chat/completions` json_schema | 1,248 | Yes | 95s, timeout risk |
| `/v1/chat/completions` schema-in-system-msg | **timeout** | — | >300s |
| Pydantic AI `NativeOutput` | 3,953 | Partial | 221s |
| Pydantic AI `PromptedOutput` | **timeout** | — | >600s |

**Target:** All agents → `/api/generate` with `format` (~143 tokens each).