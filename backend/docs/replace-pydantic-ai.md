# Replace Pydantic AI with Custom Ollama Agent Module

**Status:** Planning  
**Created:** 2026-05-24  
**Last updated:** 2026-05-24

## Goal

Remove the `pydantic-ai` dependency entirely. Replace it with a custom lightweight agent module in `app/ollama/` that directly calls Ollama's `/api/chat` (with tool calling) and `/api/generate` (with `format` parameter) endpoints, handling structured output, tool calling, retry logic, context injection, and token usage observation natively.

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

| Agent | Factory | Output Type | Tools | Dynamic Instructions | `deps_type` |
|-------|---------|-------------|-------|---------------------|-------------|
| Initial Context | `create_initial_context_agent()` | `PromptedOutput(InitialContext)` | 2 (`list_dataset_files`, `read_file_content`) | No (static instructions) | `InitialContextDeps` |
| Patch Draft | `create_patch_draft_agent()` | `PromptedOutput(FieldPatchResult)` | No | Yes (`@agent.instructions`) | `PatchDraftDeps` |
| Patch Discovery | `create_patch_discovery_agent()` | `PromptedOutput(PatchDiscoveryResult)` | No | Yes | `PatchDiscoveryDeps` |
| Schema Patch | `create_schema_patch_agent()` | `PromptedOutput(SchemaPatchResult)` | No | Yes | `SchemaPatchDeps` |
| Schema Repair | `create_schema_repair_agent()` | `PromptedOutput(SchemaPatchResult)` | No | Yes | `SchemaRepairDeps` |
| Patch Quality | `create_patch_quality_agent()` | `PromptedOutput(PatchQualityReport)` | No | Yes | `PatchQualityDeps` |
| Review Resolution | `create_patch_review_resolution_agent()` | `PromptedOutput(PatchReviewResolution)` | No | Yes | `PatchReviewResolutionDeps` |
| Schema Validated | `create_schema_validated_agent()` | `StructuredDict+PromptedOutput` or fallback `PromptedOutput(dict)` | No | Static (or schema-in-text fallback) | `deps_type` param |

### What `Agent.run()` provides that we need to replicate

1. **System message injection** — `instructions` param on `Agent()` constructor
2. **Dynamic instructions** — `@agent.instructions` callbacks that receive deps and return context strings
3. **Tool registration + routing** — `@agent.tool` decorated functions with deps access; model calls tools, results fed back
4. **Output parsing** — `PromptedOutput` → JSON schema description in system prompt, parse response text as JSON, validate against schema
5. **Output retries** — `output_retries` param, `ModelRetry` exception triggers re-prompt with error message
6. **Token usage** — `result.usage` returns `RunUsage` with `input_tokens`, `output_tokens`, `requests`, `details`
7. **Model settings** — `temperature`, `seed` per agent
8. **Error handling** — `AgentRunError` wraps model/structured output failures

### What we do NOT need from Pydantic AI

- Streaming (`run_stream`)
- Conversation history (`message_history`)
- Multi-turn beyond tool-return loop
- `NativeOutput` (tested: worse than PromptedOutput for self-hosted Ollama)
- `@agent.output_validator` — only used by `create_schema_validated_agent()` which we'll handle directly
- Graph/agent orchestration features
- Dependency injection beyond what a simple dataclass carries

---

## Architecture

### Module structure

```
app/ollama/
├── __init__.py          # Re-export public API
├── client.py            # OllamaClientWrapper (existing, unchanged except removing pydantic_ai)
├── runtime.py           # Model management, diagnostics (existing, unchanged)
├── agent.py             # NEW: Agent class replacing pydantic_ai.Agent
├── output.py            # NEW: Output parsing, schema formatting, validation
├── tools.py             # NEW: Tool definition, calling, result handling
├── retry.py             # NEW: ModelRetry exception, retry loop logic
├── usage.py             # NEW: RunUsage dataclass, token counting
└── errors.py            # NEW: AgentRunError and custom error types
```

### Key design decisions

1. **Dual API routing:** Agents without tools → `/api/generate` with `format` param (minimal token overhead). Agents with tools → `/api/chat` with tool definitions (required for tool calling protocol).
2. **Pyantic models stay:** Output types remain Pydantic `BaseModel` subclasses. We parse the LLM response JSON and validate with `model.model_validate()`. `StructuredDict` is replaced with plain JSON schema validation.
3. **Tool calling protocol:** Ollama's `/api/chat` endpoint natively supports tool definitions and tool results. We'll parse tool call responses, execute tool functions, and feed results back — exactly like pydantic-ai but without the abstraction overhead.
4. **ModelRetry as exception:** Keep the same pattern — `ModelRetry(message)` raised in validation/retry callbacks triggers a re-prompt with the error message appended.
5. **Token usage via Ollama response metadata:** Both `/api/generate` and `/api/chat` return `prompt_eval_count` and `eval_count` fields. We map these to `input_tokens` / `output_tokens`.

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
    output_type: Any = str  # Pydantic model, dict schema, or str
    output_name: str | None = None
    output_description: str | None = None
    output_retries: int = 2
    temperature: float = 0.0
    seed: int = 42
    max_context_length: int = 8192
    think: bool = False  # For qwen3.5 etc
    
class Agent(Generic[DepsT, OutputT]):
    def __init__(self, config: AgentConfig):
        self._config = config
        self._tools: list[ToolDefinition] = []
        self._dynamic_instructions: list[Callable[[DepsT], str]] = []
    
    def tool(self, fn):
        """Decorator: register a tool function. Mirrors pydantic_ai @agent.tool."""
    
    def instructions(self, fn):
        """Decorator: register dynamic instructions callback. Mirrors @agent.instructions."""
    
    async def run(self, prompt: str, *, deps: DepsT = None) -> AgentResult[OutputT]:
        """Main execution loop.
        
        For agents WITHOUT tools:
          - Use /api/generate with format={json_schema}
          - Parse response, validate output
          - Retry on ModelRetry up to output_retries times
        
        For agents WITH tools:
          - Use /api/chat with tool definitions
          - Loop: send messages → receive tool calls → execute tools → send results
          - After tool loop completes, parse final output
          - Retry on ModelRetry
        """

@dataclass
class AgentResult(Generic[OutputT]):
    output: OutputT
    usage: RunUsage
```

**Key implementation details for `/api/generate` path (extraction agents):**

```python
async def _run_generate(self, prompt: str, deps: Any) -> AgentResult:
    """Agent path via /api/generate — no tool calling, minimal token overhead."""
    schema = format_json_schema(self._config.output_type)
    system_msg = self._build_system_message(deps)
    
    response = await self._ollama_client.chat_client.generate(
        model=self._config.model,
        prompt=prompt,  # Actually we need to prepend system + schema
        system=system_msg,  # Ollama supports system param
        format=schema,  # Structured output!
        options={"temperature": self._config.temperature, "seed": self._config.seed, "num_ctx": self._config.max_context_length},
        think=self._config.think,
    )
    
    output = parse_structured_output(response.response, self._config.output_type)
    usage = RunUsage(
        input_tokens=getattr(response, 'prompt_eval_count', 0) or 0,
        output_tokens=getattr(response, 'eval_count', 0) or 0,
        requests=1,
    )
    return AgentResult(output=output, usage=usage)
```

**Key implementation details for `/api/chat` path (tool-using agents):**

```python
async def _run_chat(self, prompt: str, deps: Any) -> AgentResult:
    """Agent path via /api/chat — supports tool calling."""
    import ollama
    
    system_msg = self._build_system_message(deps)
    messages = [ollama.Message(role="system", content=system_msg),
                ollama.Message(role="user", content=prompt)]
    tools = build_ollama_tools(self._tools)
    
    total_usage = RunUsage()
    
    for attempt in range(self._config.output_retries + 1):
        response = await self._ollama_client.chat_client.chat(
            model=self._config.model,
            messages=messages,
            tools=tools if tools else None,
            options={"temperature": self._config.temperature, "seed": self._config.seed, "num_ctx": self._config.max_context_length},
        )
        
        total_usage.input_tokens += getattr(response, 'prompt_eval_count', 0) or 0
        total_usage.output_tokens += getattr(response, 'eval_count', 0) or 0
        total_usage.requests += 1
        
        # Check for tool calls
        msg = response.message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                result = self._execute_tool(tc.function.name, tc.function.arguments, deps)
                messages.append(ollama.Message(
                    role="tool",
                    content=result,
                    tool_call_id=...,
                ))
            continue  # Loop back with tool results
        
        # No tool calls — parse as final output
        try:
            output = parse_structured_output(msg.content, self._config.output_type)
            return AgentResult(output=output, usage=total_usage)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < self._config.output_retries:
                messages.append(ollama.Message(role="assistant", content=msg.content))
                messages.append(ollama.Message(role="user", content=f"Error: {e}. Please try again."))
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

**Goal:** Every agent factory has been rewritten to use `app.ollama.agent.Agent` instead of `pydantic_ai.Agent`. Existing tests pass.

#### 2a. `agents.py` refactoring

- Remove imports of `Agent`, `ModelRetry`, `RunContext`, `StructuredDict`, `PromptedOutput`
- Import from `app.ollama.agent`, `app.ollama.errors`, `app.ollama.output`
- `prompted_json_output()` → becomes a helper that returns output config (schema + name + description) instead of a `PromptedOutput` object
- `structured_profile_output()` → same
- `create_schema_validated_agent()` → uses new `Agent` class with schema validation in retry loop
- `validate_json_output_against_schema()` → keep as-is (uses `jsonschema`, not pydantic-ai)
- `ModelRetry` → import from `app.ollama.errors`

#### 2b. `initial_context.py` refactoring

- Replace `pydantic_ai.Agent` with `app.ollama.agent.Agent`
- Replace `@agent.tool` decorator with new `@agent.tool` (same API, different implementation)
- Replace `RunContext[InitialContextDeps]` — new Agent's tool functions receive deps directly
- `result.output` and `result.usage` — same attribute names on `AgentResult`

#### 2c. `patch_draft.py`, `patch_quality.py`, `review_resolution.py` refactoring

- Replace `@agent.instructions` with new `@agent.instructions`
- Replace `RunContext[Deps]` — function signatures become `(deps: DepsT) -> str`
- Same pattern for all 6 agent factories

#### 2d. `extraction.py` (API layer)

- Replace `from pydantic_ai.exceptions import AgentRunError` with `from app.ollama.errors import AgentRunError`
- Error handling stays the same

#### 2e. Update test files

- `test_extraction_agents.py`: Remove `TestModel` import, replace with mock Agent or mock Ollama responses
- `test_patch_quality.py`: Same
- `test_initial_context_extraction.py`: Same
- `test_ollama_client.py`: Remove `OllamaModel`/`OllamaProvider` assertions
- Add new test file for `app/ollama/agent.py`

#### 2f. Integration test

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

**Target:** All extraction agents (no tools) → `/api/generate` with `format` (~143 tokens). Initial context agent (has tools) → `/api/chat` with native tool calling.