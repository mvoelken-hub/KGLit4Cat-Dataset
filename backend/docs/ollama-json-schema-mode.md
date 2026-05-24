# Ollama JSON Schema Mode Investigation

Date: 2026-05-23

## Context

SIMONE reduced extraction-agent prompt context so local models can run below roughly
8k tokens per call. One remaining question was whether Ollama JSON Schema mode can
accept larger output schemas without those schemas consuming the model prompt
context.

The current SIMONE extraction helper still uses prompted schema text through
`PromptedOutput`:

- `backend/app/domain/extraction/agents.py`
- `JSON_OUTPUT_TEMPLATE`
- `structured_profile_output(...)`

The configured Ollama model is wired through `OllamaModel` in:

- `backend/app/ollama/client.py`

## Environment Tested

Values came from `.env`:

- Ollama endpoint: `http://christina-desktop:11433`
- Ollama version: `0.22.0`
- Chat model: `qwen3.5:4b`
- Configured max context: `8192`

Available relevant models at the time:

- `qwen3.5:4b`
- `qwen3.5:0.8b`
- `qwen3-embedding:0.6b`
- `gemma4:31b-cloud`

## Result

Ollama native JSON Schema mode does avoid charging the schema as prompt context.

A deliberately large JSON Schema of about 24,661 characters was tested three ways
against `/api/generate`:

| Case | Prompt tokens (`prompt_eval_count`) | Wall time |
| --- | ---: | ---: |
| No schema baseline | 29 | 26.99s |
| Schema passed via `format` | 25 | 6.76s |
| Same schema pasted into prompt text | 5,890 | 43.98s |

This strongly suggests that using Ollama's `format: <json schema>` path can support
larger schemas without consuming the prompt token budget the way prompted schema
text does.

## Additional Observations

- A smaller `/api/generate` test with `format: <json schema>` and `think: false`
  returned valid parseable JSON.
- The model filled optional nullable fields with `null`, which is valid for the
  tested schema but may not be desirable for sparse patch outputs.
- `qwen3.5` emitted separate thinking output by default. Small `num_predict`
  values can be consumed entirely by thinking before any answer content is
  produced. For schema-mode tests, disable thinking with `think: false` or allow
  a larger completion budget.
- The raw OpenAI-compatible `/v1/chat/completions` path timed out in quick tests
  with `response_format: {"type": "json_schema", ...}`, even for tiny schemas.
  This matters because SIMONE currently uses Pydantic AI's OpenAI-compatible
  `OllamaModel` provider. Do not assume Pydantic `NativeOutput` will work in this
  repo until a focused integration test confirms it against this endpoint/model.
- `/api/chat` with `format` returned quickly, but one test still wrapped JSON in
  Markdown code fences despite the schema-shaped format. `/api/generate` was the
  cleanest path in these quick checks.

## Implications For SIMONE

Potential path forward:

1. Keep local validation with `jsonschema` after model output.
2. Add a small Ollama-native structured-output adapter for extraction agents, or
   verify whether the current Pydantic AI version can use `NativeOutput` reliably
   with this server.
3. Compare current `PromptedOutput` calls with a native schema-mode variant using
   real profile schemas and record:
   - `prompt_eval_count`
   - `eval_count`
   - wall time
   - parse/validation success rate
   - retry count
4. Keep compact semantic field hints in the prompt if the model needs descriptions
   for choosing fields. Schema mode constrains output shape; it does not replace
   all extraction guidance.

## Useful References

- Ollama structured outputs: https://docs.ollama.com/capabilities/structured-outputs
- Ollama generate API: https://docs.ollama.com/api/generate
- Pydantic AI Ollama model docs: https://pydantic.dev/docs/ai/models/ollama/
