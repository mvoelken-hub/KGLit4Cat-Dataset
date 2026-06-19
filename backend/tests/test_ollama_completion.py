"""Tests for the custom Ollama structured completion module."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from app.ollama.completion import (
    CompletionResult,
    _extract_json_schema,
    _failed_response_for_repair,
    _schema_example,
    _strip_markdown_fences,
    generate_structured,
    generate_text,
    repair_structured_output,
)
from app.ollama.errors import CompletionError, EmptyResponseError, MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage


class SimpleOutput(BaseModel):
    answer: str
    score: int


class NestedOutput(BaseModel):
    items: list[SimpleOutput]
    total: int


class PrunableItem(BaseModel):
    value: str
    optional_field: str = ""


class PrunableOutput(BaseModel):
    items: list[PrunableItem]
    optional_root_field: str = ""


@dataclass
class FakeGenerateResponse:
    """Fake ollama GenerateResponse for testing."""

    response: str
    prompt_eval_count: int = 0
    eval_count: int = 0
    prompt_eval_duration: int = 0
    load_duration: int = 0
    eval_duration: int = 0
    total_duration: int = 0


class FakeOllamaClient:
    """Fake ollama.AsyncClient that returns pre-scheduled responses."""

    def __init__(self, responses: list[FakeGenerateResponse | Exception] | None = None):
        self.responses: list[FakeGenerateResponse | Exception] = responses or []
        self.calls: list[dict[str, Any]] = []
        self.call_index = 0

    @property
    def ollama_client(self) -> "FakeOllamaClient":
        return self

    async def generate(self, **kwargs: Any) -> FakeGenerateResponse:
        self.calls.append(kwargs)
        if self.call_index < len(self.responses):
            r = self.responses[self.call_index]
            self.call_index += 1
            if isinstance(r, Exception):
                raise r
            return r
        # Default empty response if exhausted
        return FakeGenerateResponse(response="{}")


class WhitespaceBudgeter:
    uses_fallback = False
    fallback_reason = None

    def count(self, value: str) -> int:
        return len(value.split()) if value else 0


class StripMarkdownFencesTests(unittest.TestCase):
    def test_strips_json_fence(self):
        self.assertEqual(
            _strip_markdown_fences('```json\n{"x": 1}\n```'),
            '{"x": 1}',
        )

    def test_strips_plain_fence(self):
        self.assertEqual(
            _strip_markdown_fences('```\n{"x": 1}\n```'),
            '{"x": 1}',
        )

    def test_strips_non_json_language_fence_on_one_line(self):
        self.assertEqual(
            _strip_markdown_fences('```python {"x": 1} ```'),
            '{"x": 1}',
        )

    def test_strips_non_json_language_fence_with_newline(self):
        self.assertEqual(
            _strip_markdown_fences('```python\n{"x": 1}\n```'),
            '{"x": 1}',
        )

    def test_no_fence_passthrough(self):
        self.assertEqual(
            _strip_markdown_fences('{"x": 1}'),
            '{"x": 1}',
        )

    def test_strips_leading_trailing_backticks(self):
        self.assertEqual(
            _strip_markdown_fences('`{"x": 1}`'),
            '{"x": 1}',
        )


class ExtractJsonSchemaTests(unittest.TestCase):
    def test_from_pydantic_model(self):
        schema = _extract_json_schema(SimpleOutput)
        self.assertIn("properties", schema)
        self.assertIn("answer", schema["properties"])

    def test_dict_passthrough(self):
        schema = _extract_json_schema({"type": "object"})
        self.assertEqual(schema, {"type": "object"})

    def test_unsupported_type_raises(self):
        with self.assertRaises(CompletionError):
            _extract_json_schema("not_a_model")


class SchemaExampleTests(unittest.TestCase):
    def test_integer_example_respects_minimum(self):
        schema = {
            "type": "object",
            "properties": {
                "rank": {"type": "integer", "minimum": 1},
                "score": {"type": "number", "minimum": 0.25},
            },
        }

        self.assertEqual(
            _schema_example(schema),
            {
                "rank": 1,
                "score": 0.25,
            },
        )

    def test_recursive_ref_cycle_is_truncated(self):
        schema = {
            "$defs": {
                "Dataset": {
                    "type": "object",
                    "properties": {
                        "dataset_distribution": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Distribution"},
                        },
                    },
                },
                "Distribution": {
                    "type": "object",
                    "properties": {
                        "access_service": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/DataService"},
                        },
                    },
                },
                "DataService": {
                    "type": "object",
                    "properties": {
                        "serves_dataset": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Dataset"},
                        },
                    },
                },
            },
            "$ref": "#/$defs/Dataset",
        }

        self.assertEqual(
            _schema_example(schema),
            {
                "dataset_distribution": [
                    {
                        "access_service": [
                            {
                                "serves_dataset": [{}],
                            }
                        ],
                    }
                ],
            },
        )


class GenerateStructuredHappyPathTests(IsolatedAsyncioTestCase):
    async def test_valid_json_to_pydantic_model(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(
                response='{"answer": "hello", "score": 42}',
                prompt_eval_count=100,
                eval_count=20,
            ),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type=SimpleOutput,
        )

        self.assertIsInstance(result, CompletionResult)
        self.assertIsInstance(result.output, SimpleOutput)
        self.assertEqual(result.output.answer, "hello")
        self.assertEqual(result.output.score, 42)
        self.assertEqual(result.usage.requests, 1)
        self.assertEqual(result.usage.input_tokens, 100)
        self.assertEqual(result.usage.output_tokens, 20)

    async def test_num_predict_passed_to_ollama_options(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "hello", "score": 42}'),
        ])

        await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type=SimpleOutput,
            num_predict=1200,
        )

        self.assertEqual(client.calls[0]["options"].num_predict, 1200)

    async def test_dict_output_type_returns_raw_dict(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(
                response='{"custom": "data", "count": 7}',
                prompt_eval_count=50,
                eval_count=10,
            ),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type={"type": "object"},
        )

        self.assertEqual(result.output, {"custom": "data", "count": 7})
        self.assertEqual(result.usage.requests, 1)

    async def test_dict_output_type_validates_json_schema(self):
        schema = {
            "type": "object",
            "properties": {"required_field": {"type": "string"}},
            "required": ["required_field"],
            "additionalProperties": False,
        }
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"required_field": "present"}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type=schema,
        )

        self.assertEqual(result.output, {"required_field": "present"})

    async def test_prompt_diagnostics_include_components_and_schema(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(
                response='{"answer": "hello", "score": 1}',
                prompt_eval_count=12,
                eval_count=3,
            ),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="base system",
            prompt="first part second part",
            system_components=[("base_system", "base system")],
            prompt_components=[
                ("first", "first part "),
                ("second", "second part"),
            ],
            token_budgeter=WhitespaceBudgeter(),
            operation_id="op-1",
            agent_name="agent",
            diagnostic_metadata={"file_path": "a.txt"},
            output_type=SimpleOutput,
        )

        diagnostics = result.prompt_diagnostics
        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(diagnostics.operation_id, "op-1")
        self.assertEqual(diagnostics.agent_name, "agent")
        self.assertEqual(diagnostics.metadata["file_path"], "a.txt")
        self.assertEqual(len(diagnostics.attempts), 1)
        component_names = [component.name for component in diagnostics.attempts[0].components]
        self.assertIn("base_system", component_names)
        self.assertIn("system_schema", component_names)
        self.assertIn("first", component_names)
        self.assertEqual(diagnostics.attempts[0].usage["input_tokens"], 12)
        first = next(component for component in diagnostics.attempts[0].components if component.name == "first")
        self.assertEqual(first.text, "first part ")
        self.assertEqual(first.tokens, 2)

    async def test_passes_options_to_generate(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            temperature=0.7,
            seed=123,
            think=True,
            num_ctx=4096,
            keep_alive=300,
        )

        call = client.calls[0]
        self.assertEqual(call["model"], "qwen3.5:4b")
        self.assertTrue(call["system"].startswith("sys"))
        self.assertIn("JSON Schema:", call["system"])
        self.assertEqual(call["prompt"], "prompt")
        self.assertIsInstance(call["format"], dict)
        self.assertEqual(
            call["options"].model_dump(exclude_none=True),
            {"num_ctx": 4096, "seed": 123, "temperature": 0.7},
        )
        self.assertTrue(call["think"])
        self.assertEqual(call["keep_alive"], 300)

    async def test_wrapper_temperature_overrides_call_temperature(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])
        client.generation_temperature = 0.0

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            temperature=0.7,
        )

        self.assertEqual(client.calls[0]["options"].temperature, 0.0)

    async def test_output_token_limit_is_derived_from_context(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])
        client.enforce_output_token_limit = True

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            num_ctx=8,
        )

        self.assertEqual(client.calls[0]["system"], "sys")
        self.assertEqual(client.calls[0]["options"].num_predict, 5)

    async def test_explicit_output_token_limit_is_only_lowered(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])
        client.enforce_output_token_limit = True

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            num_ctx=8,
            num_predict=1200,
        )

        self.assertEqual(client.calls[0]["options"].num_predict, 5)

    async def test_output_token_limit_can_be_disabled(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])
        client.enforce_output_token_limit = False

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            num_ctx=8,
            num_predict=1200,
        )

        self.assertEqual(client.calls[0]["options"].num_predict, 1200)

    async def test_generate_text_uses_runtime_temperature_and_output_cap(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="ok"),
        ])
        client.generation_temperature = 0.0
        client.enforce_output_token_limit = True

        await generate_text(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            options={"temperature": 0.7, "num_ctx": 8, "num_predict": 1200},
        )

        self.assertEqual(client.calls[0]["options"]["temperature"], 0.0)
        self.assertEqual(client.calls[0]["options"]["num_predict"], 5)

    async def test_injects_json_schema_into_system_prompt_by_default(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
        )

        self.assertIn("JSON Schema:", client.calls[0]["system"])
        self.assertIn('"answer"', client.calls[0]["system"])
        self.assertIn('"score"', client.calls[0]["system"])
        self.assertIsInstance(client.calls[0]["format"], dict)

    async def test_omitted_fields_prune_prompt_and_format_schema(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"items": [{"value": "x"}]}'),
        ])

        result = await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=PrunableOutput,
            omitted_fields={
                "PrunableItem": ["optional_field"],
                "PrunableOutput": ["optional_root_field"],
            },
        )

        self.assertEqual(result.output.items[0].optional_field, "")
        self.assertEqual(result.output.optional_root_field, "")
        self.assertNotIn("optional_field", client.calls[0]["system"])
        self.assertNotIn("optional_root_field", client.calls[0]["system"])
        self.assertNotIn("optional_field", json.dumps(client.calls[0]["format"]))
        self.assertNotIn("optional_root_field", json.dumps(client.calls[0]["format"]))

    async def test_omitted_fields_reject_required_fields(self):
        client = FakeOllamaClient()

        with self.assertRaises(CompletionError):
            await generate_structured(
                client,
                model="qwen3.5:4b",
                system="sys",
                prompt="prompt",
                output_type=PrunableOutput,
                omitted_fields={"PrunableItem": ["value"]},
            )

        self.assertEqual(client.calls, [])

    async def test_uses_example_shape_when_full_schema_exceeds_context_budget(self):
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "x" * 1000,
                }
            },
            "required": ["answer"],
        }
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x"}'),
        ])

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=schema,
            num_ctx=220,
        )

        self.assertNotIn("JSON Schema:", client.calls[0]["system"])
        self.assertIn("Example JSON shape:", client.calls[0]["system"])
        self.assertIn('"answer"', client.calls[0]["system"])
        self.assertNotIn("xxx", client.calls[0]["system"])

    async def test_omits_prompt_visible_schema_when_context_budget_is_too_small(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="sys",
            prompt="prompt",
            output_type=SimpleOutput,
            num_ctx=8,
        )

        self.assertEqual(client.calls[0]["system"], "sys")

    async def test_markdown_fences_stripped(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='```json\n{"answer": "yo", "score": 1}\n```'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type=SimpleOutput,
        )

        self.assertEqual(result.output.answer, "yo")

    async def test_non_json_markdown_fence_language_is_stripped(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='```python {"answer": "yo", "score": 1} ```'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="You are helpful.",
            prompt="Give me data.",
            output_type=SimpleOutput,
            retries=0,
        )

        self.assertEqual(result.output.answer, "yo")


class GenerateStructuredRetryTests(IsolatedAsyncioTestCase):
    async def test_retry_on_ollama_api_error(self):
        client = FakeOllamaClient([
            RuntimeError("Bad Request"),
            FakeGenerateResponse(response='{"answer": "fixed", "score": 99}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            api_retries=1,
            api_retry_backoff_seconds=0,
        )

        self.assertEqual(result.output.answer, "fixed")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result.usage.requests, 1)

    async def test_api_error_raises_completion_error_after_retries(self):
        client = FakeOllamaClient([
            RuntimeError("Bad Request"),
            RuntimeError("Bad Request"),
        ])

        with self.assertRaises(CompletionError) as error:
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                api_retries=1,
                api_retry_backoff_seconds=0,
            )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Ollama API error after 2 attempt", str(error.exception))
        self.assertEqual(error.exception.details["api_attempts"], 2)

    async def test_retry_on_json_decode_error(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="not json at all"),
            FakeGenerateResponse(response='{"answer": "fixed", "score": 99}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=2,
        )

        self.assertEqual(result.output.answer, "fixed")
        self.assertEqual(result.usage.requests, 2)
        self.assertIsNotNone(result.prompt_diagnostics)
        assert result.prompt_diagnostics is not None
        self.assertEqual(len(result.prompt_diagnostics.attempts), 2)
        self.assertEqual(result.prompt_diagnostics.attempts[0].attempt_kind, "initial")
        self.assertEqual(result.prompt_diagnostics.attempts[1].attempt_kind, "repair")
        repair_names = [
            component.name
            for component in result.prompt_diagnostics.attempts[1].components
        ]
        self.assertIn("failed_response", repair_names)

    async def test_repair_prompt_compacts_runaway_failed_response(self):
        runaway = (
            "```python\n"
            "{\n"
            '  "answer": "started",\n'
            '  "metadata_signals": [\n'
            + "\n".join('    "1632482437 timestamp",' for _ in range(300))
            + "\n  ]\n"
            "}\n"
        )
        client = FakeOllamaClient([
            FakeGenerateResponse(response=runaway, eval_count=16384),
            FakeGenerateResponse(response='{"answer": "fixed", "score": 99}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=1,
        )

        self.assertEqual(result.output.answer, "fixed")
        repair_prompt = client.calls[1]["prompt"]
        self.assertIn("failed response compacted before repair", repair_prompt)
        self.assertLess(repair_prompt.count("1632482437 timestamp"), 10)
        self.assertLess(len(repair_prompt), len(runaway))

    def test_failed_response_for_repair_keeps_small_failures_unchanged(self):
        self.assertEqual(_failed_response_for_repair("bad json"), "bad json")

    async def test_retry_on_pydantic_validation_error(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": 42, "score": "not-int"}'),
            FakeGenerateResponse(response='{"answer": "correct", "score": 1}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=2,
        )

        self.assertEqual(result.output.answer, "correct")
        self.assertEqual(result.usage.requests, 2)

    async def test_retries_propagate_error_in_prompt(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="bad json"),
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=1,
        )

        # Second call should repair only the failed response, not redo the original task.
        self.assertIn("Validation error:", client.calls[1]["prompt"])
        self.assertIn("bad json", client.calls[1]["prompt"])
        self.assertNotIn("Return JSON.", client.calls[1]["prompt"])
        self.assertIn("repair", client.calls[1]["system"].lower())

    async def test_repair_retry_keeps_schema_in_system_prompt(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="bad json"),
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="qwen3.5:4b",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=1,
        )

        self.assertIn("repair", client.calls[1]["system"].lower())
        self.assertIn("JSON Schema:", client.calls[1]["system"])
        self.assertIn('"answer"', client.calls[1]["system"])

    async def test_repair_attempt_can_use_dedicated_repair_model(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="bad json"),
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        await generate_structured(
            client,
            model="primary-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=1,
            repair_model="repair-model",
        )

        self.assertEqual(client.calls[0]["model"], "primary-model")
        self.assertEqual(client.calls[1]["model"], "repair-model")

    async def test_max_retries_exceeded_on_persistent_failure(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="still bad"),
            FakeGenerateResponse(response="still bad"),
            FakeGenerateResponse(response="still bad"),
        ])

        with self.assertRaises(MaxRetriesExceeded) as error:
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                retries=2,
            )

        self.assertEqual(client.call_index, 3)
        self.assertIsInstance(error.exception.last_error, OutputParsingError)
        self.assertEqual(error.exception.details["last_error_type"], "OutputParsingError")
        self.assertEqual(error.exception.failed_response, "still bad")
        self.assertEqual(error.exception.first_response, "still bad")
        self.assertEqual(error.exception.details["first_response"], "still bad")
        self.assertIsNotNone(error.exception.prompt_diagnostics)
        diagnostics = error.exception.prompt_diagnostics
        self.assertEqual(diagnostics.status, "failed")
        self.assertEqual(len(diagnostics.attempts), 3)

    async def test_api_error_after_first_structured_failure_keeps_first_response(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="bad json"),
            RuntimeError("repair rejected"),
            RuntimeError("repair rejected"),
            RuntimeError("repair rejected"),
        ])

        with self.assertRaises(CompletionError) as error:
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                retries=1,
            )

        self.assertEqual(client.call_index, 4)
        self.assertEqual(error.exception.details["first_response"], "bad json")
        self.assertEqual(getattr(error.exception, "first_response"), "bad json")

    async def test_failed_response_can_be_repaired_later(self):
        first_pass = FakeOllamaClient([
            FakeGenerateResponse(response="bad json", prompt_eval_count=10, eval_count=5),
        ])

        with self.assertRaises(MaxRetriesExceeded) as error:
            await generate_structured(
                first_pass,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                retries=0,
            )

        repair_client = FakeOllamaClient([
            FakeGenerateResponse(response='{"answer": "fixed", "score": 1}'),
        ])

        result = await repair_structured_output(
            repair_client,
            model="test-model",
            failed_response=error.exception.failed_response or "",
            error=error.exception.last_error or error.exception,
            output_type=SimpleOutput,
        )

        self.assertEqual(result.output.answer, "fixed")
        self.assertIn("bad json", repair_client.calls[0]["prompt"])
        self.assertNotIn("Return JSON.", repair_client.calls[0]["prompt"])

    async def test_usage_accumulates_across_retries(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response="bad", prompt_eval_count=10, eval_count=5),
            FakeGenerateResponse(response="bad", prompt_eval_count=12, eval_count=6),
            FakeGenerateResponse(response='{"answer": "ok", "score": 1}', prompt_eval_count=15, eval_count=8),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=SimpleOutput,
            retries=2,
        )

        self.assertEqual(result.usage.requests, 3)
        self.assertEqual(result.usage.input_tokens, 37)  # 10 + 12 + 15
        self.assertEqual(result.usage.output_tokens, 19)  # 5 + 6 + 8

    async def test_empty_response_fails_without_repair(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response=""),
            FakeGenerateResponse(response='{"answer": "x", "score": 1}'),
        ])

        with self.assertRaises(EmptyResponseError):
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                retries=1,
            )

        self.assertEqual(client.call_index, 1)

    async def test_empty_response_with_no_retries_raises(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response=""),
        ])

        with self.assertRaises(EmptyResponseError):
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=SimpleOutput,
                retries=0,
            )

    async def test_retry_on_json_schema_validation_error_for_dict_output_type(self):
        schema = {
            "type": "object",
            "properties": {"required_field": {"type": "string"}},
            "required": ["required_field"],
            "additionalProperties": False,
        }
        client = FakeOllamaClient([
            FakeGenerateResponse(response="{}"),
            FakeGenerateResponse(response='{"required_field": "fixed"}'),
        ])

        result = await generate_structured(
            client,
            model="test-model",
            system="Be precise.",
            prompt="Return JSON.",
            output_type=schema,
            retries=1,
        )

        self.assertEqual(result.output, {"required_field": "fixed"})
        self.assertEqual(result.usage.requests, 2)
        self.assertIn("Validation error:", client.calls[1]["prompt"])
        self.assertIn("Failed response:\n{}", client.calls[1]["prompt"])

    async def test_max_retries_exceeded_on_persistent_json_schema_validation_error(self):
        schema = {
            "type": "object",
            "properties": {"required_field": {"type": "string"}},
            "required": ["required_field"],
        }
        client = FakeOllamaClient([
            FakeGenerateResponse(response="{}"),
            FakeGenerateResponse(response="{}"),
        ])

        with self.assertRaises(MaxRetriesExceeded) as error:
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type=schema,
                retries=1,
            )

        self.assertEqual(client.call_index, 2)
        self.assertIn("required_field", str(error.exception.last_error))

    async def test_invalid_json_schema_raises_completion_error_before_calling_model(self):
        client = FakeOllamaClient([
            FakeGenerateResponse(response='{"x": 1}'),
        ])

        with self.assertRaises(CompletionError):
            await generate_structured(
                client,
                model="test-model",
                system="Be precise.",
                prompt="Return JSON.",
                output_type={"type": "not-a-json-schema-type"},
            )

        self.assertEqual(client.calls, [])


class CompletionErrorTests(unittest.TestCase):
    def test_package_all_contains_existing_exports_only(self):
        import app.ollama as ollama_module

        for name in ollama_module.__all__:
            self.assertTrue(hasattr(ollama_module, name), name)

    def test_message_and_details(self):
        err = CompletionError("boom", {"key": "value"})
        self.assertEqual(str(err), "boom")
        self.assertEqual(err.details, {"key": "value"})

    def test_model_retry_inheritance(self):
        err = MaxRetriesExceeded()
        self.assertIsInstance(err, CompletionError)

    def test_empty_response_error_inheritance(self):
        err = EmptyResponseError()
        self.assertIsInstance(err, OutputParsingError)
        self.assertIsInstance(err, CompletionError)
        self.assertEqual(str(err), "Empty response from model")

    def test_max_retries_exceeded_carries_last_error_details(self):
        cause = OutputParsingError("bad json")
        err = MaxRetriesExceeded(last_error=cause)
        self.assertIs(err.last_error, cause)
        self.assertEqual(err.details["last_error_type"], "OutputParsingError")
        self.assertEqual(err.details["last_error"], "bad json")


class RunUsageTests(unittest.TestCase):
    def test_merge_sums_attributes(self):
        a = RunUsage(requests=1, input_tokens=10, output_tokens=5)
        b = RunUsage(requests=2, input_tokens=20, output_tokens=10)
        c = a.merge(b)
        self.assertEqual(c.requests, 3)
        self.assertEqual(c.input_tokens, 30)
        self.assertEqual(c.output_tokens, 15)

    def test_add_operator(self):
        a = RunUsage(requests=1, input_tokens=5, output_tokens=3)
        b = RunUsage(requests=1, input_tokens=5, output_tokens=2)
        c = a + b
        self.assertEqual(c.requests, 2)
        self.assertEqual(c.input_tokens, 10)
        self.assertEqual(c.output_tokens, 5)

    def test_details_merge(self):
        a = RunUsage(details={"a": 1, "b": 2})
        b = RunUsage(details={"b": 3, "c": 4})
        c = a.merge(b)
        self.assertEqual(c.details, {"a": 1, "b": 5, "c": 4})

    def test_from_ollama_response(self):
        resp = FakeGenerateResponse(
            response="",
            prompt_eval_count=55,
            eval_count=12,
            prompt_eval_duration=110_000_000,
            load_duration=25_000_000,
        )
        usage = RunUsage.from_ollama_response(resp)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.input_tokens, 55)
        self.assertEqual(usage.output_tokens, 12)
        self.assertEqual(usage.prompt_eval_duration_ms, 110)
        self.assertEqual(usage.load_duration_ms, 25)


if __name__ == "__main__":
    unittest.main()
