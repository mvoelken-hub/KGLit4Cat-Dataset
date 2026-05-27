"""Tests for the custom Ollama structured completion module."""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from typing import Any
from unittest import IsolatedAsyncioTestCase

from pydantic import BaseModel

from app.ollama.completion import CompletionResult, _extract_json_schema, _strip_markdown_fences, generate_structured
from app.ollama.errors import CompletionError, EmptyResponseError, MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage


class SimpleOutput(BaseModel):
    answer: str
    score: int


class NestedOutput(BaseModel):
    items: list[SimpleOutput]
    total: int


@dataclass
class FakeGenerateResponse:
    """Fake ollama GenerateResponse for testing."""

    response: str
    prompt_eval_count: int = 0
    eval_count: int = 0


class FakeOllamaClient:
    """Fake ollama.AsyncClient that returns pre-scheduled responses."""

    def __init__(self, responses: list[FakeGenerateResponse] | None = None):
        self.responses: list[FakeGenerateResponse] = responses or []
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
            return r
        # Default empty response if exhausted
        return FakeGenerateResponse(response="{}")


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
        self.assertEqual(call["system"], "sys")
        self.assertEqual(call["prompt"], "prompt")
        self.assertIsInstance(call["format"], dict)
        self.assertEqual(
            call["options"].model_dump(exclude_none=True),
            {"num_ctx": 4096, "seed": 123, "temperature": 0.7},
        )
        self.assertTrue(call["think"])
        self.assertEqual(call["keep_alive"], 300)

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


class GenerateStructuredRetryTests(IsolatedAsyncioTestCase):
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
        resp = FakeGenerateResponse(response="", prompt_eval_count=55, eval_count=12)
        usage = RunUsage.from_ollama_response(resp)
        self.assertEqual(usage.requests, 1)
        self.assertEqual(usage.input_tokens, 55)
        self.assertEqual(usage.output_tokens, 12)


if __name__ == "__main__":
    unittest.main()
