"""Core structured completion via Ollama /api/generate.

Replaces pydantic-ai Agent.run() with a single async function that calls
Ollama's /api/generate endpoint directly, using the `format` parameter
for JSON schema validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, get_args

from pydantic import BaseModel, ValidationError

from app.ollama.errors import CompletionError, MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage

T = TypeVar("T")

# Regex to strip markdown JSON fences (```json ... ```)
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$")


@dataclass
class CompletionResult(Generic[T]):
    """Result of a successful structured completion."""

    output: T
    usage: RunUsage


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    cleaned = _MARKDOWN_FENCE_RE.sub("", text)
    # Also strip any leading/trailing whitespace and stray backticks
    cleaned = cleaned.strip().strip("`").strip()
    return cleaned


def _extract_json_schema(output_type: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """Extract a JSON Schema dict from a Pydantic model class or return a dict as-is."""
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return output_type.model_json_schema()
    if isinstance(output_type, dict):
        return output_type
    raise CompletionError(f"Unsupported output_type: {output_type!r}")


async def generate_structured(
    client: Any,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[BaseModel] | dict[str, Any],
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: bool = False,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
) -> CompletionResult[Any]:
    """Single /api/generate call with format=json_schema, parse + validate + retry.

    Args:
        client: ollama.AsyncClient instance
        model: Ollama model name (e.g. "qwen3.5:4b")
        system: System message (static + dynamic context combined)
        prompt: User message (the extraction instruction)
        output_type: Pydantic model class for structured output,
                     or dict for raw JSON schema (BaseModel validation skipped)
        retries: Max re-prompts on parse/validation failure
        temperature: Sampling temperature
        seed: Reproducibility seed
        think: Enable thinking mode (for qwen3.5 etc)
        num_ctx: Context window size; passed via options["num_ctx"].
                 If None, uses whatever the Ollama host has configured.
        keep_alive: How long to keep the model loaded. Default -1 = keep in RAM.

    Returns:
        CompletionResult with validated output and token usage.

    Raises:
        OutputParsingError: JSON decode or validation failure after all retries
        MaxRetriesExceeded: If retries exhausted
        CompletionError: For Ollama API errors
    """
    schema = _extract_json_schema(output_type)

    options: dict[str, Any] = {"temperature": temperature, "seed": seed}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx

    current_prompt = prompt
    total_usage = RunUsage()

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await client.generate(
                model=model,
                prompt=current_prompt,
                system=system,
                format=schema,
                options=options,
                think=think,
                keep_alive=keep_alive,
            )
        except Exception as e:
            raise CompletionError(f"Ollama API error: {e}") from e

        # Accumulate token usage from Ollama response metadata
        total_usage.requests += 1
        total_usage.input_tokens += getattr(response, "prompt_eval_count", 0) or 0
        total_usage.output_tokens += getattr(response, "eval_count", 0) or 0

        raw = getattr(response, "response", "")
        if not raw:
            last_error = OutputParsingError("Empty response from model")
            if attempt < retries:
                current_prompt = f"{prompt}\n\nError: Model returned empty output. Please return valid JSON conforming to the schema."
                continue
            raise OutputParsingError(str(last_error)) from last_error

        # Strip markdown fences if present
        cleaned = _strip_markdown_fences(raw)

        # Try to parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < retries:
                current_prompt = (
                    f"{prompt}\n\nError: Invalid JSON — {e}. "
                    "Please return ONLY valid JSON, no markdown fences, no commentary."
                )
                continue
            raise OutputParsingError(str(e)) from e

        # Validate against Pydantic model if provided
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            try:
                validated = output_type.model_validate(parsed)
            except ValidationError as e:
                last_error = e
                if attempt < retries:
                    current_prompt = (
                        f"{prompt}\n\nError: Schema validation failed — {e}. "
                        "Please return valid JSON that matches the schema exactly."
                    )
                    continue
                raise OutputParsingError(str(e)) from e
            return CompletionResult(output=validated, usage=total_usage)

        # Raw dict output (no Pydantic validation)
        return CompletionResult(output=parsed, usage=total_usage)

    raise MaxRetriesExceeded()
