"""Core structured completion via Ollama /api/generate.

Replaces pydantic-ai Agent.run() with a single async function that calls
Ollama's /api/generate endpoint directly, using the `format` parameter
for JSON schema validation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import ollama
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ValidationError as PydanticValidationError

from app.ollama.errors import CompletionError, EmptyResponseError, MaxRetriesExceeded, OutputParsingError
from app.ollama.usage import RunUsage

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper

JsonSchema: TypeAlias = dict[str, Any]
ThinkMode: TypeAlias = bool | Literal["low", "medium", "high"] | None
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

# Regex to strip markdown JSON fences (```json ... ```)
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$")
_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed structured JSON. Preserve the factual content and "
    "field values from the failed response. Do not solve the original task "
    "again, do not infer new facts, and do not add commentary. Return only "
    "the corrected JSON object."
)


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


def _extract_json_schema(output_type: type[BaseModel] | JsonSchema) -> JsonSchema:
    """Extract a JSON Schema dict from a Pydantic model class or return a dict as-is."""
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        return output_type.model_json_schema()
    if isinstance(output_type, dict):
        return output_type
    raise CompletionError(f"Unsupported output_type: {output_type!r}")


def _json_schema_validator(json_schema: JsonSchema):
    """Build a JSON Schema validator, failing fast for invalid schemas."""
    try:
        validator_class = validator_for(json_schema)
        validator_class.check_schema(json_schema)
        return validator_class(json_schema)
    except SchemaError as e:
        raise CompletionError(f"Invalid output JSON Schema: {e.message}") from e


def _format_json_schema_validation_errors(validator: Any, output: Any) -> str:
    """Return stable, readable validation errors for repair prompts."""
    errors = sorted(validator.iter_errors(output), key=str)
    return "\n".join(
        f"{_format_json_path(error.absolute_path)}: {error.message} "
        f"(schema: {_format_json_path(error.absolute_schema_path)})"
        for error in errors
    )


def _format_json_path(path: Any) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _max_retries_exceeded(last_error: Exception | None) -> MaxRetriesExceeded:
    return MaxRetriesExceeded(
        "Max retries exceeded while generating structured output",
        last_error=last_error,
    )


def _repair_prompt(*, failed_response: str, error: Exception) -> str:
    """Build a narrow correction prompt from only the bad output and error."""
    response = failed_response if failed_response else "[empty response]"
    return (
        "The previous response failed JSON parsing or schema validation. "
        "Repair only the JSON structure, field names, and value types needed "
        "to satisfy the schema. Preserve the existing information from the "
        "failed response. If a required field is missing and cannot be derived "
        "from the failed response, use the least-informative schema-valid "
        "value. Return only corrected JSON.\n\n"
        f"Validation error:\n{error}\n\n"
        f"Failed response:\n{response}"
    )


@overload
async def generate_structured(
    client: OllamaClientWrapper,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[ModelT],
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = False,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[ModelT]: ...


@overload
async def generate_structured(
    client: OllamaClientWrapper,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: JsonSchema,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = False,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[Any]:
    ...


async def generate_structured(
    client: OllamaClientWrapper,
    *,
    model: str,
    system: str,
    prompt: str,
    output_type: type[BaseModel] | JsonSchema,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = False,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[Any]:
    """Single /api/generate call with format=json_schema, parse + validate + retry.

    Args:
        client: OllamaClientWrapper instance
        model: Ollama model name (e.g. "qwen3.5:4b")
        system: System message (static + dynamic context combined)
        prompt: User message (the extraction instruction)
        output_type: Pydantic model class for structured output,
                     or dict for raw JSON schema validation
        retries: Max repair attempts on parse/validation failure
        temperature: Sampling temperature
        seed: Reproducibility seed
        think: Enable thinking mode (for qwen3.5 etc)
        num_ctx: Context window size; passed via options["num_ctx"].
                 If None, uses whatever the Ollama host has configured.
        keep_alive: How long to keep the model loaded. Default -1 = keep in RAM.
        repair_model: Optional model name for repair attempts. Defaults to model.

    Returns:
        CompletionResult with validated output and token usage.

    Raises:
        MaxRetriesExceeded: JSON decode or validation failure after all retries
        CompletionError: For Ollama API errors
    """
    schema = _extract_json_schema(output_type)
    raw_schema_validator = _json_schema_validator(schema) if isinstance(output_type, dict) else None

    options = ollama.Options(
        temperature=temperature,
        seed=seed,
        num_ctx=num_ctx,
    )

    current_prompt = prompt
    current_system = system
    current_model = model
    total_usage = RunUsage()

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = cast(
                ollama.GenerateResponse,
                await client.ollama_client.generate(
                    model=current_model,
                    prompt=current_prompt,
                    system=current_system,
                    format=schema,
                    options=options,
                    think=think,
                    keep_alive=keep_alive,
                ),
            )
        except Exception as e:
            raise CompletionError(f"Ollama API error: {e}") from e

        # Accumulate token usage from Ollama response metadata
        total_usage = total_usage + RunUsage.from_ollama_response(response)

        raw = response.response or ""
        if not raw:
            raise EmptyResponseError()

        # Strip markdown fences if present
        cleaned = _strip_markdown_fences(raw)

        # Try to parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = OutputParsingError(str(e))
            if attempt < retries:
                current_system = _REPAIR_SYSTEM_PROMPT
                current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                current_model = repair_model or model
                continue
            raise _max_retries_exceeded(last_error) from e

        # Validate against Pydantic model if provided
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            try:
                validated = output_type.model_validate(parsed)
            except PydanticValidationError as e:
                last_error = OutputParsingError(str(e))
                if attempt < retries:
                    current_system = _REPAIR_SYSTEM_PROMPT
                    current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                    current_model = repair_model or model
                    continue
                raise _max_retries_exceeded(last_error) from e
            return CompletionResult(output=validated, usage=total_usage)

        # Raw dict schema output validated with jsonschema.
        if raw_schema_validator is not None:
            validation_errors = _format_json_schema_validation_errors(raw_schema_validator, parsed)
            if validation_errors:
                last_error = OutputParsingError(
                    "Output did not match JSON Schema:\n" + validation_errors
                )
                if attempt < retries:
                    current_system = _REPAIR_SYSTEM_PROMPT
                    current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                    current_model = repair_model or model
                    continue
                raise _max_retries_exceeded(last_error) from last_error
        return CompletionResult(output=parsed, usage=total_usage)

    raise _max_retries_exceeded(last_error)
