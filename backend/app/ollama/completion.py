"""Core structured completion via Ollama /api/generate.

Provides one async function for schema-bound structured output through
Ollama's `format` parameter.
"""

from __future__ import annotations

import asyncio
import json
import math
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
from app.core.logging import logger

if TYPE_CHECKING:
    from app.ollama.client import OllamaClientWrapper

JsonSchema: TypeAlias = dict[str, Any]
ThinkMode: TypeAlias = bool | Literal["low", "medium", "high"] | None
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)

_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL,
)
_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed structured JSON. Preserve the factual content and "
    "field values from the failed response. Do not solve the original task "
    "again, do not infer new facts, and do not add commentary. Return only "
    "the corrected JSON object."
)
_SCHEMA_PROMPT_TEMPLATE = (
    "\n\nReturn exactly one JSON object that satisfies this JSON Schema. "
    "Do not include Markdown fences, prose, comments, or additional text."
    "\n\nJSON Schema:\n{schema}"
)
_EXAMPLE_PROMPT_TEMPLATE = (
    "\n\nReturn exactly one JSON object matching this compact example shape. "
    "Replace placeholder values with extracted values where available, and do "
    "not include Markdown fences, prose, comments, or additional text."
    "\n\nExample JSON shape:\n{example}"
)
_ESTIMATED_CHARS_PER_TOKEN = 4
# Be conservative: keep more headroom for the system-prompt schema or
# example additions, the per-request `format` parameter, and Ollama's own
# tokenization overhead. The ratio used to be 0.75 which left no headroom
# for the 73 KB dcat-ap-plus schema on a 32K-token context.
_INPUT_CONTEXT_BUDGET_RATIO = 0.5
_MAX_SCHEMA_EXAMPLE_DEPTH = 50


@dataclass
class CompletionResult(Generic[T]):
    """Result of a successful structured completion."""

    output: T
    usage: RunUsage


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    match = _MARKDOWN_FENCE_RE.match(text)
    cleaned = match.group("body") if match else text
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


def _estimated_tokens(text: str) -> int:
    return math.ceil(len(text) / _ESTIMATED_CHARS_PER_TOKEN)


def _fits_context_budget(
    *,
    system: str,
    prompt: str,
    addition: str,
    num_ctx: int | None,
) -> bool:
    if num_ctx is None:
        return True
    budget = max(1, int(num_ctx * _INPUT_CONTEXT_BUDGET_RATIO))
    return _estimated_tokens(system + addition + prompt) <= budget


def _schema_example(schema: JsonSchema) -> Any:
    return _schema_example_from_node(schema, schema, seen_refs=(), depth=0)


def _schema_example_from_node(
    node: Any,
    root: JsonSchema,
    *,
    seen_refs: tuple[str, ...],
    depth: int,
) -> Any:
    if not isinstance(node, dict):
        return None
    if depth > _MAX_SCHEMA_EXAMPLE_DEPTH:
        return None
    if "$ref" in node:
        ref = str(node["$ref"])
        if ref.startswith("#/$defs/"):
            key = ref.removeprefix("#/$defs/")
            if key in seen_refs:
                return {}
            return _schema_example_from_node(
                root.get("$defs", {}).get(key),
                root,
                seen_refs=(*seen_refs, key),
                depth=depth + 1,
            )
        return None
    for union_key in ("anyOf", "oneOf"):
        options = [option for option in node.get(union_key, []) if option.get("type") != "null"]
        if options:
            return _schema_example_from_node(
                options[0],
                root,
                seen_refs=seen_refs,
                depth=depth + 1,
            )
    if "const" in node:
        return node["const"]
    if "enum" in node and node["enum"]:
        return node["enum"][0]

    node_type = node.get("type")
    if isinstance(node_type, list):
        node_type = next((item for item in node_type if item != "null"), node_type[0])
    if node_type == "object" or "properties" in node:
        return {
            key: _schema_example_from_node(
                value,
                root,
                seen_refs=seen_refs,
                depth=depth + 1,
            )
            for key, value in node.get("properties", {}).items()
        }
    if node_type == "array":
        return [
            _schema_example_from_node(
                node.get("items", {}),
                root,
                seen_refs=seen_refs,
                depth=depth + 1,
            )
        ]
    if node_type == "integer":
        if isinstance(node.get("minimum"), int | float):
            return int(math.ceil(node["minimum"]))
        if isinstance(node.get("exclusiveMinimum"), int | float):
            return int(math.floor(node["exclusiveMinimum"]) + 1)
        return 0
    if node_type == "number":
        if isinstance(node.get("minimum"), int | float):
            return node["minimum"]
        if isinstance(node.get("exclusiveMinimum"), int | float):
            return node["exclusiveMinimum"] + 1
        return 0.0
    if node_type == "boolean":
        return False
    if node_type == "null":
        return None
    return ""


def _system_prompt_for_model(
    *,
    system: str,
    prompt: str,
    schema: JsonSchema,
    num_ctx: int | None,
) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    schema_addition = _SCHEMA_PROMPT_TEMPLATE.format(schema=schema_text)
    if _fits_context_budget(
        system=system,
        prompt=prompt,
        addition=schema_addition,
        num_ctx=num_ctx,
    ):
        return system + schema_addition

    example_text = json.dumps(
        _schema_example(schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    example_addition = _EXAMPLE_PROMPT_TEMPLATE.format(example=example_text)
    if _fits_context_budget(
        system=system,
        prompt=prompt,
        addition=example_addition,
        num_ctx=num_ctx,
    ):
        return system + example_addition
    return system


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


def _structured_failure(
    last_error: Exception | None,
    *,
    failed_response: str,
    first_response: str,
    usage: RunUsage,
) -> MaxRetriesExceeded:
    return MaxRetriesExceeded(
        "Max retries exceeded while generating structured output",
        last_error=last_error,
        failed_response=failed_response,
        first_response=first_response,
        usage=usage,
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
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
    api_retries: int = 2,
    api_retry_backoff_seconds: float = 0.5,
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
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
    api_retries: int = 2,
    api_retry_backoff_seconds: float = 0.5,
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
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
    api_retries: int = 2,
    api_retry_backoff_seconds: float = 0.5,
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
        api_retries: Max retries for Ollama API/transport failures per structured attempt.
        api_retry_backoff_seconds: Initial exponential backoff delay for API retries.

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
    current_model = model
    current_system = _system_prompt_for_model(
        system=system,
        prompt=current_prompt,
        schema=schema,
        num_ctx=num_ctx,
    )
    total_usage = RunUsage()

    last_error: Exception | None = None
    first_response = ""

    for attempt in range(retries + 1):
        try:
            response = await _generate_with_api_retries(
                client,
                model=current_model,
                prompt=current_prompt,
                system=current_system,
                schema=schema,
                options=options,
                think=think,
                keep_alive=keep_alive,
                api_retries=api_retries,
                api_retry_backoff_seconds=api_retry_backoff_seconds,
            )
        except CompletionError as exc:
            if first_response:
                exc.details.setdefault("first_response", first_response)
                setattr(exc, "first_response", first_response)
            raise

        # Accumulate token usage from Ollama response metadata
        total_usage = total_usage + RunUsage.from_ollama_response(response)

        raw = response.response or ""
        if not raw:
            raise EmptyResponseError()
        if not first_response:
            first_response = raw

        # Strip markdown fences if present
        cleaned = _strip_markdown_fences(raw)

        # Try to parse JSON
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = OutputParsingError(str(e))
            if attempt < retries:
                current_system = _system_prompt_for_model(
                    system=_REPAIR_SYSTEM_PROMPT,
                    prompt=_repair_prompt(failed_response=raw, error=last_error),
                    schema=schema,
                    num_ctx=num_ctx,
                )
                current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                current_model = repair_model or model
                continue
            raise _structured_failure(
                last_error,
                failed_response=raw,
                first_response=first_response,
                usage=total_usage,
            ) from e

        # Validate against Pydantic model if provided
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            try:
                validated = output_type.model_validate(parsed)
            except PydanticValidationError as e:
                last_error = OutputParsingError(str(e))
                if attempt < retries:
                    current_system = _system_prompt_for_model(
                        system=_REPAIR_SYSTEM_PROMPT,
                        prompt=_repair_prompt(failed_response=raw, error=last_error),
                        schema=schema,
                        num_ctx=num_ctx,
                    )
                    current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                    current_model = repair_model or model
                    continue
                raise _structured_failure(
                    last_error,
                    failed_response=raw,
                    first_response=first_response,
                    usage=total_usage,
                ) from e
            return CompletionResult(output=validated, usage=total_usage)

        # Raw dict schema output validated with jsonschema.
        if raw_schema_validator is not None:
            validation_errors = _format_json_schema_validation_errors(raw_schema_validator, parsed)
            if validation_errors:
                last_error = OutputParsingError(
                    "Output did not match JSON Schema:\n" + validation_errors
                )
                if attempt < retries:
                    current_system = _system_prompt_for_model(
                        system=_REPAIR_SYSTEM_PROMPT,
                        prompt=_repair_prompt(failed_response=raw, error=last_error),
                        schema=schema,
                        num_ctx=num_ctx,
                    )
                    current_prompt = _repair_prompt(failed_response=raw, error=last_error)
                    current_model = repair_model or model
                    continue
                raise _structured_failure(
                    last_error,
                    failed_response=raw,
                    first_response=first_response,
                    usage=total_usage,
                ) from last_error
        return CompletionResult(output=parsed, usage=total_usage)

    raise _max_retries_exceeded(last_error)


async def _generate_with_api_retries(
    client: OllamaClientWrapper,
    *,
    model: str,
    prompt: str,
    system: str,
    schema: JsonSchema,
    options: ollama.Options,
    think: ThinkMode,
    keep_alive: float | str | None,
    api_retries: int,
    api_retry_backoff_seconds: float,
) -> ollama.GenerateResponse:
    max_attempts = max(0, api_retries) + 1
    last_error: Exception | None = None
    for api_attempt in range(max_attempts):
        try:
            kwargs = dict(
                model=model,
                prompt=prompt,
                system=system,
                format=schema,
                options=options,
                keep_alive=keep_alive,
            )
            if think is not None:
                kwargs["think"] = think
            return cast(
                ollama.GenerateResponse,
                await client.ollama_client.generate(**kwargs),
            )
        except Exception as exc:
            last_error = exc
            if api_attempt >= max_attempts - 1:
                break
            delay = max(0.0, api_retry_backoff_seconds) * (2**api_attempt)
            logger.warning(
                "Ollama structured generation failed; retrying",
                extra={
                    "model": model,
                    "api_attempt": api_attempt + 1,
                    "api_retries": api_retries,
                    "delay_seconds": delay,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if delay:
                await asyncio.sleep(delay)
    raise CompletionError(
        f"Ollama API error after {max_attempts} attempt(s): {last_error}",
        {
            "error_type": type(last_error).__name__ if last_error else None,
            "api_attempts": max_attempts,
            "model": model,
        },
    ) from last_error


@overload
async def repair_structured_output(
    client: OllamaClientWrapper,
    *,
    model: str,
    failed_response: str,
    error: Exception,
    output_type: type[ModelT],
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[ModelT]: ...


@overload
async def repair_structured_output(
    client: OllamaClientWrapper,
    *,
    model: str,
    failed_response: str,
    error: Exception,
    output_type: JsonSchema,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[Any]:
    ...


async def repair_structured_output(
    client: OllamaClientWrapper,
    *,
    model: str,
    failed_response: str,
    error: Exception,
    output_type: type[BaseModel] | JsonSchema,
    retries: int = 2,
    temperature: float = 0.0,
    seed: int = 42,
    think: ThinkMode = None,
    num_ctx: int | None = None,
    keep_alive: float | str | None = -1,
    repair_model: str | None = None,
) -> CompletionResult[Any]:
    """Repair a previously failed structured output without rerunning the task prompt."""
    return await generate_structured(
        client,
        model=repair_model or model,
        system=_REPAIR_SYSTEM_PROMPT,
        prompt=_repair_prompt(failed_response=failed_response, error=error),
        output_type=output_type,
        retries=retries,
        temperature=temperature,
        seed=seed,
        think=think,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        repair_model=repair_model,
    )
