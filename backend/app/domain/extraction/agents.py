from __future__ import annotations

from copy import deepcopy
from typing import Any

from jsonschema import ValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from pydantic_ai import Agent, ModelRetry, RunContext, StructuredDict
from pydantic_ai.output import PromptedOutput


DEFAULT_OUTPUT_RETRIES = 2
JSON_OUTPUT_TEMPLATE = (
    "Return a valid JSON object with no markdown, no code fences, "
    "and no surrounding commentary.\n\n"
    "The JSON object must conform to this JSON Schema:\n"
    "{schema}"
)


def prompted_json_output(
    output_type: Any,
    *,
    name: str | None = None,
    description: str | None = None,
    template: str = JSON_OUTPUT_TEMPLATE,
) -> PromptedOutput[Any]:
    return PromptedOutput(
        output_type,
        name=name,
        description=description,
        template=template,
    )


def structured_profile_output(
    json_schema: dict[str, Any],
    *,
    name: str,
    description: str | None = None,
    template: str = JSON_OUTPUT_TEMPLATE,
) -> PromptedOutput[Any]:
    output_type = StructuredDict(
        deepcopy(json_schema),
        name=name,
        description=description,
    )
    return prompted_json_output(
        output_type,
        name=name,
        description=description,
        template=template,
    )


def validate_json_output_against_schema(
    output: dict[str, Any],
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    validator = _schema_validator(json_schema)
    errors = [
        _format_validation_error(error)
        for error in sorted(validator.iter_errors(output), key=str)
    ]
    if errors:
        raise ModelRetry(
            "Output did not match JSON Schema:\n" + "\n".join(errors)
        )
    return output


def create_schema_validated_agent(
    *,
    model: Any,
    json_schema: dict[str, Any],
    output_name: str,
    output_description: str | None = None,
    instructions: str | None = None,
    deps_type: type[Any] | None = None,
    model_settings: dict[str, Any] | None = None,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
    tool_retries: int | None = None,
) -> Agent[Any, Any]:
    agent_kwargs: dict[str, Any] = {
        "model": model,
        "output_type": structured_profile_output(
            json_schema,
            name=output_name,
            description=output_description,
        ),
        "instructions": instructions,
        "model_settings": model_settings,
        "output_retries": output_retries,
    }
    if tool_retries is not None:
        agent_kwargs["tool_retries"] = tool_retries
    if deps_type is not None:
        agent_kwargs["deps_type"] = deps_type

    agent = Agent(**agent_kwargs)

    @agent.output_validator
    def validate_against_schema(
        ctx: RunContext[Any],
        output: dict[str, Any],
    ) -> dict[str, Any]:
        return validate_json_output_against_schema(output, json_schema)

    return agent


def _schema_validator(json_schema: dict[str, Any]):
    try:
        validator_class = validator_for(json_schema)
        validator_class.check_schema(json_schema)
        return validator_class(json_schema)
    except SchemaError as exc:
        raise ValueError(f"Invalid output JSON Schema: {exc.message}") from exc


def _format_validation_error(error: ValidationError) -> str:
    return (
        f"{_format_json_path(error.absolute_path)}: {error.message} "
        f"(schema: {_format_json_path(error.absolute_schema_path)})"
    )


def _format_json_path(path: Any) -> str:
    result = "$"
    for part in path:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result
