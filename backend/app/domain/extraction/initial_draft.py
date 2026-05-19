from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent, RunContext

from app.domain.datasources import DataPackage
from app.domain.extraction.agents import (
    DEFAULT_OUTPUT_RETRIES,
    create_schema_validated_agent,
    validate_json_output_against_schema,
)
from app.domain.extraction.artifacts import InitialContext
from app.domain.extraction.schema_utils import (
    resolve_effective_schema as _resolve_effective_schema,
    resolve_ref as _resolve_ref,
    schema_allows_null as _schema_allows_null,
    schema_allows_type as _schema_allows_type,
)
from app.domain.profiles import ProfileManifest, validation_schema_for_target_class


INITIAL_DRAFT_INSTRUCTIONS = (
    "You are an assistant that initializes a schema-conformant DCAT dataset "
    "draft from a previously extracted InitialContext. Use the context to fill "
    "high-level dataset fields such as title, description, creator when clearly "
    "supported, theme, keywords, analytical technique, instrument or device, "
    "and high-level data-generating activity. Create context-supported shell "
    "objects for obvious resources such as analyzed entities, instruments, "
    "activities, and distributions when the schema supports them. Use null for "
    "unknown fields only when null is valid for that field in the JSON Schema. "
    "Do not fill specific fields that require detailed document chunks, "
    "quantitative attributes, calibration values, exact measurements, "
    "variable-level descriptions, or detailed distribution metadata. Do not "
    "invent creators, organizations, instruments, methods, sample counts, "
    "identifiers, measurements, or provenance details."
)

INITIAL_DRAFT_SKELETON_MAX_DEPTH = 3


@dataclass
class InitialDraftDeps:
    initial_context: InitialContext
    data_package: DataPackage
    profile_manifest: ProfileManifest
    profile_json_schema: dict[str, Any]


class InitialContextRequiredError(Exception):
    """Raised when InitialDraft is requested before InitialContext exists."""


def create_initial_draft_agent(
    *,
    model: Any,
    profile_json_schema: dict[str, Any],
    target_class: str,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[InitialDraftDeps, dict[str, Any]]:
    output_schema = validation_schema_for_target_class(
        json_schema=profile_json_schema,
        target_class=target_class,
    )
    agent = cast(
        Agent[InitialDraftDeps, dict[str, Any]],
        create_schema_validated_agent(
            model=model,
            json_schema=output_schema,
            output_name="InitialDraft",
            output_description=(
                "Initial schema-conformant dataset draft generated from an "
                "InitialContext."
            ),
            instructions=INITIAL_DRAFT_INSTRUCTIONS,
            deps_type=InitialDraftDeps,
            model_settings={
                "temperature": 0.0,
                "seed": 42,
            },
            output_retries=output_retries,
        ),
    )

    @agent.instructions
    def add_initial_draft_context(ctx: RunContext[InitialDraftDeps]) -> str:
        initial_context_json = ctx.deps.initial_context.model_dump_json(
            indent=2,
        )
        return (
            f"DCAT schema profile: {ctx.deps.profile_manifest.identifier}\n"
            f"Target class: {ctx.deps.profile_manifest.target_class}\n\n"
            "Profile JSON Schema:\n"
            f"{json.dumps(ctx.deps.profile_json_schema, indent=2)}\n\n"
            "Dataset source name:\n"
            f"{ctx.deps.data_package.file_name}\n\n"
            "Previously extracted InitialContext:\n"
            f"{initial_context_json}\n\n"
            "Create the initial dataset draft now. Only fill fields that can "
            "be inferred from the InitialContext or the dataset source name. "
            "Create context-supported shell objects for obvious resources when "
            "they are supported by the schema. Set unknown fields to null when "
            "null is valid for that field. Keep later-stage, chunk-specific, "
            "quantitative, or highly detailed fields generic, null, or omitted "
            "when the schema allows it. "
            "Return only the final JSON object."
        )

    return agent


async def initialize_draft_from_initial_context(
    *,
    initial_context: InitialContext,
    data_package: DataPackage,
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    agent = create_initial_draft_agent(
        model=model,
        profile_json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
    )
    deps = InitialDraftDeps(
        initial_context=initial_context,
        data_package=data_package,
        profile_manifest=profile_manifest,
        profile_json_schema=profile_json_schema,
    )
    result = await agent.run(
        (
            "Initialize the DCAT dataset draft from the existing InitialContext. "
            "Return a single schema-conformant JSON object."
        ),
        deps=deps,
    )
    expanded = expand_schema_placeholders(
        draft=result.output,
        profile_json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
        max_depth=INITIAL_DRAFT_SKELETON_MAX_DEPTH,
    )
    return validate_json_output_against_schema(
        expanded,
        validation_schema_for_target_class(
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        ),
    )


def expand_schema_placeholders(
    *,
    draft: dict[str, Any],
    profile_json_schema: dict[str, Any],
    target_class: str,
    max_depth: int = INITIAL_DRAFT_SKELETON_MAX_DEPTH,
) -> dict[str, Any]:
    """Add schema-valid null placeholders to an initial draft.

    The expander is intentionally conservative: it preserves inferred values,
    defaults missing nullable fields to null, and only recurses into nested
    object/list structures the agent already created.
    """

    root_schema = validation_schema_for_target_class(
        json_schema=profile_json_schema,
        target_class=target_class,
    )
    expanded = deepcopy(draft)
    target_schema = _resolve_effective_schema(root_schema, root_schema)
    if not isinstance(target_schema, dict):
        return expanded

    return _expand_value_for_schema(
        value=expanded,
        schema=target_schema,
        root_schema=root_schema,
        depth=1,
        max_depth=max_depth,
    )


def _expand_value_for_schema(
    *,
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    depth: int,
    max_depth: int,
) -> Any:
    effective_schema = _resolve_effective_schema(schema, root_schema)

    if isinstance(value, dict):
        return _expand_object_for_schema(
            value=value,
            schema=effective_schema,
            root_schema=root_schema,
            depth=depth,
            max_depth=max_depth,
        )

    if isinstance(value, list):
        item_schema = effective_schema.get("items")
        if not isinstance(item_schema, dict) or depth > max_depth:
            return value
        return [
            _expand_value_for_schema(
                value=item,
                schema=item_schema,
                root_schema=root_schema,
                depth=depth,
                max_depth=max_depth,
            )
            if isinstance(item, (dict, list))
            else item
            for item in value
        ]

    return value


def _expand_object_for_schema(
    *,
    value: dict[str, Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    if depth > max_depth or not _schema_allows_type(schema, "object"):
        return value

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return value

    for property_name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue

        if property_name not in value:
            if _schema_allows_null(property_schema, root_schema):
                value[property_name] = None
            continue

        property_value = value[property_name]
        if property_value is None or depth >= max_depth:
            continue

        value[property_name] = _expand_value_for_schema(
            value=property_value,
            schema=property_schema,
            root_schema=root_schema,
            depth=depth + 1,
            max_depth=max_depth,
        )

    return value

# Schema utility functions are now imported from schema_utils:
#   _resolve_effective_schema, _resolve_ref, _schema_allows_null, _schema_allows_type
