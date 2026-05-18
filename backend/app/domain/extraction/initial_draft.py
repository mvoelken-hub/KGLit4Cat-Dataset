from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Agent, RunContext

from app.domain.datasources import DataPackage
from app.domain.extraction.agents import (
    DEFAULT_OUTPUT_RETRIES,
    create_schema_validated_agent,
)
from app.domain.extraction.artifacts import InitialContext
from app.domain.extraction.profiles import ProfileManifest, validation_schema_for_target_class


INITIAL_DRAFT_INSTRUCTIONS = (
    "You are an assistant that initializes a schema-conformant DCAT dataset "
    "draft from a previously extracted InitialContext. Use the context to fill "
    "high-level dataset fields such as title, description, creator when clearly "
    "supported, theme, keywords, analytical technique, instrument or device, "
    "and high-level data-generating activity. Do not fill specific fields that "
    "require detailed document chunks, quantitative attributes, calibration "
    "values, exact measurements, variable-level descriptions, or detailed "
    "distribution metadata. Do not invent creators, organizations, instruments, "
    "methods, sample counts, identifiers, measurements, or provenance details."
)


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
            "Keep later-stage, chunk-specific, quantitative, or highly "
            "detailed fields generic or omitted when the schema allows it. "
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
    return result.output
