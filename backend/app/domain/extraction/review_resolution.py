from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.profiles import ProfileManifest


ReviewItemKind = Literal["matched", "unmapped"]


class PatchReviewItem(BaseModel):
    id: str
    kind: ReviewItemKind
    path: str
    detail: str | None = None
    issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    patch: dict[str, Any] | None = None
    fact: str | None = None
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    file_name: str | None = None


class PatchReviewResolution(BaseModel):
    draft_patch: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON merge patch to apply to the current draft. Use only fields "
            "allowed by the profile schema. Leave empty when no draft edit is needed."
        ),
    )
    resolved_item_ids: list[str] = Field(
        default_factory=list,
        description="Review item IDs that are resolved by the draft_patch or already-resolved draft content.",
    )
    unmapped_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Assignments for unmapped review item IDs to existing top-level draft fields.",
    )
    resolution_notes: dict[str, str] = Field(
        default_factory=dict,
        description="Short note for each resolved item explaining what was done.",
    )
    unresolved_item_ids: list[str] = Field(
        default_factory=list,
        description="Review item IDs the agent could not resolve safely.",
    )


@dataclass
class PatchReviewResolutionDeps:
    current_draft: dict[str, Any]
    review_items: list[PatchReviewItem]
    profile_manifest: ProfileManifest
    profile_json_schema: dict[str, Any]
    existing_review_state: dict[str, Any]


PATCH_REVIEW_RESOLUTION_INSTRUCTIONS = """
You are a metadata patch review resolution agent.

You receive the current DCAT-style metadata draft, unresolved review items, the
profile JSON Schema, and the existing review state.

Your task:
- Resolve as many review items as possible by producing a schema-valid draft_patch.
- For matched items, inspect the proposed patch, issues, evidence, and current draft.
- When a proposed patch adds a refined description next to an older generic
  description, replace the older description with the refined one. Do not
  accumulate long lists of near-duplicate descriptions.
- For unmapped items, assign each fact to the best existing top-level draft field
  in unmapped_assignments and add any necessary schema-valid draft_patch content.
- Only mark a review item resolved when the current draft already handles it or
  your draft_patch handles it.
- If an item cannot be resolved without inventing facts, inventing schema fields,
  or making a low-confidence semantic choice, put its ID in unresolved_item_ids.
- Do not use description, title, or keyword as dumping grounds for structured facts.
- Never invent source facts, URIs, schema fields, files, instruments, or values.
- Never create namespace-looking identifiers such as https://w3id.org/... .
  Use ids that are scoped to the dataset or experiment, for example
  "1h-nmr-clean/activity/1h-nmr-acquisition" or "sample-a/agent/bruker-nmr".
- draft_patch must be a JSON merge patch object rooted at the Dataset draft.
- Return only the PatchReviewResolution JSON object.
"""


def create_patch_review_resolution_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[PatchReviewResolutionDeps, PatchReviewResolution]:
    agent = Agent(
        model,
        deps_type=PatchReviewResolutionDeps,
        output_type=prompted_json_output(
            PatchReviewResolution,
            name="PatchReviewResolution",
            description="Agent-authored resolution for unresolved patch review items.",
        ),
        instructions=PATCH_REVIEW_RESOLUTION_INSTRUCTIONS,
        model_settings={
            "temperature": 0.0,
            "seed": 42,
        },
        output_retries=output_retries,
    )

    @agent.instructions
    def add_resolution_context(ctx: RunContext[PatchReviewResolutionDeps]) -> str:
        return "".join(
            [
                f"Profile identifier:\n{ctx.deps.profile_manifest.identifier}\n\n",
                f"Target class:\n{ctx.deps.profile_manifest.target_class}\n\n",
                "Profile JSON Schema:\n"
                f"{json.dumps(ctx.deps.profile_json_schema, indent=2, ensure_ascii=False)}\n\n",
                "Current draft JSON:\n"
                f"{json.dumps(ctx.deps.current_draft, indent=2, ensure_ascii=False)}\n\n",
                "Existing review state JSON:\n"
                f"{json.dumps(ctx.deps.existing_review_state, indent=2, ensure_ascii=False)}\n\n",
                "Unresolved review items JSON:\n"
                f"{json.dumps([item.model_dump(mode='json') for item in ctx.deps.review_items], indent=2, ensure_ascii=False)}\n\n",
                "Resolve the review items with a schema-valid draft patch and review-state updates.",
            ]
        )

    return agent


async def resolve_patch_review_items(
    *,
    current_draft: dict[str, Any],
    review_items: list[PatchReviewItem],
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    existing_review_state: dict[str, Any],
    model: Any,
) -> PatchReviewResolution:
    agent = create_patch_review_resolution_agent(model=model)
    deps = PatchReviewResolutionDeps(
        current_draft=current_draft,
        review_items=review_items,
        profile_manifest=profile_manifest,
        profile_json_schema=profile_json_schema,
        existing_review_state=existing_review_state,
    )
    result = await agent.run(
        "Resolve the unresolved patch review items. Update the draft only where "
        "the source evidence and profile schema support the change.",
        deps=deps,
    )
    return result.output
