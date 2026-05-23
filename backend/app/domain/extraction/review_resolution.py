from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.profiles import ProfileManifest


ReviewItemKind = Literal["matched", "unmapped"]
ReviewItemOutcome = Literal["included", "already_present", "excluded", "unresolved"]
TokenUsageCallback = Callable[[str, Any, int], None]


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


class PatchReviewDecision(BaseModel):
    id: str = Field(description="Review item ID this decision applies to.")
    outcome: ReviewItemOutcome = Field(
        description=(
            "included when the final draft now represents the information; "
            "already_present when the current draft already represented it; "
            "excluded when it was intentionally kept out. The service may "
            "normalize unresolved decisions to excluded."
        ),
    )
    note: str = Field(description="Short explanation of the decision.")
    target_path: str | None = Field(
        default=None,
        description="Draft path where the item was handled, when applicable.",
    )


class PatchReviewResolution(BaseModel):
    draft_patch: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "JSON merge patch to apply to the current Dataset draft. Prefer this "
            "over returning the complete draft."
        ),
    )
    final_draft: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Backward-compatible complete schema-valid Dataset draft after "
            "resolving review items. Prefer draft_patch for new resolver output."
        ),
    )
    item_decisions: list[PatchReviewDecision] = Field(
        default_factory=list,
        description="Explicit outcome for each submitted review item.",
    )
    unmapped_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Assignments for unmapped review item IDs to existing top-level draft fields.",
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
- Resolve every review item by producing a JSON merge patch in draft_patch.
- Preserve all valid current draft content unless a review item supports a correction.
- For matched items, inspect the proposed patch, issues, evidence, and current draft.
- When a proposed patch adds a refined description next to an older generic
  description, replace the older description with the refined one. Do not
  accumulate long lists of near-duplicate descriptions.
- For unmapped items, assign each fact to the best existing top-level draft field
  in unmapped_assignments when it can be represented safely.
- For every submitted review item, add exactly one item_decisions entry.
- Return final_draft only if a merge patch cannot express the change.
- Use outcome "included" only when final_draft contains the information.
- Use outcome "already_present" only when the current draft already contains it.
- Use outcome "excluded" when the source fact or patch is invalid, semantically
  wrong, schema-incompatible, unsupported, low confidence, or cannot be safely
  attached to the draft, and you intentionally keep it out.
- Do not return outcome "unresolved"; choose "excluded" with a clear note when
  you cannot safely include or confirm the item.
- Do not use description, title, or keyword as dumping grounds for structured facts.
- Never invent source facts, URIs, schema fields, files, instruments, or values.
- Never create namespace-looking identifiers such as https://w3id.org/... .
  Use ids that are scoped to the dataset or experiment, for example
  "1h-nmr-clean/activity/1h-nmr-acquisition" or "sample-a/agent/bruker-nmr".
- For ChecksumAlgorithm, follow the schema exactly. If the schema only allows
  title/description, do not add an id field to the algorithm object.
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
                "Resolve the review items with a complete schema-valid final draft and per-item decisions.",
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
    on_token_usage: TokenUsageCallback | None = None,
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
        "Resolve the unresolved patch review items. Return a JSON merge patch "
        "and update it only where the source evidence and profile schema "
        "support the change.",
        deps=deps,
    )
    if on_token_usage is not None:
        patch_count = len({item.file_name for item in review_items if item.file_name}) or 1
        on_token_usage("auto_resolve", result.usage, patch_count)
    return result.output
