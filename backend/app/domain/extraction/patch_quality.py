from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from app.domain.datasources import ContentChunk
from app.domain.extraction.agents import DEFAULT_OUTPUT_RETRIES, prompted_json_output
from app.domain.extraction.artifacts import InitialContext, PatchCandidate
from app.domain.profiles import ProfileManifest, validate_document_against_profile


PatchQualityDecision = Literal["accept", "revise", "reject"]
TokenUsageCallback = Callable[[str, Any, int], None]


class PatchQualityIssue(BaseModel):
    path: str
    issue_type: Literal[
        "description_abuse",
        "wrong_field",
        "unsupported_fact",
        "duplicate_information",
        "overly_granular",
        "schema_mismatch",
        "source_fidelity_risk",
        "fabricated_uri",
        "needs_review",
    ]
    severity: Literal["minor", "major", "critical"] = "major"
    explanation: str
    suggested_target_path: str | None = None


class UnmappedFact(BaseModel):
    fact: str
    reason: str
    source_hint: str | None = None
    suggested_schema_extension: str | None = None


class CandidateQualityRating(BaseModel):
    """Per-candidate quality decision for a single field-level patch candidate."""

    field_path: str = Field(
        description="The top-level Dataset field this rating applies to.",
    )
    decision: PatchQualityDecision = Field(
        description="Whether to accept, revise, or reject this candidate.",
    )
    issues: list[PatchQualityIssue] = Field(
        default_factory=list,
        description="Issues found with this candidate.",
    )
    revised_patch: dict[str, Any] | None = Field(
        default=None,
        description=(
            "If decision='revise', a corrected merge-patch scoped to this "
            "field. Must be None for 'accept' or 'reject'."
        ),
    )
    unmapped_facts: list[UnmappedFact] = Field(
        default_factory=list,
        description="Facts from the source that could not be mapped to schema fields.",
    )


class PatchQualityReport(BaseModel):
    """Batch quality review of all field-level patch candidates from a chunk."""

    overall_decision: PatchQualityDecision = Field(
        description=(
            "Overall assessment: 'accept' if all candidates are acceptable, "
            "'revise' if at least one candidate needs revision, 'reject' if "
            "the batch is fundamentally flawed."
        ),
    )
    candidate_ratings: list[CandidateQualityRating] = Field(
        default_factory=list,
        description="Per-candidate quality decisions.",
    )
    summary: str = Field(
        description="Human-readable summary of the quality review.",
    )

    # Legacy fields kept for backward compatibility with existing callers.
    # These are populated from candidate_ratings on demand.
    @property
    def decision(self) -> PatchQualityDecision:
        return self.overall_decision

    @property
    def issues(self) -> list[PatchQualityIssue]:
        return [issue for r in self.candidate_ratings for issue in r.issues]

    @property
    def unmapped_facts(self) -> list[UnmappedFact]:
        return [f for r in self.candidate_ratings for f in r.unmapped_facts]


@dataclass
class PatchQualityDeps:
    initial_context: InitialContext
    current_draft: dict[str, Any]
    proposed_patch: dict[str, Any]
    candidate_draft: dict[str, Any]
    schema_validation_errors: list[str]
    chunk_batch: list[ContentChunk]
    document_file_path: str
    profile_manifest: ProfileManifest
    profile_json_schema: dict[str, Any]
    candidates: list[PatchCandidate] | None = None


PATCH_QUALITY_INSTRUCTIONS = """
You are a semantic quality-control agent for DCAT-style metadata extraction.

You receive:
1. the current metadata draft,
2. a proposed JSON merge patch (merged from all field-level candidates),
3. the candidate draft after applying that patch,
4. schema validation errors, if any,
5. the profile JSON Schema,
6. source chunk evidence,
7. the previously extracted InitialContext,
8. a list of field-level PatchCandidate objects, each with field_path, patch, confidence, reasoning, and source_evidence.

Your task is to review each candidate individually and produce a CandidateQualityRating for each one, plus an overall decision.

Important principles:

- The final draft is a curated metadata representation, not a raw extraction log.
- Do not allow broad text fields such as description, title, or keyword to become fallback containers for facts that belong in structured fields.
- Description fields should summarize the resource in natural language.
- Description fields must not accumulate low-level parameter lists, audit trails, file logs, command traces, internal software paths, or unrelated operational details.
- If detailed facts are supported by the source but the current schema has no suitable field, move them to unmapped_facts instead of forcing them into description.
- Prefer structured fields such as activities, entities, distributions, agentic entities, qualitative attributes, quantitative attributes, checksums, identifiers, file roles, dates, and defined terms when the schema supports them.
- Flag fabricated URIs: candidates that invent https://w3id.org/ or other namespace URIs not present in the current draft or InitialContext should be flagged with issue_type='fabricated_uri'. Stable local IDs and descriptive labels are acceptable.
- If a candidate is semantically useful but misplaces content, set decision='revise' and provide revised_patch scoped to that field.
- If a candidate is mostly unsupported, overly noisy, or cannot be safely mapped, set decision='reject'.
- If schema_validation_errors are present, do not set overall_decision='accept' unless all candidates resolve the issues.
- Never invent facts.
- Never invent schema fields.
- revised_patch must only use fields allowed by the provided profile schema.
- Return only the PatchQualityReport JSON object.
"""


def create_patch_quality_agent(
    *,
    model: Any,
    output_retries: int = DEFAULT_OUTPUT_RETRIES,
) -> Agent[PatchQualityDeps, PatchQualityReport]:
    agent = Agent(
        model,
        deps_type=PatchQualityDeps,
        output_type=prompted_json_output(
            PatchQualityReport,
            name="PatchQualityReport",
            description="Semantic quality review of field-level metadata patch candidates.",
        ),
        instructions=PATCH_QUALITY_INSTRUCTIONS,
        model_settings={
            "temperature": 0.0,
            "seed": 42,
        },
        output_retries=output_retries,
    )

    @agent.instructions
    def add_patch_quality_context(ctx: RunContext[PatchQualityDeps]) -> str:
        parts = [
            f"Profile identifier:\n{ctx.deps.profile_manifest.identifier}\n\n",
            f"Target class:\n{ctx.deps.profile_manifest.target_class}\n\n",
            "Profile JSON Schema:\n"
            f"{json.dumps(ctx.deps.profile_json_schema, indent=2, ensure_ascii=False)}\n\n",
            f"Document file path:\n{ctx.deps.document_file_path}\n\n",
            "InitialContext JSON:\n"
            f"{ctx.deps.initial_context.model_dump_json(indent=2)}\n\n",
            "Current draft JSON:\n"
            f"{json.dumps(ctx.deps.current_draft, indent=2, ensure_ascii=False)}\n\n",
            "Proposed patch JSON (merged from all candidates):\n"
            f"{json.dumps(ctx.deps.proposed_patch, indent=2, ensure_ascii=False)}\n\n",
            "Candidate draft after applying proposed patch:\n"
            f"{json.dumps(ctx.deps.candidate_draft, indent=2, ensure_ascii=False)}\n\n",
            "Schema validation errors:\n"
            f"{json.dumps(ctx.deps.schema_validation_errors, indent=2, ensure_ascii=False)}\n\n",
            "Source chunk batch:\n"
            f"{json.dumps([chunk.model_dump(exclude={'embedding'}) for chunk in ctx.deps.chunk_batch], indent=2, ensure_ascii=False)}\n\n",
        ]

        if ctx.deps.candidates:
            parts.append(
                "Field-level patch candidates:\n"
                f"{json.dumps([c.model_dump() for c in ctx.deps.candidates], indent=2, ensure_ascii=False)}\n\n"
            )

        parts.append("Review each candidate and produce a PatchQualityReport.")
        return "".join(parts)

    return agent


async def review_patch_semantic_quality(
    *,
    initial_context: InitialContext,
    current_draft: dict[str, Any],
    proposed_patch: dict[str, Any],
    candidate_draft: dict[str, Any],
    schema_validation_errors: list[str],
    chunk_batch: list[ContentChunk],
    document_file_path: str,
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
    model: Any,
    candidates: list[PatchCandidate] | None = None,
    on_token_usage: TokenUsageCallback | None = None,
) -> PatchQualityReport:
    agent = create_patch_quality_agent(model=model)

    deps = PatchQualityDeps(
        initial_context=initial_context,
        current_draft=current_draft,
        proposed_patch=proposed_patch,
        candidate_draft=candidate_draft,
        schema_validation_errors=schema_validation_errors,
        chunk_batch=chunk_batch,
        document_file_path=document_file_path,
        profile_manifest=profile_manifest,
        profile_json_schema=profile_json_schema,
        candidates=candidates,
    )

    result = await agent.run(
        "Review the field-level patch candidates for semantic quality "
        "and schema-appropriate field usage. Provide a per-candidate "
        "rating and an overall decision.",
        deps=deps,
    )
    if on_token_usage is not None:
        on_token_usage("patch_quality", result.usage, 1)
    return result.output


def validate_candidate_draft(
    *,
    candidate: dict[str, Any],
    profile_json_schema: dict[str, Any],
    profile_manifest: ProfileManifest,
) -> list[str]:
    result = validate_document_against_profile(
        document=candidate,
        json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
    )
    return [
        f"{issue.path}: {issue.message} (schema: {issue.schema_path})"
        for issue in result.errors
    ]
