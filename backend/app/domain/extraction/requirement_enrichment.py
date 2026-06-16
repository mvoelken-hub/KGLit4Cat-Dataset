from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.extraction.evidence_context import EvidenceCandidate, RoutedEvidenceContext


RequirementStatus = Literal["fulfilled", "partial", "missing", "not_applicable"]


class DcatRequirement(BaseModel):
    requirement_id: str
    label: str
    description: str
    weight: float = Field(default=1.0, gt=0)
    target_paths: list[str] = Field(default_factory=list)
    expected_target_class: str | None = None
    evidence_hints: list[str] = Field(default_factory=list)
    allow_not_applicable: bool = False


class RequirementAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_id: str
    status: RequirementStatus
    quality: float = Field(default=0.0, ge=0.0, le=1.0)
    applicable: bool = True
    rationale: str = ""
    target_paths: list[str] = Field(default_factory=list)
    evidence_search_hints: list[str] = Field(default_factory=list)
    expected_target_class: str | None = None


class RequirementEvaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessments: list[RequirementAssessment] = Field(default_factory=list)


class RequirementEvidenceItem(BaseModel):
    candidate_id: str
    category: str
    claim: str
    evidence_text: str
    file_path: str = ""
    start_idx: int = 0
    end_idx: int = 0


class RequirementPatchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    should_patch: bool = True
    target_path: str = ""
    target_class: str = ""
    instance: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class RequirementPatchAttempt(BaseModel):
    attempted: bool = False
    status: Literal["not_attempted", "applied", "failed", "rolled_back"] = "not_attempted"
    target_path: str | None = None
    target_class: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    reason: str = ""


class RequirementReportItem(BaseModel):
    requirement_id: str
    label: str
    weight: float
    status: RequirementStatus
    applicable: bool
    quality: float
    weighted_score: float
    rationale: str = ""
    target_paths: list[str] = Field(default_factory=list)
    evidence_search_hints: list[str] = Field(default_factory=list)
    selected_evidence: list[RequirementEvidenceItem] = Field(default_factory=list)
    context_window: list[RequirementEvidenceItem] = Field(default_factory=list)
    patch: RequirementPatchAttempt = Field(default_factory=RequirementPatchAttempt)


class RequirementReport(BaseModel):
    schema_valid: bool = True
    metadata_completeness_score: float = 0.0
    applicable_weight: float = 0.0
    earned_weight: float = 0.0
    requirements: list[RequirementReportItem] = Field(default_factory=list)


DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS: tuple[DcatRequirement, ...] = (
    DcatRequirement(
        requirement_id="dataset_identity",
        label="Dataset identity",
        description="Dataset has non-placeholder id, title, and meaningful description.",
        weight=1.0,
        target_paths=["/id", "/title", "/description"],
        evidence_hints=["dataset title", "dataset description", "dataset identifier"],
    ),
    DcatRequirement(
        requirement_id="dataset_distributions",
        label="Dataset distributions",
        description="Dataset exposes relevant distributions with access resources and descriptions.",
        weight=1.0,
        target_paths=["/dataset_distribution/-"],
        expected_target_class="Distribution",
        evidence_hints=["file", "format", "distribution", "download", "access", "resource"],
    ),
    DcatRequirement(
        requirement_id="dataset_generation",
        label="Dataset generation activity",
        description="Dataset has one or more data-generating activities that produced source content.",
        weight=1.25,
        target_paths=["/was_generated_by/-"],
        expected_target_class="DataGeneratingActivity",
        evidence_hints=["acquisition", "measurement", "processing", "generated", "experiment"],
    ),
    DcatRequirement(
        requirement_id="about_entity_or_activity",
        label="Dataset aboutness",
        description="Dataset states evaluated entity or evaluated activity it is about.",
        weight=1.25,
        target_paths=["/is_about_entity/-", "/is_about_activity/-"],
        expected_target_class="EvaluatedEntity",
        evidence_hints=["sample", "entity", "spectrum", "evaluated", "measured", "activity"],
    ),
    DcatRequirement(
        requirement_id="generation_activity_detail",
        label="Generation activity detail",
        description="Generation activity has meaningful title, description, and type.",
        weight=1.0,
        target_paths=["/was_generated_by/-"],
        expected_target_class="DataGeneratingActivity",
        evidence_hints=["acquisition", "processing", "experiment type", "method"],
    ),
    DcatRequirement(
        requirement_id="agents_instruments_software",
        label="Agents, instruments, and software",
        description="Generation activity names instruments, software, people, or organizations via carried_out_by.",
        weight=1.25,
        target_paths=["/was_generated_by/0/carried_out_by/-"],
        expected_target_class="AgenticEntity",
        evidence_hints=["instrument", "spectrometer", "software", "TopSpin", "Bruker", "owner", "origin"],
    ),
    DcatRequirement(
        requirement_id="method_plan",
        label="Method or plan",
        description="Generation activity links to method, protocol, pulse sequence, plan, or procedure.",
        weight=1.0,
        target_paths=["/was_generated_by/0/realized_plan"],
        expected_target_class="Plan",
        evidence_hints=["method", "protocol", "plan", "procedure", "pulse sequence", "program"],
    ),
    DcatRequirement(
        requirement_id="semantic_attributes",
        label="Quantitative and qualitative attributes",
        description="Evaluated entities, evaluated activities, generation activities, or instruments include key quantitative or qualitative attributes.",
        weight=1.5,
        target_paths=["/was_generated_by/0/has_quantitative_attribute/-", "/is_about_entity/0/has_quantitative_attribute/-"],
        expected_target_class="QuantitativeAttribute",
        evidence_hints=["temperature", "frequency", "width", "points", "unit", "parameter", "value"],
    ),
    DcatRequirement(
        requirement_id="source_trace",
        label="Source evidence trace",
        description="Fulfilled semantic requirements can be traced to source evidence snippets.",
        weight=1.0,
        target_paths=[],
        evidence_hints=["evidence", "source", "file path"],
    ),
)


REQUIREMENT_EVALUATOR_SYSTEM_PROMPT = """
You evaluate DCAT-AP+ scientific metadata completeness for one Dataset draft.
Return only JSON matching the supplied schema.
Return exactly one assessment for every supplied requirement_id. Do not omit requirements.
Assess applicability and quality, not JSON Schema validity.
Use statuses: fulfilled, partial, missing, not_applicable.
quality must be 1 for fulfilled, 0.5 for partial, 0 for missing/not_applicable.
For missing/partial requirements, provide evidence_search_hints and target_paths.
Prefer not_applicable only when source package type makes requirement irrelevant, not when evidence is absent.
"""


REQUIREMENT_PATCH_SYSTEM_PROMPT = """
You create one schema-constrained DCAT-AP+ patch object for one missing requirement.
Return only JSON matching the supplied schema.
Use only selected evidence. Do not invent facts.
Set should_patch=false if evidence is insufficient.
Return exactly one target_path from allowed_target_paths and one object/value instance for that path.
"""


def build_requirement_evaluation_prompt(
    *,
    document: dict[str, Any],
    requirements: list[DcatRequirement],
) -> str:
    payload = {
        "draft_dataset": document,
        "requirements": [req.model_dump(mode="json") for req in requirements],
    }
    return (
        "Evaluate draft_dataset against requirements. "
        "Do not require FAIR/access/legal fields outside supplied requirements.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_requirement_patch_prompt(
    *,
    requirement: DcatRequirement,
    assessment: RequirementAssessment,
    selected_evidence: list[RequirementEvidenceItem],
    context_window: list[RequirementEvidenceItem],
    draft_excerpt: Any,
    schema_branch: dict[str, Any],
) -> str:
    payload = {
        "requirement": requirement.model_dump(mode="json"),
        "assessment": assessment.model_dump(mode="json"),
        "allowed_target_paths": assessment.target_paths or requirement.target_paths,
        "expected_target_class": assessment.expected_target_class or requirement.expected_target_class,
        "draft_excerpt": draft_excerpt,
        "schema_branch": schema_branch,
        "selected_evidence": [item.model_dump(mode="json") for item in selected_evidence],
        "context_window": [item.model_dump(mode="json") for item in context_window],
    }
    return "Create one grounded patch for this missing/partial requirement.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def score_requirement_report(items: list[RequirementReportItem], *, schema_valid: bool = True) -> RequirementReport:
    _score_source_trace(items)
    applicable = [item for item in items if item.applicable and item.status != "not_applicable"]
    applicable_weight = sum(item.weight for item in applicable)
    earned_weight = sum(max(0.0, min(1.0, item.quality)) * item.weight for item in applicable)
    score = earned_weight / applicable_weight if applicable_weight else 0.0
    for item in items:
        item.weighted_score = 0.0 if item.status == "not_applicable" else item.weight * item.quality
    return RequirementReport(
        schema_valid=schema_valid,
        metadata_completeness_score=score,
        applicable_weight=applicable_weight,
        earned_weight=earned_weight,
        requirements=items,
    )


def normalized_requirement_evaluation(
    *,
    requirements: list[DcatRequirement],
    evaluation: RequirementEvaluation,
) -> RequirementEvaluation:
    by_id: dict[str, RequirementAssessment] = {}
    for assessment in evaluation.assessments:
        if assessment.requirement_id in by_id:
            continue
        by_id[assessment.requirement_id] = _normalized_assessment(assessment)
    normalized: list[RequirementAssessment] = []
    for requirement in requirements:
        assessment = by_id.get(requirement.requirement_id)
        if assessment is None:
            assessment = RequirementAssessment(
                requirement_id=requirement.requirement_id,
                status="missing",
                quality=0.0,
                applicable=True,
                rationale="Requirement evaluator did not return this requirement.",
                target_paths=requirement.target_paths,
                evidence_search_hints=requirement.evidence_hints,
                expected_target_class=requirement.expected_target_class,
            )
        elif assessment.status == "not_applicable" and not requirement.allow_not_applicable:
            assessment = RequirementAssessment(
                requirement_id=requirement.requirement_id,
                status="missing",
                quality=0.0,
                applicable=True,
                rationale=(
                    "Requirement evaluator marked this core DCAT-AP+ scientific requirement "
                    "not_applicable; treated as missing because v1 requirements are applicable by default."
                ),
                target_paths=assessment.target_paths or requirement.target_paths,
                evidence_search_hints=assessment.evidence_search_hints or requirement.evidence_hints,
                expected_target_class=assessment.expected_target_class or requirement.expected_target_class,
            )
        if not assessment.target_paths:
            assessment.target_paths = list(requirement.target_paths)
        if not assessment.evidence_search_hints:
            assessment.evidence_search_hints = list(requirement.evidence_hints)
        if not assessment.expected_target_class:
            assessment.expected_target_class = requirement.expected_target_class
        normalized.append(assessment)
    return RequirementEvaluation(assessments=normalized)


def report_items_from_evaluation(
    *,
    requirements: list[DcatRequirement],
    evaluation: RequirementEvaluation,
) -> list[RequirementReportItem]:
    evaluation = normalized_requirement_evaluation(
        requirements=requirements,
        evaluation=evaluation,
    )
    by_id = {item.requirement_id: item for item in evaluation.assessments}
    items: list[RequirementReportItem] = []
    for req in requirements:
        assessment = by_id.get(req.requirement_id)
        if assessment is None:
            assessment = RequirementAssessment(
                requirement_id=req.requirement_id,
                status="missing",
                quality=0.0,
                applicable=True,
                rationale="Requirement evaluator did not return this requirement.",
                target_paths=req.target_paths,
                evidence_search_hints=req.evidence_hints,
                expected_target_class=req.expected_target_class,
            )
        status: RequirementStatus = assessment.status
        applicable = bool(assessment.applicable) and status != "not_applicable"
        quality = 0.0 if status == "not_applicable" else max(0.0, min(1.0, assessment.quality))
        items.append(
            RequirementReportItem(
                requirement_id=req.requirement_id,
                label=req.label,
                weight=req.weight,
                status=status,
                applicable=applicable,
                quality=quality,
                weighted_score=req.weight * quality if applicable else 0.0,
                rationale=assessment.rationale,
                target_paths=assessment.target_paths or req.target_paths,
                evidence_search_hints=assessment.evidence_search_hints or req.evidence_hints,
            )
        )
    return items


def select_requirement_evidence_packet(
    *,
    requirement: DcatRequirement,
    assessment: RequirementAssessment | RequirementReportItem,
    evidence_context: RoutedEvidenceContext,
    max_selected: int = 5,
    max_context: int = 12,
) -> tuple[list[RequirementEvidenceItem], list[RequirementEvidenceItem]]:
    hints = [
        token
        for hint in ((assessment.evidence_search_hints or requirement.evidence_hints))
        for token in _tokenize(hint)
    ]
    candidates = list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence)
    scored: list[tuple[int, EvidenceCandidate]] = []
    for candidate in candidates:
        text = _candidate_search_text(candidate)
        score = sum(1 for token in hints if token in text)
        if requirement.expected_target_class:
            score += _class_hint_score(requirement.expected_target_class, candidate)
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].file_path, item[1].start_idx, item[1].candidate_id))
    selected_candidates = _dedupe_candidates([candidate for _, candidate in scored[:max_selected]])
    context_candidates: list[EvidenceCandidate] = []
    for selected in selected_candidates:
        context_candidates.extend(
            candidate
            for candidate in candidates
            if candidate.file_path == selected.file_path
            and abs(candidate.start_idx - selected.start_idx) <= 500
        )
    context_candidates = _dedupe_candidates(context_candidates)[:max_context]
    return (
        [_evidence_item(candidate) for candidate in selected_candidates],
        [_evidence_item(candidate) for candidate in context_candidates],
    )


def assessment_for_report_item(item: RequirementReportItem) -> RequirementAssessment:
    return RequirementAssessment(
        requirement_id=item.requirement_id,
        status=item.status,
        quality=item.quality,
        applicable=item.applicable,
        rationale=item.rationale,
        target_paths=item.target_paths,
        evidence_search_hints=item.evidence_search_hints,
    )


def merge_requirement_assessment(
    *,
    item: RequirementReportItem,
    assessment: RequirementAssessment,
    requirement: DcatRequirement,
) -> RequirementReportItem:
    assessment = _normalized_assessment(assessment)
    status: RequirementStatus = assessment.status
    applicable = bool(assessment.applicable) and status != "not_applicable"
    quality = 0.0 if status == "not_applicable" else max(0.0, min(1.0, assessment.quality))
    item.status = status
    item.applicable = applicable
    item.quality = quality
    item.weighted_score = requirement.weight * quality if applicable else 0.0
    item.rationale = assessment.rationale
    item.target_paths = assessment.target_paths or requirement.target_paths
    item.evidence_search_hints = assessment.evidence_search_hints or requirement.evidence_hints
    return item


def _normalized_assessment(assessment: RequirementAssessment) -> RequirementAssessment:
    status = assessment.status
    if status == "fulfilled":
        quality = 1.0
        applicable = True
    elif status == "partial":
        quality = 0.5
        applicable = bool(assessment.applicable)
    elif status == "not_applicable":
        quality = 0.0
        applicable = False
    else:
        quality = 0.0
        applicable = bool(assessment.applicable)
    assessment.status = status
    assessment.quality = quality
    assessment.applicable = applicable
    return assessment


def _score_source_trace(items: list[RequirementReportItem]) -> None:
    trace = next((item for item in items if item.requirement_id == "source_trace"), None)
    if trace is None or not trace.applicable:
        return
    traceable = [
        item
        for item in items
        if item.requirement_id != "source_trace"
        and item.applicable
        and item.status in {"fulfilled", "partial"}
        and item.selected_evidence
    ]
    if traceable:
        trace.status = "fulfilled"
        trace.quality = 1.0
        trace.rationale = "Fulfilled semantic requirements include selected evidence packets in requirement_report.json."
    else:
        trace.status = "missing"
        trace.quality = 0.0
        trace.rationale = trace.rationale or "No fulfilled semantic requirement has selected evidence yet."


def _candidate_search_text(candidate: EvidenceCandidate) -> str:
    return " ".join(
        [
            candidate.candidate_id,
            candidate.category,
            candidate.claim,
            candidate.evidence_text,
            candidate.file_path,
        ]
    ).lower()


def _class_hint_score(target_class: str, candidate: EvidenceCandidate) -> int:
    category = candidate.category
    text = _candidate_search_text(candidate)
    if target_class == "DataGeneratingActivity" and (
        category == "method_signal" or any(term in text for term in ("acquisition", "processing", "experiment"))
    ):
        return 2
    if target_class == "AgenticEntity" and (
        category == "agent_signal" or any(term in text for term in ("instrument", "software", "bruker", "topspin"))
    ):
        return 2
    if target_class == "EvaluatedEntity" and category in {"entity_signal", "measurement_signal"}:
        return 2
    if target_class == "Distribution" and (
        category == "resource_signal" or any(term in text for term in ("file", "format", "download"))
    ):
        return 2
    if target_class == "QuantitativeAttribute" and category == "measurement_signal":
        return 2
    return 0


def _tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 3]


def _dedupe_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    seen: set[str] = set()
    result: list[EvidenceCandidate] = []
    for candidate in candidates:
        key = f"{candidate.candidate_id}|{candidate.file_path}|{candidate.start_idx}|{candidate.end_idx}"
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _evidence_item(candidate: EvidenceCandidate) -> RequirementEvidenceItem:
    return RequirementEvidenceItem(
        candidate_id=candidate.candidate_id,
        category=str(candidate.category),
        claim=candidate.claim,
        evidence_text=candidate.evidence_text,
        file_path=candidate.file_path,
        start_idx=candidate.start_idx,
        end_idx=candidate.end_idx,
    )
