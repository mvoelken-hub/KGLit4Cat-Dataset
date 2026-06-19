from __future__ import annotations

import json
from hashlib import sha1
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
    allowed_categories: list[str] = Field(default_factory=list)


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
    evidence_id: str = ""
    candidate_id: str
    category: str
    claim: str
    evidence_text: str
    file_path: str = ""
    start_idx: int = 0
    end_idx: int = 0
    evidence_match_score: float = 0.0


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


class CoverageFieldReport(BaseModel):
    path: str
    score: float
    present: bool
    kind: str = "field"
    children: list["CoverageFieldReport"] = Field(default_factory=list)


class CoverageReport(BaseModel):
    score: float = 0.0
    filled_fields: int = 0
    total_fields: int = 0
    fields: list[CoverageFieldReport] = Field(default_factory=list)


class SourceTraceReport(BaseModel):
    score: float = 0.0
    used_evidence_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)


class RequirementReport(BaseModel):
    schema_valid: bool = True
    coverage_score: float = 0.0
    semantic_requirements_score: float = 0.0
    source_trace_score: float = 0.0
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    semantic_requirements: list[RequirementReportItem] = Field(default_factory=list)
    source_trace: SourceTraceReport = Field(default_factory=SourceTraceReport)
    coverage_patches: list[RequirementReportItem] = Field(default_factory=list)


def compute_coverage_report(document: dict[str, Any], schema: dict[str, Any]) -> CoverageReport:
    root_schema = _resolve_schema_node(schema, schema)
    fields = _coverage_fields_for_object(
        document if isinstance(document, dict) else {},
        root_schema,
        schema,
        path="",
        seen=set(),
    )
    filled_fields, total_fields = _coverage_field_counts(fields)
    return CoverageReport(score=float(filled_fields), filled_fields=filled_fields, total_fields=total_fields, fields=fields)


def compute_source_trace_report(
    evidence_context: RoutedEvidenceContext,
    used_evidence_ids: list[str],
) -> SourceTraceReport:
    by_id: dict[str, EvidenceCandidate] = {}
    for candidate in list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence):
        evidence_id = stable_evidence_id(candidate)
        by_id[evidence_id] = candidate
        explicit_id = getattr(candidate, "evidence_id", "")
        if explicit_id:
            by_id[str(explicit_id)] = candidate
    unique_ids = list(dict.fromkeys(item for item in used_evidence_ids if item))
    scores = [
        float(getattr(by_id[evidence_id], "evidence_match_score", 0.0) or 0.0)
        for evidence_id in unique_ids
        if evidence_id in by_id
    ]
    score = sum(scores) / len(scores) if scores else 0.0
    return SourceTraceReport(score=score, used_evidence_count=len(scores), evidence_ids=unique_ids)


def _coverage_fields_for_object(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    path: str,
    seen: set[str],
) -> list[CoverageFieldReport]:
    schema = _resolve_schema_node(schema, root_schema)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return []
    fields: list[CoverageFieldReport] = []
    for name, child_schema in properties.items():
        child_path = f"{path}/{_escape_json_pointer(name)}"
        child_value = value.get(name) if isinstance(value, dict) else None
        fields.append(
            _coverage_field(
                child_value,
                child_schema if isinstance(child_schema, dict) else {},
                root_schema,
                path=child_path,
                seen=seen,
            )
        )
    return fields


def _coverage_field(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    path: str,
    seen: set[str],
) -> CoverageFieldReport:
    schema = _resolve_schema_node(schema, root_schema)
    schema_key = json.dumps(schema, sort_keys=True, default=str)[:500]
    if schema_key in seen:
        return CoverageFieldReport(path=path, score=0.0 if _is_missing_value(value) else 1.0, present=not _is_missing_value(value))
    next_seen = {*seen, schema_key}
    schema_type = schema.get("type")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if schema_type == "array" or "items" in schema:
        if not isinstance(value, list) or not value:
            return CoverageFieldReport(path=path, score=0.0, present=False, kind="array")
        item_schema = _resolve_schema_node(schema.get("items", {}), root_schema)
        item_properties = item_schema.get("properties") if isinstance(item_schema, dict) else None
        if not isinstance(item_properties, dict):
            return CoverageFieldReport(path=path, score=1.0, present=True, kind="array")
        children = [
            CoverageFieldReport(
                path=f"{path}/{index}",
                score=_average([child.score for child in item_children]),
                present=not _is_missing_value(item),
                kind="object",
                children=item_children,
            )
            for index, item in enumerate(value)
            for item_children in [
                _coverage_fields_for_object(item, item_schema, root_schema, path=f"{path}/{index}", seen=next_seen)
            ]
        ]
        return CoverageFieldReport(path=path, score=_average([child.score for child in children]), present=True, kind="array", children=children)
    if isinstance(properties, dict):
        if not isinstance(value, dict):
            children = _coverage_fields_for_object({}, schema, root_schema, path=path, seen=next_seen)
            return CoverageFieldReport(path=path, score=0.0, present=False, kind="object", children=children)
        children = _coverage_fields_for_object(value, schema, root_schema, path=path, seen=next_seen)
        return CoverageFieldReport(path=path, score=_average([child.score for child in children]), present=True, kind="object", children=children)
    present = not _is_missing_value(value)
    return CoverageFieldReport(path=path, score=1.0 if present else 0.0, present=present)


def _resolve_schema_node(schema: Any, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        value: Any = root_schema
        for token in ref[2:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict):
                return schema
            value = value.get(token)
        return _resolve_schema_node(value, root_schema)
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        choices = [item for item in schema["anyOf"] if isinstance(item, dict) and item.get("type") != "null"]
        return _resolve_schema_node(choices[0], root_schema) if choices else schema
    if "oneOf" in schema and isinstance(schema["oneOf"], list):
        choices = [item for item in schema["oneOf"] if isinstance(item, dict) and item.get("type") != "null"]
        return _resolve_schema_node(choices[0], root_schema) if choices else schema
    if "allOf" in schema and isinstance(schema["allOf"], list):
        merged: dict[str, Any] = {}
        for item in schema["allOf"]:
            resolved = _resolve_schema_node(item, root_schema)
            if isinstance(resolved, dict):
                merged.update(resolved)
        return merged or schema
    return schema


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _coverage_field_counts(fields: list[CoverageFieldReport]) -> tuple[int, int]:
    filled = 0
    total = 0
    for field in fields:
        total += 1
        if field.present:
            filled += 1
        child_filled, child_total = _coverage_field_counts(field.children)
        filled += child_filled
        total += child_total
    return filled, total


def _escape_json_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


DCAT_AP_PLUS_COVERAGE_REQUIREMENTS: tuple[DcatRequirement, ...] = (
    DcatRequirement(
        requirement_id="dataset_identity",
        label="Dataset identity",
        description="Dataset has non-placeholder id, title, and meaningful description.",
        weight=1.0,
        target_paths=["/id", "/title", "/description"],
        evidence_hints=["dataset title", "dataset description", "dataset identifier"],
        allowed_categories=["resource_signal", "surrounding_signal", "other"],
    ),
    DcatRequirement(
        requirement_id="dataset_generation",
        label="Dataset generation activity",
        description="Dataset has one or more data-generating activities that produced source content.",
        weight=1.25,
        target_paths=["/was_generated_by/-"],
        expected_target_class="DataGeneratingActivity",
        evidence_hints=["acquisition", "measurement", "processing", "generated", "experiment"],
        allowed_categories=["activity_signal"],
    ),
    DcatRequirement(
        requirement_id="about_entity_or_activity",
        label="Dataset aboutness",
        description="Dataset states evaluated entity or evaluated activity it is about.",
        weight=1.25,
        target_paths=["/is_about_entity/-", "/is_about_activity/-"],
        expected_target_class="EvaluatedEntity",
        evidence_hints=["sample", "entity", "spectrum", "evaluated", "measured", "activity"],
        allowed_categories=["activity_signal", "resource_signal", "surrounding_signal"],
    ),
    DcatRequirement(
        requirement_id="generation_activity_detail",
        label="Generation activity detail",
        description="Generation activity has meaningful title, description, and type.",
        weight=1.0,
        target_paths=["/was_generated_by/-"],
        expected_target_class="DataGeneratingActivity",
        evidence_hints=["acquisition", "processing", "experiment type", "method"],
        allowed_categories=["activity_signal"],
    ),
    DcatRequirement(
        requirement_id="technical_agents",
        label="Technical agents",
        description="Generation activity names instruments, software, or devices via carried_out_by.",
        weight=1.25,
        target_paths=["/was_generated_by/0/carried_out_by/-"],
        expected_target_class="AgenticEntity",
        evidence_hints=["instrument", "software", "device"],
        allowed_categories=["agent_signal"],
    ),
    DcatRequirement(
        requirement_id="method_plan",
        label="Method or plan",
        description="Generation activity links to method, protocol, plan, or procedure.",
        weight=1.0,
        target_paths=["/was_generated_by/0/realized_plan"],
        expected_target_class="Plan",
        evidence_hints=["method", "protocol", "plan", "procedure", "program"],
        allowed_categories=["method_signal"],
    ),
    DcatRequirement(
        requirement_id="instrument_settings_attributes",
        label="Instrument settings attributes",
        description="Generation activity or instruments include key setting/configuration attributes.",
        weight=1.5,
        target_paths=["/was_generated_by/0/has_quantitative_attribute/-", "/is_about_entity/0/has_quantitative_attribute/-"],
        expected_target_class="QuantitativeAttribute",
        evidence_hints=["temperature", "frequency", "width", "unit", "parameter", "threshold", "setting"],
        allowed_categories=["instrument_signal"],
    ),
)


DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS: tuple[DcatRequirement, ...] = (
    DcatRequirement(
        requirement_id="dataset_identity_semantics",
        label="Dataset identity semantics",
        description="Title and description identify the dataset rather than the extraction/projection process.",
        weight=1.0,
        target_paths=["/title", "/description"],
        evidence_hints=["dataset title", "dataset description", "identifier"],
    ),
    DcatRequirement(
        requirement_id="dataset_generation_semantics",
        label="Dataset generation semantics",
        description="was_generated_by describes a real data-generating activity.",
        weight=1.25,
        target_paths=["/was_generated_by"],
        evidence_hints=["activity", "acquisition", "processing", "generated"],
        allowed_categories=["activity_signal"],
    ),
    DcatRequirement(
        requirement_id="aboutness_semantics",
        label="Aboutness semantics",
        description="Aboutness identifies a concrete evaluated entity/activity; generic inferred subject is partial; file names are not entities.",
        weight=1.25,
        target_paths=["/is_about_entity", "/is_about_activity"],
        evidence_hints=["sample", "entity", "spectrum", "evaluated", "activity"],
        allowed_categories=["activity_signal", "resource_signal", "surrounding_signal"],
    ),
    DcatRequirement(
        requirement_id="technical_agents_semantics",
        label="Technical agents semantics",
        description="carried_out_by contains instruments, software, or devices rather than people or provenance metadata.",
        weight=1.25,
        target_paths=["/was_generated_by/0/carried_out_by"],
        evidence_hints=["instrument", "software", "device"],
        allowed_categories=["agent_signal"],
    ),
    DcatRequirement(
        requirement_id="method_plan_semantics",
        label="Method plan semantics",
        description="realized_plan is grounded in explicit method/procedure/plan evidence; activity-only inference is partial at most.",
        weight=1.0,
        target_paths=["/was_generated_by/0/realized_plan"],
        evidence_hints=["method", "protocol", "plan", "procedure", "program"],
        allowed_categories=["method_signal"],
    ),
    DcatRequirement(
        requirement_id="instrument_settings_semantics",
        label="Instrument settings semantics",
        description="Instrument/configuration settings are represented as suitable attributes.",
        weight=1.5,
        target_paths=["/was_generated_by/0/has_quantitative_attribute", "/is_about_entity/0/has_quantitative_attribute"],
        evidence_hints=["temperature", "frequency", "width", "unit", "parameter", "threshold", "setting"],
        allowed_categories=["instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="provenance_context_semantics",
        label="Provenance context semantics",
        description="Dates, people, labs, teams, and origins from surrounding evidence are placed in suitable generic provenance/context fields when present.",
        weight=1.0,
        target_paths=["/creator", "/modification_date"],
        evidence_hints=["date", "creator", "owner", "origin", "laboratory", "team"],
        allowed_categories=["surrounding_signal"],
    ),
)


DCAT_AP_PLUS_SCIENTIFIC_REQUIREMENTS = DCAT_AP_PLUS_COVERAGE_REQUIREMENTS


REQUIREMENT_EVALUATOR_SYSTEM_PROMPT = """
You evaluate one DCAT-AP+ scientific semantic requirement for one Dataset draft slice.
Return only JSON matching the supplied schema.
Return exactly one assessment for every supplied requirement_id. Do not omit requirements.
Assess semantic adequacy, not JSON Schema validity or field coverage.
Use statuses: fulfilled, partial, missing, not_applicable.
quality must be 1 for fulfilled, 0.5 for partial, 0 for missing/not_applicable.
Prefer not_applicable only when the supplied evidence categories make the semantic requirement irrelevant.
For aboutness, file names are not evaluated entities. For method plans, explicit method_signal evidence is required for fulfilled.
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
    selected_evidence: list[RequirementEvidenceItem] | None = None,
    context_window: list[RequirementEvidenceItem] | None = None,
) -> str:
    payload = {
        "draft_excerpt": document,
        "requirements": [req.model_dump(mode="json") for req in requirements],
        "selected_evidence": [item.model_dump(mode="json") for item in selected_evidence or []],
        "context_window": [item.model_dump(mode="json") for item in context_window or []],
        "category_meanings": {
            "resource_signal": "files, distributions, formats, access paths, and resource-scoped notes",
            "method_signal": "explicit realized plans, protocols, methods, procedures",
            "activity_signal": "data-generating activities and other activities",
            "agent_signal": "software, devices, instruments, machines, services, executable systems",
            "instrument_signal": "acquisition, instrument, processing, calibration, unit, threshold, and configuration settings",
            "surrounding_signal": "dates, labs, teams, people, organizations, ownership, origin, authorship, provenance context",
            "measurement_signal": "primary/raw data values; downstream inert",
        },
    }
    return (
        "Evaluate the draft excerpt against the supplied semantic requirement. "
        "Use only supplied evidence and category meanings.\n\n"
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


def score_requirement_items(items: list[RequirementReportItem]) -> float:
    applicable = [item for item in items if item.applicable and item.status != "not_applicable"]
    total_weight = sum(item.weight for item in applicable)
    scored_weight = sum(max(0.0, min(1.0, item.quality)) * item.weight for item in applicable)
    score = scored_weight / total_weight if total_weight else 0.0
    for item in items:
        item.weighted_score = 0.0 if item.status == "not_applicable" else item.weight * item.quality
    return score


def build_requirement_report(
    *,
    schema_valid: bool,
    coverage: CoverageReport,
    semantic_requirements: list[RequirementReportItem],
    source_trace: SourceTraceReport,
    coverage_patches: list[RequirementReportItem],
) -> RequirementReport:
    semantic_score = score_requirement_items(semantic_requirements)
    return RequirementReport(
        schema_valid=schema_valid,
        coverage_score=coverage.score,
        semantic_requirements_score=semantic_score,
        source_trace_score=source_trace.score,
        coverage=coverage,
        semantic_requirements=semantic_requirements,
        source_trace=source_trace,
        coverage_patches=coverage_patches,
    )


def score_requirement_report(items: list[RequirementReportItem], *, schema_valid: bool = True) -> RequirementReport:
    semantic_score = score_requirement_items(items)
    return RequirementReport(
        schema_valid=schema_valid,
        semantic_requirements_score=semantic_score,
        semantic_requirements=items,
        coverage_patches=items,
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
    candidates = [
        candidate
        for candidate in list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence)
        if candidate.category != "measurement_signal"
        and (not requirement.allowed_categories or str(candidate.category) in requirement.allowed_categories)
    ]
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
        category == "activity_signal" or any(term in text for term in ("acquisition", "processing", "experiment"))
    ):
        return 2
    if target_class == "AgenticEntity" and (
        category in {"agent_signal", "instrument_signal"} or any(term in text for term in ("instrument", "software", "device", "equipment", "sensor"))
    ):
        return 2
    if target_class == "EvaluatedEntity" and category == "activity_signal":
        return 2
    if target_class == "Distribution" and (
        category == "resource_signal" or any(term in text for term in ("file", "format", "download"))
    ):
        return 2
    if target_class == "Plan" and category == "method_signal":
        return 2
    if target_class == "QuantitativeAttribute" and category == "instrument_signal":
        score = 2
        if any(term in text for term in ("frequency", "temperature", "width", "count", "threshold", "unit", "setting", "parameter")):
            score += 1
        if any(term in text for term in ("formula", "subrange", "range of", "0..", "rel ")):
            score -= 1
        return score
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
        evidence_id=stable_evidence_id(candidate),
        candidate_id=candidate.candidate_id,
        category=str(candidate.category),
        claim=candidate.claim,
        evidence_text=candidate.evidence_text,
        file_path=candidate.file_path,
        start_idx=candidate.start_idx,
        end_idx=candidate.end_idx,
        evidence_match_score=float(getattr(candidate, "evidence_match_score", 0.0) or 0.0),
    )


def stable_evidence_id(candidate: EvidenceCandidate, *, run_id: str = "") -> str:
    payload = "|".join(
        [
            run_id,
            candidate.file_path,
            str(candidate.start_idx),
            str(candidate.end_idx),
            str(candidate.category),
            candidate.candidate_id,
            candidate.claim,
            candidate.evidence_text,
        ]
    )
    return f"ev:{sha1(payload.encode('utf-8')).hexdigest()[:16]}"
