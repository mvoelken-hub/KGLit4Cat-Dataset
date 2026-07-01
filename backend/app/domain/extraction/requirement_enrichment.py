from __future__ import annotations

import json
from hashlib import sha1
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.extraction.evidence_context import EvidenceCandidate, RoutedEvidenceContext


RequirementStatus = Literal["fulfilled", "partial", "missing", "unanswered", "unresolved", "not_applicable"]
RequirementIterationKind = Literal["coverage", "semantic_diagnosis", "semantic_reconstruction"]
SemanticDefectType = Literal[
    "duplicate_attribute",
    "wrong_parent",
    "bad_range",
    "bad_label",
    "missing_plan",
    "bad_identity",
    "bad_agent",
    "bad_aboutness",
    "bad_provenance",
    "no_defect",
]
SemanticRecommendedAction = Literal["merge", "remove", "move", "replace", "append", "no_action"]


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
    role: str = ""
    claim: str
    evidence_text: str
    source_context: str = ""
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
    status: Literal["not_attempted", "applied", "failed", "rolled_back", "unresolved"] = "not_attempted"
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
    iteration_kind: RequirementIterationKind = "coverage"
    defect_type: str = ""
    diagnosed_defects_count: int = 0
    compiled_actions_count: int = 0
    synthesis_calls_count: int = 0
    diagnosed_defects: list[dict[str, Any]] = Field(default_factory=list)
    compiled_actions: list[dict[str, Any]] = Field(default_factory=list)


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


class SemanticReconstructionRecord(BaseModel):
    requirement_id: str
    status: Literal["applied", "skipped", "failed", "rolled_back", "unresolved"] = "skipped"
    target_paths: list[str] = Field(default_factory=list)
    changed_paths: list[str] = Field(default_factory=list)
    reason: str = ""
    validation_errors: list[str] = Field(default_factory=list)
    applied_actions_count: int = 0
    rejected_actions_count: int = 0
    rejected_reasons: list[str] = Field(default_factory=list)
    iteration_kind: RequirementIterationKind = "semantic_reconstruction"
    defect_type: str = ""
    diagnosed_defects_count: int = 0
    compiled_actions_count: int = 0
    synthesis_calls_count: int = 0
    diagnosed_defects: list[dict[str, Any]] = Field(default_factory=list)
    compiled_actions: list[dict[str, Any]] = Field(default_factory=list)


class SemanticReconstructionDefect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defect_type: SemanticDefectType
    target_path: str
    entry_indices: list[int] = Field(default_factory=list)
    recommended_action: SemanticRecommendedAction = "no_action"
    needs_synthesis: bool = False
    reason: str = ""


class SemanticReconstructionDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defects: list[SemanticReconstructionDefect] = Field(default_factory=list)
    reason: str = ""


class JsonPatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    op: Literal["add", "remove", "replace", "move", "copy", "test"]
    path: str
    value: Any = None
    from_: str | None = Field(default=None, alias="from")

    @model_validator(mode="after")
    def require_operation_members(self) -> "JsonPatchOperation":
        if self.op in {"add", "replace", "test"} and "value" not in self.model_fields_set:
            raise ValueError(f"{self.op} operation requires value")
        if self.op in {"move", "copy"} and not self.from_:
            raise ValueError(f"{self.op} operation requires from")
        return self


class SemanticReconstructionPatchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operations: list[JsonPatchOperation] = Field(default_factory=list)
    reason: str = ""


class RequirementReport(BaseModel):
    schema_valid: bool = True
    coverage_score: float = 0.0
    semantic_requirements_score: float = 0.0
    source_trace_score: float = 0.0
    coverage: CoverageReport = Field(default_factory=CoverageReport)
    semantic_requirements: list[RequirementReportItem] = Field(default_factory=list)
    source_trace: SourceTraceReport = Field(default_factory=SourceTraceReport)
    coverage_patches: list[RequirementReportItem] = Field(default_factory=list)
    semantic_reconstructions: list[SemanticReconstructionRecord] = Field(default_factory=list)


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
    source_evidence_ids = [evidence_id for evidence_id in unique_ids if evidence_id in by_id]
    scores = [
        float(getattr(by_id[evidence_id], "evidence_match_score", 0.0) or 0.0)
        for evidence_id in source_evidence_ids
    ]
    score = sum(scores) / len(scores) if scores else 0.0
    return SourceTraceReport(
        score=score,
        used_evidence_count=len(scores),
        evidence_ids=source_evidence_ids,
    )


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
        evidence_hints=["acquisition", "measurement", "processing", "generated", "experiment", "method", "procedure", "instrument", "software", "setting"],
        allowed_categories=["activity_signal", "method_signal", "software_signal", "instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="generation_activity_detail",
        label="Generation activity detail",
        description="Generation activity has meaningful title, description, and type.",
        weight=1.0,
        target_paths=["/was_generated_by/-"],
        expected_target_class="DataGeneratingActivity",
        evidence_hints=["acquisition", "processing", "experiment type", "method"],
        allowed_categories=["activity_signal", "method_signal", "software_signal", "instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="technical_agents",
        label="Technical agents",
        description="Generation activity names instruments, software, or devices via carried_out_by.",
        weight=1.25,
        target_paths=["/was_generated_by/0/carried_out_by/-"],
        expected_target_class="AgenticEntity",
        evidence_hints=["instrument", "software", "device"],
        allowed_categories=["software_signal", "instrument_signal"],
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
)


DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS: tuple[DcatRequirement, ...] = (
    DcatRequirement(
        requirement_id="dataset_title_identity",
        label="Dataset title identity",
        description="Title identifies the dataset itself rather than only a source file or projection process.",
        weight=0.75,
        target_paths=["/title"],
        evidence_hints=["dataset title", "dataset identifier", "dataset name"],
    ),
    DcatRequirement(
        requirement_id="dataset_description_identity",
        label="Dataset description identity",
        description="Description summarizes dataset content and visible source-backed facts.",
        weight=1.0,
        target_paths=["/description"],
        evidence_hints=["dataset description", "summary", "content"],
    ),
    DcatRequirement(
        requirement_id="generation_activity_reality",
        label="Generation activity reality",
        description="was_generated_by represents a real acquisition, measurement, processing, or generation activity.",
        weight=1.0,
        target_paths=["/was_generated_by"],
        evidence_hints=["activity", "acquisition", "processing", "generated", "method", "procedure", "instrument", "software", "setting"],
        allowed_categories=["activity_signal", "method_signal", "software_signal", "instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="technical_agent_kind",
        label="Technical agent kind",
        description="carried_out_by contains instruments, software, or devices rather than people or provenance metadata.",
        weight=1.0,
        target_paths=["/was_generated_by/0/carried_out_by"],
        evidence_hints=["instrument", "software", "device"],
        allowed_categories=["software_signal", "instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="method_plan_presence",
        label="Method plan presence",
        description="Explicit method or procedure evidence maps to realized_plan when available.",
        weight=1.0,
        target_paths=["/was_generated_by/0/realized_plan"],
        evidence_hints=["method", "protocol", "plan", "procedure", "program"],
        allowed_categories=["method_signal"],
    ),
    DcatRequirement(
        requirement_id="activity_evaluation_target",
        label="Activity evaluation target",
        description="Every data-generating activity identifies at least one concrete entity or other activity it directly measured, observed, analysed, or studied.",
        weight=1.25,
        target_paths=[
            "/was_generated_by/0/evaluated_entity",
            "/was_generated_by/0/evaluated_activity",
        ],
        evidence_hints=[
            "evaluated entity",
            "evaluated activity",
            "measured sample",
            "observed process",
            "analysed input",
            "studied target",
        ],
        allowed_categories=[
            "activity_signal",
            "resource_signal",
            "measurement_signal",
            "measurement_condition",
            "method_signal",
        ],
    ),
    DcatRequirement(
        requirement_id="attribute_duplicate_coherence",
        label="Attribute duplicate coherence",
        description="Duplicate quantitative or qualitative attributes under the same parent or across parents are merged or removed.",
        weight=1.0,
        target_paths=[
            "/was_generated_by/0/has_quantitative_attribute",
            "/was_generated_by/0/carried_out_by/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_activity/0/has_quantitative_attribute",
        ],
        evidence_hints=["duplicate", "same value", "same quantity", "attribute"],
        allowed_categories=["instrument_signal", "measurement_condition", "measurement_signal", "software_signal", "activity_signal"],
    ),
    DcatRequirement(
        requirement_id="attribute_range_decomposition",
        label="Attribute range decomposition",
        description="Ranges are represented as separate minimum and maximum attributes, not string fragments or invalid units.",
        weight=1.0,
        target_paths=[
            "/was_generated_by/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_activity/0/has_quantitative_attribute",
        ],
        evidence_hints=["range", "minimum", "maximum", "from", "to", "spanning"],
        allowed_categories=["measurement_condition", "measurement_signal", "instrument_signal"],
    ),
    DcatRequirement(
        requirement_id="attribute_label_quality",
        label="Attribute label quality",
        description="Attribute labels are concise semantic quantities rather than copied sentence fragments.",
        weight=0.75,
        target_paths=[
            "/was_generated_by/0/has_quantitative_attribute",
            "/was_generated_by/0/carried_out_by/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_activity/0/has_quantitative_attribute",
        ],
        evidence_hints=["quantity", "attribute", "label", "unit"],
        allowed_categories=["instrument_signal", "measurement_condition", "measurement_signal", "software_signal", "activity_signal"],
    ),
    DcatRequirement(
        requirement_id="attribute_parent_placement",
        label="Attribute parent placement",
        description="Attributes attach to activity, entity, or agent parents according to generic evidence ownership cues.",
        weight=1.25,
        target_paths=[
            "/was_generated_by/0/has_quantitative_attribute",
            "/was_generated_by/0/carried_out_by/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute",
            "/was_generated_by/0/evaluated_activity/0/has_quantitative_attribute",
        ],
        evidence_hints=["attribute parent", "instrument setting", "measurement condition", "device", "software", "evaluated entity", "evaluated activity"],
        allowed_categories=["instrument_signal", "measurement_condition", "measurement_signal", "software_signal", "activity_signal"],
    ),
    DcatRequirement(
        requirement_id="provenance_context_placement",
        label="Provenance context placement",
        description="Dates, people, labs, teams, and origins from surrounding evidence are placed in suitable generic provenance/context fields when present.",
        weight=0.75,
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
Evidence establishes applicability and supports interpretation. Only values visibly present at the supplied draft target paths can establish fulfillment.
Never mark a requirement fulfilled when all supplied target paths are empty or absent.
Use statuses: fulfilled, partial, missing, not_applicable.
quality must be 1 for fulfilled, 0.5 for partial, 0 for missing/not_applicable.
Prefer not_applicable only when the supplied evidence categories make the semantic requirement irrelevant.
For activity targets, evaluated_entity/evaluated_activity answer what that specific DataGeneratingActivity directly measured, observed, analysed, or studied. Every data-generating activity needs at least one concrete target. A merely generated output is not an evaluated target; an input file is valid only when evidence says it was directly analysed.
Forbid a DataGeneratingActivity from referencing its own id through evaluated_activity.
For method plans, explicit method/procedure/protocol/plan evidence is required for fulfilled; prefer method_signal, but accept another category when the claim/evidence explicitly says method, procedure, protocol, plan, sampling, or acquisition.
For attribute parent semantics, device/software configuration belongs to agent parents; acquisition/processing settings and thresholds belong to activities; axis bounds, point counts, transmittance/intensity extents, resolution, and data scaling belong to the evaluated data entity unless evidence explicitly says otherwise.
For technical agents, instruments, software, and devices are all valid carried_out_by entries; people and provenance-only origins are not.
"""


REQUIREMENT_PATCH_SYSTEM_PROMPT = """
You create one schema-constrained DCAT-AP+ write envelope for one missing requirement.
Return only JSON matching the supplied schema.
Use only selected evidence. Do not invent facts.
Return an empty writes array if evidence is insufficient.
Return writes only for allowed_target_paths. Use mode=append for array targets and mode=replace for scalar/object targets.
"""


SEMANTIC_RECONSTRUCTION_SYSTEM_PROMPT = """
You reconstruct one DCAT-AP+ Dataset draft slice from one semantic requirement review.
Return only JSON matching the supplied schema.
Use only the current draft, selected evidence, and context window. Do not invent facts.
Return schema-constrained write envelopes only for allowed_target_paths.
Keep edits minimal: replace or append fields only when the semantic requirement justifies it.
Preserve valid numeric instrument/configuration settings. Remove only obvious qualitative/default/placeholders from quantitative attributes.
Do not satisfy missing role=parameter evidence by generalizing one existing quantitative attribute; add separate schema-valid attributes or return empty writes.
For attribute parents, device/software configuration belongs to agent parents; acquisition/processing settings and thresholds belong to activities; axis bounds, point counts, transmittance/intensity extents, resolution, and data scaling belong to the evaluated data entity unless evidence explicitly says otherwise.
Represent numeric ranges as separate schema-valid minimum and maximum quantitative attributes with numeric values and source units when present; never put a range string in a quantitative value.
For activity evaluation targets, write a lean evaluated_entity or evaluated_activity only when evidence directly identifies what that activity measured, observed, analysed, or studied. Never use a generated output merely because it was generated.
Do not use new evidence search.
Return an empty writes array when no safe semantic reconstruction is available.
"""


SEMANTIC_DIAGNOSIS_SYSTEM_PROMPT = """
You diagnose one semantic defect class in one DCAT-AP+ Dataset draft slice.
Return only JSON matching the supplied diagnosis schema with top-level keys defects and reason.
Each defect must use exactly: defect_type, target_path, entry_indices, recommended_action, needs_synthesis, and reason.
Do not return writes, operations, JSON Patch, action envelopes, replacement values, or profile objects.
Use only the supplied draft excerpt and evidence. Do not invent facts.
Use no_action when the defect is real but the supplied context cannot justify a safe repair.
When a missing target is directly supported by selected evidence, recommend append or replace and set needs_synthesis true.
Any append or replace that needs a new value must set needs_synthesis true.
For activity evaluation targets, diagnose every generating activity independently and forbid evaluated_activity self-reference.
"""


SEMANTIC_SYNTHESIS_SYSTEM_PROMPT = """
You synthesize exactly one small value for a backend-validated semantic reconstruction action.
Return only JSON matching the supplied target schema.
Do not return writes, operations, JSON Patch, action envelopes, target paths, or commentary.
Use only the supplied current target value and evidence. Preserve schema-required value shapes such as arrays.
Do not invent facts. Return the smallest value that resolves the diagnosed defect.
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
        "selected_evidence": _prompt_evidence_payload(selected_evidence or [], include_source_context=True),
        "context_window": _prompt_evidence_payload(context_window or [], include_source_context=False),
        "category_meanings": {
            "resource_signal": "files, distributions, formats, access paths, and resource-scoped notes",
            "method_signal": "explicit realized plans, protocols, methods, procedures",
            "activity_signal": "data-generating activities and other activities",
            "software_signal": "software, scripts, executable systems, services, processing applications",
            "instrument_signal": "acquisition, instrument, processing, calibration, unit, threshold, and configuration settings",
            "surrounding_signal": "dates, labs, teams, people, organizations, ownership, origin, authorship, provenance context",
            "measurement_signal": "measurement-related evidence that can route to quantitative or qualitative activity/entity attributes",
            "measurement_condition": "axis bounds, axis units, point counts, ranges, scales, and dataset-level measurement descriptors",
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
    allowed_target_paths = _canonical_target_paths(assessment.target_paths or requirement.target_paths)
    payload = {
        "requirement": requirement.model_dump(mode="json"),
        "assessment": assessment.model_dump(mode="json"),
        "allowed_target_paths": allowed_target_paths,
        "expected_target_class": assessment.expected_target_class or requirement.expected_target_class,
        "draft_excerpt": draft_excerpt,
        "schema_branch": schema_branch,
        "selected_evidence": [item.model_dump(mode="json") for item in selected_evidence],
        "context_window": [item.model_dump(mode="json") for item in context_window],
    }
    return "Create one grounded schema-constrained write envelope for this missing/partial requirement.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def build_semantic_reconstruction_prompt(
    *,
    requirement: DcatRequirement,
    item: RequirementReportItem,
    document: dict[str, Any],
    draft_excerpt: dict[str, Any],
    schema_branches: dict[str, Any],
) -> str:
    allowed_target_paths = _canonical_target_paths(item.target_paths or requirement.target_paths)
    payload = {
        "requirement": requirement.model_dump(mode="json"),
        "assessment": {
            "requirement_id": item.requirement_id,
            "status": item.status,
            "quality": item.quality,
            "rationale": item.rationale,
            "target_paths": item.target_paths,
        },
        "allowed_target_paths": allowed_target_paths,
        "current_document": document,
        "draft_excerpt": draft_excerpt,
        "schema_branches": schema_branches,
        "reconstruction_rules": [
            "Return only the action envelope with writes[]. Do not return JSON Patch operations. Never use keys named op or path.",
            "Allowed write modes are append, replace, remove, and merge.",
            "Use remove only for exact target paths that are semantically unsupported or duplicate after merge/cleanup.",
            "Use merge for duplicate entries in the same array: target_path is the array path, survivor_index is the entry to keep, merged_indices are duplicate entries to remove.",
            "Do not satisfy missing role=parameter evidence by generalizing one existing quantitative attribute; add separate schema-valid attributes or return empty writes.",
            "Device/software cues belong to agent parents; role=parameter evidence with instrument_signal, measurement_condition, resource_signal, or activity_signal belongs to data-generating activity by default unless explicit evidence says it belongs to an agent or evaluated subject/activity.",
            "Represent numeric ranges as separate schema-valid minimum and maximum quantitative attributes with numeric values and source units when present; never put a range string in a quantitative value.",
            "A merely generated output is not an evaluated target. A file may be evaluated_entity only when evidence states that the activity directly analysed it.",
            "Never create evaluated_activity self-reference.",
        ],
        "write_examples": [
            {
                "writes": [
                    {
                        "target_path": "/was_generated_by/0/has_quantitative_attribute",
                        "mode": "merge",
                        "survivor_index": 0,
                        "merged_indices": [2],
                        "reason": "Both entries describe the same point count.",
                    }
                ],
                "reason": "Merged duplicate attributes.",
            },
            {
                "writes": [
                    {
                        "target_path": "/was_generated_by/0/evaluated_entity/0/has_quantitative_attribute/3",
                        "mode": "remove",
                        "reason": "Entry duplicates a better-supported attribute.",
                    }
                ],
                "reason": "Removed duplicate attribute.",
            },
        ],
        "selected_evidence": [evidence.model_dump(mode="json") for evidence in item.selected_evidence],
        "context_window": [evidence.model_dump(mode="json") for evidence in item.context_window],
        "category_meanings": {
            "resource_signal": "files, distributions, formats, access paths, and resource-scoped notes",
            "method_signal": "explicit realized plans, protocols, methods, procedures",
            "activity_signal": "data-generating activities and other activities",
            "software_signal": "software, scripts, executable systems, services, processing applications",
            "instrument_signal": "acquisition, instrument, processing, calibration, unit, threshold, and configuration settings",
            "surrounding_signal": "dates, labs, teams, people, organizations, ownership, origin, authorship, provenance context",
            "measurement_signal": "measurement-related evidence that can route to quantitative or qualitative activity/entity attributes",
            "measurement_condition": "axis bounds, axis units, point counts, ranges, scales, and dataset-level measurement descriptors",
        },
    }
    return "Create a minimal schema-constrained semantic reconstruction write envelope for this requirement.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def build_semantic_diagnosis_prompt(
    *,
    requirement: DcatRequirement,
    item: RequirementReportItem,
    draft_excerpt: dict[str, Any],
) -> str:
    allowed_target_paths = _canonical_target_paths(item.target_paths or requirement.target_paths)
    payload = {
        "requirement": requirement.model_dump(mode="json"),
        "assessment": {
            "requirement_id": item.requirement_id,
            "status": item.status,
            "quality": item.quality,
            "rationale": item.rationale,
            "target_paths": item.target_paths,
        },
        "allowed_target_paths": allowed_target_paths,
        "draft_excerpt": draft_excerpt,
        "selected_evidence": _prompt_evidence_payload(item.selected_evidence, include_source_context=True),
        "context_window": _prompt_evidence_payload(item.context_window, include_source_context=False),
        "diagnosis_rules": [
            "Return only defects[]. Do not return JSON Patch operations or action envelopes.",
            "Use entry_indices only for entries already visible in the target array excerpt.",
            "Set needs_synthesis true only when a new value or object must be generated from evidence.",
            "For append or replace of a missing/incorrect value, set needs_synthesis true unless the value already exists visibly in the draft excerpt.",
            "When selected evidence directly supports a missing required target, prefer append or replace with needs_synthesis true over no_action.",
            "Instruments, software, and devices are all valid technical agents; do not diagnose software as invalid merely because it is not a physical instrument.",
            "Use no_action when the requirement is unresolved but no safe repair can be identified.",
        ],
    }
    return "Diagnose semantic draft defects for this one requirement. Do not write the profile.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def semantic_diagnosis_output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2019-09/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "defects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "defect_type": {
                            "enum": [
                                "duplicate_attribute",
                                "wrong_parent",
                                "bad_range",
                                "bad_label",
                                "missing_plan",
                                "bad_identity",
                                "bad_agent",
                                "bad_aboutness",
                                "bad_provenance",
                                "no_defect",
                            ]
                        },
                        "target_path": {"type": "string"},
                        "entry_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "recommended_action": {
                            "enum": ["merge", "remove", "move", "replace", "append", "no_action"]
                        },
                        "needs_synthesis": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "defect_type",
                        "target_path",
                        "entry_indices",
                        "recommended_action",
                        "needs_synthesis",
                        "reason",
                    ],
                },
            },
            "reason": {"type": "string"},
        },
        "required": ["defects", "reason"],
    }


def _canonical_target_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(path[:-2] if path.endswith("/-") else path for path in paths if path))


def compact_requirement_evidence(
    items: list[RequirementEvidenceItem],
    *,
    include_source_context: bool,
) -> list[dict[str, Any]]:
    return _prompt_evidence_payload(items, include_source_context=include_source_context)


def _prompt_evidence_payload(
    items: list[RequirementEvidenceItem],
    *,
    include_source_context: bool,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items:
        value: dict[str, Any] = {
            "evidence_id": item.evidence_id,
            "category": item.category,
            "role": item.role,
            "claim": item.claim,
            "evidence_text": item.evidence_text,
            "file_path": item.file_path,
        }
        if include_source_context and item.source_context:
            value["source_context"] = item.source_context
        payload.append(value)
    return payload


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
    semantic_reconstructions: list[SemanticReconstructionRecord] | None = None,
) -> RequirementReport:
    for record in semantic_reconstructions or []:
        if record.status == "skipped":
            continue
        for item in semantic_requirements:
            if item.requirement_id != record.requirement_id:
                continue
            item.patch = RequirementPatchAttempt(
                attempted=True,
                status=record.status,
                target_path=record.changed_paths[0] if record.changed_paths else (record.target_paths[0] if record.target_paths else None),
                validation_errors=record.validation_errors,
                reason=record.reason,
            )
            break
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
        semantic_reconstructions=semantic_reconstructions or [],
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
                status="unanswered",
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
                status="unanswered",
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
                iteration_kind="semantic_diagnosis" if req in DCAT_AP_PLUS_SEMANTIC_REQUIREMENTS else "coverage",
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
    selected_limit, context_limit = _requirement_evidence_limits(
        requirement.requirement_id,
        max_selected=max_selected,
        max_context=max_context,
    )
    hints = [
        token
        for hint in ((assessment.evidence_search_hints or requirement.evidence_hints))
        for token in _tokenize(hint)
    ]
    candidates = [
        candidate
        for candidate in list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence)
        if _candidate_allowed_for_requirement(requirement, candidate)
    ]
    scored: list[tuple[int, EvidenceCandidate]] = []
    for candidate in candidates:
        text = _candidate_search_text(candidate)
        score = sum(1 for token in hints if token in text)
        if requirement.expected_target_class:
            score += _class_hint_score(requirement.expected_target_class, candidate)
        if requirement.requirement_id == "attribute_range_decomposition" and (
            "range" in text
            or "spanning" in text
            or re.search(r"\bfrom\b.+\bto\b", text)
        ):
            score += 100
        if score > 0:
            scored.append((score, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].file_path, item[1].start_idx, item[1].candidate_id))
    selected_candidates = _dedupe_candidates([candidate for _, candidate in scored[:selected_limit]])
    context_candidates: list[EvidenceCandidate] = []
    if context_limit:
        compact_semantic_packet = requirement.requirement_id in {
            "attribute_duplicate_coherence",
            "attribute_label_quality",
            "attribute_range_decomposition",
            "attribute_parent_placement",
            "dataset_title_identity",
            "dataset_description_identity",
            "generation_activity_reality",
            "technical_agent_kind",
            "method_plan_presence",
            "activity_evaluation_target",
            "provenance_context_placement",
        }
        selected_keys = {
            f"{candidate.candidate_id}|{candidate.file_path}|{candidate.start_idx}|{candidate.end_idx}"
            for candidate in selected_candidates
        }
        for selected in selected_candidates:
            context_candidates.extend(
                candidate
                for candidate in candidates
                if candidate.file_path == selected.file_path
                and abs(candidate.start_idx - selected.start_idx) <= 500
                and (
                    not compact_semantic_packet
                    or f"{candidate.candidate_id}|{candidate.file_path}|{candidate.start_idx}|{candidate.end_idx}"
                    not in selected_keys
                )
            )
        context_candidates = _dedupe_candidates(context_candidates)[:context_limit]
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
    elif status in {"unanswered", "unresolved"}:
        quality = 0.0
        applicable = bool(assessment.applicable)
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
            candidate.role,
            candidate.claim,
            candidate.evidence_text,
            candidate.source_context,
            candidate.file_path,
        ]
    ).lower()


def _candidate_allowed_for_requirement(requirement: DcatRequirement, candidate: EvidenceCandidate) -> bool:
    category = str(candidate.category)
    if not requirement.allowed_categories or category in requirement.allowed_categories:
        return True
    if requirement.expected_target_class == "QuantitativeAttribute" and str(candidate.role) == "parameter":
        return True
    if requirement.expected_target_class == "Plan" and _candidate_has_method_cue(candidate):
        return True
    return False


def _candidate_has_method_cue(candidate: EvidenceCandidate) -> bool:
    text = _candidate_search_text(candidate)
    return bool(re.search(r"\b(method|procedure|protocol|plan|sampling|acquisition)\b", text))


def _class_hint_score(target_class: str, candidate: EvidenceCandidate) -> int:
    category = candidate.category
    text = _candidate_search_text(candidate)
    if target_class == "DataGeneratingActivity" and (
        category == "activity_signal" or any(term in text for term in ("acquisition", "processing", "experiment"))
    ):
        return 2
    if target_class == "AgenticEntity" and (
        category in {"software_signal", "instrument_signal"} or any(term in text for term in ("instrument", "software", "device", "equipment", "sensor"))
    ):
        return 2
    if target_class == "EvaluatedEntity" and category == "activity_signal":
        return 2
    if target_class == "Distribution" and (
        category == "resource_signal" or any(term in text for term in ("file", "format", "download"))
    ):
        return 2
    if target_class == "Plan" and (category == "method_signal" or _candidate_has_method_cue(candidate)):
        return 2
    if target_class == "QuantitativeAttribute" and (
        category in {"instrument_signal", "measurement_condition"}
        or str(candidate.role) == "parameter"
    ):
        score = 2
        if any(term in text for term in ("frequency", "temperature", "width", "count", "points", "range", "axis", "threshold", "unit", "setting", "parameter")):
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
        role=str(candidate.role),
        claim=candidate.claim,
        evidence_text=candidate.evidence_text,
        source_context=candidate.source_context,
        file_path=candidate.file_path,
        start_idx=candidate.start_idx,
        end_idx=candidate.end_idx,
        evidence_match_score=float(getattr(candidate, "evidence_match_score", 0.0) or 0.0),
    )


def _requirement_evidence_limits(
    requirement_id: str,
    *,
    max_selected: int,
    max_context: int,
) -> tuple[int, int]:
    limits = {
        "attribute_duplicate_coherence": (0, 0),
        "attribute_label_quality": (0, 0),
        "attribute_range_decomposition": (3, 0),
        "attribute_parent_placement": (4, 2),
        "dataset_title_identity": (3, 1),
        "dataset_description_identity": (3, 1),
        "generation_activity_reality": (3, 2),
        "technical_agent_kind": (3, 2),
        "method_plan_presence": (3, 2),
        "activity_evaluation_target": (4, 2),
        "provenance_context_placement": (3, 2),
    }
    selected_limit, context_limit = limits.get(requirement_id, (max_selected, max_context))
    return min(max_selected, selected_limit), min(max_context, context_limit)


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
            candidate.source_context,
        ]
    )
    return f"ev:{sha1(payload.encode('utf-8')).hexdigest()[:16]}"
