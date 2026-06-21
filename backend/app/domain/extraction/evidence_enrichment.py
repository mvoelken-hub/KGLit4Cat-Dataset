from __future__ import annotations

import json
import re
from hashlib import sha1
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from app.domain.extraction.evidence_context import (
    EvidenceCandidate,
    RoutedEvidenceContext,
)
from app.domain.extraction.workflow import ProjectionLedgerRecord


EVIDENCE_ENRICHMENT_CONTEXT_CHARS = 500
BLOCKED_ENRICHMENT_PATHS = {"/title", "/description", "/keyword"}


class EvidenceNoveltyDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    is_novel: bool
    reason: str = ""
    corrected_target_path: str | None = None
    corrected_target_class: str | None = None


class MeasurementSemanticRouteDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    target_path: str | None = None
    merge_key: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class EvidenceEnrichmentEvaluatedEntity(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    title: str | None = None
    description: str | None = None
    type: dict[str, Any] | None = None
    rdf_type: dict[str, Any] | None = None
    has_quantitative_attribute: list[dict[str, Any]] = Field(default_factory=list)
    has_qualitative_attribute: list[dict[str, Any]] = Field(default_factory=list)
    was_generated_by: list[dict[str, Any]] = Field(default_factory=list)
    part_of: list[dict[str, Any]] = Field(default_factory=list)
    has_part: list[dict[str, Any]] = Field(default_factory=list)
    other_identifier: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceEnrichmentAgenticEntity(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    title: str | None = None
    description: str | None = None
    type: dict[str, Any] | None = None
    rdf_type: dict[str, Any] | None = None
    has_quantitative_attribute: list[dict[str, Any]] = Field(default_factory=list)
    has_qualitative_attribute: list[dict[str, Any]] = Field(default_factory=list)
    part_of: list[dict[str, Any]] = Field(default_factory=list)
    has_part: list[dict[str, Any]] = Field(default_factory=list)
    other_identifier: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceEnrichmentAgent(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    name: list[str] = Field(default_factory=list)
    type: dict[str, Any] | None = None


class EvidenceEnrichmentDataGeneratingActivity(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    title: list[str] = Field(default_factory=list)
    description: list[str] = Field(default_factory=list)
    type: dict[str, Any] | None = None
    rdf_type: dict[str, Any] | None = None
    carried_out_by: list[dict[str, Any]] = Field(default_factory=list)
    evaluated_entity: list[dict[str, Any]] = Field(default_factory=list)
    evaluated_activity: list[dict[str, Any]] = Field(default_factory=list)
    has_part: list[dict[str, Any]] = Field(default_factory=list)
    part_of: list[dict[str, Any]] = Field(default_factory=list)
    has_quantitative_attribute: list[dict[str, Any]] = Field(default_factory=list)
    has_qualitative_attribute: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceEnrichmentDistribution(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str | None = None
    title: list[str] = Field(default_factory=list)
    description: list[str] = Field(default_factory=list)
    access_URL: list[dict[str, Any]] = Field(default_factory=list)
    download_URL: list[dict[str, Any]] = Field(default_factory=list)
    format: dict[str, Any] | None = None
    media_type: dict[str, Any] | None = None
    byte_size: int | None = None


_BUILDER_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "EvaluatedEntity": EvidenceEnrichmentEvaluatedEntity,
    "AgenticEntity": EvidenceEnrichmentAgenticEntity,
    "Agent": EvidenceEnrichmentAgent,
    "DataGeneratingActivity": EvidenceEnrichmentDataGeneratingActivity,
    "Distribution": EvidenceEnrichmentDistribution,
}


def builder_output_model_for_target(
    target_class: str,
    target_schema: dict[str, Any] | None = None,
) -> type[BaseModel]:
    if target_schema and target_schema.get("properties"):
        properties = target_schema.get("properties", {})
        fields: dict[str, Any] = {}
        for name, prop in properties.items():
            fields[name] = _builder_field_for_schema(prop)
        return create_model(
            f"EvidenceEnrichment{target_class}",
            __config__=ConfigDict(extra="ignore", populate_by_name=True),
            **fields,
        )
    if target_class in _BUILDER_MODEL_REGISTRY:
        return _BUILDER_MODEL_REGISTRY[target_class]
    return EvidenceEnrichmentEvaluatedEntity


def _schema_types(schema: dict[str, Any]) -> set[str]:
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        return {str(item) for item in raw_type}
    if isinstance(raw_type, str):
        return {raw_type}
    for key in ("anyOf", "oneOf"):
        types: set[str] = set()
        for branch in schema.get(key, []):
            if isinstance(branch, dict):
                types.update(_schema_types(branch))
        if types:
            return types
    if "$ref" in schema or "properties" in schema:
        return {"object"}
    return {"string"}


def _builder_field_for_schema(schema: dict[str, Any]) -> tuple[Any, Any]:
    types = _schema_types(schema)
    if "array" in types:
        return (list[Any], Field(default_factory=list))
    if "object" in types:
        return (dict[str, Any] | None, None)
    if "integer" in types:
        return (int | None, None)
    if "number" in types:
        return (float | int | None, None)
    if "boolean" in types:
        return (bool | None, None)
    return (str | None, None)


def _note_search_text(note: EvidenceCandidate) -> str:
    return f"{note.candidate_id} {note.category} {note.claim} {note.evidence_text}".lower()


def _note_has_device_signal(note: EvidenceCandidate) -> bool:
    text = _note_search_text(note)
    return any(
        term in text
        for term in (
            "instrument",
            "device",
            "equipment",
            "sensor",
            "apparatus",
        )
    )


def route_evidence_note_to_target(
    note: EvidenceCandidate,
) -> tuple[str | None, str | None]:
    text = _note_search_text(note)
    target_path: str | None = None
    target_class: str | None = None

    if note.category == "measurement_signal":
        return None, None
    if note.category in {"software_signal", "instrument_signal"} or _note_has_device_signal(note):
        target_path = "/was_generated_by/0/carried_out_by/-"
        target_class = "AgenticEntity"
    elif note.category == "surrounding_signal" and any(term in text for term in ("date", "timestamp", "modified", "modification")):
        target_path = "/modification_date"
    elif note.category == "surrounding_signal" and any(term in text for term in ("origin", "owner", "creator", "author", "team", "laboratory")):
        target_path = "/creator/-"
        target_class = "Agent"
    elif note.category == "activity_signal" or any(
        term in text
        for term in ("experiment", "acquisition", "generation", "measurement activity", "workflow")
    ):
        target_path = "/was_generated_by/-"
        target_class = "DataGeneratingActivity"
    elif note.category == "method_signal":
        target_path = "/was_generated_by/0/realized_plan"
        target_class = "Plan"
    elif any(term in text for term in ("dataset name", "title", "name")):
        target_path = "/title"
    elif any(term in text for term in ("date", "timestamp", "modified", "modification")):
        target_path = "/modification_date"

    if target_path in BLOCKED_ENRICHMENT_PATHS:
        return None, None
    return target_path, target_class


def build_context_window_for_note(
    note: EvidenceCandidate,
    evidence_context: RoutedEvidenceContext,
    window_chars: int = EVIDENCE_ENRICHMENT_CONTEXT_CHARS,
) -> list[EvidenceCandidate]:
    start = note.start_idx
    end = note.end_idx
    candidates = [
        candidate
        for candidate in list(evidence_context.portable_evidence) + list(evidence_context.contextual_evidence)
    ]
    matches: list[EvidenceCandidate] = []
    seen: set[tuple[str, str, int, int]] = set()
    key = lambda c: (c.candidate_id, c.file_path, c.start_idx, c.end_idx)
    for other in candidates:
        if key(other) == key(note):
            continue
        if other.file_path != note.file_path:
            continue
        overlap = other.end_idx >= start and other.start_idx <= end
        near = (
            abs(other.start_idx - start) <= window_chars
            or abs(other.end_idx - end) <= window_chars
        )
        if overlap or near:
            k = key(other)
            if k not in seen:
                seen.add(k)
                matches.append(other)
    return matches


def _clone_json_object(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _json_pointer_tokens(path: str) -> list[str]:
    if not path or path == "/":
        return []
    return [
        token.replace("~1", "/").replace("~0", "~")
        for token in path.lstrip("/").split("/")
    ]


def _value_at_json_pointer(document: Any, path: str) -> Any:
    value = document
    for token in _json_pointer_tokens(path):
        if isinstance(value, list) and token.isdigit():
            index = int(token)
            if index >= len(value):
                return None
            value = value[index]
        elif isinstance(value, dict):
            value = value.get(token)
        else:
            return None
    return value


def _set_json_pointer_value(
    document: dict[str, Any],
    path: str,
    value: Any,
) -> dict[str, Any]:
    if path in ("", "/"):
        if not isinstance(value, dict):
            raise ValueError("Root replacement must be an object.")
        return _clone_json_object(value)
    result = _clone_json_object(document)
    parts = _json_pointer_tokens(path)
    current: Any = result
    for index, part in enumerate(parts[:-1]):
        next_part = parts[index + 1]
        if isinstance(current, dict):
            if part not in current or current[part] is None:
                current[part] = [] if next_part.isdigit() else {}
            current = current[part]
            continue
        if isinstance(current, list):
            item_index = int(part)
            while len(current) <= item_index:
                current.append({} if not next_part.isdigit() else [])
            current = current[item_index]
            continue
        raise ValueError(f"Cannot set JSON Pointer path '{path}'.")
    final_part = parts[-1]
    if isinstance(current, dict):
        current[final_part] = value
    elif isinstance(current, list) and final_part.isdigit():
        idx = int(final_part)
        while len(current) <= idx:
            current.append({})
        current[idx] = value
    else:
        raise ValueError(f"Cannot set JSON Pointer path '{path}'.")
    return result


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return normalized[:40] or "unknown"


def _build_evidence_id(
    data_package_id: str,
    kind: str,
    label: str,
    index: int | None = None,
) -> str:
    normalized = _slug(label)
    if index is not None:
        normalized = f"{normalized}-{index}"
    digest = sha1(f"{data_package_id}|{kind}|{label}".encode("utf-8")).hexdigest()[:8]
    return f"{data_package_id}:{kind}:{normalized}:{digest}"


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _instance_label(instance: dict[str, Any]) -> str | None:
    for key in ("title", "name", "description", "id"):
        val = instance.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
            if isinstance(first, dict):
                label = _instance_label(first)
                if label:
                    return label
    return None


def _kind_from_target_path(target_path: str) -> str:
    if "creator" in target_path:
        return "agent"
    if "carried_out_by" in target_path:
        return "agentic-entity"
    if "is_about_entity" in target_path:
        return "entity"
    if "was_generated_by" in target_path or "is_about_activity" in target_path:
        return "activity"
    return "enrichment"


def _nested_kind(key: str) -> str:
    mapping = {
        "type": "defined-term",
        "rdf_type": "defined-term",
        "access_URL": "resource",
        "download_URL": "resource",
        "format": "concept",
        "media_type": "concept",
    }
    return mapping.get(key, "nested")


def _normalize_instance_ids(
    instance: dict[str, Any],
    data_package_id: str,
    target_path: str,
    index: int,
    target_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(instance, dict):
        return instance
    instance = _clone_json_object(instance)
    allowed_properties = (
        set(target_schema.get("properties", {}).keys())
        if isinstance(target_schema, dict)
        else set()
    )
    if allowed_properties and "id" not in allowed_properties:
        instance.pop("id", None)
    kind = _kind_from_target_path(target_path)
    should_fill_root_id = not allowed_properties or "id" in allowed_properties
    if should_fill_root_id and not _non_empty_string(instance.get("id")):
        label = _instance_label(instance) or f"enriched-{index}"
        instance["id"] = _build_evidence_id(data_package_id, kind, label)
    for key, value in list(instance.items()):
        if isinstance(value, dict) and not _non_empty_string(value.get("id")):
            value["id"] = _build_evidence_id(
                data_package_id, _nested_kind(key), instance.get("id", "")
            )
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict) and not _non_empty_string(item.get("id")):
                    item["id"] = _build_evidence_id(
                        data_package_id, _nested_kind(key), instance.get("id", ""), i
                    )
    return instance


def apply_evidence_instance(
    document: dict[str, Any],
    target_path: str,
    instance: dict[str, Any],
    data_package_id: str,
    target_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_path in BLOCKED_ENRICHMENT_PATHS or not target_path:
        raise ValueError(f"Target path blocked or empty: {target_path}")
    if target_path.endswith("/-"):
        array_path = target_path[:-2]
        array_value = _value_at_json_pointer(document, array_path)
        if array_value is None:
            document = _set_json_pointer_value(document, array_path, [])
            array_value = _value_at_json_pointer(document, array_path)
        if not isinstance(array_value, list):
            raise ValueError(f"Schema branch target is not an array: {array_path}")
        index = len(array_value)
        instance = _normalize_instance_ids(instance, data_package_id, target_path, index, target_schema)
        array_value.append(_clone_json_object(instance))
        return document
    instance = _normalize_instance_ids(instance, data_package_id, target_path, 0, target_schema)
    return _set_json_pointer_value(document, target_path, _clone_json_object(instance))


EVIDENCE_NOVELTY_EVALUATOR_SYSTEM_PROMPT = (
    "You compare a portable evidence note against existing DCAT-AP+ profile objects and decide if it is semantically novel."
)

EVIDENCE_INSTANCE_BUILDER_SYSTEM_PROMPT = (
    "You emit a schema-constrained DCAT-AP+ write envelope derived from the provided evidence notes."
)

EVIDENCE_INSTANCE_REPAIR_SYSTEM_PROMPT = (
    "You fix schema validation errors in a schema-constrained DCAT-AP+ write envelope."
)

MEASUREMENT_SEMANTIC_ROUTER_SYSTEM_PROMPT = (
    "You route one measurement-related note into a DCAT-AP+ profile draft. "
    "Choose only among the allowed activity/entity attribute paths, or return null when the note is not safely projectable. "
    "Also return a stable merge key for semantically equivalent notes."
)


def build_novelty_evaluator_prompt(
    note: EvidenceCandidate,
    contextual_notes: list[EvidenceCandidate],
    draft_excerpt: Any,
    schema_branch: dict[str, Any],
    target_path: str,
    target_class: str,
) -> str:
    lines = [
        "You are a semantic novelty detector for a DCAT-AP+ metadata profile draft.",
        f"Target JSON Pointer path: {target_path}",
        f"Target schema class: {target_class}",
        "Current objects already at the target path:",
        json.dumps(draft_excerpt, ensure_ascii=False, indent=2)
        if draft_excerpt is not None
        else "(none)",
        "Compact schema branch for the target class:",
        json.dumps(schema_branch, ensure_ascii=False, indent=2),
        "Portable evidence note to evaluate:",
        f"  category: {note.category}",
        f"  claim: {note.claim}",
        f"  evidence_text: {note.evidence_text}",
        f"  source_context: {note.source_context or note.evidence_text}",
        f"  file_path: {note.file_path}",
        "Contextual notes from the same source chunk (compact, no raw chunk text):",
    ]
    for context in contextual_notes:
        lines.append(
            f"  - [{context.category}] {context.claim} | evidence: {context.evidence_text} | source_context: {context.source_context or context.evidence_text}"
        )
    lines.extend(
        [
            "Question: Is the portable evidence note already represented by an existing object at the target path?",
            "Answer with is_novel=true only if it brings a new, distinct object or a new property value not already present. Provide a concise reason.",
        ]
    )
    return "\n".join(lines)


def build_instance_builder_prompt(
    target_path: str,
    target_class: str,
    schema_branch: dict[str, Any],
    draft_excerpt: Any,
    note: EvidenceCandidate,
    contextual_notes: list[EvidenceCandidate],
) -> str:
    lines = [
        "You are building a single DCAT-AP+ instance for a metadata profile draft.",
        f"Target path: {target_path}",
        f"Target class: {target_class}",
        "Schema branch (compact example of typical fields):",
        json.dumps(schema_branch, ensure_ascii=False, indent=2),
        "Existing objects at the target path (for reference, do not duplicate):",
        json.dumps(draft_excerpt, ensure_ascii=False, indent=2)
        if draft_excerpt is not None
        else "(none)",
        "Portable evidence note:",
        f"  category: {note.category}",
        f"  claim: {note.claim}",
        f"  evidence_text: {note.evidence_text}",
        f"  source_context: {note.source_context or note.evidence_text}",
        f"  file_path: {note.file_path}",
        "Contextual notes:",
    ]
    for context in contextual_notes:
        lines.append(
            f"  - [{context.category}] {context.claim} | evidence: {context.evidence_text} | source_context: {context.source_context or context.evidence_text}"
        )
    lines.extend(
        [
            "Emit one write envelope containing one JSON object matching the target class. Omit fields you cannot ground in the evidence.",
            "Include an `id` field only if a real stable identifier is present in the evidence; otherwise omit it and the backend will assign one.",
        ]
    )
    return "\n".join(lines)


def build_measurement_semantic_route_prompt(
    *,
    note: EvidenceCandidate,
    contextual_notes: list[EvidenceCandidate],
    draft_excerpt: dict[str, Any],
    allowed_target_paths: list[str],
) -> str:
    lines = [
        "You are routing one measurement-related evidence note into a DCAT-AP+ profile draft.",
        "Allowed target paths:",
        json.dumps(allowed_target_paths, ensure_ascii=False, indent=2),
        "Current draft excerpt around the allowed measurement parent branches:",
        json.dumps(draft_excerpt, ensure_ascii=False, indent=2),
        "Portable evidence note:",
        f"  category: {note.category}",
        f"  role: {note.role}",
        f"  claim: {note.claim}",
        f"  evidence_text: {note.evidence_text}",
        f"  source_context: {note.source_context or note.evidence_text}",
        f"  file_path: {note.file_path}",
        "Nearby context notes from the same source chunk:",
    ]
    for context in contextual_notes:
        lines.append(
            f"  - [{context.category}] {context.claim} | evidence: {context.evidence_text} | source_context: {context.source_context or context.evidence_text}"
        )
    lines.extend(
        [
            "Routing rules:",
            "- Choose exactly one allowed target path when the note is safely projectable.",
            "- Use only activity/entity parents and only has_quantitative_attribute or has_qualitative_attribute terminals.",
            "- Return target_path=null and merge_key=null when the note is too ambiguous, unsupported, or unsafe to project.",
            "- confidence must be between 0 and 1. Use confidence below 0.7 when ownership or attribute kind is uncertain.",
            "- merge_key must be stable across semantically equivalent notes and should ignore surface formatting differences.",
            "Return concise reason text for the routing choice or skip decision.",
        ]
    )
    return "\n".join(lines)


def build_instance_repair_prompt(
    target_path: str,
    target_class: str,
    schema_branch: dict[str, Any],
    instance: dict[str, Any],
    validation_errors: list[str],
) -> str:
    return "\n".join(
        [
            "Repair the following DCAT-AP+ instance so it validates against the schema.",
            f"Target path: {target_path}",
            f"Target class: {target_class}",
            "Schema branch:",
            json.dumps(schema_branch, ensure_ascii=False, indent=2),
            "Validation errors:",
            json.dumps(validation_errors, ensure_ascii=False, indent=2),
            "Instance to repair:",
            json.dumps(instance, ensure_ascii=False, indent=2),
            "Return the repaired write envelope only, with no extra commentary.",
        ]
    )
