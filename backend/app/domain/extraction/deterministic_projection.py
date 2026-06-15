from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.extraction.evidence_context import EvidenceContext, EvidenceNote, RoutedEvidenceContext
from app.domain.extraction.workflow import ProjectionLedgerRecord


HYBRID_INSTANCE_BUILDER_SYSTEM_PROMPT = """
You build exactly one DCAT-AP-plus schema class instance from evidence selected by SIMONE.
Return only the structured wrapper JSON. Use portable evidence as the object anchor.
Use contextual evidence only to enrich that same object; contextual evidence must not create a new object.
Do not copy all evidence notes into broad description fields. Prefer concise summaries and supported schema slots.
Return status='skip' when the evidence does not improve the target instance or cannot be represented safely.
Never invent values, dates, identifiers, units, or defaults.
"""

HYBRID_CLASS_DECIDER_SYSTEM_PROMPT = """
You resolve one ambiguous portable evidence note to a DCAT-AP-plus target class.
Select exactly one candidate class or return skip. Do not generate metadata values.
"""

ProjectionTargetClass = Literal[
    "Dataset",
    "Distribution",
    "Agent",
    "AgenticEntity",
    "DataGeneratingActivity",
    "EvaluatedEntity",
    "Concept",
]


@dataclass
class InstanceProjectionCandidate:
    candidate_id: str
    portable_note: EvidenceNote
    target_class: str | None
    target_path: str | None
    reason: str
    identity_key: str | None = None
    candidate_classes: list[str] = field(default_factory=list)
    duplicate_of: str | None = None


@dataclass
class InstanceProjectionGroup:
    group_id: str
    target_class: str
    target_path: str
    portable_notes: list[EvidenceNote]
    contextual_notes: list[EvidenceNote]
    identity_key: str | None
    reason: str


@dataclass
class ProjectedInstance:
    group: InstanceProjectionGroup
    value: Any
    status: Literal["projected", "not_projected", "user_edit_required"]
    reason: str
    used_portable_note_ids: list[str] = field(default_factory=list)
    used_contextual_note_ids: list[str] = field(default_factory=list)
    skipped_note_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class HybridProjectionPreparation:
    candidates: list[InstanceProjectionCandidate]
    groups: list[InstanceProjectionGroup]
    ledger: list[ProjectionLedgerRecord]


@dataclass
class HybridProjectionResult:
    document: dict[str, Any]
    ledger: list[ProjectionLedgerRecord]


class ProjectionClassDecision(BaseModel):
    status: Literal["targeted", "skip"] = "targeted"
    target_class: str | None = None
    reason: str = ""


class ProjectionInstanceWriteDocument(BaseModel):
    status: Literal["write", "skip"] = "write"
    value: Any = None
    used_portable_note_ids: list[str] = Field(default_factory=list)
    used_contextual_note_ids: list[str] = Field(default_factory=list)
    skipped_note_ids: list[str] = Field(default_factory=list)
    reason: str = ""


def prepare_hybrid_projection(
    *,
    data_package_id: str,
    evidence_context: EvidenceContext | RoutedEvidenceContext,
    resolved_class_choices: dict[str, str | None] | None = None,
) -> HybridProjectionPreparation:
    resolved_class_choices = resolved_class_choices or {}
    candidates: list[InstanceProjectionCandidate] = []
    ledger: list[ProjectionLedgerRecord] = []
    seen: dict[str, str] = {}

    for note in evidence_context.portable_evidence:
        candidate = route_portable_evidence_note(note, data_package_id=data_package_id)
        if candidate.target_class is None and candidate.candidate_classes:
            selected_class = resolved_class_choices.get(candidate.candidate_id)
            if selected_class:
                candidate.target_class = selected_class
                candidate.target_path = target_path_for_class(selected_class)
                candidate.reason = "Ambiguous target class resolved by class-choice model."
            else:
                ledger.append(
                    ledger_record_for_candidate(
                        candidate,
                        data_package_id=data_package_id,
                        status="not_projected",
                        reason=f"{candidate.reason} Ambiguous target class was not resolved.",
                    )
                )
                candidates.append(candidate)
                continue

        duplicate_key = _duplicate_key(note)
        if duplicate_key in seen:
            candidate.duplicate_of = seen[duplicate_key]
            ledger.append(
                ledger_record_for_candidate(
                    candidate,
                    data_package_id=data_package_id,
                    status="not_projected",
                    reason=f"Duplicate portable evidence note of {candidate.duplicate_of}.",
                )
            )
            candidates.append(candidate)
            continue
        seen[duplicate_key] = candidate.candidate_id

        if candidate.target_class is None or candidate.target_path is None:
            ledger.append(
                ledger_record_for_candidate(
                    candidate,
                    data_package_id=data_package_id,
                    status="not_projected",
                    reason=candidate.reason,
                )
            )
        candidates.append(candidate)

    groups = group_instance_candidates(
        data_package_id=data_package_id,
        candidates=[
            candidate
            for candidate in candidates
            if candidate.target_class and candidate.target_path and not candidate.duplicate_of
        ],
        contextual_notes=_contextual_notes(evidence_context),
    )
    return HybridProjectionPreparation(candidates=candidates, groups=groups, ledger=ledger)


def route_portable_evidence_note(
    note: EvidenceNote,
    *,
    data_package_id: str,
) -> InstanceProjectionCandidate:
    target_class, reason, candidates = _target_class_for_note(note)
    target_path = target_path_for_class(target_class) if target_class else None
    identity_key = _identity_key_for_note(note, target_class) if target_class else None
    return InstanceProjectionCandidate(
        candidate_id=_candidate_identifier(note, data_package_id=data_package_id),
        portable_note=note,
        target_class=target_class,
        target_path=target_path,
        reason=reason,
        identity_key=identity_key,
        candidate_classes=candidates,
    )


def group_instance_candidates(
    *,
    data_package_id: str,
    candidates: list[InstanceProjectionCandidate],
    contextual_notes: list[EvidenceNote],
) -> list[InstanceProjectionGroup]:
    keyed: dict[tuple[str, str], list[InstanceProjectionCandidate]] = {}
    groups: list[InstanceProjectionGroup] = []
    for candidate in candidates:
        if candidate.identity_key:
            keyed.setdefault((candidate.target_class or "", candidate.identity_key), []).append(candidate)
        else:
            groups.append(_group_for_candidates(data_package_id, [candidate], contextual_notes))
    for merged in keyed.values():
        groups.append(_group_for_candidates(data_package_id, merged, contextual_notes))
    return groups


def _group_for_candidates(
    data_package_id: str,
    candidates: list[InstanceProjectionCandidate],
    contextual_notes: list[EvidenceNote],
) -> InstanceProjectionGroup:
    portable_notes = [candidate.portable_note for candidate in candidates]
    first = candidates[0]
    contextual = select_contextual_evidence_for_group(
        portable_notes=portable_notes,
        contextual_notes=contextual_notes,
        target_class=first.target_class or "",
    )
    source = "|".join(candidate.candidate_id for candidate in candidates)
    digest = sha1(f"{data_package_id}|{first.target_class}|{source}".encode("utf-8")).hexdigest()[:12]
    return InstanceProjectionGroup(
        group_id=f"instance:{(first.target_class or 'unknown').lower()}:{digest}",
        target_class=first.target_class or "",
        target_path=first.target_path or "",
        portable_notes=portable_notes,
        contextual_notes=contextual,
        identity_key=first.identity_key,
        reason="; ".join(_dedupe_strings([candidate.reason for candidate in candidates])),
    )


def select_contextual_evidence_for_group(
    *,
    portable_notes: list[EvidenceNote],
    contextual_notes: list[EvidenceNote],
    target_class: str,
) -> list[EvidenceNote]:
    ranked: list[tuple[int, int, str, EvidenceNote]] = []
    for note in contextual_notes:
        distances = [
            _span_distance(anchor, note)
            for anchor in portable_notes
            if anchor.file_path and anchor.file_path == note.file_path
        ]
        if not distances:
            continue
        distance = min(distances)
        compatibility = _contextual_category_rank(note, target_class)
        ranked.append((distance, compatibility, _projection_identifier_for_evidence_note(note), note))
    ranked.sort(key=lambda item: item[:3])
    if not ranked:
        return []
    cap = min(10, max(3, len(portable_notes) * 2))
    return [note for *_prefix, note in ranked[:cap]]


def build_instance_builder_prompt_components(
    *,
    data_package_id: str,
    group: InstanceProjectionGroup,
    target_schema: dict[str, Any],
) -> list[tuple[str, str]]:
    portable = [_note_prompt_record(note) for note in group.portable_notes]
    contextual = [_note_prompt_record(note) for note in group.contextual_notes]
    return [
        (
            "task",
            (
                f"Data package id: {data_package_id}\n"
                f"Instance group id: {group.group_id}\n"
                f"Target class: {group.target_class}\n"
                f"Target dataset path: {group.target_path}\n"
                f"Identity key: {group.identity_key or ''}\n\n"
            ),
        ),
        ("target_schema_json", json.dumps(target_schema, ensure_ascii=False, indent=2)),
        ("portable_evidence_json", json.dumps(portable, ensure_ascii=False, indent=2)),
        ("contextual_evidence_json", json.dumps(contextual, ensure_ascii=False, indent=2)),
        (
            "instructions",
            (
                "Build one target class instance. Use portable evidence as the anchor. "
                "Use contextual evidence only when it clearly enriches that same instance. "
                "If you skip any evidence, include its note_id in skipped_note_ids. "
                "When status is skip, set value to null."
            ),
        ),
    ]


def build_class_decision_prompt_components(
    *,
    note: EvidenceNote,
    candidate_classes: list[str],
    class_summaries: dict[str, str],
) -> list[tuple[str, str]]:
    return [
        ("candidate_classes_json", json.dumps(candidate_classes, ensure_ascii=False, indent=2)),
        ("class_summaries_json", json.dumps(class_summaries, ensure_ascii=False, indent=2)),
        ("portable_evidence_json", json.dumps(_note_prompt_record(note), ensure_ascii=False, indent=2)),
        ("instructions", "Select one target_class from candidate_classes or status='skip'."),
    ]


def instance_builder_output_schema(
    *,
    json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    return {
        "$schema": json_schema.get("$schema", "https://json-schema.org/draft/2019-09/schema"),
        "$defs": json_schema.get("$defs", {}),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "value",
            "used_portable_note_ids",
            "used_contextual_note_ids",
            "skipped_note_ids",
            "reason",
        ],
        "properties": {
            "status": {"type": "string", "enum": ["write", "skip"]},
            "value": {
                "anyOf": [
                    {"$ref": f"#/$defs/{target_class}"},
                    {"type": "null"},
                ]
            },
            "used_portable_note_ids": {"type": "array", "items": {"type": "string"}},
            "used_contextual_note_ids": {"type": "array", "items": {"type": "string"}},
            "skipped_note_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
    }


def assemble_hybrid_dataset_projection(
    *,
    data_package_id: str,
    base_document: dict[str, Any],
    evidence_context: EvidenceContext | RoutedEvidenceContext,
    projected_instances: list[ProjectedInstance],
    preparation_ledger: list[ProjectionLedgerRecord],
) -> HybridProjectionResult:
    document = _clone(base_document)
    ledger = list(preparation_ledger)
    scaffold_records: list[ProjectionLedgerRecord] = []
    dataset_instances = [
        item for item in projected_instances if item.status == "projected" and item.group.target_class == "Dataset"
    ]
    for item in dataset_instances:
        if isinstance(item.value, dict):
            _merge_dataset_identity(document, item.value)

    fallback_title = _dataset_title_from_notes(evidence_context.portable_evidence) or data_package_id
    if not _has_meaningful_value(document.get("title")):
        document["title"] = [fallback_title]
        scaffold_records.append(
            _scaffold_record(
                object_identifier="scaffold:/title",
                target_path="/title",
                target_class="Dataset",
                reason="Required Dataset.title scaffold fallback.",
            )
        )
    if not _has_meaningful_value(document.get("description")):
        document["description"] = [f"SIMONE metadata draft for {fallback_title}."]
        scaffold_records.append(
            _scaffold_record(
                object_identifier="scaffold:/description",
                target_path="/description",
                target_class="Dataset",
                reason="Required Dataset.description scaffold fallback.",
            )
        )
    document["id"] = str(document.get("id") or data_package_id)
    if "identifier" in document:
        document["identifier"] = _merge_strings(_as_list(document.get("identifier")), [data_package_id])

    for item in projected_instances:
        ledger.append(ledger_record_for_projected_instance(item))
        if item.status != "projected":
            continue
        if item.group.target_class == "Dataset":
            continue
        _insert_projected_instance(document, item)

    if not _has_meaningful_value(document.get("was_generated_by")):
        document["was_generated_by"] = [_primary_activity(data_package_id)]
        scaffold_records.append(
            _scaffold_record(
                object_identifier="scaffold:/was_generated_by/0",
                target_path="/was_generated_by/0",
                target_class="DataGeneratingActivity",
                reason="Required Dataset.was_generated_by scaffold fallback.",
            )
        )
    ledger.extend(scaffold_records)
    return HybridProjectionResult(document=_remove_empty_strings(document), ledger=ledger)


def ledger_record_for_candidate(
    candidate: InstanceProjectionCandidate,
    *,
    data_package_id: str,
    status: Literal["projected", "not_projected", "user_edit_required"],
    reason: str,
    error: str | None = None,
) -> ProjectionLedgerRecord:
    return ProjectionLedgerRecord(
        object_identifier=candidate.candidate_id,
        object_kind="InstanceProjectionCandidate",
        source_evidence=candidate.portable_note.evidence_text,
        evidence_note_identifiers=[_projection_identifier_for_evidence_note(candidate.portable_note)],
        status=status,
        projected_paths=[],
        target_path=candidate.target_path,
        target_class=candidate.target_class,
        planner_status="skip" if status != "projected" else "deterministic",
        planner_reason=candidate.reason,
        evidence_quality={
            "note_count": 1,
            "routes": {"portable_evidence": 1},
            "candidate_classes": candidate.candidate_classes,
            "data_package_id": data_package_id,
        },
        reason=reason,
        error=error,
    )


def ledger_record_for_projected_instance(item: ProjectedInstance) -> ProjectionLedgerRecord:
    status = item.status
    return ProjectionLedgerRecord(
        object_identifier=item.group.group_id,
        object_kind="InstanceProjectionGroup",
        source_evidence="\n".join(note.evidence_text for note in item.group.portable_notes if note.evidence_text),
        evidence_note_identifiers=[
            _projection_identifier_for_evidence_note(note)
            for note in item.group.portable_notes
        ],
        status=status,
        projected_paths=[item.group.target_path] if status == "projected" else [],
        target_path=item.group.target_path,
        target_class=item.group.target_class,
        planner_status="hybrid_instance_builder",
        planner_reason=item.group.reason,
        evidence_quality={
            "note_count": len(item.group.portable_notes) + len(item.group.contextual_notes),
            "routes": {
                "portable_evidence": len(item.group.portable_notes),
                "contextual_evidence": len(item.group.contextual_notes),
            },
            "used_portable_note_ids": item.used_portable_note_ids,
            "used_contextual_note_ids": item.used_contextual_note_ids,
            "skipped_note_ids": item.skipped_note_ids,
            "identity_key": item.group.identity_key,
        },
        reason=item.reason,
        error=item.error,
    )


def target_path_for_class(target_class: str | None) -> str | None:
    return {
        "Dataset": "/",
        "Distribution": "/dataset_distribution",
        "Agent": "/creator",
        "AgenticEntity": "/was_generated_by/0/carried_out_by",
        "DataGeneratingActivity": "/was_generated_by",
        "EvaluatedEntity": "/is_about_entity",
        "Concept": "/type",
    }.get(target_class or "")


def ambiguous_candidates(preparation: HybridProjectionPreparation) -> list[InstanceProjectionCandidate]:
    return [
        candidate
        for candidate in preparation.candidates
        if candidate.target_class is None and candidate.candidate_classes and not candidate.duplicate_of
    ]


def _target_class_for_note(note: EvidenceNote) -> tuple[str | None, str, list[str]]:
    text = _note_search_text(note)
    normalized_key, value = _assignment_from_note(note)
    key = normalized_key.lower().strip("$#.") if normalized_key else ""
    if _dataset_title_from_note(note):
        return "Dataset", "Portable dataset identity evidence.", []
    if key in {"origin", "owner", "author", "creator"} and value:
        return "Agent", "Portable creator/owner evidence.", []
    if _has_device_signal(note):
        return "AgenticEntity", "Portable device or instrument evidence.", []
    if _looks_like_distribution_evidence(note):
        return "Distribution", "Portable distribution or resource evidence.", []
    if _looks_like_entity_evidence(note):
        return "EvaluatedEntity", "Portable evaluated-entity evidence.", []
    if _looks_like_activity_evidence(note):
        return "DataGeneratingActivity", "Portable data-generating activity evidence.", []
    if note.category == "measurement_signal":
        classes = ["DataGeneratingActivity", "EvaluatedEntity", "Distribution"]
        if _activity_measurement_signal(text) and not _entity_measurement_signal(text):
            return "DataGeneratingActivity", "Portable activity-scoped measurement evidence.", []
        if _entity_measurement_signal(text) and not _activity_measurement_signal(text):
            return "EvaluatedEntity", "Portable entity-scoped measurement evidence.", []
        return None, "Ambiguous measurement evidence needs target-class decision.", classes
    if _looks_like_date_evidence(note):
        return "Dataset", "Portable date-like dataset evidence.", []
    if "type" in text or "category" in text or "class" in text:
        return "Concept", "Portable type/category evidence.", []
    return None, "No deterministic DCAT-AP-plus target class matched the evidence.", []


def _identity_key_for_note(note: EvidenceNote, target_class: str | None) -> str | None:
    if target_class == "Dataset":
        return "dataset"
    if target_class == "Distribution":
        resource = _resource_name_from_note(note) or note.file_path
        return f"distribution:{_normalize_identity(resource)}" if resource else None
    if target_class == "Agent":
        key, value = _assignment_from_note(note)
        return f"agent:{_normalize_identity(value)}" if value else None
    if target_class == "AgenticEntity":
        title = _device_title_from_note(note)
        return f"agentic-entity:{_normalize_identity(title)}" if title else None
    if target_class == "DataGeneratingActivity":
        method = _method_identifier_from_note(note)
        return f"activity:{_normalize_identity(method)}" if method else None
    if target_class == "EvaluatedEntity":
        entity = _entity_identifier_from_note(note)
        return f"entity:{_normalize_identity(entity)}" if entity else None
    if target_class == "Concept":
        label = _short_value_from_note(note)
        return f"concept:{_normalize_identity(label)}" if label else None
    return None


def _insert_projected_instance(document: dict[str, Any], item: ProjectedInstance) -> None:
    target_class = item.group.target_class
    value = _clone(item.value)
    if not _has_meaningful_value(value):
        return
    if target_class == "Distribution":
        document["dataset_distribution"] = _merge_dicts(_as_list(document.get("dataset_distribution")), [value])
    elif target_class == "Agent":
        document["creator"] = _merge_dicts(_as_list(document.get("creator")), [value])
    elif target_class == "AgenticEntity":
        activities = _as_list(document.get("was_generated_by"))
        if not activities or not isinstance(activities[0], dict):
            activities = [_primary_activity("generated")]
        activities[0]["carried_out_by"] = _merge_dicts(_as_list(activities[0].get("carried_out_by")), [value])
        document["was_generated_by"] = activities
    elif target_class == "DataGeneratingActivity":
        existing = [
            item
            for item in _as_list(document.get("was_generated_by"))
            if not (
                isinstance(item, dict)
                and str(item.get("id", "")).endswith(":activity:metadata-extraction")
            )
        ]
        document["was_generated_by"] = _merge_dicts([value], [item for item in existing if isinstance(item, dict)])
    elif target_class == "EvaluatedEntity":
        document["is_about_entity"] = _merge_dicts(_as_list(document.get("is_about_entity")), [value])
    elif target_class == "Concept":
        document["type"] = _merge_dicts(_as_list(document.get("type")), [value])


def _merge_dataset_identity(document: dict[str, Any], value: dict[str, Any]) -> None:
    for key in ("title", "description", "identifier"):
        if key in value:
            document[key] = _merge_strings(_as_list(document.get(key)), _as_list(value.get(key)))
    for key, item in value.items():
        if key in {"id", "title", "description", "identifier"}:
            continue
        if key not in document and _has_meaningful_value(item):
            document[key] = _clone(item)


def _scaffold_record(
    *,
    object_identifier: str,
    target_path: str,
    target_class: str,
    reason: str,
) -> ProjectionLedgerRecord:
    return ProjectionLedgerRecord(
        object_identifier=object_identifier,
        object_kind="ScaffoldFact",
        status="projected",
        projected_paths=[target_path],
        target_path=target_path,
        target_class=target_class,
        planner_status="scaffold",
        planner_reason=reason,
        reason=reason,
    )


def _primary_activity(data_package_id: str) -> dict[str, Any]:
    return {
        "id": f"{data_package_id}:activity:metadata-extraction",
        "title": ["Metadata extraction activity"],
        "description": ["Required scaffold activity for a schema-valid Dataset draft."],
        "has_qualitative_attribute": [],
        "has_quantitative_attribute": [],
        "evaluated_activity": [],
        "evaluated_entity": [],
        "carried_out_by": [],
    }


def _contextual_notes(evidence_context: EvidenceContext | RoutedEvidenceContext) -> list[EvidenceNote]:
    return list(getattr(evidence_context, "contextual_evidence", []))


def _candidate_identifier(note: EvidenceNote, *, data_package_id: str) -> str:
    digest = sha1(f"{data_package_id}|{_projection_identifier_for_evidence_note(note)}".encode("utf-8")).hexdigest()[:12]
    return f"candidate:{digest}"


def _projection_identifier_for_evidence_note(note: EvidenceNote) -> str:
    return f"{note.file_path or 'unknown-file'}#{note.start_idx}-{note.end_idx}#{note.note_id or 'unnamed-note'}"


def _note_prompt_record(note: EvidenceNote) -> dict[str, Any]:
    return {
        "note_id": _projection_identifier_for_evidence_note(note),
        "candidate_id": note.note_id,
        "category": note.category,
        "claim": note.claim,
        "evidence_text": note.evidence_text,
        "file_path": note.file_path,
        "start_idx": note.start_idx,
        "end_idx": note.end_idx,
    }


def _duplicate_key(note: EvidenceNote) -> str:
    return re.sub(r"\s+", " ", f"{note.category}|{note.file_path}|{note.claim}|{note.evidence_text}".lower()).strip()


def _span_distance(anchor: EvidenceNote, note: EvidenceNote) -> int:
    if note.end_idx < anchor.start_idx:
        return anchor.start_idx - note.end_idx
    if anchor.end_idx < note.start_idx:
        return note.start_idx - anchor.end_idx
    return 0


def _contextual_category_rank(note: EvidenceNote, target_class: str) -> int:
    compatible = {
        "Dataset": {"resource_signal", "data_quality_signal", "entity_signal"},
        "Distribution": {"resource_signal", "data_quality_signal"},
        "Agent": {"agent_signal"},
        "AgenticEntity": {"agent_signal", "method_signal"},
        "DataGeneratingActivity": {"method_signal", "measurement_signal"},
        "EvaluatedEntity": {"entity_signal", "measurement_signal"},
        "Concept": {"entity_signal", "resource_signal", "method_signal"},
    }.get(target_class, set())
    return 0 if note.category in compatible else 1


def _dataset_title_from_notes(notes: list[EvidenceNote]) -> str | None:
    for note in notes:
        title = _dataset_title_from_note(note)
        if title:
            return title
    return None


def _dataset_title_from_note(note: EvidenceNote) -> str | None:
    for text in (note.evidence_text, note.claim):
        if not text:
            continue
        match = re.search(r"\bdataset\s+(?:name|title)\s*[:=]\s*([^\r\n.;]+)", text, re.IGNORECASE)
        if match:
            return _clean_title(match.group(1))
        match = re.search(r"\bdataset\s+(?:name|title)\s+is\s+([^\r\n.;]+)", text, re.IGNORECASE)
        if match:
            return _clean_title(match.group(1))
    key, value = _assignment_from_note(note)
    if key and key.lower().strip("$#.") == "title" and value:
        return _clean_title(value)
    return None


def _clean_title(value: str) -> str | None:
    cleaned = value.strip().strip('"')
    if not cleaned or cleaned.isdigit() or len(cleaned) < 2:
        return None
    if cleaned.lower() in {"parameter file", "spectrum title", "dataset"}:
        return None
    return cleaned


def _assignment_from_note(note: EvidenceNote) -> tuple[str | None, str | None]:
    text = (note.evidence_text or "").strip()
    if not text:
        return None, None
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(r"^##\.?\$?", "", first_line).strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_ .-]{0,48})\s*[:=]\s*<?([^>\r\n]+)>?", first_line)
    if match:
        return match.group(1).strip(), match.group(2).strip().strip('"')
    match = re.match(r"([A-Z][A-Z0-9_]{1,32})\s+(.+)$", first_line)
    if match:
        return match.group(1).strip(), match.group(2).strip().strip('"')
    return None, None


def _note_search_text(note: EvidenceNote) -> str:
    return " ".join(
        part for part in (note.note_id, note.category, note.claim, note.evidence_text, note.file_path) if part
    ).lower()


def _has_device_signal(note: EvidenceNote) -> bool:
    return any(term in _note_search_text(note) for term in ("instrument", "device", "equipment", "sensor", "apparatus"))


def _device_title_from_note(note: EvidenceNote) -> str | None:
    text = f"{note.evidence_text}\n{note.claim}"
    match = re.search(
        r"(?:instrument|device|equipment|sensor|apparatus)\s*(?:used\s*)?(?:is|:|=)\s*<?([^>\r\n.;]+)>?",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _looks_like_distribution_evidence(note: EvidenceNote) -> bool:
    if note.category == "method_signal":
        return False
    text = _note_search_text(note)
    return note.category == "resource_signal" or any(
        term in text for term in ("file", "format", "distribution", "download", "archive", "media type")
    )


def _resource_name_from_note(note: EvidenceNote) -> str | None:
    match = re.search(r"\b([\w.-]+\.[A-Za-z0-9]{1,8})\b", f"{note.evidence_text}\n{note.claim}")
    return match.group(1) if match else None


def _looks_like_activity_evidence(note: EvidenceNote) -> bool:
    text = _note_search_text(note)
    return note.category == "method_signal" or any(
        term in text for term in ("method", "procedure", "workflow", "experiment", "acquisition", "generation", "activity", "mode")
    )


def _method_identifier_from_note(note: EvidenceNote) -> str | None:
    key, value = _assignment_from_note(note)
    if key and key.lower().strip("$#.") in {"pulprog", "method", "procedure", "program", "sequence"} and value:
        return value
    text = f"{note.evidence_text}\n{note.claim}"
    match = re.search(r"\b(?:method|procedure|program|sequence|workflow)\s*(?:is|:|=)\s*([^\r\n.;]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _looks_like_entity_evidence(note: EvidenceNote) -> bool:
    text = _note_search_text(note)
    if "dataset name" in text or "dataset title" in text:
        return False
    return note.category == "entity_signal" or any(
        term in text for term in ("sample", "specimen", "subject", "target entity", "material", "entity")
    )


def _entity_identifier_from_note(note: EvidenceNote) -> str | None:
    key, value = _assignment_from_note(note)
    if key and key.lower().strip("$#.") in {"sample", "sample_id", "specimen", "subject", "entity", "material"} and value:
        return value
    text = f"{note.evidence_text}\n{note.claim}"
    match = re.search(r"\b(?:sample|specimen|subject|entity)\s*(?:is|:|=)\s*([^\r\n.;]+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _looks_like_date_evidence(note: EvidenceNote) -> bool:
    return any(term in _note_search_text(note) for term in ("date", "timestamp", "modified", "modification date"))


def _activity_measurement_signal(text: str) -> bool:
    return any(term in text for term in ("acquisition", "generation", "activity", "experiment", "method", "procedure"))


def _entity_measurement_signal(text: str) -> bool:
    return any(term in text for term in ("sample", "entity", "subject", "specimen", "observed", "measurement result"))


def _short_value_from_note(note: EvidenceNote) -> str | None:
    _key, value = _assignment_from_note(note)
    if value:
        return value
    claim = (note.claim or "").strip()
    return claim if claim and len(claim) <= 120 else None


def _normalize_identity(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def _merge_strings(existing: list[Any], incoming: list[Any]) -> list[str]:
    return _dedupe_strings([str(item).strip() for item in existing + incoming if str(item).strip()])


def _merge_dicts(existing: list[Any], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in existing + incoming:
        if not isinstance(item, dict) or not _has_meaningful_value(item):
            continue
        key = str(item.get("id") or item.get("title") or item.get("name") or item.get("value") or item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(_clone(item))
    return merged


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _dedupe_strings(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _has_meaningful_value(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_has_meaningful_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_meaningful_value(item) for item in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None


def _remove_empty_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _remove_empty_strings(item)) != ""
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _remove_empty_strings(item)) != ""]
    if isinstance(value, str):
        return value.strip()
    return value


def _clone(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return value
