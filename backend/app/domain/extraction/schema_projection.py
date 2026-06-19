from __future__ import annotations

import re
from typing import Any

from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.loaders import yaml_loader
from linkml_runtime.utils.schemaview import SchemaView
from pydantic import BaseModel, Field

from app.domain.extraction.evidence_context import EvidenceCandidate


class SchemaSubclassCandidate(BaseModel):
    class_name: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    exact_mappings: list[str] = Field(default_factory=list)
    ancestors: list[str] = Field(default_factory=list)


class SchemaBranch(BaseModel):
    path: str
    traversal_path: str
    slot_name: str
    slot_description: str = ""
    range_class: str | None = None
    range_description: str = ""
    required: bool = False
    recommended: bool = False
    multivalued: bool = False
    inlined_as_list: bool = False
    ancestors: list[str] = Field(default_factory=list)
    subclass_candidates: list[SchemaSubclassCandidate] = Field(default_factory=list)
    depth: int = 0
    shape: dict[str, Any] = Field(default_factory=dict)


class SchemaSearchQuery(BaseModel):
    text: str
    semantic_hints: list[str] = Field(default_factory=list)
    max_depth: int = 3


class SchemaSearchResult(BaseModel):
    query: SchemaSearchQuery
    candidates: list[SchemaBranch] = Field(default_factory=list)


class SchemaBranchDecision(BaseModel):
    status: str = "targeted"
    branch: SchemaBranch | None = None
    reason: str = ""


class TypedSubObjectWrite(BaseModel):
    status: str = "write"
    path: str
    value: Any = None
    reason: str = ""


def build_schema_branch_index(
    merged_schema: str,
    *,
    target_class: str,
    max_depth: int = 3,
) -> list[SchemaBranch]:
    schema = yaml_loader.loads(merged_schema, target_class=SchemaDefinition)
    schema_view = SchemaView(schema)
    subclasses_by_parent = _subclasses_by_parent(schema_view)
    branches: list[SchemaBranch] = []

    def walk(class_name: str, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            slots = schema_view.class_induced_slots(class_name)
        except Exception:
            return
        for slot in slots:
            slot_name = slot.name
            if not slot_name:
                continue
            slot_range = slot.range
            multivalued = bool(slot.multivalued)
            slot_path = f"{path}/{slot_name}" if path else f"/{slot_name}"
            traversal_path = f"{slot_path}/0" if multivalued and _is_class_range(schema_view, slot_range) else slot_path
            branch_path = f"{slot_path}/-" if multivalued and _is_class_range(schema_view, slot_range) else traversal_path
            range_class = slot_range if _is_class_range(schema_view, slot_range) else None
            subclass_candidates = [
                _subclass_candidate(schema_view, subclass)
                for subclass in subclasses_by_parent.get(range_class or "", [])
            ]
            branches.append(
                SchemaBranch(
                    path=branch_path,
                    traversal_path=traversal_path,
                    slot_name=slot_name,
                    slot_description=slot.description or "",
                    range_class=range_class,
                    range_description=_class_description(schema_view, range_class),
                    required=bool(slot.required),
                    recommended=bool(slot.recommended),
                    multivalued=multivalued,
                    inlined_as_list=bool(slot.inlined_as_list),
                    ancestors=list(schema_view.class_ancestors(range_class)) if range_class else [],
                    subclass_candidates=subclass_candidates,
                    depth=depth,
                    shape={
                        "range": slot_range,
                        "required": bool(slot.required),
                        "recommended": bool(slot.recommended),
                        "multivalued": multivalued,
                        "inlined_as_list": bool(slot.inlined_as_list),
                    },
                )
            )
            if range_class and depth < max_depth:
                walk(range_class, traversal_path, depth + 1)

    walk(target_class, "", 0)
    return _dedupe_branches(branches)


def build_schema_search_query(
    notes: list[EvidenceCandidate],
    *,
    max_depth: int = 3,
) -> SchemaSearchQuery:
    text = " ".join(
        part
        for note in notes
        for part in (
            note.candidate_id,
            note.category,
            note.claim,
            note.evidence_text,
            note.file_path,
        )
        if part
    )
    return SchemaSearchQuery(
        text=text,
        semantic_hints=_semantic_hints_from_text(text),
        max_depth=max_depth,
    )


def search_schema_branches(
    branches: list[SchemaBranch],
    query: SchemaSearchQuery,
    *,
    top_k: int = 8,
) -> SchemaSearchResult:
    scored = [
        (score_schema_branch(branch, query), branch)
        for branch in branches
        if branch.depth <= query.max_depth
    ]
    candidates = [
        branch
        for score, branch in sorted(scored, key=lambda item: item[0], reverse=True)
        if score > 0
    ][:top_k]
    return SchemaSearchResult(query=query, candidates=candidates)


def score_schema_branch(branch: SchemaBranch, query: SchemaSearchQuery) -> int:
    text = _normalize(query.text)
    branch_text = _normalize(
        " ".join(
            [
                branch.path,
                branch.slot_name,
                branch.slot_description,
                branch.range_class or "",
                branch.range_description,
                " ".join(branch.ancestors),
                " ".join(
                    " ".join(
                        [
                            candidate.class_name,
                            candidate.description,
                            " ".join(candidate.aliases),
                            " ".join(candidate.exact_mappings),
                        ]
                    )
                    for candidate in branch.subclass_candidates
                ),
            ]
        )
    )
    score = 0
    semantic_hints = set(query.semantic_hints)
    if "device" in semantic_hints:
        if any(candidate.class_name == "Device" for candidate in branch.subclass_candidates):
            score += 70
        if branch.range_class == "AgenticEntity":
            score += 30
        if branch.slot_name == "carried_out_by":
            score += 45
    if "method" in semantic_hints:
        if branch.slot_name in {"was_generated_by", "is_about_activity", "realized_plan"}:
            score += 30
        if branch.range_class in {"DataGeneratingActivity", "EvaluatedActivity", "Plan"}:
            score += 20
    if "file_format" in semantic_hints:
        if branch.slot_name in {"dataset_distribution", "format", "media_type", "type"}:
            score += 35
    if "dataset_identity" in semantic_hints:
        if branch.slot_name in {"title", "identifier"}:
            score += 40
    if "measured_entity" in semantic_hints:
        if branch.slot_name in {"evaluated_entity", "is_about_entity"}:
            score += 35
        if branch.range_class in {"EvaluatedEntity", "Entity"}:
            score += 20
    if "date" in semantic_hints and branch.slot_name in {"modification_date", "release_date"}:
        score += 35
    for token in _query_tokens(text):
        if token in branch_text:
            score += 2
    if branch.recommended:
        score += 5
    score -= branch.depth * 20
    if branch.slot_name in {"description", "title", "keyword"} and not semantic_hints & {"dataset_identity"}:
        score -= 25
    if "parameter_setting" in semantic_hints and not semantic_hints - {"parameter_setting"}:
        score -= 25
    return score


def _semantic_hints_from_text(text: str) -> list[str]:
    normalized = _normalize(text)
    hints: set[str] = set()
    if any(term in normalized for term in ("dataset name", "dataset title", "study title", "package title")):
        hints.add("dataset_identity")
    if any(term in normalized for term in ("instrument", "device", "equipment", "sensor", "apparatus")):
        hints.add("device")
    if any(term in normalized for term in ("file format", "format is", "media type", "distribution")):
        hints.add("file_format")
    if re.search(r"\b(method|procedure|workflow|protocol|acquisition|generation)\b", normalized):
        hints.add("method")
    if any(term in normalized for term in ("sample", "specimen", "material", "subject", "target entity")):
        hints.add("measured_entity")
    if any(term in normalized for term in ("identifier", "checksum", " id ")):
        hints.add("identifier")
    if any(term in normalized for term in ("date", "timestamp", "modified", "modification")):
        hints.add("date")
    if "parameter" in normalized or re.search(r"##\$?[a-z][a-z0-9_]{1,32}\s*=", normalized):
        hints.add("parameter_setting")
    if any(term in normalized for term in ("quality", "uncertain", "warning")):
        hints.add("quality_signal")
    return sorted(hints)


def schema_branches_to_catalog(branches: list[SchemaBranch]) -> list[dict[str, Any]]:
    return [
        {
            "path": branch.path,
            "label": branch.slot_name.replace("_", " "),
            "field_name": branch.path.strip("/").split("/")[0],
            "target_class": branch.range_class,
            "description": branch.slot_description,
            "current_value": None,
            "scaffold_status": "schema_branch",
            "category_affinities": _branch_category_affinities(branch),
            "description_last_resort": branch.slot_name == "description",
            "schema_branch": branch.model_dump(mode="json"),
        }
        for branch in branches
    ]


def _branch_category_affinities(branch: SchemaBranch) -> list[str]:
    affinities: list[str] = []
    if branch.range_class == "AgenticEntity" or any(
        candidate.class_name == "Device" for candidate in branch.subclass_candidates
    ):
        affinities.append("agent_signal")
    if branch.range_class in {"DataGeneratingActivity", "EvaluatedActivity"}:
        affinities.append("activity_signal")
    if branch.range_class == "Plan":
        affinities.append("method_signal")
    if branch.range_class in {"EvaluatedEntity", "Entity"}:
        affinities.append("activity_signal")
    if branch.slot_name in {"dataset_distribution", "format", "media_type"}:
        affinities.append("resource_signal")
    return sorted(set(affinities))


def _subclasses_by_parent(schema_view: SchemaView) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for class_name, class_def in schema_view.all_classes().items():
        parent = class_def.is_a
        if parent:
            mapping.setdefault(parent, []).append(class_name)
    return mapping


def _subclass_candidate(schema_view: SchemaView, class_name: str) -> SchemaSubclassCandidate:
    class_def = schema_view.get_class(class_name)
    return SchemaSubclassCandidate(
        class_name=class_name,
        description=class_def.description or "",
        aliases=list(class_def.aliases or []),
        exact_mappings=list(class_def.exact_mappings or []),
        ancestors=list(schema_view.class_ancestors(class_name)),
    )


def _is_class_range(schema_view: SchemaView, class_name: str | None) -> bool:
    return bool(class_name and schema_view.get_class(class_name) is not None)


def _class_description(schema_view: SchemaView, class_name: str | None) -> str:
    if not class_name:
        return ""
    class_def = schema_view.get_class(class_name)
    return class_def.description or "" if class_def else ""


def _dedupe_branches(branches: list[SchemaBranch]) -> list[SchemaBranch]:
    seen: set[str] = set()
    deduped: list[SchemaBranch] = []
    for branch in branches:
        if branch.path in seen:
            continue
        seen.add(branch.path)
        deduped.append(branch)
    return deduped


def _query_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token not in {"the", "and", "with", "that", "this", "signal", "note"}
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())
