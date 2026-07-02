from __future__ import annotations

import json
import re
from hashlib import sha1
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.domain.extraction.overview import ExtractionFileSummary
from app.domain.extraction.overview import ExtractionOverview
from app.domain.extraction.file_ranking import RankedFile
from app.domain.extraction.workflow import ProjectionLedgerRecord, ProjectionLedgerStatus


DATASET_SUMMARY_SYSTEM_PROMPT = """
You write compact natural-language dataset accounts for DCAT Dataset metadata.
Return only JSON matching the supplied schema.
Use dense, sectioned fragments like caveman mode: high information, low filler.
Stay domain-agnostic: name generic roles/classes, keep concrete terms only as evidence labels.
Cover exactly: dataset catalog basics, one data-generating activity, its directly evaluated
entity/activity, inputs, outputs, instruments/software, method/plan, and compact characteristics.
Do not create Dataset is_about_entity/is_about_activity claims. Omit unsupported facts.
Treat SIMONE metadata extraction, profile projection, and this LLM workflow as out of scope:
they are not dataset-generating activities and must not appear in the summary.
"""


DATASET_LEVEL_PROJECTION_SYSTEM_PROMPT = """
You create dataset-level shallow DCAT-AP-plus Dataset metadata from a compact dataset account.
Return only JSON matching the supplied schema.
Use concise dataset-level metadata. Do not generate a Dataset description.
The backend uses the dataset summary as the Dataset description.
Infer dataset title, keywords, and one type at most from the dataset summary and ranked file summaries.
The title should name the dataset content or data product, not copy a single instrument/software sentence.
Populate creators only when supported.
Do not model SIMONE metadata extraction, profile projection, or this LLM workflow as a dataset
activity. was_generated_by is only for the scientific or data production activity that created
the source dataset content.
For this dataset-level projection, return was_generated_by as an empty list.
Do not populate Dataset is_about_entity/is_about_activity. DataGeneratingActivity evaluated targets
state what the activity directly measured, observed, or analysed. A merely generated output
belongs in had_output_entity, not evaluated_entity; inputs belong in had_input_entity/activity.
Prefer omission over invention for optional fields.
"""


OVERVIEW_SHALLOW_PROJECTION_SYSTEM_PROMPT = DATASET_LEVEL_PROJECTION_SYSTEM_PROMPT


class DatasetSummaryProjection(BaseModel):
    summary: str = Field(
        default="",
        description="Compressed sectioned natural-language account for DCAT Dataset metadata, about 400 tokens or less.",
    )


class ShallowResourceProjection(BaseModel):
    id: str | None = Field(
        default=None,
        description="Stable URI, path, or generated identifier for the resource. Omit when unknown.",
    )
    title: str | None = Field(default=None, description="Concise resource label.")
    description: str | None = Field(default=None, description="Short resource description.")

    @field_validator("title", "description", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        return _coerce_optional_text(value)


class ShallowConceptProjection(BaseModel):
    preferred_label: list[str] = Field(
        default_factory=list,
        description="Human-readable concept labels such as dataset type, method, or topic.",
    )
    title: str | None = Field(default=None, description="Optional concept title.")
    description: str | None = Field(default=None, description="Optional concept description.")


class ShallowAgentProjection(BaseModel):
    name: list[str] = Field(
        default_factory=list,
        description="Names of creators, organizations, labs, software agents, or people explicitly supported by summaries.",
    )
    type: ShallowConceptProjection | None = Field(
        default=None,
        description="Agent type, for example organization, person, software, or instrument vendor.",
    )


class ShallowAgenticEntityProjection(BaseModel):
    id: str | None = Field(default=None, description="Identifier for an actor, instrument, software, or organization.")
    title: str | None = Field(default=None, description="Concise actor or instrument name.")
    description: str | None = Field(default=None, description="Short actor or instrument description.")
    type: ShallowResourceProjection | None = Field(
        default=None,
        description="Actor/entity class such as instrument, software, organization, or laboratory.",
    )

    @field_validator("title", "description", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        return _coerce_optional_text(value)


class ShallowEvaluatedEntityProjection(BaseModel):
    id: str | None = Field(default=None, description="Identifier for sample, material, signal, file, or measured entity.")
    title: str | None = Field(default=None, description="Concise evaluated entity name.")
    description: str | None = Field(default=None, description="Short evaluated entity description.")
    type: ShallowResourceProjection | None = Field(
        default=None,
        description="Entity class such as sample, material, signal, or directly analysed input data.",
    )

    @field_validator("title", "description", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> Any:
        return _coerce_optional_text(value)


class ShallowEvaluatedActivityProjection(BaseModel):
    id: str | None = Field(default=None, description="Identifier for an activity or process being evaluated.")
    title: list[str] = Field(default_factory=list, description="Concise evaluated activity titles.")
    description: list[str] = Field(default_factory=list, description="Short evaluated activity descriptions.")
    type: ShallowResourceProjection | None = Field(
        default=None,
        description="Evaluated activity type such as reaction, process, protocol execution, or behavior.",
    )


class ShallowDataGeneratingActivityProjection(BaseModel):
    id: str | None = Field(default=None, description="Identifier for source-data acquisition, processing, or analysis activity.")
    title: list[str] = Field(default_factory=list, description="Concise activity titles.")
    description: list[str] = Field(default_factory=list, description="Short activity descriptions; avoid parameter dumps.")
    type: ShallowResourceProjection | None = Field(
        default=None,
        description="Activity type such as acquisition, processing, conversion, or analysis.",
    )
    carried_out_by: list[ShallowAgenticEntityProjection] = Field(
        default_factory=list,
        description="Instruments, software, organizations, or agents that carried out the activity.",
    )
    evaluated_entity: list[ShallowEvaluatedEntityProjection] = Field(
        default_factory=list,
        description="Entities directly measured, observed, or analysed by this activity; generated output alone is insufficient.",
    )
    evaluated_activity: list[ShallowEvaluatedActivityProjection] = Field(
        default_factory=list,
        description="Other activities or processes directly evaluated or studied by this activity; never the activity itself.",
    )
    had_input_entity: list[ShallowEvaluatedEntityProjection] = Field(
        default_factory=list,
        description="Input entities used by this activity, such as raw files, sample material, or source data.",
    )
    had_input_activity: list[ShallowEvaluatedActivityProjection] = Field(
        default_factory=list,
        description="Prior activities used as input, such as preparation, synthesis, or preprocessing.",
    )
    had_output_entity: list[ShallowEvaluatedEntityProjection] = Field(
        default_factory=list,
        description="Entities produced by this activity, such as generated files, spectra, processed results, or datasets.",
    )


class ShallowDatasetProjection(BaseModel):
    id: str | None = Field(default=None, description="Dataset identifier. Use package id if no better identifier is supported.")
    title: list[str] = Field(default_factory=list, description="Concise dataset title inferred from description and summaries.")
    description: list[str] = Field(
        default_factory=list,
        description="Meaningful 1-3 sentence dataset description. Never copy SIMONE scaffold placeholder text.",
    )
    identifier: list[str] = Field(default_factory=list, description="Dataset identifiers explicitly supported by package metadata.")
    keyword: list[str] = Field(default_factory=list, description="Broad search keywords: method, instrument, software, data type, domain.")
    creator: list[ShallowAgentProjection] = Field(default_factory=list, description="Creators or responsible agents explicitly supported.")
    type: list[ShallowConceptProjection] = Field(default_factory=list, description="At most one dataset type/topic concept.")
    modification_date: str | None = Field(default=None, description="Dataset-level modification date when explicitly supported.")
    was_generated_by: list[ShallowDataGeneratingActivityProjection] = Field(
        default_factory=list,
        description="Scientific/data-production activities that generated the source dataset content. Do not use for SIMONE metadata extraction or profile projection.",
    )


class ShallowDatasetLevelProjection(BaseModel):
    id: str | None = Field(default=None, description="Dataset identifier. Use package id if no better identifier is supported.")
    title: list[str] = Field(default_factory=list, description="Concise dataset title inferred from dataset summary.")
    identifier: list[str] = Field(default_factory=list, description="Dataset identifiers explicitly supported by package metadata.")
    keyword: list[str] = Field(default_factory=list, description="Broad search keywords: method, instrument, software, data type, domain.")
    creator: list[ShallowAgentProjection] = Field(default_factory=list, description="Creators or responsible agents explicitly supported.")
    type: list[ShallowConceptProjection] = Field(default_factory=list, description="At most one dataset type/topic concept.")
    modification_date: str | None = Field(default=None, description="Dataset-level modification date when explicitly supported.")
    was_generated_by: list[ShallowDataGeneratingActivityProjection] = Field(
        default_factory=list,
        description="Scientific/data-production activities that generated the source dataset content. Do not use for SIMONE metadata extraction or profile projection.",
    )


class ShallowDatasetLevelProjectionForPrompt(BaseModel):
    """Slimmed schema for LLM prompts: drops nested provenance defs.

    The dataset-level projection should always return was_generated_by: [] .
    Including the full ShallowDataGeneratingActivityProjection schema in the
    prompt wastes ~62% of the schema tokens on a field the model is told to
    leave empty. This model uses list[dict[str, Any]] instead, which keeps the
    field in the schema without pulling in 5 nested $defs.
    """
    id: str | None = Field(default=None, description="Dataset identifier. Use package id if no better identifier is supported.")
    title: list[str] = Field(default_factory=list, description="Concise dataset title inferred from dataset summary.")
    identifier: list[str] = Field(default_factory=list, description="Dataset identifiers explicitly supported by package metadata.")
    keyword: list[str] = Field(default_factory=list, description="Broad search keywords: method, instrument, software, data type, domain.")
    creator: list[ShallowAgentProjection] = Field(default_factory=list, description="Creators or responsible agents explicitly supported.")
    type: list[ShallowConceptProjection] = Field(default_factory=list, description="At most one dataset type/topic concept.")
    modification_date: str | None = Field(default=None, description="Dataset-level modification date when explicitly supported.")
    was_generated_by: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Always return an empty array. Provenance is filled by a later stage.",
    )


def build_dataset_summary_prompt_components(
    *,
    data_package_id: str,
    initial_file_summaries: list[ExtractionFileSummary],
    ranked_files: list[RankedFile],
) -> list[tuple[str, str]]:
    payload = {
        "data_package_id": data_package_id,
        "file_summaries": compact_file_summaries_for_shallow_projection(
            initial_file_summaries=initial_file_summaries,
            ranked_files=ranked_files,
        ),
    }
    return [
        (
            "task",
            "Write a compressed natural-language dataset summary for downstream DCAT Dataset projection. "
            "Use labeled fragments: dataset_subject, generated_by, activity_target, instruments, setting_plan, characteristics. "
            "Keep about 400 tokens or less. Domain-agnostic perspective; no exhaustive parameter dump. "
            "File summaries use one line per file: rank|path|fmt|pur|ev|meta|tool|num. "
            "Ignore SIMONE extraction/projection as workflow metadata, not dataset content. "
            "Keep descriptions concise.\n\n",
        ),
        ("dataset_summary_input_json", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


def build_dataset_level_projection_prompt_components(
    *,
    data_package_id: str,
    dataset_summary: str,
    skeleton: dict[str, Any],
    file_summaries: str | None = None,
) -> list[tuple[str, str]]:
    payload = {
        "data_package_id": data_package_id,
        "dataset_summary": dataset_summary,
        "required_skeleton": skeleton,
    }
    if file_summaries:
        payload["file_summaries"] = file_summaries
    return [
        (
            "task",
            "Infer dataset-level shallow DCAT-AP-plus metadata from dataset_summary and file_summaries. "
            "Do not return a description; the backend uses dataset_summary as the Dataset description. "
            "Infer title, keywords, and one dataset type from the combined context. "
            "Use required_skeleton only for mandatory shape; replace placeholder title. "
            "Do not treat SIMONE extraction/projection as a dataset-generating activity. "
            "Return was_generated_by as an empty list; provenance is filled by a later construction step. "
            "Do not create Dataset is_about_* relations. "
            "Nested resource/entity title and description are scalar strings. "
            "Prefer omission over unsupported invention.\n\n",
        ),
        ("dataset_level_projection_input_json", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


def build_overview_shallow_projection_prompt_components(
    *,
    data_package_id: str,
    initial_overview: ExtractionOverview | None,
    initial_file_summaries: list[ExtractionFileSummary],
    ranked_files: list[RankedFile],
    skeleton: dict[str, Any],
) -> list[tuple[str, str]]:
    del initial_overview
    dataset_summary = compact_file_summaries_for_shallow_projection(
        initial_file_summaries=initial_file_summaries,
        ranked_files=ranked_files,
    )
    return build_dataset_level_projection_prompt_components(
        data_package_id=data_package_id,
        dataset_summary=dataset_summary,
        skeleton=skeleton,
        file_summaries=dataset_summary,
    )


def compact_overview_for_shallow_projection(overview: ExtractionOverview | None) -> dict[str, Any]:
    if overview is None:
        return {"nodes": [], "edges": [], "uncertainties": []}
    return {
        "source_file_paths": overview.source_file_paths[:24],
        "nodes": [
            {
                "id": node.node_id,
                "label": node.label,
                "kind": node.kind,
                "path": node.file_path,
                "summary": _shorten(node.summary, 260),
            }
            for node in overview.nodes[:48]
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "relation": edge.relation,
                "evidence": [_shorten(item, 120) for item in edge.evidence[:3]],
                "note": _shorten(edge.note, 180),
            }
            for edge in overview.edges[:96]
        ],
        "uncertainties": [_shorten(item, 220) for item in overview.uncertainties[:12]],
    }


def compact_file_summaries_for_shallow_projection(
    *,
    initial_file_summaries: list[ExtractionFileSummary],
    ranked_files: list[RankedFile],
) -> str:
    rank_by_path = {item.file_path: item.rank for item in ranked_files}
    ranked = sorted(
        initial_file_summaries,
        key=lambda item: (rank_by_path.get(item.file_path, 9999), item.file_path),
    )
    return "\n".join(
        _compact_summary_line(summary, rank=rank_by_path.get(summary.file_path))
        for summary in ranked[:10]
    )


def shallow_required_skeleton(data_package_id: str, fallback_title: str | None = None) -> dict[str, Any]:
    return {
        "id": data_package_id,
        "title": _clean_string_list([fallback_title]),
        "description": [],
        "was_generated_by": [],
    }


def shallow_projection_to_dcat_document(
    projection: ShallowDatasetProjection,
    *,
    data_package_id: str,
    fallback_title: str | None = None,
    fallback_description: str | None = None,
) -> tuple[dict[str, Any], list[ProjectionLedgerRecord]]:
    records: list[ProjectionLedgerRecord] = []
    title = _first_nonempty(projection.title) or fallback_title or ""
    projection_description = [
        item
        for item in _clean_string_list(projection.description)
        if not _is_scaffold_description(item)
    ]
    description = _clean_string_list([fallback_description]) or projection_description
    if not description or all(_is_scaffold_description(item) for item in description):
        description = projection_description or ([title] if title else [])
    document: dict[str, Any] = {
        "id": _clean_string(projection.id)
        or _scaffold_id(data_package_id, "dataset", title or "dataset", records, "/id"),
        "title": _clean_string_list(projection.title) or ([title] if title else []),
        "description": description,
        "was_generated_by": [
            _activity_to_document(
                item,
                data_package_id=data_package_id,
                records=records,
                index=index,
                base_path="/was_generated_by",
            )
            for index, item in enumerate(projection.was_generated_by[:1])
        ],
    }
    if not document["was_generated_by"]:
        document["was_generated_by"] = [
            {"id": _scaffold_id(data_package_id, "activity", "dataset-generation", records, "/was_generated_by/0/id")}
        ]

    optional_values: dict[str, Any] = {
        "identifier": _clean_string_list(projection.identifier) or [data_package_id],
        "keyword": _clean_string_list(projection.keyword),
        "creator": [_agent_to_document(item) for item in projection.creator],
        "type": [_concept_to_document(item) for item in projection.type[:1]],
    }
    if _clean_string(projection.modification_date):
        optional_values["modification_date"] = _clean_string(projection.modification_date)

    for key, value in optional_values.items():
        cleaned = _remove_empty_values(value)
        if _has_meaningful_value(cleaned):
            document[key] = cleaned
    return _remove_empty_values(document), records


def overview_projection_record(
    *,
    status: ProjectionLedgerStatus,
    reason: str,
    error: str | None = None,
    projected_paths: list[str] | None = None,
) -> ProjectionLedgerRecord:
    return projection_stage_record(
        stage="dataset_level_projection",
        object_kind="DatasetLevelProjection",
        status=status,
        reason=reason,
        error=error,
        projected_paths=projected_paths,
    )


def projection_stage_record(
    *,
    stage: str,
    object_kind: str,
    status: ProjectionLedgerStatus,
    reason: str,
    error: str | None = None,
    projected_paths: list[str] | None = None,
) -> ProjectionLedgerRecord:
    return ProjectionLedgerRecord(
        object_identifier=stage,
        object_kind=object_kind,
        status=status,
        projected_paths=projected_paths or [],
        target_path="/",
        target_class="Dataset",
        planner_status=stage,
        reason=reason,
        error=error,
    )


def overview_projection_repair_record(
    *,
    status: ProjectionLedgerStatus,
    reason: str,
    error: str | None = None,
    projected_paths: list[str] | None = None,
) -> ProjectionLedgerRecord:
    return projection_stage_record(
        stage="dataset_level_projection_repair",
        object_kind="DatasetLevelProjectionRepair",
        status=status,
        reason=reason,
        error=error,
        projected_paths=projected_paths,
    )


def _activity_to_document(
    item: ShallowDataGeneratingActivityProjection,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    index: int,
    base_path: str,
) -> dict[str, Any]:
    item_path = f"{base_path}/{index}"
    label = _first_nonempty(item.title) or _first_nonempty(item.description) or f"activity-{index + 1}"
    doc = {
        "id": _clean_string(item.id)
        or _scaffold_id(data_package_id, "activity", label, records, f"{item_path}/id"),
        "title": _clean_string_list(item.title),
        "description": _clean_string_list(item.description),
        "type": _defined_term_to_document(
            item.type,
            data_package_id=data_package_id,
            records=records,
            path=f"{item_path}/type/id",
        ),
        "carried_out_by": [
            _agentic_entity_to_document(entity, data_package_id=data_package_id, records=records, index=i)
            for i, entity in enumerate(item.carried_out_by[:3])
        ],
        "evaluated_entity": [
            _entity_to_document(
                entity,
                data_package_id=data_package_id,
                records=records,
                index=i,
                base_path=f"{item_path}/evaluated_entity",
            )
            for i, entity in enumerate(item.evaluated_entity[:3])
        ],
        "evaluated_activity": [
            _evaluated_activity_to_document(
                activity,
                data_package_id=data_package_id,
                records=records,
                index=i,
                base_path=f"{item_path}/evaluated_activity",
            )
            for i, activity in enumerate(item.evaluated_activity[: max(0, 3 - len(item.evaluated_entity[:3]))])
        ],
        "had_input_entity": [
            _entity_to_document(
                entity,
                data_package_id=data_package_id,
                records=records,
                index=i,
                base_path=f"{item_path}/had_input_entity",
            )
            for i, entity in enumerate(item.had_input_entity[:3])
        ],
        "had_input_activity": [
            _evaluated_activity_to_document(
                activity,
                data_package_id=data_package_id,
                records=records,
                index=i,
                base_path=f"{item_path}/had_input_activity",
            )
            for i, activity in enumerate(item.had_input_activity[:3])
        ],
        "had_output_entity": [
            _entity_to_document(
                entity,
                data_package_id=data_package_id,
                records=records,
                index=i,
                base_path=f"{item_path}/had_output_entity",
            )
            for i, entity in enumerate(item.had_output_entity[:3])
        ],
    }
    return _remove_empty_values(doc)


def _evaluated_activity_to_document(
    item: ShallowEvaluatedActivityProjection,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    index: int,
    base_path: str,
) -> dict[str, Any]:
    item_path = f"{base_path}/{index}"
    label = _first_nonempty(item.title) or _first_nonempty(item.description) or f"evaluated-activity-{index + 1}"
    return _remove_empty_values(
        {
            "id": _clean_string(item.id)
            or _scaffold_id(
                data_package_id,
                "evaluated-activity",
                label,
                records,
                f"{item_path}/id",
                identity_scope=item_path,
            ),
            "title": _clean_string_list(item.title),
            "description": _clean_string_list(item.description),
            "type": _defined_term_to_document(
                item.type,
                data_package_id=data_package_id,
                records=records,
                path=f"{item_path}/type/id",
            ),
        }
    )


def _agentic_entity_to_document(
    item: ShallowAgenticEntityProjection,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    index: int,
) -> dict[str, Any]:
    label = _clean_string(item.title) or _clean_string(item.description) or f"agentic-entity-{index + 1}"
    return _remove_empty_values(
        {
            "id": _clean_string(item.id)
            or _scaffold_id(data_package_id, "agentic-entity", label, records, f"/agentic_entity/{index}/id"),
            "title": _clean_string(item.title),
            "description": _clean_string(item.description),
            "type": _defined_term_to_document(item.type, data_package_id=data_package_id, records=records, path=f"/agentic_entity/{index}/type/id"),
        }
    )


def _entity_to_document(
    item: ShallowEvaluatedEntityProjection,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    index: int,
    base_path: str,
) -> dict[str, Any]:
    item_path = f"{base_path}/{index}"
    label = _clean_string(item.title) or _clean_string(item.description) or f"entity-{index + 1}"
    return _remove_empty_values(
        {
            "id": _clean_string(item.id)
            or _scaffold_id(
                data_package_id,
                "entity",
                label,
                records,
                f"{item_path}/id",
                identity_scope=item_path,
            ),
            "title": _clean_string(item.title),
            "description": _clean_string(item.description),
            "type": _defined_term_to_document(
                item.type,
                data_package_id=data_package_id,
                records=records,
                path=f"{item_path}/type/id",
            ),
        }
    )


def _agent_to_document(item: ShallowAgentProjection) -> dict[str, Any]:
    names = _clean_string_list(item.name)
    if not names:
        return {}
    return _remove_empty_values({"name": names, "type": _concept_to_document(item.type) if item.type else None})


def _concept_to_document(item: ShallowConceptProjection | None) -> dict[str, Any]:
    if item is None:
        return {}
    labels = _clean_string_list(item.preferred_label)
    if not labels:
        labels = _clean_string_list([item.title])
    if not labels:
        return {}
    return _remove_empty_values(
        {
            "preferred_label": labels,
            "title": _clean_string(item.title),
            "description": _clean_string(item.description),
        }
    )


def _resource_to_document(
    item: ShallowResourceProjection,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    path: str,
    kind: str,
) -> dict[str, Any]:
    label = _clean_string(item.title) or _clean_string(item.id) or kind
    return _remove_empty_values(
        {
            "id": _clean_string(item.id)
            or _scaffold_id(data_package_id, "resource", label, records, path),
            "title": _clean_string(item.title),
            "description": _clean_string(item.description),
        }
    )


def _term_resource_to_document(item: ShallowResourceProjection | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return _remove_empty_values({"title": _clean_string(item.title), "description": _clean_string(item.description)}) or None


def _defined_term_to_document(
    item: ShallowResourceProjection | None,
    *,
    data_package_id: str,
    records: list[ProjectionLedgerRecord],
    path: str,
) -> dict[str, Any] | None:
    if item is None:
        return None
    label = _clean_string(item.title) or _clean_string(item.id) or "type"
    return _remove_empty_values(
        {
            "id": _clean_string(item.id)
            or _scaffold_id(data_package_id, "defined-term", label, records, path),
            "title": _clean_string(item.title),
        }
    ) or None


def _scaffold_id(
    data_package_id: str,
    kind: str,
    label: str,
    records: list[ProjectionLedgerRecord],
    path: str,
    identity_scope: str | None = None,
) -> str:
    normalized = _slug(label) or kind
    digest = sha1(
        f"{data_package_id}|{kind}|{label}|{identity_scope or ''}".encode("utf-8")
    ).hexdigest()[:8]
    value = f"{data_package_id}:{kind}:{normalized}:{digest}"
    records.append(
        ProjectionLedgerRecord(
            object_identifier=f"scaffold:{path}",
            object_kind="ScaffoldFact",
            status="projected",
            projected_paths=[path],
            target_path=path,
            planner_status="scaffold",
            reason="Backend filled missing required shallow projection identifier.",
        )
    )
    return value


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


def _coerce_optional_text(value: Any) -> str | None:
    if isinstance(value, list):
        return _first_nonempty(value)
    return _clean_string(value)


def _clean_string_list(values: list[Any] | Any) -> list[str]:
    if values is None:
        return []
    raw_values = values if isinstance(values, list) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        cleaned = _clean_string(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _first_nonempty(values: list[Any] | Any) -> str | None:
    return next(iter(_clean_string_list(values)), None)


def _remove_empty_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _remove_empty_values(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := _remove_empty_values(item)) not in (None, "", [], {})
        ]
    if isinstance(value, str):
        return value.strip()
    return value


def _has_meaningful_value(value: Any) -> bool:
    return _remove_empty_values(value) not in (None, "", [], {})


def _is_scaffold_description(value: str) -> bool:
    return value.strip().lower().startswith("simone metadata draft for ")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64]


def _compact_summary_line(summary: ExtractionFileSummary, *, rank: int | None) -> str:
    parts = [f"r{rank}" if rank is not None else "r?", _compact_signal(summary.file_path, 70)]
    if summary.data_format:
        parts.append(f"fmt={_compact_signal(summary.data_format, 50)}")
    if summary.explicit_purpose:
        parts.append(f"pur={_compact_signal(summary.explicit_purpose, 70)}")
    if summary.information_summary:
        parts.append(f"summary={_compact_signal(summary.information_summary, 180)}")
    return "|".join(parts)


def _compact_signal(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").replace("|", "/").replace(";", ",").strip()
    return _shorten(cleaned, limit)


def _shorten(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value or "").strip()
    return cleaned[:limit]
