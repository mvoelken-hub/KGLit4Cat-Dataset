from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.domain.datasources import DataPackage
from app.domain.extraction.agents import (
    validate_json_output_against_schema,
)
from app.domain.extraction.artifacts import InitialContext
from app.domain.extraction.schema_utils import (
    resolve_effective_schema as _resolve_effective_schema,
    resolve_ref as _resolve_ref,
    schema_allows_null as _schema_allows_null,
    schema_allows_type as _schema_allows_type,
)
from app.domain.profiles import ProfileManifest, validation_schema_for_target_class


INITIAL_DRAFT_SKELETON_MAX_DEPTH = 3


class InitialContextRequiredError(Exception):
    """Raised when InitialDraft is requested before InitialContext exists."""


async def initialize_draft_from_initial_context(
    *,
    initial_context: InitialContext,
    data_package: DataPackage,
    profile_manifest: ProfileManifest,
    profile_json_schema: dict[str, Any],
) -> dict[str, Any]:
    draft = build_initial_draft_from_context(
        initial_context=initial_context,
        data_package=data_package,
        profile_json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
    )
    expanded = expand_schema_placeholders(
        draft=draft,
        profile_json_schema=profile_json_schema,
        target_class=profile_manifest.target_class,
        max_depth=INITIAL_DRAFT_SKELETON_MAX_DEPTH,
    )
    return validate_json_output_against_schema(
        expanded,
        validation_schema_for_target_class(
            json_schema=profile_json_schema,
            target_class=profile_manifest.target_class,
        ),
    )


def build_initial_draft_from_context(
    *,
    initial_context: InitialContext,
    data_package: DataPackage,
    profile_json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    root_schema = validation_schema_for_target_class(
        json_schema=profile_json_schema,
        target_class=target_class,
    )
    target_schema = _resolve_effective_schema(root_schema, root_schema)
    properties = target_schema.get("properties")
    if not isinstance(properties, dict):
        return {}

    dataset_id = _dataset_id(data_package.file_name)
    draft: dict[str, Any] = {}
    _set_if_supported(
        draft,
        properties=properties,
        root_schema=root_schema,
        field_name="id",
        value=dataset_id,
    )
    _set_if_supported(
        draft,
        properties=properties,
        root_schema=root_schema,
        field_name="title",
        value=_draft_title(initial_context, data_package),
    )
    _set_if_supported(
        draft,
        properties=properties,
        root_schema=root_schema,
        field_name="description",
        value=initial_context.summary,
    )
    _set_if_supported(
        draft,
        properties=properties,
        root_schema=root_schema,
        field_name="keyword",
        value=initial_context.keywords,
    )
    _set_if_supported(
        draft,
        properties=properties,
        root_schema=root_schema,
        field_name="keywords",
        value=initial_context.keywords,
    )
    if initial_context.entities_analyzed:
        _set_if_supported(
            draft,
            properties=properties,
            root_schema=root_schema,
            field_name="is_about_entity",
            value=[
                _context_entity(
                    entity=entity,
                    field_schema=properties["is_about_entity"],
                    root_schema=root_schema,
                    dataset_id=dataset_id,
                )
                for entity in initial_context.entities_analyzed
            ]
            if "is_about_entity" in properties
            else [],
        )
    if (
        initial_context.analytical_technique
        or initial_context.device_name
        or initial_context.device_model
    ):
        _set_if_supported(
            draft,
            properties=properties,
            root_schema=root_schema,
            field_name="was_generated_by",
            value=[
                _context_activity(
                    initial_context=initial_context,
                    field_schema=properties["was_generated_by"],
                    root_schema=root_schema,
                    dataset_id=dataset_id,
                )
            ]
            if "was_generated_by" in properties
            else [],
        )

    for field_name in target_schema.get("required", []):
        if field_name not in draft and field_name in properties:
            draft[field_name] = _default_value_for_schema(
                properties[field_name],
                root_schema=root_schema,
                fallback=_fallback_text(field_name, data_package),
            )
    return draft


def expand_schema_placeholders(
    *,
    draft: dict[str, Any],
    profile_json_schema: dict[str, Any],
    target_class: str,
    max_depth: int = INITIAL_DRAFT_SKELETON_MAX_DEPTH,
) -> dict[str, Any]:
    """Add schema-valid null placeholders to an initial draft.

    The expander is intentionally conservative: it preserves inferred values,
    defaults missing nullable fields to null, and only recurses into nested
    object/list structures the agent already created.
    """

    root_schema = validation_schema_for_target_class(
        json_schema=profile_json_schema,
        target_class=target_class,
    )
    expanded = deepcopy(draft)
    target_schema = _resolve_effective_schema(root_schema, root_schema)
    if not isinstance(target_schema, dict):
        return expanded

    return _expand_value_for_schema(
        value=expanded,
        schema=target_schema,
        root_schema=root_schema,
        depth=1,
        max_depth=max_depth,
    )


def _expand_value_for_schema(
    *,
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    depth: int,
    max_depth: int,
) -> Any:
    effective_schema = _resolve_effective_schema(schema, root_schema)

    if isinstance(value, dict):
        return _expand_object_for_schema(
            value=value,
            schema=effective_schema,
            root_schema=root_schema,
            depth=depth,
            max_depth=max_depth,
        )

    if isinstance(value, list):
        item_schema = effective_schema.get("items")
        if not isinstance(item_schema, dict) or depth > max_depth:
            return value
        return [
            _expand_value_for_schema(
                value=item,
                schema=item_schema,
                root_schema=root_schema,
                depth=depth,
                max_depth=max_depth,
            )
            if isinstance(item, (dict, list))
            else item
            for item in value
        ]

    return value


def _context_entity(
    *,
    entity: str,
    field_schema: dict[str, Any],
    root_schema: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    item_schema = _array_item_schema(field_schema, root_schema)
    result = _object_shell(
        item_schema,
        root_schema=root_schema,
        fallback=entity,
    )
    properties = _schema_properties(item_schema, root_schema)
    entity_id = f"{dataset_id}/entity/{_slugify(entity)}"
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="id",
        value=entity_id,
    )
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="title",
        value=entity,
    )
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="name",
        value=entity,
    )
    return result


def _context_activity(
    *,
    initial_context: InitialContext,
    field_schema: dict[str, Any],
    root_schema: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    item_schema = _array_item_schema(field_schema, root_schema)
    title = _activity_title(initial_context)
    result = _object_shell(
        item_schema,
        root_schema=root_schema,
        fallback=title,
    )
    properties = _schema_properties(item_schema, root_schema)
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="id",
        value=f"{dataset_id}/activity/initial-context",
    )
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="title",
        value=title,
    )
    _set_if_supported(
        result,
        properties=properties,
        root_schema=root_schema,
        field_name="description",
        value=_activity_description(initial_context),
    )
    agent = _context_agent(
        initial_context,
        properties=properties,
        root_schema=root_schema,
        dataset_id=dataset_id,
    )
    if agent is not None:
        for field_name in ("agent", "carried_out_by"):
            _set_if_supported(
                result,
                properties=properties,
                root_schema=root_schema,
                field_name=field_name,
                value=[agent],
            )
    return result


def _context_agent(
    initial_context: InitialContext,
    *,
    properties: dict[str, Any],
    root_schema: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any] | None:
    label = initial_context.device_name or initial_context.device_model
    if not label:
        return None
    agent_field_schema = properties.get("agent") or properties.get("carried_out_by")
    if not isinstance(agent_field_schema, dict):
        return None
    item_schema = _array_item_schema(agent_field_schema, root_schema)
    result = _object_shell(item_schema, root_schema=root_schema, fallback=label)
    agent_props = _schema_properties(item_schema, root_schema)
    _set_if_supported(
        result,
        properties=agent_props,
        root_schema=root_schema,
        field_name="id",
        value=f"{dataset_id}/agent/{_slugify(label)}",
    )
    _set_if_supported(
        result,
        properties=agent_props,
        root_schema=root_schema,
        field_name="name",
        value=label,
    )
    _set_if_supported(
        result,
        properties=agent_props,
        root_schema=root_schema,
        field_name="title",
        value=label,
    )
    if initial_context.device_model and initial_context.device_model != label:
        _set_if_supported(
            result,
            properties=agent_props,
            root_schema=root_schema,
            field_name="description",
            value=f"Device model: {initial_context.device_model}",
        )
    return result


def _set_if_supported(
    document: dict[str, Any],
    *,
    properties: dict[str, Any],
    root_schema: dict[str, Any],
    field_name: str,
    value: Any,
) -> None:
    field_schema = properties.get(field_name)
    if not isinstance(field_schema, dict):
        return
    coerced = _coerce_value_for_schema(value, field_schema, root_schema)
    if coerced is not None or _schema_allows_null(field_schema, root_schema):
        document[field_name] = coerced


def _coerce_value_for_schema(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> Any:
    if value is None:
        if _schema_allows_null(schema, root_schema):
            return None
        return _default_value_for_schema(schema, root_schema=root_schema)
    effective = _resolve_effective_schema(schema, root_schema)
    if _schema_allows_type(effective, "array"):
        items_schema = effective.get("items")
        values = value if isinstance(value, list) else [value]
        if not isinstance(items_schema, dict):
            return values
        return [
            _coerce_value_for_schema(item, items_schema, root_schema)
            for item in values
            if item is not None
        ]
    if _schema_allows_type(effective, "object"):
        if isinstance(value, dict):
            return value
        return _object_shell(
            effective,
            root_schema=root_schema,
            fallback=str(value),
        )
    if _schema_allows_type(effective, "string"):
        if isinstance(value, list):
            value = next((item for item in value if item), "")
        return str(value)
    if _schema_allows_type(effective, "number") or _schema_allows_type(effective, "integer"):
        return value
    if _schema_allows_type(effective, "boolean"):
        return bool(value)
    return value


def _default_value_for_schema(
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    fallback: str = "",
) -> Any:
    if _schema_allows_null(schema, root_schema):
        return None
    effective = _resolve_effective_schema(schema, root_schema)
    if _schema_allows_type(effective, "array"):
        return []
    if _schema_allows_type(effective, "object"):
        return _object_shell(effective, root_schema=root_schema, fallback=fallback)
    if _schema_allows_type(effective, "string"):
        return fallback
    if _schema_allows_type(effective, "integer"):
        return 0
    if _schema_allows_type(effective, "number"):
        return 0
    if _schema_allows_type(effective, "boolean"):
        return False
    return None


def _object_shell(
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    fallback: str,
) -> dict[str, Any]:
    properties = _schema_properties(schema, root_schema)
    required = _resolve_effective_schema(schema, root_schema).get("required", [])
    result: dict[str, Any] = {}
    for field_name in required if isinstance(required, list) else []:
        field_schema = properties.get(field_name)
        if isinstance(field_schema, dict):
            result[field_name] = _default_value_for_schema(
                field_schema,
                root_schema=root_schema,
                fallback=fallback,
            )
    return result


def _schema_properties(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    effective = _resolve_effective_schema(schema, root_schema)
    properties = effective.get("properties")
    return properties if isinstance(properties, dict) else {}


def _array_item_schema(schema: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    effective = _resolve_effective_schema(schema, root_schema)
    items = effective.get("items")
    return _resolve_effective_schema(items, root_schema) if isinstance(items, dict) else {}


def _dataset_id(file_name: str) -> str:
    return f"dataset-{_slugify(file_name)}"


def _draft_title(initial_context: InitialContext, data_package: DataPackage) -> str:
    subject = ", ".join(initial_context.entities_analyzed[:2])
    if initial_context.analytical_technique and subject:
        return f"{_sentence_case(initial_context.analytical_technique)} dataset for {subject}"
    if initial_context.analytical_technique:
        return f"{_sentence_case(initial_context.analytical_technique)} dataset"
    if subject:
        return f"Dataset for {subject}"
    return data_package.file_name


def _activity_title(initial_context: InitialContext) -> str:
    if initial_context.analytical_technique:
        return f"{_sentence_case(initial_context.analytical_technique)} activity"
    return "Initial data-generating activity"


def _activity_description(initial_context: InitialContext) -> str:
    parts = []
    if initial_context.analytical_technique:
        parts.append(f"Analytical technique: {initial_context.analytical_technique}")
    if initial_context.device_name:
        parts.append(f"Device: {initial_context.device_name}")
    if initial_context.device_model:
        parts.append(f"Model: {initial_context.device_model}")
    return "; ".join(parts) if parts else initial_context.summary


def _fallback_text(field_name: str, data_package: DataPackage) -> str:
    if field_name == "id":
        return _dataset_id(data_package.file_name)
    return data_package.file_name


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "dataset"


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def _expand_object_for_schema(
    *,
    value: dict[str, Any],
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    depth: int,
    max_depth: int,
) -> dict[str, Any]:
    if depth > max_depth or not _schema_allows_type(schema, "object"):
        return value

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return value

    for property_name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue

        if property_name not in value:
            if _schema_allows_null(property_schema, root_schema):
                value[property_name] = None
            continue

        property_value = value[property_name]
        if property_value is None or depth >= max_depth:
            continue

        value[property_name] = _expand_value_for_schema(
            value=property_value,
            schema=property_schema,
            root_schema=root_schema,
            depth=depth + 1,
            max_depth=max_depth,
        )

    return value

# Schema utility functions are now imported from schema_utils:
#   _resolve_effective_schema, _resolve_ref, _schema_allows_null, _schema_allows_type
