"""Shared JSON Schema utilities for extraction agents.

Provides schema resolution, type checking, and field introspection used by
both the initial-draft expander and the field-level patch extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.profiles import validation_schema_for_target_class


@dataclass
class FieldInfo:
    """Metadata about a top-level field in the target class schema."""

    name: str
    schema: dict[str, Any]
    is_array: bool
    is_object: bool
    is_nullable: bool


def resolve_ref(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a ``$ref`` pointer inside *root_schema*.

    Follows ``#/defs/ClassName`` style references used by the LinkML
    JSON Schema generator.  Returns *schema* unchanged if it contains
    no ``$ref``.
    """
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema

    resolved: Any = root_schema
    for part in ref.removeprefix("#/").split("/"):
        if not isinstance(resolved, dict):
            return schema
        resolved = resolved.get(part)

    return resolved if isinstance(resolved, dict) else schema


def resolve_effective_schema(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    """Resolve ``$ref`` and collapse single-non-null ``anyOf``/``oneOf``.

    When a schema uses ``anyOf`` or ``oneOf`` with exactly one non-null
    option (the common nullable pattern), merge that option into the
    effective schema so callers can treat it as a plain type.
    """
    resolved = resolve_ref(schema, root_schema)

    for keyword in ("anyOf", "oneOf"):
        options = resolved.get(keyword)
        if not isinstance(options, list):
            continue
        non_null_options = [
            resolve_ref(option, root_schema)
            for option in options
            if isinstance(option, dict) and option.get("type") != "null"
        ]
        if len(non_null_options) == 1:
            merged = {
                key: value
                for key, value in resolved.items()
                if key not in {keyword, "type"}
            }
            merged.update(non_null_options[0])
            return resolve_effective_schema(merged, root_schema)

    return resolved


def schema_allows_null(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> bool:
    """Return ``True`` if *schema* accepts ``null`` as a value."""
    resolved = resolve_ref(schema, root_schema)
    schema_type = resolved.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True

    for keyword in ("anyOf", "oneOf"):
        options = resolved.get(keyword)
        if isinstance(options, list) and any(
            isinstance(option, dict)
            and schema_allows_null(option, root_schema)
            for option in options
        ):
            return True

    return False


def schema_allows_type(schema: dict[str, Any], schema_type: str) -> bool:
    """Return ``True`` if *schema* declares *schema_type* as an allowed type."""
    declared_type = schema.get("type")
    if declared_type == schema_type:
        return True
    if isinstance(declared_type, list) and schema_type in declared_type:
        return True
    return "properties" in schema if schema_type == "object" else False


def get_top_level_fields(
    *,
    profile_json_schema: dict[str, Any],
    target_class: str,
) -> list[FieldInfo]:
    """Return ordered metadata for each top-level property of *target_class*.

    Walks the ``properties`` of the resolved target class schema and
    classifies each field as array/object/nullable for downstream use
    by the patch extraction agent.
    """
    root_schema = validation_schema_for_target_class(
        json_schema=profile_json_schema,
        target_class=target_class,
    )
    target_schema = resolve_effective_schema(root_schema, root_schema)
    if not isinstance(target_schema, dict):
        return []

    properties = target_schema.get("properties")
    if not isinstance(properties, dict):
        return []

    fields: list[FieldInfo] = []
    for name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        effective = resolve_effective_schema(prop_schema, root_schema)

        is_nullable = schema_allows_null(prop_schema, root_schema)

        # Determine the "inner" type for arrays (unwrap items).
        inner_schema = effective
        is_array = False
        if effective.get("type") == "array" or (
            isinstance(effective.get("type"), list) and "array" in effective["type"]
        ):
            is_array = True
            items = effective.get("items")
            if isinstance(items, dict):
                inner_schema = resolve_effective_schema(items, root_schema)

        is_object = schema_allows_type(inner_schema, "object")

        fields.append(
            FieldInfo(
                name=name,
                schema=prop_schema,
                is_array=is_array,
                is_object=is_object,
                is_nullable=is_nullable,
            )
        )

    return fields