from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from jsonschema import Draft201909Validator

from app.domain.extraction.schema_utils import (
    resolve_effective_schema,
    resolve_ref,
    schema_allows_null,
)
from app.domain.profiles import validation_schema_for_target_class


FABRICATED_NFDI_ID_PATTERN = re.compile(
    r"^https://w3id\.org/nfdi-de/(?P<kind>activity|agent|entity|resource)/(?P<slug>[^/#?]+)$",
    re.IGNORECASE,
)


def normalize_review_draft(
    document: dict[str, Any],
    *,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Curate resolver output for stable draft shape and local identifiers."""

    normalized = deepcopy(document)
    scope = _slugify(dataset_id or str(normalized.get("id") or "dataset"))
    return _normalize_review_value(normalized, scope=scope, parent_key=None)


def sanitize_document_against_schema(
    *,
    document: dict[str, Any],
    json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    """Remove optional values that cannot validate against the profile schema."""

    root_schema = validation_schema_for_target_class(
        json_schema=json_schema,
        target_class=target_class,
    )
    target_schema = resolve_effective_schema(root_schema, root_schema)
    sanitized = _sanitize_value(
        deepcopy(document),
        target_schema,
        root_schema,
        required=True,
    )
    return sanitized if isinstance(sanitized, dict) else deepcopy(document)


def _normalize_review_value(value: Any, *, scope: str, parent_key: str | None) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "id" and isinstance(item, str):
                normalized[key] = _normalize_identifier(item, scope=scope)
            else:
                normalized[key] = _normalize_review_value(item, scope=scope, parent_key=key)
        return normalized

    if isinstance(value, list):
        normalized_items = [
            _normalize_review_value(item, scope=scope, parent_key=parent_key)
            for item in value
        ]
        if parent_key == "description" and all(
            isinstance(item, str) for item in normalized_items
        ):
            return _curate_description_list(normalized_items)
        return normalized_items

    return value


def _normalize_identifier(value: str, *, scope: str) -> str:
    match = FABRICATED_NFDI_ID_PATTERN.match(value)
    if match:
        kind = match.group("kind").lower()
        if kind == "resource":
            kind = "distribution"
        return f"{scope}/{kind}/{_slugify(match.group('slug'))}"

    if value.startswith(f"{scope}/"):
        return value
    for kind in ("activity", "agent", "entity", "distribution"):
        prefix = f"{kind}-"
        if value.lower().startswith(prefix):
            return f"{scope}/{kind}/{_slugify(value[len(prefix):])}"

    return value


def _curate_description_list(values: list[str]) -> list[str]:
    cleaned = [value.strip() for value in values if value.strip()]
    if not cleaned:
        return []

    result: list[str] = []
    for value in cleaned:
        replacement_index = _find_description_replacement_index(result, value)
        if replacement_index is None:
            if value not in result:
                result.append(value)
        else:
            result[replacement_index] = value
    return result


def _find_description_replacement_index(
    existing_values: list[str],
    candidate: str,
) -> int | None:
    candidate_tokens = _description_tokens(candidate)
    if not candidate_tokens:
        return None

    best_index: int | None = None
    best_overlap = 0.0
    for index, existing in enumerate(existing_values):
        existing_tokens = _description_tokens(existing)
        if not existing_tokens:
            continue
        overlap = len(candidate_tokens & existing_tokens) / len(existing_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    if best_index is None:
        return None
    existing = existing_values[best_index]
    if best_overlap >= 0.55 and len(candidate) >= len(existing):
        return best_index
    return None


def _description_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", value.lower())
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "for",
            "with",
            "using",
            "data",
            "description",
        }
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "item"


def _sanitize_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    required: bool,
) -> Any:
    if value is None:
        return None if required or schema_allows_null(schema, root_schema) else _Drop

    resolved = resolve_ref(schema, root_schema)
    union_options = _non_null_union_options(resolved, root_schema)
    if union_options:
        best = value
        for option in union_options:
            candidate = _sanitize_value(
                value,
                option,
                root_schema,
                required=required,
            )
            if candidate is _Drop:
                continue
            if _is_valid(candidate, option, root_schema):
                return candidate
            best = candidate
        return best if required else _Drop

    effective = resolve_effective_schema(resolved, root_schema)
    schema_type = effective.get("type")
    if schema_type == "object" or "properties" in effective:
        return _sanitize_object(value, effective, root_schema, required=required)
    if schema_type == "array":
        return _sanitize_array(value, effective, root_schema, required=required)
    if _is_valid(value, effective, root_schema):
        return value
    return value if required else _Drop


def _sanitize_object(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    required: bool,
) -> Any:
    if not isinstance(value, dict):
        return value if required else _Drop

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return value

    required_fields = set(schema.get("required", []))
    allow_extra = schema.get("additionalProperties", True) is not False
    sanitized: dict[str, Any] = {}

    for key, item in value.items():
        prop_schema = properties.get(key)
        if not isinstance(prop_schema, dict):
            if allow_extra:
                sanitized[key] = item
            continue

        cleaned = _sanitize_value(
            item,
            prop_schema,
            root_schema,
            required=key in required_fields,
        )
        if cleaned is _Drop:
            continue
        sanitized[key] = cleaned

    return sanitized if required or sanitized else _Drop


def _sanitize_array(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    *,
    required: bool,
) -> Any:
    if not isinstance(value, list):
        return value if required else _Drop

    item_schema = schema.get("items")
    if not isinstance(item_schema, dict):
        return value

    cleaned_items = []
    for item in value:
        cleaned = _sanitize_value(item, item_schema, root_schema, required=False)
        if cleaned is not _Drop and _is_valid(cleaned, item_schema, root_schema):
            cleaned_items.append(cleaned)

    return cleaned_items if required or cleaned_items else _Drop


def _non_null_union_options(
    schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            return [
                resolve_ref(option, root_schema)
                for option in options
                if isinstance(option, dict) and option.get("type") != "null"
            ]
    return []


def _is_valid(value: Any, schema: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    validation_schema = {
        "$schema": root_schema.get("$schema", "https://json-schema.org/draft/2019-09/schema"),
        "$defs": root_schema.get("$defs", {}),
        **schema,
    }
    return Draft201909Validator(validation_schema).is_valid(value)


class _DropType:
    pass


_Drop = _DropType()
