from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PatchWriteMode = Literal["append", "replace", "remove", "merge"]


class SchemaConstrainedWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_path: str
    mode: PatchWriteMode
    items: list[Any] = Field(default_factory=list)
    value: Any = None
    survivor_index: int | None = None
    merged_indices: list[int] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def require_payload_for_mode(self) -> "SchemaConstrainedWrite":
        if self.mode == "append" and not self.items:
            raise ValueError("append write requires at least one item")
        if self.mode == "replace" and "value" not in self.model_fields_set:
            raise ValueError("replace write requires value")
        if self.mode == "merge":
            if self.survivor_index is None:
                raise ValueError("merge write requires survivor_index")
            if not self.merged_indices:
                raise ValueError("merge write requires merged_indices")
            if self.survivor_index in self.merged_indices:
                raise ValueError("merge write cannot merge survivor_index into itself")
        return self


class SchemaConstrainedPatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    writes: list[SchemaConstrainedWrite] = Field(default_factory=list)
    reason: str = ""


def build_schema_constrained_patch_schema(
    *,
    validation_schema: dict[str, Any],
    allowed_target_paths: list[str],
) -> dict[str, Any]:
    canonical_paths = _dedupe_paths(allowed_target_paths)
    branches = [
        _write_branch_schema(validation_schema, target_path)
        for target_path in canonical_paths
    ]
    branches = [branch for branch in branches if branch]
    defs = _reachable_defs(validation_schema, branches)
    return {
        "$schema": validation_schema.get(
            "$schema",
            "https://json-schema.org/draft/2019-09/schema",
        ),
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "writes": {
                "type": "array",
                "items": {"oneOf": branches} if branches else False,
            },
            "reason": {"type": "string"},
        },
        "required": ["writes", "reason"],
        "$defs": defs,
    }


def parse_schema_constrained_patch_result(output: Any) -> SchemaConstrainedPatchResult:
    result = SchemaConstrainedPatchResult.model_validate(output)
    result.writes = [_normalize_write_target_path(write) for write in result.writes]
    return result


def apply_schema_constrained_writes(
    *,
    document: dict[str, Any],
    writes: list[SchemaConstrainedWrite],
    data_package_id: str,
    validation_schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    updated = deepcopy(document)
    changed_paths: list[str] = []
    for write in writes:
        write = _normalize_write_target_path(write)
        if write.mode == "append":
            array_path = _array_path(write.target_path)
            _ensure_container_path(
                updated,
                validation_schema=validation_schema,
                path=array_path,
                data_package_id=data_package_id,
            )
            array_value = _value_at_json_pointer(updated, array_path)
            if not isinstance(array_value, list):
                raise ValueError(f"Target path is not an array: {array_path}")
            item_schema = _array_item_schema(validation_schema, array_path)
            for item in write.items:
                index = len(array_value)
                normalized = _normalize_instance_ids(
                    item,
                    data_package_id=data_package_id,
                    target_path=write.target_path,
                    index=index,
                    target_schema=item_schema,
                )
                array_value.append(deepcopy(normalized))
                changed_paths.append(f"{array_path}/{index}")
            continue

        if write.mode == "remove":
            updated = _remove_json_pointer_value(updated, write.target_path)
            changed_paths.append(write.target_path)
            continue

        if write.mode == "merge":
            updated, merge_paths = _merge_json_pointer_array_items(
                updated,
                write.target_path,
                survivor_index=write.survivor_index,
                merged_indices=write.merged_indices,
            )
            changed_paths.extend(merge_paths)
            continue

        _ensure_container_path(
            updated,
            validation_schema=validation_schema,
            path=_parent_path(write.target_path),
            data_package_id=data_package_id,
        )
        value_schema = schema_for_json_pointer(validation_schema, write.target_path)
        value = _normalize_instance_ids(
            write.value,
            data_package_id=data_package_id,
            target_path=write.target_path,
            index=0,
            target_schema=value_schema,
        )
        updated = _set_json_pointer_value(updated, write.target_path, deepcopy(value))
        changed_paths.append(write.target_path)
    return updated, changed_paths


def schema_for_json_pointer(validation_schema: dict[str, Any], path: str) -> dict[str, Any]:
    current = resolve_schema_node(validation_schema, validation_schema)
    for raw_part in path.strip("/").split("/") if path.strip("/") else []:
        part = _json_pointer_unescape(raw_part)
        current = resolve_schema_node(current, validation_schema)
        if not isinstance(current, dict):
            return {}
        if part.isdigit():
            current = current.get("items", {})
            continue
        properties = current.get("properties", {})
        if not isinstance(properties, dict):
            return {}
        current = properties.get(part, {})
    current = resolve_schema_node(current, validation_schema)
    return current if isinstance(current, dict) else {}


def resolve_schema_node(node: Any, root: dict[str, Any]) -> Any:
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), node)
    return node


def _write_branch_schema(
    validation_schema: dict[str, Any],
    target_path: str,
) -> dict[str, Any] | None:
    target_path = _canonical_target_path(target_path)
    schema_path = _array_path(target_path) if target_path.endswith("/-") else target_path
    target_schema = schema_for_json_pointer(validation_schema, schema_path)
    if not target_schema:
        return None
    if _schema_is_array(target_schema, validation_schema):
        item_schema = target_schema.get("items", {})
        return {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_path": {"const": target_path},
                        "mode": {"const": "append"},
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            "items": deepcopy(item_schema),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["target_path", "mode", "items", "reason"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_path": {"const": target_path},
                        "mode": {"const": "replace"},
                        "value": deepcopy(target_schema),
                        "reason": {"type": "string"},
                    },
                    "required": ["target_path", "mode", "value", "reason"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_path": {"const": target_path},
                        "mode": {"const": "remove"},
                        "reason": {"type": "string"},
                    },
                    "required": ["target_path", "mode", "reason"],
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "target_path": {"const": target_path},
                        "mode": {"const": "merge"},
                        "survivor_index": {"type": "integer", "minimum": 0},
                        "merged_indices": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "integer", "minimum": 0},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["target_path", "mode", "survivor_index", "merged_indices", "reason"],
                },
            ],
        }
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_path": {"const": target_path},
                    "mode": {"const": "replace"},
                    "value": deepcopy(target_schema),
                    "reason": {"type": "string"},
                },
                "required": ["target_path", "mode", "value", "reason"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_path": {"const": target_path},
                    "mode": {"const": "remove"},
                    "reason": {"type": "string"},
                },
                "required": ["target_path", "mode", "reason"],
            },
        ],
    }


def _reachable_defs(root: dict[str, Any], schema_nodes: list[Any]) -> dict[str, Any]:
    all_defs = root.get("$defs", {})
    if not isinstance(all_defs, dict):
        return {}
    needed: set[str] = set()
    stack = list(schema_nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.removeprefix("#/$defs/")
                if name not in needed and name in all_defs:
                    needed.add(name)
                    stack.append(all_defs[name])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return {name: deepcopy(all_defs[name]) for name in sorted(needed)}


def _ensure_container_path(
    document: dict[str, Any],
    *,
    validation_schema: dict[str, Any],
    path: str,
    data_package_id: str,
) -> None:
    if not path or path == "/":
        return
    current: Any = document
    parts = [_json_pointer_unescape(part) for part in path.strip("/").split("/")]
    for index, part in enumerate(parts):
        current_path = "/" + "/".join(_json_pointer_escape(p) for p in parts[: index + 1])
        node_schema = schema_for_json_pointer(validation_schema, current_path)
        if part.isdigit():
            if not isinstance(current, list):
                raise ValueError(f"Parent path is not an array: {current_path}")
            item_index = int(part)
            while len(current) <= item_index:
                current.append(_schema_default_value(node_schema, validation_schema, data_package_id, current_path))
            current = current[item_index]
            continue
        if not isinstance(current, dict):
            raise ValueError(f"Parent path is not an object: {current_path}")
        if part not in current or current[part] is None:
            current[part] = _schema_default_value(
                node_schema,
                validation_schema,
                data_package_id,
                current_path,
            )
        current = current[part]


def _schema_default_value(
    schema: dict[str, Any],
    root: dict[str, Any],
    data_package_id: str,
    path: str,
) -> Any:
    schema = resolve_schema_node(schema, root)
    if _schema_is_array(schema, root):
        return []
    if isinstance(schema, dict) and schema.get("type") == "object":
        value: dict[str, Any] = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if isinstance(properties, dict) and isinstance(required, list):
            for field in required:
                field_schema = resolve_schema_node(properties.get(field, {}), root)
                if field == "id":
                    value[field] = _build_patch_id(data_package_id, path)
                else:
                    value[field] = _schema_default_value(field_schema, root, data_package_id, f"{path}/{field}")
        return value
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(schema_type, list):
        schema_type = next((item for item in schema_type if item != "null"), None)
    if schema_type == "string":
        return ""
    if schema_type == "number":
        return 0
    if schema_type == "integer":
        return 0
    if schema_type == "boolean":
        return False
    return {}


def _schema_is_array(schema: dict[str, Any], root: dict[str, Any]) -> bool:
    schema = resolve_schema_node(schema, root)
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    return schema_type == "array" or (
        isinstance(schema_type, list) and "array" in schema_type
    )


def _array_item_schema(validation_schema: dict[str, Any], array_path: str) -> dict[str, Any]:
    schema = schema_for_json_pointer(validation_schema, array_path)
    if isinstance(schema.get("items"), dict):
        return resolve_schema_node(schema["items"], validation_schema)
    return {}


def _normalize_instance_ids(
    instance: Any,
    *,
    data_package_id: str,
    target_path: str,
    index: int,
    target_schema: dict[str, Any] | None = None,
) -> Any:
    if not isinstance(instance, dict):
        return instance
    value = deepcopy(instance)
    allowed_properties = (
        set(target_schema.get("properties", {}).keys())
        if isinstance(target_schema, dict)
        else set()
    )
    if allowed_properties and "id" not in allowed_properties:
        value.pop("id", None)
    if (not allowed_properties or "id" in allowed_properties) and not _non_empty_string(value.get("id")):
        label = _instance_label(value) or f"patched-{index}"
        value["id"] = _build_patch_id(data_package_id, f"{target_path}/{label}")
    for key, child in list(value.items()):
        if isinstance(child, dict) and not _non_empty_string(child.get("id")):
            child["id"] = _build_patch_id(data_package_id, f"{target_path}/{key}")
        elif isinstance(child, list):
            for child_index, item in enumerate(child):
                if isinstance(item, dict) and not _non_empty_string(item.get("id")):
                    item["id"] = _build_patch_id(data_package_id, f"{target_path}/{key}/{child_index}")
    return value


def _value_at_json_pointer(document: Any, path: str) -> Any:
    if not path or path == "/":
        return document
    current = document
    for raw_part in path.strip("/").split("/"):
        part = _json_pointer_unescape(raw_part)
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _set_json_pointer_value(document: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    if not path or path == "/":
        if not isinstance(value, dict):
            raise ValueError("Root replacement requires an object value.")
        return value
    result = deepcopy(document)
    current: Any = result
    parts = [_json_pointer_unescape(part) for part in path.strip("/").split("/")]
    for part in parts[:-1]:
        if isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise ValueError(f"Cannot traverse JSON Pointer path '{path}'.")
    leaf = parts[-1]
    if isinstance(current, dict):
        current[leaf] = value
    elif isinstance(current, list) and leaf.isdigit():
        index = int(leaf)
        while len(current) <= index:
            current.append(None)
        current[index] = value
    else:
        raise ValueError(f"Cannot set JSON Pointer path '{path}'.")
    return result


def _remove_json_pointer_value(document: dict[str, Any], path: str) -> dict[str, Any]:
    if not path or path == "/":
        raise ValueError("Root removal is not supported.")
    result = deepcopy(document)
    current: Any = result
    parts = [_json_pointer_unescape(part) for part in path.strip("/").split("/")]
    for part in parts[:-1]:
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise ValueError(f"Cannot traverse JSON Pointer path '{path}'.")
            current = current[index]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Cannot traverse JSON Pointer path '{path}'.")
    leaf = parts[-1]
    if isinstance(current, dict):
        if leaf not in current:
            raise ValueError(f"Cannot remove missing JSON Pointer path '{path}'.")
        del current[leaf]
        return result
    if isinstance(current, list) and leaf.isdigit():
        index = int(leaf)
        if index >= len(current):
            raise ValueError(f"Cannot remove missing JSON Pointer path '{path}'.")
        current.pop(index)
        return result
    raise ValueError(f"Cannot remove JSON Pointer path '{path}'.")


def _merge_json_pointer_array_items(
    document: dict[str, Any],
    path: str,
    *,
    survivor_index: int | None,
    merged_indices: list[int],
) -> tuple[dict[str, Any], list[str]]:
    if survivor_index is None:
        raise ValueError("merge write requires survivor_index")
    result = deepcopy(document)
    array_value = _value_at_json_pointer(result, path)
    if not isinstance(array_value, list):
        raise ValueError(f"Merge target path is not an array: {path}")
    indices = list(dict.fromkeys(merged_indices))
    if survivor_index in indices:
        raise ValueError("merge write cannot merge survivor_index into itself")
    all_indices = [survivor_index] + indices
    missing = [index for index in all_indices if index < 0 or index >= len(array_value)]
    if missing:
        raise ValueError(f"Merge index out of range at {path}: {missing}")
    changed_paths = [f"{path}/{index}" for index in sorted(indices)]
    for index in sorted(indices, reverse=True):
        array_value.pop(index)
    changed_paths.append(path)
    return result, changed_paths


def _dedupe_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(_canonical_target_path(path) for path in paths if path))


def _array_path(path: str) -> str:
    return path[:-2] if path.endswith("/-") else path


def _canonical_target_path(path: str) -> str:
    return _array_path(path)


def _normalize_write_target_path(write: SchemaConstrainedWrite) -> SchemaConstrainedWrite:
    canonical = _canonical_target_path(write.target_path)
    if canonical == write.target_path:
        return write
    return write.model_copy(update={"target_path": canonical})


def _parent_path(path: str) -> str:
    if not path or path == "/":
        return "/"
    return "/" + "/".join(path.strip("/").split("/")[:-1])


def _json_pointer_unescape(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def _json_pointer_escape(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _instance_label(instance: dict[str, Any]) -> str:
    for key in ("title", "name", "label", "has_quantity_type", "description"):
        value = instance.get(key)
        if isinstance(value, list):
            value = next((item for item in value if isinstance(item, str) and item.strip()), "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_patch_id(data_package_id: str, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:64] or "object"
    return f"{data_package_id}:{slug}"
