from __future__ import annotations

import json
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from jsonschema import Draft201909Validator
from linkml.generators.jsonldcontextgen import ContextGenerator
from linkml.generators.jsonschemagen import JsonSchemaGenerator
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.utils.schemaview import SchemaView
from pydantic import BaseModel, Field
from rdflib import Graph


PROFILE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_ENRICHABLE_FIELD_CANDIDATES = frozenset(
    {"type", "rdf_type", "has_quantity_type", "unit"}
)


class ProfileError(Exception):
    """Base class for profile-core errors."""


class InvalidProfileIdentifierError(ProfileError):
    pass


class ProfileAlreadyExistsError(ProfileError):
    pass


class ProfileNotFoundError(ProfileError):
    pass


class ProfileSourceError(ProfileError):
    pass


class ProfileCompatibilityError(ProfileError):
    pass


ProfileSourceType = Literal["url", "upload"]


class ProfileManifest(BaseModel):
    identifier: str
    source: str
    source_type: ProfileSourceType
    schema_url: str | None = None
    schema_file_name: str | None = None
    target_class: str = "Dataset"
    checksum: str
    version: str | None = None
    enrichable_fields: list[str] = Field(default_factory=list)


class GeneratedProfileArtifacts(BaseModel):
    manifest: ProfileManifest
    source_schema: str
    merged_schema: str
    json_schema: dict[str, Any]
    jsonld_context: dict[str, Any]


class ProfileValidationIssue(BaseModel):
    path: str
    message: str
    schema_path: str


class ProfileValidationResult(BaseModel):
    valid: bool
    errors: list[ProfileValidationIssue] = Field(default_factory=list)


class JsonLdExportResult(BaseModel):
    document: dict[str, Any]
    triple_count: int


def validate_profile_identifier(identifier: str) -> str:
    if not PROFILE_IDENTIFIER_PATTERN.fullmatch(identifier):
        raise InvalidProfileIdentifierError(
            "Profile identifier must start with a letter or digit and may only "
            "contain letters, digits, dots, underscores, and hyphens."
        )
    return identifier


def generate_profile_artifacts(
    *,
    identifier: str,
    source_schema: str,
    source: str,
    source_type: ProfileSourceType,
    target_class: str = "Dataset",
    version: str | None = None,
    enrichable_fields: list[str] | None = None,
    schema_url: str | None = None,
    schema_file_name: str | None = None,
    schema_location: str | None = None,
    source_suffix: str = ".yaml",
) -> GeneratedProfileArtifacts:
    """Generate persistent profile artifacts from a LinkML schema source."""

    validate_profile_identifier(identifier)

    if source_type == "url" and not schema_url:
        raise ProfileSourceError("URL profile registrations require schema_url.")
    if source_type == "upload" and not schema_file_name:
        raise ProfileSourceError("Uploaded profile registrations require schema_file_name.")

    with TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        effective_schema_location = schema_location
        if effective_schema_location is None:
            schema_path = temporary_path / f"source{_normalized_schema_suffix(source_suffix)}"
            schema_path.write_text(source_schema, encoding="utf-8")
            effective_schema_location = str(schema_path)

        try:
            schema_view = SchemaView(effective_schema_location, merge_imports=True)
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to load LinkML schema from {source!r}: {exc}"
            ) from exc

        if schema_view.get_class(target_class) is None:
            raise ProfileCompatibilityError(
                f"Target class '{target_class}' was not found in the LinkML profile."
            )

        try:
            induced_slots = schema_view.class_induced_slots(target_class)
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to inspect slots for target class '{target_class}'."
            ) from exc

        if not induced_slots:
            raise ProfileCompatibilityError(
                f"Target class '{target_class}' does not define any effective slots."
            )

        # LinkML records the source file path on loaded schemas. For uploaded
        # schemas this is a temporary file, so remove it before persisting and
        # hashing to keep artifacts reproducible.
        schema_view.schema.source_file = None
        merged_schema = yaml_dumper.dumps(schema_view.schema)
        checksum = "sha256:" + sha256(
            merged_schema.encode("utf-8")
        ).hexdigest()

        merged_schema_path = temporary_path / "merged_schema.yaml"
        merged_schema_path.write_text(merged_schema, encoding="utf-8")

        try:
            json_schema = json.loads(
                JsonSchemaGenerator(
                    str(merged_schema_path),
                    top_class=target_class,
                ).serialize()
            )
            jsonld_context = json.loads(
                ContextGenerator(str(merged_schema_path)).serialize()
            )
        except Exception as exc:
            raise ProfileCompatibilityError(
                f"Failed to generate profile artifacts for '{identifier}': {exc}"
            ) from exc

    detected_enrichable_fields = detect_enrichable_fields(
        induced_slot_names=[slot.name for slot in induced_slots],
        requested_fields=enrichable_fields,
    )

    profile_version = version or getattr(schema_view.schema, "version", None)

    manifest = ProfileManifest(
        identifier=identifier,
        source=source,
        source_type=source_type,
        schema_url=schema_url,
        schema_file_name=schema_file_name,
        target_class=target_class,
        checksum=checksum,
        version=profile_version,
        enrichable_fields=detected_enrichable_fields,
    )

    return GeneratedProfileArtifacts(
        manifest=manifest,
        source_schema=source_schema,
        merged_schema=merged_schema,
        json_schema=json_schema,
        jsonld_context=jsonld_context,
    )


def detect_enrichable_fields(
    *,
    induced_slot_names: list[str],
    requested_fields: list[str] | None = None,
) -> list[str]:
    candidates = (
        set(requested_fields)
        if requested_fields is not None
        else set(DEFAULT_ENRICHABLE_FIELD_CANDIDATES)
    )
    present_fields = set(induced_slot_names)
    return sorted(candidates & present_fields)


def validate_document_against_profile(
    *,
    document: dict[str, Any],
    json_schema: dict[str, Any],
    target_class: str,
) -> ProfileValidationResult:
    validation_schema = validation_schema_for_target_class(
        json_schema=json_schema,
        target_class=target_class,
    )
    validator = Draft201909Validator(validation_schema)
    issues = [
        ProfileValidationIssue(
            path=_format_json_path(error.absolute_path),
            message=error.message,
            schema_path=_format_json_path(error.absolute_schema_path),
        )
        for error in sorted(validator.iter_errors(document), key=str)
    ]
    return ProfileValidationResult(valid=not issues, errors=issues)


def validation_schema_for_target_class(
    *,
    json_schema: dict[str, Any],
    target_class: str,
) -> dict[str, Any]:
    if target_class not in json_schema.get("$defs", {}):
        return json_schema

    validation_schema = {
        "$schema": json_schema.get(
            "$schema",
            "https://json-schema.org/draft/2019-09/schema",
        ),
        "$defs": deepcopy(json_schema["$defs"]),
        "$ref": f"#/$defs/{target_class}",
    }
    if "$id" in json_schema:
        validation_schema["$id"] = json_schema["$id"]
    return validation_schema


def export_document_to_jsonld(
    *,
    document: dict[str, Any],
    jsonld_context: dict[str, Any],
    target_class: str,
) -> JsonLdExportResult:
    clean_document = remove_null_values(document)
    if not isinstance(clean_document, dict):
        raise ValueError("JSON-LD export requires an object document.")

    context_body = jsonld_context.get("@context", jsonld_context)
    jsonld_document = {
        **clean_document,
        "@context": context_body,
    }
    jsonld_document.setdefault("@type", target_class)

    graph = Graph()
    graph.parse(data=json.dumps(jsonld_document), format="json-ld")
    serialized = graph.serialize(
        format="json-ld",
        context=context_body,
        auto_compact=True,
        indent=2,
    )
    compact_document = json.loads(serialized)
    return JsonLdExportResult(
        document=compact_document,
        triple_count=len(graph),
    )


def remove_null_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := remove_null_values(item)) is not None
        }
    if isinstance(value, list):
        return [
            cleaned
            for item in value
            if (cleaned := remove_null_values(item)) is not None
        ]
    return value


def _format_json_path(path: Any) -> str:
    result = "$"
    for part in path:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


def _normalized_schema_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    if normalized not in {".yaml", ".yml", ".json"}:
        return ".yaml"
    return normalized
