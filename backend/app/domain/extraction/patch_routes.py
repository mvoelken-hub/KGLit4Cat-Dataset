from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.extraction.evidence_context import EvidenceCandidate
from app.domain.extraction.evidence_enrichment import route_evidence_note_to_target
from app.domain.extraction.schema_constrained_patch import (
    build_schema_constrained_patch_schema,
)


@dataclass(frozen=True)
class PatchTargetRoute:
    target_path: str | None
    target_class: str | None = None


class LegacyEvidencePatchRoute:
    """Current heuristic evidence target route, isolated for later removal."""

    @staticmethod
    def route(note: EvidenceCandidate) -> PatchTargetRoute:
        target_path, target_class = route_evidence_note_to_target(note)
        return PatchTargetRoute(target_path=target_path, target_class=target_class)


class SchemaConstrainedPatchRoute:
    """Builds the small output schema for schema-constrained patch writes."""

    @staticmethod
    def output_schema(
        *,
        validation_schema: dict[str, Any],
        allowed_target_paths: list[str],
    ) -> dict[str, Any]:
        return build_schema_constrained_patch_schema(
            validation_schema=validation_schema,
            allowed_target_paths=allowed_target_paths,
        )
