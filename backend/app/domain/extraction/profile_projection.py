from __future__ import annotations

from typing import Any

from app.domain.extraction.extraction_context import ExtractionContext
from app.domain.extraction.vocabulary import ExtractionNormalization


PROFILE_PROJECTION_SYSTEM_PROMPT = """
You transform normalized scientific extraction context into the selected application profile JSON document.
Use the supplied target profile schema and profile metadata to choose where facts belong.
Prefer schema-valid concise metadata over exhaustive copying. Return only JSON that satisfies the target schema.
"""


def build_profile_projection_prompt(
    *,
    data_package_id: str,
    profile_identifier: str,
    profile_target_class: str,
    extraction_context: ExtractionContext,
    normalization: ExtractionNormalization,
    warnings: list[str],
    profile_schema: dict[str, Any],
) -> str:
    return (
        f"Data package id: {data_package_id}\n"
        f"Profile identifier: {profile_identifier}\n"
        f"Profile target class: {profile_target_class}\n\n"
        "Merged ExtractionContext JSON:\n"
        f"{extraction_context.model_dump(mode='json')}\n\n"
        "Vocabulary normalization JSON:\n"
        f"{normalization.model_dump(mode='json')}\n\n"
        "Normalization warnings JSON:\n"
        f"{warnings}\n\n"
        "Target profile JSON Schema:\n"
        f"{profile_schema}\n\n"
        "Create the final profile document. Preserve source-supported facts, use normalized vocabulary URIs where available, "
        "and include raw unmatched values only when schema-valid."
    )

