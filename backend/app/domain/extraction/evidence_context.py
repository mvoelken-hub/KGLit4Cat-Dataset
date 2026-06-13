from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.extraction.overview import (
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewStatus,
    file_summary_to_prompt_text,
    overview_to_prompt_text,
)


EvidenceCategory = Literal[
    "resource_signal",
    "method_signal",
    "measurement_signal",
    "entity_signal",
    "agent_signal",
    "data_quality_signal",
    "uncertainty",
    "other",
]
EvidenceConfidence = Literal["high", "medium", "low"]

EVIDENCE_MATCH_THRESHOLD = 0.90


class EvidenceNote(BaseModel):
    note_id: str = Field(..., description="Stable note identifier within the chunk.")
    category: EvidenceCategory = Field(
        "other",
        description="Broad evidence category. Do not use ontology class labels here.",
    )
    observation: str = Field(
        ...,
        description="Concise observation supported by the evidence text.",
    )
    evidence_text: str = Field(
        ...,
        description="Text copied from the current chunk that supports the observation.",
    )
    uncertainty: str = Field(
        "",
        description="Short uncertainty note; empty string when the evidence is straightforward.",
    )
    interpretation_confidence: EvidenceConfidence = Field(
        "medium",
        description="Confidence that the observation correctly interprets the copied evidence text.",
    )
    profile_worthiness: EvidenceConfidence = Field(
        "medium",
        description="How useful the note is for filling the selected metadata profile.",
    )
    file_path: str = ""
    start_idx: int = Field(0, ge=0)
    end_idx: int = Field(0, ge=0)
    evidence_match_score: float = Field(0.0, ge=0.0, le=1.0)


class FileInventoryItem(BaseModel):
    file_path: str
    byte_size: int = 0
    file_type: str | None = None
    rank: int | None = None
    summary: str | None = None


class EvidenceContext(BaseModel):
    notes: list[EvidenceNote] = Field(default_factory=list)
    file_inventory: list[FileInventoryItem] = Field(default_factory=list)


class EvidenceChunkMetadata(BaseModel):
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)
    file_path: str
    data_package_name: str


class EvidenceChunkContext(BaseModel):
    content: str
    metadata: EvidenceChunkMetadata


EVIDENCE_CONTEXT_SYSTEM_PROMPT = f"""You extract broad, traceable evidence observations from scientific data-package chunks.
Your output is NOT an ontology object model and NOT a final profile document. Do not classify observations into classes such as DataGeneratingActivity, Resource, Method, AgenticEntity, or EvaluatedEntity.

Return only a valid EvidenceContext JSON object.
Use this output schema: {EvidenceContext.model_json_schema()}

Rules:
- Produce concise evidence notes only when the current chunk contains human-readable evidence.
- Every evidence_text must be a short substring copied from the current chunk. Do not paraphrase evidence_text.
- Use initial overview and file summary only to understand context; do not cite them as evidence.
- If the chunk contains only encoded payload, raw numeric signal rows, checksums, empty declarations, or unreadable data, return notes: [].
- Prefer a broad perspective: one note may summarize several adjacent headers or settings when they clearly describe the same signal.
- Suppress repeated boilerplate/header-only facts such as repeated JCAMP version, empty origin, generic DATA TYPE, or repeated parameter-file titles unless they add new profile-worthy meaning.
- Group adjacent low-level parameters into one broader note when they describe the same method, instrument configuration, processing step, or resource characteristic.
- Use only these categories: resource_signal, method_signal, measurement_signal, entity_signal, agent_signal, data_quality_signal, uncertainty, other.
- Record uncertainty when labels are ambiguous, evidence is only technical, or a setting cannot be safely interpreted.
- Set interpretation_confidence to high only when the observation follows directly from clear labels or prose; use medium/low for technical parameters.
- Set profile_worthiness to high only for evidence likely to fill dataset title, creator, distribution, method/activity, entity, measurement, type, keyword, identifier, or modification-date fields.
"""


def build_evidence_context_prompt(chunk_context: EvidenceChunkContext) -> str:
    return (
        "Chunk context metadata:\n"
        f"{chunk_context.metadata.model_dump_json()}\n\n"
        f"Chunk content (residual lines after text-quality filtering):\n{chunk_context.content}\n"
        "Extract broad evidence notes from the current chunk content."
    )


def build_evidence_system_prompt_with_overview(
    base_prompt: str,
    *,
    overview: ExtractionOverview | None,
    overview_status: ExtractionOverviewStatus | None,
    file_summary: ExtractionFileSummary | None = None,
) -> str:
    sections: list[str] = []
    overview_text = overview_to_prompt_text(overview, status=overview_status)
    if overview_text:
        sections.append(
            "\n\nInitial extraction overview (orientation only; not evidence):\n"
            + overview_text
        )
    file_summary_text = file_summary_to_prompt_text(file_summary)
    if file_summary_text:
        sections.append(
            "\n\nCurrent file summary (orientation only; not evidence):\n"
            + file_summary_text
        )
    return base_prompt + "".join(sections)


def merge_evidence_contexts(contexts: list[EvidenceContext]) -> EvidenceContext:
    notes: list[EvidenceNote] = []
    inventory_by_path: dict[str, FileInventoryItem] = {}
    seen_notes: set[tuple[str, int, int, str]] = set()
    for context in contexts:
        for item in context.file_inventory:
            inventory_by_path.setdefault(item.file_path, item)
        for note in context.notes:
            key = (
                note.file_path,
                note.start_idx,
                note.end_idx,
                _normalize_evidence_text(note.evidence_text),
            )
            if key in seen_notes:
                continue
            seen_notes.add(key)
            notes.append(note)
    return EvidenceContext(
        notes=notes,
        file_inventory=sorted(inventory_by_path.values(), key=lambda item: item.file_path),
    )


def validate_evidence_context_for_chunk(
    context: EvidenceContext,
    *,
    chunk_content: str,
    file_path: str,
    start_idx: int,
    end_idx: int,
    threshold: float = EVIDENCE_MATCH_THRESHOLD,
) -> tuple[EvidenceContext, list[EvidenceNote]]:
    kept: list[EvidenceNote] = []
    dropped: list[EvidenceNote] = []
    for index, note in enumerate(context.notes):
        score = evidence_text_match_score(note.evidence_text, chunk_content)
        updated = note.model_copy(
            update={
                "note_id": note.note_id or f"{file_path}:{start_idx}:{end_idx}:{index}",
                "file_path": file_path,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "evidence_match_score": round(score, 4),
            }
        )
        if score >= threshold:
            kept.append(updated)
        else:
            dropped.append(updated)
    return EvidenceContext(notes=kept, file_inventory=context.file_inventory), dropped


def evidence_text_match_score(evidence_text: str, chunk_content: str) -> float:
    evidence = _normalize_evidence_text(evidence_text)
    content = _normalize_evidence_text(chunk_content)
    if not evidence:
        return 0.0
    if evidence in content:
        return 1.0
    if len(evidence) < 12:
        return 0.0
    best = 0.0
    window = len(evidence)
    step = max(1, window // 4)
    for start in range(0, max(1, len(content) - window + 1), step):
        candidate = content[start : start + window]
        best = max(best, SequenceMatcher(None, evidence, candidate).ratio())
        if best >= EVIDENCE_MATCH_THRESHOLD:
            return best
    return best


def is_noisy_payload_chunk(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    total_chars = sum(len(line) for line in lines)
    if total_chars <= 0:
        return False
    encoded_lines = [line for line in lines if _looks_like_encoded_signal_row(line)]
    encoded_chars = sum(len(line) for line in encoded_lines)
    has_payload_marker = any(
        marker in line.upper()
        for line in lines
        for marker in ("##XYDATA", "##PEAKTABLE")
    )
    if len(encoded_lines) >= 3 and encoded_chars / total_chars >= 0.45:
        return True
    if has_payload_marker and encoded_chars / total_chars >= 0.65:
        return True
    return False


def _looks_like_encoded_signal_row(line: str) -> bool:
    if line.startswith("#") or line.startswith("$"):
        return False
    if len(line) < 60:
        return False
    compact = re.sub(r"\s+", "", line)
    if not re.match(r"^[0-9][0-9A-Za-z@+\-.]+$", compact):
        return False
    alphaish = sum(1 for char in compact if char.isalpha() or char == "@")
    return alphaish / max(1, len(compact)) >= 0.35


def _normalize_evidence_text(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    return normalized.strip()
