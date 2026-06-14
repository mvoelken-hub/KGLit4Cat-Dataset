from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.token_budget import PromptTokenBudgeter

from app.domain.extraction.overview import (
    ExtractionFileSummary,
    ExtractionOverview,
    ExtractionOverviewStatus,
    file_summary_to_prompt_text,
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
EvidenceSignalLevel = Literal["high", "medium", "low"]
FilteredEvidenceReason = Literal[
    "signal_level_filtered",
    "duplicate_evidence",
    "evidence_text_unsupported",
]

EVIDENCE_MATCH_THRESHOLD = 0.90
EVIDENCE_OVERVIEW_PROMPT_BUDGET_TOKENS = 220
EVIDENCE_FILE_SUMMARY_PROMPT_BUDGET_TOKENS = 200
EVIDENCE_ORIENTATION_LINE_TOKENS = 45
EVIDENCE_ORIENTATION_VALUES_PER_SECTION = 4


class EvidenceNote(BaseModel):
    note_id: str = Field(..., description="Stable note identifier within the chunk.")
    category: EvidenceCategory = Field(
        "other",
        description="Broad evidence category. Do not use ontology class labels here.",
    )
    observation: str = Field(
        ...,
        description=(
            "Concise free-text observation supported by the evidence text. "
            "Include the useful semantic role in natural language, such as "
            "'instrument/device', 'dataset title', 'file format', 'method', "
            "'measured entity', or 'low-level parameter', when the evidence supports it."
        ),
    )
    evidence_text: str = Field(
        ...,
        description="Text copied from the current chunk that supports the observation.",
    )
    uncertainty: str = Field(
        "",
        description="Short uncertainty note; empty string when the evidence is straightforward.",
    )
    signal_level: EvidenceSignalLevel = Field(
        "medium",
        description=(
            "Profile-agnostic semantic level. Use high for stable metadata facts worth "
            "forwarding, medium for interpretable technical context, and low for raw "
            "parameters, boilerplate, or unclear settings."
        ),
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


class FilteredEvidenceNote(BaseModel):
    reason: FilteredEvidenceReason
    note: EvidenceNote
    file_path: str = ""
    start_idx: int = Field(0, ge=0)
    end_idx: int = Field(0, ge=0)
    chunk_index: int | None = None
    duplicate_representative_id: str | None = None


class FilteredEvidenceLedger(BaseModel):
    filtered_notes: list[FilteredEvidenceNote] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class EvidenceChunkMetadata(BaseModel):
    start_idx: int = Field(..., ge=0)
    end_idx: int = Field(..., ge=0)
    file_path: str
    data_package_name: str


class EvidenceChunkContext(BaseModel):
    content: str
    metadata: EvidenceChunkMetadata


EVIDENCE_CONTEXT_SYSTEM_PROMPT = """
You extract broad, traceable evidence observations from scientific data-package chunks. These observations later feed into a DCAT application profile for dataset metadata.
Your output is NOT an ontology object model and NOT a final profile document.

Return only a valid EvidenceContext JSON object.

Rules:
- Produce concise evidence notes only when the current chunk contains human-readable evidence.
- Every evidence_text must be a short substring copied from the current chunk. Do not paraphrase evidence_text.
- Use initial overview and file summary only to understand context; do not cite them as evidence.
- If the chunk contains only encoded payload, raw numeric signal rows, checksums, empty declarations, or unreadable data, return notes: [].
- Prefer a broad perspective: one note may summarize several adjacent headers or settings when they clearly describe the same scientific method, instrument, file format, sample, or dataset-level signal.
- Suppress repeated boilerplate/header-only facts such as repeated versions, empty origin, generic DATA TYPE, generic OWNER, local paths, audit labels, or repeated parameter-file titles unless they add new profile-worthy meaning.
- Group adjacent low-level parameters into one broader note when they describe the same interpretable method or instrument configuration; otherwise leave raw parameters as medium or low signal.
- Use only these categories: resource_signal, method_signal, measurement_signal, entity_signal, agent_signal, data_quality_signal, uncertainty, other.
- Put the useful semantic role into observation as natural language, not as structured fields.
- Dataset-level identity, creator, date, origin, owner, or title claims require evidence that explicitly names the whole dataset, package, study, or submission. Do not promote a file-local header, local path, parameter-file label, audit label, or resource title into a dataset-level claim.
- File-local headers should be described as file/resource-local when useful, and should be medium or low unless they identify a meaningful distribution/resource format or a clearly reusable dataset resource.
- Generic owner, origin, vendor, manufacturer, software, or organization labels identify provenance context only. Do not infer an instrument/device, dataset creator, or dataset owner from such labels unless the evidence explicitly states that role.
- Instruments, spectrometers, probes, software-controlled acquisition hardware, and named devices should use category agent_signal and an observation that explicitly says instrument/device.
- Scientific method facts such as pulse sequence, observed nucleus, solvent, and observation frequency may be high when the evidence supports an interpretable method statement.
- Numeric geometry, point counts, axis/range min/max values, scaling factors, thresholds, routing keys, checksums, local file paths, raw instrument parameters, repeated file headers, and empty/off toggles should be described as technical or low-level parameters and should usually have medium or low signal_level.
- Numbered, channel-specific, namespace-prefixed, or code-like parameters are technical settings. Keep them medium or low unless grouped into a clearly supported higher-level method observation.
- Record uncertainty when labels are ambiguous, evidence is only technical, or a setting cannot be safely interpreted.
- Set signal_level to high only for stable, profile-agnostic metadata facts likely to describe explicit dataset identity, creator/agent, instrument/device, distribution/resource format, interpretable method/activity, measured entity, sample, identifier, date, or meaningful file syntax.
- Set signal_level to medium for interpretable technical context that may help debugging but should not drive profile generation by itself.
- Set signal_level to low for raw parameters, boilerplate, repeated syntax/header declarations, empty settings, or unclear technical fields.
- Only high signal notes are forwarded downstream; medium and low notes are retained in debug artifacts only.
"""

def build_evidence_context_prompt(chunk_context: EvidenceChunkContext) -> str:
    normalized_content = normalize_chunk_text_for_evidence_prompt(chunk_context.content)
    return (
        "Chunk context metadata:\n"
        f"{chunk_context.metadata.model_dump_json()}\n\n"
        "Chunk content (normalized residual lines after text-quality filtering):\n"
        f"{normalized_content}\n"
        "Extract broad evidence notes from the current chunk content."
    )


def build_evidence_system_prompt_with_overview(
    base_prompt: str,
    *,
    overview: ExtractionOverview | None,
    overview_status: ExtractionOverviewStatus | None,
    file_summary: ExtractionFileSummary | None = None,
    token_budgeter: PromptTokenBudgeter | None = None,
    max_overview_tokens: int = EVIDENCE_OVERVIEW_PROMPT_BUDGET_TOKENS,
    max_file_summary_tokens: int = EVIDENCE_FILE_SUMMARY_PROMPT_BUDGET_TOKENS,
) -> str:
    token_budgeter = token_budgeter or PromptTokenBudgeter()
    sections: list[str] = []
    overview_text = compact_evidence_overview_to_prompt_text(
        overview,
        status=overview_status,
        current_file_path=file_summary.file_path if file_summary else None,
        max_tokens=max_overview_tokens,
        token_budgeter=token_budgeter,
    )
    if overview_text:
        sections.append(
            "\n\nInitial extraction overview (orientation only; not evidence):\n"
            + overview_text
        )
    file_summary_text = _cap_prompt_text(
        file_summary_to_prompt_text(file_summary),
        max_tokens=max_file_summary_tokens,
        token_budgeter=token_budgeter,
    )
    if file_summary_text:
        sections.append(
            "\n\nCurrent file summary (orientation only; not evidence):\n"
            + file_summary_text
        )
    return base_prompt + "".join(sections)


def compact_evidence_overview_to_prompt_text(
    overview: ExtractionOverview | None,
    *,
    status: ExtractionOverviewStatus | None = None,
    current_file_path: str | None = None,
    max_tokens: int = EVIDENCE_OVERVIEW_PROMPT_BUDGET_TOKENS,
    token_budgeter: PromptTokenBudgeter | None = None,
) -> str:
    if overview is None:
        return ""
    token_budgeter = token_budgeter or PromptTokenBudgeter()
    parts: list[str] = []
    if status:
        parts.append(f"Overview status: {status}")
    if current_file_path and overview.file_roles:
        matching_roles = [
            role for role in overview.file_roles if role.file_path == current_file_path
        ]
        if matching_roles:
            role_lines = []
            for role in matching_roles[:2]:
                notes = _compact_prompt_values(
                    role.extraction_notes,
                    limit=2,
                    token_budgeter=token_budgeter,
                )
                role_lines.append(
                    f"- {role.file_path}: {role.role}"
                    + (f" ({'; '.join(notes)})" if notes else "")
                )
            parts.append("Current file role:\n" + "\n".join(role_lines))
    for label, values in (
        ("Observed signals", overview.observed_signals),
        ("Suggested interpretations", overview.suggested_interpretations),
        ("Conflicts/uncertainties", overview.conflicts_or_uncertainties),
    ):
        compact_values = _compact_prompt_values(
            values,
            limit=EVIDENCE_ORIENTATION_VALUES_PER_SECTION,
            token_budgeter=token_budgeter,
        )
        if compact_values:
            parts.append(label + ":\n" + "\n".join(f"- {value}" for value in compact_values))
    return _cap_prompt_text(
        "\n\n".join(parts),
        max_tokens=max_tokens,
        token_budgeter=token_budgeter,
    )


def normalize_chunk_text_for_evidence_prompt(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)
    normalized = "\n".join(
        re.sub(r"[ ]{2,}", " ", line).rstrip()
        for line in normalized.split("\n")
    )
    return normalized.strip()


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


def filter_evidence_context_by_signal_level(
    context: EvidenceContext,
    *,
    keep_level: EvidenceSignalLevel = "high",
    chunk_index: int | None = None,
) -> tuple[EvidenceContext, list[FilteredEvidenceNote]]:
    kept: list[EvidenceNote] = []
    dropped: list[FilteredEvidenceNote] = []
    for note in context.notes:
        if note.signal_level == keep_level:
            kept.append(note)
            continue
        dropped.append(
            _filtered_record(
                note,
                reason="signal_level_filtered",
                chunk_index=chunk_index,
            )
        )
    return (
        context.model_copy(update={"notes": kept}),
        dropped,
    )


def dedupe_repeated_evidence_notes(
    context: EvidenceContext,
    *,
    file_rank_by_path: dict[str, int] | None = None,
) -> tuple[EvidenceContext, list[FilteredEvidenceNote]]:
    file_rank_by_path = file_rank_by_path or {}
    grouped: dict[str, list[tuple[int, EvidenceNote]]] = {}
    unique_without_key: list[tuple[int, EvidenceNote]] = []
    for index, note in enumerate(context.notes):
        key = _normalize_evidence_text(note.evidence_text)
        if not key:
            unique_without_key.append((index, note))
            continue
        grouped.setdefault(key, []).append((index, note))

    kept_with_order: list[tuple[int, EvidenceNote]] = list(unique_without_key)
    dropped: list[FilteredEvidenceNote] = []
    for group in grouped.values():
        if len(group) == 1:
            kept_with_order.append(group[0])
            continue
        representative_index, representative = min(
            group,
            key=lambda item: _dedupe_representative_rank(
                item[1],
                original_index=item[0],
                file_rank_by_path=file_rank_by_path,
            ),
        )
        kept_with_order.append((representative_index, representative))
        for _, note in group:
            if note is representative:
                continue
            dropped.append(
                _filtered_record(
                    note,
                    reason="duplicate_evidence",
                    duplicate_representative_id=representative.note_id,
                )
            )
    kept = [note for _, note in sorted(kept_with_order, key=lambda item: item[0])]
    return context.model_copy(update={"notes": kept}), dropped


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


def filtered_evidence_ledger(
    records: list[FilteredEvidenceNote],
) -> FilteredEvidenceLedger:
    summary: dict[str, int] = {}
    for record in records:
        summary[record.reason] = summary.get(record.reason, 0) + 1
    return FilteredEvidenceLedger(filtered_notes=records, summary=summary)


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


def _compact_prompt_values(
    values: list[str],
    *,
    limit: int,
    token_budgeter: PromptTokenBudgeter,
) -> list[str]:
    compact: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _compact_prompt_line(value, token_budgeter=token_budgeter)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        compact.append(cleaned)
        if len(compact) >= limit:
            break
    return compact


def _compact_prompt_line(value: str, *, token_budgeter: PromptTokenBudgeter) -> str:
    cleaned = normalize_chunk_text_for_evidence_prompt(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return token_budgeter.truncate(
        cleaned,
        max_tokens=EVIDENCE_ORIENTATION_LINE_TOKENS,
    )


def _cap_prompt_text(
    value: str,
    *,
    max_tokens: int,
    token_budgeter: PromptTokenBudgeter,
) -> str:
    cleaned = normalize_chunk_text_for_evidence_prompt(value)
    if not cleaned or max_tokens <= 0:
        return ""
    return token_budgeter.truncate(cleaned, max_tokens=max_tokens)


def _filtered_record(
    note: EvidenceNote,
    *,
    reason: FilteredEvidenceReason,
    chunk_index: int | None = None,
    duplicate_representative_id: str | None = None,
) -> FilteredEvidenceNote:
    return FilteredEvidenceNote(
        reason=reason,
        note=note,
        file_path=note.file_path,
        start_idx=note.start_idx,
        end_idx=note.end_idx,
        chunk_index=chunk_index,
        duplicate_representative_id=duplicate_representative_id,
    )


def _dedupe_representative_rank(
    note: EvidenceNote,
    *,
    original_index: int,
    file_rank_by_path: dict[str, int],
) -> tuple[int, int, int, int]:
    signal_rank = {"high": 0, "medium": 1, "low": 2}.get(note.signal_level, 3)
    file_rank = file_rank_by_path.get(note.file_path, 10_000)
    richness = len(note.observation.strip())
    return (signal_rank, file_rank, -richness, original_index)
