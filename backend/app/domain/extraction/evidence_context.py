from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    "measurement_condition",
    "software_signal",
    "instrument_signal",
    "activity_signal",
    "surrounding_signal",
    "other",
]
EvidenceRole = Literal[
    "qualitative_attribute",
    "identity",
    "descriptor",
    "context",
    "parameter",
    "other_metadata",
]
EvidenceRoute = Literal["portable_evidence", "contextual_evidence", "rejected_evidence"]
EvidenceCriticGranularity = Literal["per_chunk", "per_candidate", "disabled"]
EvidenceAssessmentJudgement = Literal["yes", "partial", "no"]
EvidenceAssessmentUncertainty = Literal["low", "medium", "high"]
FilteredEvidenceReason = Literal[
    "candidate_rejected",
    "duplicate_evidence",
    "evidence_text_unsupported",
    "similarity_merged",
]


# ─── Similarity-based deduplication thresholds ───
SIMILARITY_GROUP_THRESHOLD = 0.5   # notes with similarity above this are grouped
SIMILARITY_DETERMINISTIC_THRESHOLD = 0.9  # groups with max pairwise similarity above
                                            # this use deterministic triage (file_rank,
                                            # original_index); below this an LLM picks

EVIDENCE_MATCH_THRESHOLD = 0.90
EVIDENCE_OVERVIEW_PROMPT_BUDGET_TOKENS = 220
EVIDENCE_FILE_SUMMARY_PROMPT_BUDGET_TOKENS = 200
EVIDENCE_ORIENTATION_LINE_TOKENS = 45
EVIDENCE_ORIENTATION_VALUES_PER_SECTION = 4
EVIDENCE_SOURCE_CONTEXT_WINDOW_LINES = 2


class EvidenceCandidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    candidate_id: str = Field("", description="Stable candidate identifier within the chunk.")
    category: EvidenceCategory = Field(
        ...,
        description="Broad evidence category.",
    )
    role: EvidenceRole = Field(
        ...,
        description=(
            "Profile-routing role. Use parameter for quantitative attributes, "
            "qualitative_attribute for non-numeric attributes, identity for names/identifiers, "
            "descriptor for descriptive type/class/summary facts, context for provenance or surrounding facts, "
            "and other_metadata for reusable metadata that does not fit another role."
        ),
    )
    claim: str = Field(
        ...,
        description=(
            "A concise, molecular claim supported by the evidence text. "
            "The claim should be specific enough to verify but contextualized enough "
            "to stand alone without hidden local prompt context."
        ),
    )
    evidence_text: str = Field(
        ...,
        description="Text copied from the current chunk that supports the observation.",
    )
    source_context: str = Field(
        "",
        description=(
            "Smallest contiguous text span copied from the current chunk that preserves "
            "the local source scope for this evidence. It may equal evidence_text when "
            "no extra section, block, resource, method, instrument, software, or activity "
            "context is needed."
        ),
    )
    uncertainty: str = Field(
        "",
        description="Short uncertainty note; empty string when the evidence is straightforward.",
    )
    file_path: str = ""
    start_idx: int = Field(0, ge=0)
    end_idx: int = Field(0, ge=0)
    evidence_match_score: float = Field(0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _swap_misplaced_category_role(cls, data: Any) -> Any:
        """Auto-fix when the LLM puts a category value in the role field or vice versa."""
        if not isinstance(data, dict):
            return data
        cat = str(data.get("category", "") or "")
        role = str(data.get("role", "") or "")
        valid_categories = {
            "resource_signal", "method_signal", "measurement_signal",
            "measurement_condition", "software_signal", "instrument_signal",
            "activity_signal", "surrounding_signal", "other",
        }
        valid_roles = {
            "qualitative_attribute", "identity", "descriptor",
            "context", "parameter", "other_metadata",
        }
        # If role contains a category value, swap
        if role in valid_categories and cat in valid_roles:
            data["category"], data["role"] = role, cat
        elif role in valid_categories and cat not in valid_categories:
            # role has category value, category has something else — move it
            data["category"] = role
            data["role"] = cat if cat in valid_roles else "descriptor"
        elif cat in valid_roles and role not in valid_roles:
            # category has role value, role has something else — move it
            data["role"] = cat
            data["category"] = role if role in valid_categories else "other"
        return data

class EvidenceContext(BaseModel):
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    file_inventory: list["FileInventoryItem"] = Field(default_factory=list)


class EvidenceAssessment(BaseModel):
    candidate_id: str
    groundedness: EvidenceAssessmentJudgement = "partial"
    self_containedness: EvidenceAssessmentJudgement = "partial"
    scope_clarity: EvidenceAssessmentJudgement = "partial"
    portability: EvidenceAssessmentJudgement = "partial"
    semantic_interpretability: EvidenceAssessmentJudgement = "partial"
    environment_dependence: EvidenceAssessmentUncertainty = "medium"
    specificity: EvidenceAssessmentJudgement = "partial"
    novelty: EvidenceAssessmentJudgement = "partial"
    uncertainty: EvidenceAssessmentUncertainty = "medium"
    rationale: str = ""


class EvidenceAssessmentContext(BaseModel):
    assessments: list[EvidenceAssessment] = Field(default_factory=list)


class RoutedEvidenceRecord(BaseModel):
    route: EvidenceRoute
    reason: str
    candidate: EvidenceCandidate
    assessment: EvidenceAssessment | None = None
    chunk_index: int | None = None


class RoutedEvidenceContext(BaseModel):
    portable_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    contextual_evidence: list[EvidenceCandidate] = Field(default_factory=list)
    rejected_evidence: list[RoutedEvidenceRecord] = Field(default_factory=list)
    assessments: list[EvidenceAssessment] = Field(default_factory=list)
    file_inventory: list["FileInventoryItem"] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_candidate_context(cls, data):
        if isinstance(data, EvidenceContext):
            return {
                "portable_evidence": data.candidates,
                "file_inventory": data.file_inventory,
            }
        if isinstance(data, dict) and "portable_evidence" not in data and "candidates" in data:
            return {**data, "portable_evidence": data["candidates"]}
        return data

    @property
    def notes(self) -> list[EvidenceCandidate]:
        return self.portable_evidence


class FileInventoryItem(BaseModel):
    file_path: str
    byte_size: int = 0
    file_type: str | None = None
    rank: int | None = None
    summary: str | None = None


class FilteredEvidenceNote(BaseModel):
    reason: FilteredEvidenceReason
    note: EvidenceCandidate
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
You extract high-recall, traceable evidence candidates from scientific data-package chunks.
Your output is NOT an ontology object model, NOT a final profile document, and NOT a value-ranking decision.

Return only a valid EvidenceContext JSON object.

Rules:
- Produce concise evidence candidates only when the current chunk contains human-readable evidence.
- Every evidence_text must be a short substring copied from the current chunk. Do not paraphrase, normalize, reorder, or stitch together non-contiguous source text.
- If a fact depends on multiple nearby source lines, set evidence_text to one exact contiguous source span that contains the supporting lines, or emit separate candidates for the separate lines. Never compose evidence_text by selecting only separated min/max or key/value lines while omitting intervening source text.
- Do not populate source_context; the backend derives source_context from the validated evidence_text and local chunk. Spend output budget on complete candidates instead.
- Use initial overview and file summary only to understand context; do not cite them as evidence.
- If the chunk contains only encoded payload, raw numeric signal rows, checksums, empty declarations, or unreadable data, return candidates: [].
- Extract reasonably interpretable candidates even when they are technical, local, operational, or ambiguous. A later critic will route them.
- Do not decide whether a candidate is valuable enough for profile projection.
- Do not use domain-specific key names, file formats, instruments, vendors, or scientific concepts as quality criteria.
- Use only these categories: resource_signal, method_signal, measurement_signal, measurement_condition, software_signal, instrument_signal, activity_signal, surrounding_signal, other.
- Set role to one of: qualitative_attribute, identity, descriptor, context, parameter, other_metadata.
- Use resource_signal for files, distributions, formats, access paths, and resource-scoped notes.
- Use method_signal for realized plans, protocols, procedures, and methods.
- Use measurement_signal for primary/raw observed values, row-like observations, table cells, or data extrema; later semantic routing may project eligible notes into quantitative or qualitative activity/entity attributes.
- A numeric value that encodes a format version, type code, enumeration index, file identifier, or status flag is NOT a measurement — use resource_signal with role=descriptor even when the value is numeric (e.g. JCAMPDX=5.0 is a file format version, not an observed value).
- Use measurement_condition for measurement descriptors such as axis bounds, axis units, point counts, sampling ranges, measurement scale labels, or dataset-level measurement conditions that describe how observations are organized.
- Header/title-like identifiers are resource_signal or other unless the surrounding text explicitly says they name a device, instrument, software system, machine, or service.
- Use software_signal for software, scripts, executable systems, services, or processing applications that explicitly carry out work.
- Use instrument_signal for devices, instruments, machines, sensors, acquisition hardware, processing/configuration settings, calibration, units, thresholds, and parameters that can describe an instrument or activity.
- Use activity_signal for data-generating activities and other activities.
- Do not use instrument_signal for qualitative labels, names, categories, statuses, placeholder values, unset values, null/default markers, or encoded enum values; use resource_signal, surrounding_signal, measurement_signal, or other as appropriate.
- Do not use instrument_signal for data type, data class, spectrum type, table type, file type, or resource title; use resource_signal with role=descriptor or role=identity.
- Use surrounding_signal for dates, laboratories, research teams, people, organizations, ownership, origin, authorship, creators, and other provenance/context metadata.
- People, teams, organizations, dates, ownership, authorship, and origins are surrounding_signal even when the claim wording says "acquired by", "generated by", or "created by".
- Numeric values are measurement_signal when they are observed data values, extrema, or row-like observations; use measurement_condition when the number describes an axis bound, point count, sampling range, or measurement scale; use instrument_signal only when the number configures a setting, parameter, threshold, unit, or processing choice. However, a numeric value that is a version number, format identifier, type code, or enumeration index is NOT an observation — use resource_signal or instrument_signal with an appropriate non-parameter role.
- File formats, data classes, table categories, and serialization versions are resource_signal unless they explicitly describe a realized method/procedure.
- Use role=parameter for values that may become quantitative attributes, including settings, thresholds, units, point counts, ranges, axis descriptors, measurement scales, and numeric method parameters.
- Use role=qualitative_attribute for categorical values, type codes, format versions, statuses, names, and other non-measurement qualities — even when the value is numeric (e.g. JCAMPDX=5.0, POWCHK=enabled, ShimPowRcbType=3).
- Use role=identity for names, identifiers, titles, labels, or explicit identities of datasets, agents, activities, resources, or subjects.
- Use role=descriptor for types, classes, summaries, formats, and descriptive characteristics.
- Use role=context for every surrounding_signal candidate: provenance, authorship, dates, people, organizations, locations, ownership, or other surrounding context.
- Use role=other_metadata only when the candidate is reusable metadata but no more precise role applies.
- Use other only when none of the categories applies.
- Use claim for the candidate fact. Prefer molecular claims that can be checked against evidence_text and preserve explicit local source scope in the claim when that scope affects meaning.
- Record uncertainty when labels are ambiguous, evidence is only technical, or a setting cannot be safely interpreted.
"""


EVIDENCE_CRITIC_SYSTEM_PROMPT = """
You assess grounded evidence candidates using only domain-agnostic quality criteria.
Do not decide relevance by recognizing a particular scientific domain, instrument, vendor, file format, or key name.
Return only a valid EvidenceAssessmentContext JSON object.

Assess each candidate independently using categorical judgments:
- groundedness: whether evidence_text supports the claim.
- self_containedness: whether the claim can be understood without hidden prompt context.
- scope_clarity: whether package/resource/section/environment scope is clear.
- portability: whether the claim remains meaningful when the package is copied elsewhere.
- semantic_interpretability: whether the claim is meaningful rather than raw unexplained syntax.
- environment_dependence: low/medium/high dependence on local machine, session, user, installation, or network context.
- specificity: whether the claim is concrete without overinterpretation.
- novelty: whether it adds information beyond repeated template content.
- uncertainty: low/medium/high ambiguity.
"""

def build_evidence_context_prompt(chunk_context: EvidenceChunkContext) -> str:
    return "".join(text for _, text in build_evidence_context_prompt_components(chunk_context))


def build_evidence_context_prompt_components(
    chunk_context: EvidenceChunkContext,
) -> list[tuple[str, str]]:
    normalized_content = normalize_chunk_text_for_evidence_prompt(chunk_context.content)
    return [
        (
            "chunk_metadata",
            "Chunk context metadata:\n"
            f"{chunk_context.metadata.model_dump_json()}\n\n",
        ),
        (
            "chunk_content",
            "Chunk content (normalized residual lines after text-quality filtering):\n"
            f"{normalized_content}\n",
        ),
        (
            "evidence_task_instruction",
            "Extract high-recall evidence candidates from the current chunk content.",
        ),
    ]


def build_evidence_critic_prompt(
    *,
    candidates: list[EvidenceCandidate],
    chunk_context: EvidenceChunkContext,
) -> str:
    return "".join(
        text
        for _, text in build_evidence_critic_prompt_components(
            candidates=candidates,
            chunk_context=chunk_context,
        )
    )


def build_evidence_critic_prompt_components(
    *,
    candidates: list[EvidenceCandidate],
    chunk_context: EvidenceChunkContext,
) -> list[tuple[str, str]]:
    normalized_content = normalize_chunk_text_for_evidence_prompt(chunk_context.content)
    return [
        (
            "chunk_metadata",
            "Chunk context metadata:\n"
            f"{chunk_context.metadata.model_dump_json()}\n\n",
        ),
        (
            "validated_candidates",
            "Validated evidence candidates JSON:\n"
            f"{[candidate.model_dump(mode='json', exclude={'source_context'}) for candidate in candidates]}\n\n",
        ),
        (
            "local_chunk_context",
            "Local chunk context for assessment:\n"
            f"{normalized_content}\n",
        ),
    ]


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
    return "".join(
        text
        for _, text in build_evidence_system_prompt_components_with_overview(
            base_prompt,
            overview=overview,
            overview_status=overview_status,
            file_summary=file_summary,
            token_budgeter=token_budgeter,
            max_overview_tokens=max_overview_tokens,
            max_file_summary_tokens=max_file_summary_tokens,
        )
    )


def build_evidence_system_prompt_components_with_overview(
    base_prompt: str,
    *,
    overview: ExtractionOverview | None,
    overview_status: ExtractionOverviewStatus | None,
    file_summary: ExtractionFileSummary | None = None,
    token_budgeter: PromptTokenBudgeter | None = None,
    max_overview_tokens: int = EVIDENCE_OVERVIEW_PROMPT_BUDGET_TOKENS,
    max_file_summary_tokens: int = EVIDENCE_FILE_SUMMARY_PROMPT_BUDGET_TOKENS,
) -> list[tuple[str, str]]:
    token_budgeter = token_budgeter or PromptTokenBudgeter()
    sections: list[tuple[str, str]] = [("evidence_system_prompt", base_prompt)]
    overview_text = compact_evidence_overview_to_prompt_text(
        overview,
        status=overview_status,
        current_file_path=file_summary.file_path if file_summary else None,
        max_tokens=max_overview_tokens,
        token_budgeter=token_budgeter,
    )
    if overview_text:
        sections.append(
            (
                "initial_overview_orientation",
                "\n\nInitial extraction overview (orientation only; not evidence):\n"
                + overview_text,
            )
        )
    file_summary_text = _cap_prompt_text(
        file_summary_to_prompt_text(file_summary),
        max_tokens=max_file_summary_tokens,
        token_budgeter=token_budgeter,
    )
    if file_summary_text:
        sections.append(
            (
                "current_file_summary_orientation",
                "\n\nCurrent file summary (orientation only; not evidence):\n"
                + file_summary_text,
            )
        )
    return sections


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
    node_by_id = {node.node_id: node for node in overview.nodes}
    selected_node_ids: set[str] = set()
    if current_file_path:
        selected_node_ids.update(
            node.node_id
            for node in overview.nodes
            if node.file_path == current_file_path or node.node_id == f"file:{current_file_path}"
        )
    if selected_node_ids:
        neighborhood_edges = [
            edge
            for edge in overview.edges
            if edge.source in selected_node_ids or edge.target in selected_node_ids
        ][:EVIDENCE_ORIENTATION_VALUES_PER_SECTION]
        for edge in neighborhood_edges:
            selected_node_ids.add(edge.source)
            selected_node_ids.add(edge.target)
        node_lines: list[str] = []
        for node_id in list(selected_node_ids)[: EVIDENCE_ORIENTATION_VALUES_PER_SECTION * 2]:
            node = node_by_id.get(node_id)
            if not node:
                continue
            summary = _cap_prompt_text(
                node.summary,
                max_tokens=18,
                token_budgeter=token_budgeter,
            )
            node_lines.append(
                f"- {node.node_id}: {node.label} [{node.kind}]"
                + (f" path={node.file_path}" if node.file_path else "")
                + (f" - {summary}" if summary else "")
            )
        if node_lines:
            parts.append("Current file graph neighborhood nodes:\n" + "\n".join(node_lines))
        if neighborhood_edges:
            parts.append(
                "Current file graph neighborhood relations:\n"
                + "\n".join(
                    f"- {edge.source} -[{edge.relation}]-> {edge.target}"
                    + (f" ({_cap_prompt_text(edge.note, max_tokens=18, token_budgeter=token_budgeter)})" if edge.note else "")
                    for edge in neighborhood_edges
                )
            )
    compact_uncertainties = _compact_prompt_values(
        overview.uncertainties,
        limit=EVIDENCE_ORIENTATION_VALUES_PER_SECTION,
        token_budgeter=token_budgeter,
    )
    if compact_uncertainties:
        parts.append(
            "Package graph uncertainties:\n"
            + "\n".join(f"- {value}" for value in compact_uncertainties)
        )
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


def merge_evidence_contexts(contexts: list[RoutedEvidenceContext]) -> RoutedEvidenceContext:
    portable: list[EvidenceCandidate] = []
    contextual: list[EvidenceCandidate] = []
    rejected: list[RoutedEvidenceRecord] = []
    assessments: list[EvidenceAssessment] = []
    inventory_by_path: dict[str, FileInventoryItem] = {}
    seen_portable: set[tuple[str, int, int, str]] = set()
    seen_contextual: set[tuple[str, int, int, str]] = set()
    for context in contexts:
        for item in context.file_inventory:
            inventory_by_path.setdefault(item.file_path, item)
        for note in context.portable_evidence:
            key = (
                note.file_path,
                note.start_idx,
                note.end_idx,
                _normalize_evidence_text(note.evidence_text),
            )
            if key in seen_portable:
                continue
            seen_portable.add(key)
            portable.append(note)
        for note in context.contextual_evidence:
            key = (
                note.file_path,
                note.start_idx,
                note.end_idx,
                _normalize_evidence_text(note.evidence_text),
            )
            if key in seen_contextual:
                continue
            seen_contextual.add(key)
            contextual.append(note)
        rejected.extend(context.rejected_evidence)
        assessments.extend(context.assessments)
    return RoutedEvidenceContext(
        portable_evidence=portable,
        contextual_evidence=contextual,
        rejected_evidence=rejected,
        assessments=assessments,
        file_inventory=sorted(inventory_by_path.values(), key=lambda item: item.file_path),
    )


def dedupe_repeated_evidence_notes(
    context: RoutedEvidenceContext | EvidenceContext,
    *,
    file_rank_by_path: dict[str, int] | None = None,
) -> tuple[RoutedEvidenceContext, list[FilteredEvidenceNote]]:
    if isinstance(context, EvidenceContext):
        context = RoutedEvidenceContext(portable_evidence=context.candidates, file_inventory=context.file_inventory)
    file_rank_by_path = file_rank_by_path or {}
    grouped: dict[str, list[tuple[int, EvidenceCandidate]]] = {}
    unique_without_key: list[tuple[int, EvidenceCandidate]] = []
    for index, note in enumerate(context.portable_evidence):
        key = _dedupe_evidence_note_key(note)
        if not key:
            unique_without_key.append((index, note))
            continue
        grouped.setdefault(key, []).append((index, note))

    kept_with_order: list[tuple[int, EvidenceCandidate]] = list(unique_without_key)
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
                    duplicate_representative_id=representative.candidate_id,
                )
            )
    kept = [note for _, note in sorted(kept_with_order, key=lambda item: item[0])]
    return context.model_copy(update={"portable_evidence": kept}), dropped


def validate_evidence_candidates(
    context: EvidenceContext,
    *,
    chunk_content: str,
    file_path: str,
    start_idx: int,
    end_idx: int,
    threshold: float = EVIDENCE_MATCH_THRESHOLD,
) -> tuple[EvidenceContext, list[EvidenceCandidate]]:
    kept: list[EvidenceCandidate] = []
    dropped: list[EvidenceCandidate] = []
    for index, candidate in enumerate(context.candidates):
        score = evidence_text_match_score(candidate.evidence_text, chunk_content)
        source_context = derive_source_context_for_evidence(candidate.evidence_text, chunk_content)
        candidate_id = candidate.candidate_id or f"{file_path}:{start_idx}:{end_idx}:{index}"
        updated = candidate.model_copy(
            update={
                "candidate_id": candidate_id,
                "source_context": source_context,
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
    return EvidenceContext(candidates=kept, file_inventory=context.file_inventory), dropped


def validate_evidence_context_for_chunk(
    context: EvidenceContext,
    *,
    chunk_content: str,
    file_path: str,
    start_idx: int,
    end_idx: int,
    threshold: float = EVIDENCE_MATCH_THRESHOLD,
) -> tuple[EvidenceContext, list[EvidenceCandidate]]:
    return validate_evidence_candidates(
        context,
        chunk_content=chunk_content,
        file_path=file_path,
        start_idx=start_idx,
        end_idx=end_idx,
        threshold=threshold,
    )


def derive_source_context_for_evidence(
    evidence_text: str,
    chunk_content: str,
    *,
    window_lines: int = EVIDENCE_SOURCE_CONTEXT_WINDOW_LINES,
) -> str:
    evidence = normalize_chunk_text_for_evidence_prompt(evidence_text)
    content = normalize_chunk_text_for_evidence_prompt(chunk_content)
    if not evidence:
        return ""
    match_index = content.find(evidence)
    if match_index < 0:
        return evidence_text
    match_end = match_index + len(evidence)
    start_line = content[:match_index].count("\n")
    end_line = content[:match_end].count("\n")
    lines = content.split("\n")
    context_start = max(0, start_line - max(0, window_lines))
    context_end = min(len(lines), end_line + max(0, window_lines) + 1)
    return "\n".join(lines[context_start:context_end]).strip() or evidence_text


def route_evidence_candidates(
    context: EvidenceContext,
    *,
    assessments: list[EvidenceAssessment],
    rejected_candidates: list[EvidenceCandidate] | None = None,
    chunk_index: int | None = None,
) -> RoutedEvidenceContext:
    rejected_candidates = rejected_candidates or []
    assessment_by_id = {assessment.candidate_id: assessment for assessment in assessments}
    portable: list[EvidenceCandidate] = []
    contextual: list[EvidenceCandidate] = []
    rejected: list[RoutedEvidenceRecord] = [
        RoutedEvidenceRecord(
            route="rejected_evidence",
            reason="evidence_text_unsupported",
            candidate=candidate,
            chunk_index=chunk_index,
        )
        for candidate in rejected_candidates
    ]
    seen: set[tuple[str, str]] = set()
    for candidate in context.candidates:
        key = (candidate.file_path, _normalize_evidence_text(candidate.evidence_text))
        if key in seen:
            rejected.append(
                RoutedEvidenceRecord(
                    route="rejected_evidence",
                    reason="duplicate_evidence",
                    candidate=candidate,
                    assessment=assessment_by_id.get(candidate.candidate_id),
                    chunk_index=chunk_index,
                )
            )
            continue
        seen.add(key)
        assessment = assessment_by_id.get(candidate.candidate_id) or EvidenceAssessment(
            candidate_id=candidate.candidate_id,
            rationale="No critic assessment was available; routed conservatively.",
        )
        route, reason = _route_for_assessment(assessment)
        if route == "portable_evidence":
            portable.append(candidate)
        elif route == "contextual_evidence":
            contextual.append(candidate)
        else:
            rejected.append(
                RoutedEvidenceRecord(
                    route=route,
                    reason=reason,
                    candidate=candidate,
                    assessment=assessment,
                    chunk_index=chunk_index,
                )
            )
    return RoutedEvidenceContext(
        portable_evidence=portable,
        contextual_evidence=contextual,
        rejected_evidence=rejected,
        assessments=assessments,
        file_inventory=context.file_inventory,
    )


def _route_for_assessment(assessment: EvidenceAssessment) -> tuple[EvidenceRoute, str]:
    if assessment.groundedness == "no":
        return "rejected_evidence", "critic_not_grounded"
    if (
        assessment.self_containedness == "no"
        or assessment.scope_clarity == "no"
        or assessment.uncertainty == "high"
    ):
        return "contextual_evidence", "borderline_or_ambiguous"
    if (
        assessment.groundedness == "yes"
        and assessment.self_containedness == "yes"
        and assessment.scope_clarity in {"yes", "partial"}
        and assessment.portability == "yes"
        and assessment.semantic_interpretability == "yes"
        and assessment.environment_dependence != "high"
        and assessment.specificity == "yes"
        and assessment.uncertainty != "high"
    ):
        return "portable_evidence", "portable_grounded_evidence"
    return "contextual_evidence", "conservative_contextual_default"


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


def _dedupe_evidence_note_key(note: EvidenceCandidate) -> str:
    evidence = _normalize_evidence_text(note.evidence_text)
    if not evidence:
        return ""
    claim = _normalize_evidence_text(note.claim)
    file_path = (note.file_path or "").strip().lower()
    span = f"{note.start_idx}:{note.end_idx}"
    category = str(getattr(note, "category", "") or "")
    role = str(getattr(note, "role", "") or "")
    return "|".join([file_path, span, category, role, claim, evidence])


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
    note: EvidenceCandidate,
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
    note: EvidenceCandidate,
    *,
    original_index: int,
    file_rank_by_path: dict[str, int],
) -> tuple[int, int, int, int]:
    file_rank = file_rank_by_path.get(note.file_path, 10_000)
    richness = len(note.claim.strip())
    return (file_rank, -richness, original_index, 0)


# ──────────────────────────────────────────────────────────────────────────
#  Similarity-based deduplication
# ──────────────────────────────────────────────────────────────────────────





def deterministic_triage(
    group: list[tuple[int, EvidenceCandidate]],
    *,
    file_rank_by_path: dict[str, int],
) -> tuple[tuple[int, EvidenceCandidate], list[tuple[int, EvidenceCandidate]]]:
    """Pick representative by (file_rank asc, original_index asc). Drop the rest."""
    representative = min(
        group,
        key=lambda item: (
            file_rank_by_path.get(item[1].file_path, 10_000),
            item[0],
        ),
    )
    dropped = [item for item in group if item is not representative]
    return representative, dropped


# ─── LLM triage prompt ───

_DEDUPE_TRIAGE_SYSTEM_PROMPT = (
    "You are an evidence deduplication triage assistant.\n"
    "\n"
    "You receive a group of evidence notes that have similar evidence text but\n"
    "may differ in category, role, claim, or source file. Your job is to pick the\n"
    "single best representative — the note whose classification (category + role)\n"
    "and claim most accurately describe the underlying evidence.\n"
    "\n"
    "Selection criteria (in order of importance):\n"
    "1. Correctness of category: resource_signal > measurement_signal when the\n"
    "   value is a format version, type code, status flag, or other non-measurement.\n"
    "   measurement_signal > resource_signal when the value is a genuine observed\n"
    "   measurement or data point.\n"
    "2. Correctness of role: qualitative_attribute is correct for categorical/type\n"
    "   values; parameter is correct for genuine numeric measurements.\n"
    "3. Claim richness: a note with a non-empty, specific claim is preferred over\n"
    "   one with an empty claim.\n"
    "4. File rank: notes from higher-ranked (more important) source files are\n"
    "   preferred.\n"
    "\n"
    "Return ONLY the index (0-based) of the selected note."
)


def build_triage_prompt(
    group: list[tuple[int, EvidenceCandidate]],
    *,
    file_rank_by_path: dict[str, int],
) -> str:
    """Build the user prompt for LLM-based triage."""
    lines = [
        "Select the best representative from these similar evidence notes.",
        "Return a JSON object {\"selected_index\": <integer>}.", "",
    ]
    for i, (orig_idx, note) in enumerate(group):
        file_rank = file_rank_by_path.get(note.file_path, 10_000)
        lines.append(f"Note {i}:")
        lines.append(f"  evidence_text: {note.evidence_text}")
        lines.append(f"  claim: {note.claim}")
        lines.append(f"  category: {note.category}")
        lines.append(f"  role: {note.role}")
        lines.append(f"  file: {note.file_path}")
        lines.append(f"  file_rank: {file_rank}")
        lines.append("")
    return "\n".join(lines)


class TriageSelection(BaseModel):
    """LLM output schema for triage."""
    selected_index: int = Field(ge=0, description="0-based index of the selected note.")


def triage_parse_response(
    selection: TriageSelection,
    group: list[tuple[int, EvidenceCandidate]],
) -> tuple[tuple[int, EvidenceCandidate], list[tuple[int, EvidenceCandidate]]]:
    """Resolve the LLM selection into (representative, dropped)."""
    idx = selection.selected_index
    if idx < 0 or idx >= len(group):
        # Fallback: deterministic
        return deterministic_triage(group, file_rank_by_path={})
    representative = group[idx]
    dropped = [item for item in group if item is not representative]
    return representative, dropped
