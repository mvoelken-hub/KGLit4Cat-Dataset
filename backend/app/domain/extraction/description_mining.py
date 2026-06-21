from __future__ import annotations

import json
from hashlib import sha1
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.extraction.evidence_context import (
    EvidenceCandidate,
    EvidenceCategory,
    EvidenceRole,
    RoutedEvidenceContext,
    derive_source_context_for_evidence,
)


DESCRIPTION_FACT_MINING_SYSTEM_PROMPT = """
Extract atomic evidence facts from the supplied dataset-level description texts.
Use only explicit description content. Do not infer facts or route them to a schema.
Split compound statements into separate, complete claims.
Copy evidence_text verbatim from the supplied description. Do not paraphrase,
normalize, reorder, or stitch together non-contiguous text.
Return each source_description_path exactly as supplied.
Use only these categories: resource_signal, method_signal, measurement_signal,
measurement_condition, software_signal, activity_signal, instrument_signal,
surrounding_signal, other.
Set role to one of: qualitative_attribute, identity, descriptor, context,
parameter, other_metadata.
Use measurement_signal for observed/raw values. Use measurement_condition for
axis bounds, point counts, ranges, axis units, and other dataset-level
measurement descriptors. Use instrument_signal only for settings, parameters,
thresholds, units, calibration, or processing choices.
Use resource_signal for resources/formats, method_signal for realized procedures,
instrument_signal for devices/instruments that carry out work, software_signal for
software/executable systems that carry out work, activity_signal for activities,
and surrounding_signal for people, organizations, dates, ownership, or origin.
Use role=parameter for quantitative values that may become attributes,
role=qualitative_attribute for non-numeric attributes, role=identity for names
or identifiers, role=descriptor for type/class/summary/format facts, role=context
for provenance/surrounding facts, and role=other_metadata only when no precise
role applies.
Ignore prose that does not contain a reusable structured fact.
"""

DESCRIPTION_EVIDENCE_PREFIX = "draft-description:"
_EVIDENCE_CATEGORIES = {
    "resource_signal",
    "method_signal",
    "measurement_signal",
    "measurement_condition",
    "software_signal",
    "activity_signal",
    "instrument_signal",
    "surrounding_signal",
    "other",
}
_EVIDENCE_ROLES = {
    "qualitative_attribute",
    "identity",
    "descriptor",
    "context",
    "parameter",
    "other_metadata",
}


class DescriptionSource(BaseModel):
    path: str
    text: str


class RawDescriptionFact(BaseModel):
    source_description_path: str
    category: str
    role: str
    claim: str
    evidence_text: str
    source_context: str = ""


class RawDescriptionFacts(BaseModel):
    facts: list[RawDescriptionFact] = Field(default_factory=list)


class DescriptionFact(BaseModel):
    candidate_id: str
    source_description_path: str
    category: EvidenceCategory
    role: EvidenceRole
    claim: str
    evidence_text: str
    source_context: str = ""
    start_idx: int = Field(ge=0)
    end_idx: int = Field(ge=0)


class DescriptionFactRejection(BaseModel):
    fact_index: int = Field(ge=0)
    source_description_path: str = ""
    reason: str


class DescriptionMiningArtifact(BaseModel):
    status: Literal["completed", "skipped", "failed"]
    source_description_paths: list[str] = Field(default_factory=list)
    facts: list[DescriptionFact] = Field(default_factory=list)
    rejected_count: int = Field(default=0, ge=0)
    rejection_reasons: list[DescriptionFactRejection] = Field(default_factory=list)
    reason: str = ""


def collect_dataset_description_sources(document: dict) -> list[DescriptionSource]:
    description = document.get("description")
    if isinstance(description, str):
        return [DescriptionSource(path="/description", text=description)] if description.strip() else []
    if isinstance(description, list):
        return [
            DescriptionSource(path=f"/description/{index}", text=text)
            for index, text in enumerate(description)
            if isinstance(text, str) and text.strip()
        ]
    return []


def build_description_mining_prompt(sources: list[DescriptionSource]) -> str:
    return json.dumps(
        {"description_sources": [source.model_dump(mode="json") for source in sources]},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def validate_description_facts(
    raw_facts: RawDescriptionFacts,
    sources: list[DescriptionSource],
) -> tuple[list[DescriptionFact], list[DescriptionFactRejection]]:
    source_by_path = {source.path: source for source in sources}
    facts: list[DescriptionFact] = []
    rejections: list[DescriptionFactRejection] = []
    seen: set[str] = set()

    for index, raw in enumerate(raw_facts.facts):
        source = source_by_path.get(raw.source_description_path)
        reason = ""
        if source is None:
            reason = "unknown_source_description_path"
        elif raw.category not in _EVIDENCE_CATEGORIES:
            reason = "unknown_evidence_category"
        elif raw.role not in _EVIDENCE_ROLES:
            reason = "unknown_evidence_role"
        elif not raw.claim.strip():
            reason = "empty_claim"
        elif not raw.evidence_text.strip():
            reason = "empty_evidence_text"
        elif raw.evidence_text not in source.text:
            reason = "evidence_text_not_verbatim"

        if reason:
            rejections.append(
                DescriptionFactRejection(
                    fact_index=index,
                    source_description_path=raw.source_description_path,
                    reason=reason,
                )
            )
            continue

        assert source is not None
        start_idx = source.text.index(raw.evidence_text)
        source_context = derive_source_context_for_evidence(raw.evidence_text, source.text)
        candidate_id = _description_fact_id(
            source_path=source.path,
            category=raw.category,
            role=raw.role,
            claim=raw.claim.strip(),
            evidence_text=raw.evidence_text,
            source_context=source_context,
        )
        if candidate_id in seen:
            rejections.append(
                DescriptionFactRejection(
                    fact_index=index,
                    source_description_path=raw.source_description_path,
                    reason="duplicate_fact",
                )
            )
            continue
        seen.add(candidate_id)
        facts.append(
            DescriptionFact(
                candidate_id=candidate_id,
                source_description_path=source.path,
                category=raw.category,
                role=raw.role,
                claim=raw.claim.strip(),
                evidence_text=raw.evidence_text,
                source_context=source_context,
                start_idx=start_idx,
                end_idx=start_idx + len(raw.evidence_text),
            )
        )
    return facts, rejections


def augment_evidence_context_with_description_facts(
    evidence_context: RoutedEvidenceContext,
    facts: list[DescriptionFact],
) -> RoutedEvidenceContext:
    augmented = evidence_context.model_copy(deep=True)
    augmented.portable_evidence.extend(description_fact_to_evidence_candidate(fact) for fact in facts)
    return augmented


def description_fact_to_evidence_candidate(fact: DescriptionFact) -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_id=fact.candidate_id,
        category=fact.category,
        role=fact.role,
        claim=fact.claim,
        evidence_text=fact.evidence_text,
        source_context=fact.source_context,
        file_path=f"{DESCRIPTION_EVIDENCE_PREFIX}{fact.source_description_path}",
        start_idx=fact.start_idx,
        end_idx=fact.end_idx,
        evidence_match_score=0.0,
    )


def is_description_derived_path(file_path: str) -> bool:
    return file_path.startswith(DESCRIPTION_EVIDENCE_PREFIX)


def _description_fact_id(
    *,
    source_path: str,
    category: str,
    role: str,
    claim: str,
    evidence_text: str,
    source_context: str,
) -> str:
    payload = "|".join([source_path, category, role, claim, evidence_text, source_context])
    return f"description:{sha1(payload.encode('utf-8')).hexdigest()[:16]}"
