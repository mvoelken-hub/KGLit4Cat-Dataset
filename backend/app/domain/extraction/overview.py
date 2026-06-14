from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.extraction.file_ranking import RankedFile


ExtractionOverviewStatus = Literal["structured", "unstructured_fallback", "failed"]
ExtractionFileSummaryStatus = Literal["summarized", "failed"]
InitialFileSummaryStatus = Literal["completed", "partial", "failed"]


class ExtractionOverviewFileRole(BaseModel):
    file_path: str
    role: str = Field(..., description="Likely role of the file in the dataset.")
    extraction_notes: list[str] = Field(
        default_factory=list,
        description="Hints for interpreting chunks from this file.",
    )
    read_reason: str = Field(
        "",
        description="Why this file was informative for the run overview.",
    )


class ExtractionOverviewInspectedFile(BaseModel):
    file_path: str
    byte_size: int | None = None
    chars_read: int = 0
    reason: str = ""


class ExtractionFileContentWindow(BaseModel):
    label: str
    start_char_idx: int = Field(..., ge=0)
    end_char_idx: int = Field(..., ge=0)
    omitted_before_chars: int = Field(default=0, ge=0)
    omitted_after_chars: int = Field(default=0, ge=0)
    text: str


class ExtractionFileSummary(BaseModel):
    source_fingerprint: str = Field(
        "",
        description="Backend-generated fingerprint of the sampled file content.",
    )
    file_path: str
    rank: int = Field(..., ge=1)
    status: ExtractionFileSummaryStatus = "summarized"
    data_format: str = Field(
        "",
        description="Obvious file format or syntax visible in the sampled content.",
    )
    data_characteristics: list[str] = Field(
        default_factory=list,
        description="Observable content characteristics, not inferred dataset intent.",
    )
    explicit_purpose: str = Field(
        "",
        description="Purpose only when explicitly stated by file content or filename.",
    )
    purpose_evidence: list[str] = Field(
        default_factory=list,
        description="Short snippets from this file supporting explicit_purpose.",
    )
    metadata_signals: list[str] = Field(default_factory=list)
    detected_identifiers: list[str] = Field(default_factory=list)
    instrument_or_software_terms: list[str] = Field(default_factory=list)
    parameter_terms: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)


class ExtractionOverview(BaseModel):
    source_fingerprint: str = Field(
        "",
        description="Backend-generated fingerprint of the ranked files and preview input.",
    )
    source_file_paths: list[str] = Field(
        default_factory=list,
        description="Backend-generated file paths used to create this overview.",
    )
    inspected_files: list[ExtractionOverviewInspectedFile] = Field(
        default_factory=list,
        description="Files whose extracted text was included in the overview input.",
    )
    file_roles: list[ExtractionOverviewFileRole] = Field(default_factory=list)
    observed_signals: list[str] = Field(
        default_factory=list,
        description="Directly visible file/package signals such as extensions, syntax markers, labels, filenames, exact terms, and identifiers.",
    )
    suggested_interpretations: list[str] = Field(
        default_factory=list,
        description="Plausible but unverified interpretations to guide retrieval and chunk ordering.",
    )
    conflicts_or_uncertainties: list[str] = Field(
        default_factory=list,
        description="Contradictions, missing modalities, ambiguous labels, and facts that must not be resolved without chunk evidence.",
    )


class InitialOverviewFailureDiagnostic(BaseModel):
    status: Literal["structured_failure"] = "structured_failure"
    error_type: str
    message: str
    last_error_type: str = ""
    last_error: str = ""
    failed_response_excerpt: str = ""
    prompt_budget: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)


class ExtractionOverviewFilePreview(BaseModel):
    rank: int = Field(..., ge=1)
    file_path: str
    byte_size: int | None = None
    first_lines: list[str] = Field(default_factory=list)


EXTRACTION_OVERVIEW_SYSTEM_PROMPT = """You are a scientific data archivist creating a conservative package triage map for heterogeneous scientific data.
Your output is guidance only. It is not evidence. Later extraction calls must still cite source_text from their current chunk.

Separate every claim by epistemic status:
- observed_signals: directly visible file/package signals such as extensions, syntax markers, labels, filenames, exact terms, identifiers, and explicit metadata;
- suggested_interpretations: plausible but unverified interpretations that can guide retrieval or chunk ordering;
- conflicts_or_uncertainties: contradictions, missing modalities, ambiguous labels, unsupported links, and facts that must not be resolved at overview stage.

Use broad research-data-management categories rather than dataset-specific examples:
- experimental data: raw measurements, processed measurements, derived results;
- metadata: sample identifiers, measurement context, instrument or software settings, timestamps, contributors;
- workflow/provenance: generation steps, processing history, handovers;
- relationships: sample-to-measurement, measurement-to-file, raw-to-processed, parameter-to-activity/resource;
- quality and uncertainty: missing metadata, undocumented uncertainty, inconsistent labels, ambiguous modality.

Focus on file roles, observed syntax, explicit terms, retrieval hints, provenance, and uncertainty. Do not resolve scientific meaning, instrument identity, modality, sample identity, or method identity unless explicitly supported by the summaries/previews.
When one explicit signal appears inconsistent with another, put that mismatch in conflicts_or_uncertainties. Common examples of mismatch types include instrument term versus measurement modality, file extension versus visible syntax, filename versus content labels, or raw-data versus processed-data role signals. Preserve the unresolved relationship instead of selecting one interpretation.

Do not invent final metadata.
Do not mention files that are not present in the ranked files or per-file summaries.
Fill every field with concise evidence-grounded values. Use empty lists when no file roles, observations, suggested interpretations, or uncertainties can be identified.
"""


EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT = """You are a scientific data archivist summarizing one file from a research artifact bundle.
Your output is guidance only. It is not extraction evidence for later chunk calls.

Summarize only facts visible in this file's sampled content, filename, or obvious syntax.
The explicit_purpose field is strict: fill it only when the sampled content or filename directly states the file's purpose. Otherwise leave it empty and add an uncertainty note.
Data format and data characteristics may be inferred from obvious syntax, extension, and visible content.
Evidence fields must quote short snippets from this file only.
Do not invent dataset purpose, instrument names, file roles, sample identities, or software-project files.
"""


EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT = """You are a scientific data archivist. Write a compact plain-text orientation for chunk-level metadata extraction.
The text is guidance only and must not be treated as evidence. Keep it concise and profile-independent.
Emphasize observed file roles, direct file/package signals, cautious suggested interpretations, and unresolved conflicts or limitations.
"""


def build_extraction_overview_prompt(
    *,
    data_package_name: str,
    ranked_files: list[RankedFile],
    file_previews: list[ExtractionOverviewFilePreview],
    file_summaries: list[ExtractionFileSummary] | None = None,
) -> str:
    preview_json = ",\n".join(preview.model_dump_json() for preview in file_previews)
    ranked_json = ",\n".join(file.model_dump_json() for file in ranked_files)
    summary_json = ",\n".join(
        summary.model_dump_json(exclude_defaults=True) for summary in (file_summaries or [])
    )
    summary_section = (
        "Validated per-file summaries JSON:\n"
        f"[{summary_json}]\n\n"
        if file_summaries
        else ""
    )
    return (
        "Analyze the research artifact archive and extract an ExtractionOverview. "
        "The backend has listed all ranked files and summarized the top ranked files.\n\n"
        f"Data package name: {data_package_name}\n"
        f"Ranked file count: {len(ranked_files)}\n"
        f"Per-file summary count: {len(file_summaries or [])}\n"
        f"Fallback preview file count: {len(file_previews)}\n\n"
        "Ranked files JSON:\n"
        f"[{ranked_json}]\n\n"
        f"{summary_section}"
        "Fallback raw file previews JSON:\n"
        f"[{preview_json}]\n\n"
        "Create an ExtractionOverview that will orient later one-shot chunk extraction calls. "
        "Use the per-file summaries as the primary input and previews only as fallback context when summaries are absent. "
        "Treat this as package triage, not final scientific interpretation. "
        "Populate observed_signals, suggested_interpretations, and conflicts_or_uncertainties separately. "
        "If explicit metadata and file syntax point in different scientific directions, preserve the conflict instead of resolving it; for example, do not connect an instrument term to a technique when the summaries only show them as separate signals. "
        "Every file_roles entry must reference a file_path from the ranked files JSON. "
        "Do not create file roles for missing files. Later extracted objects must still be supported "
        "by source_text from the current chunk only."
    )


def build_extraction_file_summary_prompt(
    *,
    data_package_name: str,
    rank: int,
    file_path: str,
    byte_size: int | None,
    extracted_char_count: int,
    content_windows: list[ExtractionFileContentWindow],
) -> str:
    windows_json = ",\n".join(window.model_dump_json() for window in content_windows)
    return (
        "Summarize one ranked file for later extraction orientation.\n\n"
        f"Data package name: {data_package_name}\n"
        f"Rank: {rank}\n"
        f"File path: {file_path}\n"
        f"Byte size: {byte_size if byte_size is not None else 'unknown'}\n"
        f"Extracted character count: {extracted_char_count}\n\n"
        "Sampled content windows JSON:\n"
        f"[{windows_json}]\n\n"
        "Return an ExtractionFileSummary for this exact file_path and rank. "
        "Use common metadata categories as orientation only, such as instrument settings, "
        "software settings, acquisition settings, processing settings, calibration or reference settings, "
        "sample conditions, identifiers, units, and quantity labels. "
        "These categories are examples only: do not copy them into the output and do not invent parameter_terms. "
        "Only list parameter_terms when the exact labels appear in the sampled content, and warn when they are "
        "settings or labels rather than standalone scientific objects."
    )


def build_extraction_overview_fallback_prompt(
    *,
    data_package_name: str,
    ranked_files: list[RankedFile],
    file_previews: list[ExtractionOverviewFilePreview],
) -> str:
    preview_blocks: list[str] = []
    for preview in file_previews:
        lines = "\n".join(preview.first_lines)
        preview_blocks.append(
            f"Rank {preview.rank}: {preview.file_path}"
            + (f" ({preview.byte_size} bytes)" if preview.byte_size is not None else "")
            + f"\n{lines}"
        )
    ranked_lines = "\n".join(
        f"{file.rank}. {file.file_path}" for file in ranked_files
    )
    return (
        f"Data package name: {data_package_name}\n\n"
        f"Ranked files:\n{ranked_lines}\n\n"
        "Top ranked raw file previews:\n\n"
        + "\n\n---\n\n".join(preview_blocks)
        + "\n\nWrite the orientation text."
    )


def overview_to_prompt_text(
    overview: ExtractionOverview | None,
    *,
    status: ExtractionOverviewStatus | None = None,
) -> str:
    if overview is None:
        return ""
    parts: list[str] = []
    if status:
        parts.append(f"Overview status: {status}")
    if overview.file_roles:
        parts.append(
            "File roles:\n"
            + "\n".join(
                f"- {role.file_path}: {role.role}"
                + (
                    f" ({'; '.join(role.extraction_notes)})"
                    if role.extraction_notes
                    else ""
                )
                for role in overview.file_roles
            )
        )
    for label, values in (
        ("Observed signals", overview.observed_signals),
        ("Suggested interpretations", overview.suggested_interpretations),
        ("Conflicts/uncertainties", overview.conflicts_or_uncertainties),
    ):
        if values:
            parts.append(label + ":\n" + "\n".join(f"- {value}" for value in values))
    return "\n\n".join(parts)


def file_summary_to_prompt_text(summary: ExtractionFileSummary | None) -> str:
    if summary is None:
        return ""
    parts = [
        f"File path: {summary.file_path}",
        f"Rank: {summary.rank}",
        f"Summary status: {summary.status}",
    ]
    if summary.data_format:
        parts.append(f"Data format: {summary.data_format}")
    if summary.explicit_purpose:
        parts.append(f"Explicit purpose: {summary.explicit_purpose}")
    for label, values in (
        ("Data characteristics", summary.data_characteristics),
        ("Purpose evidence", summary.purpose_evidence),
        ("Metadata signals", summary.metadata_signals),
        ("Detected identifiers", summary.detected_identifiers),
        ("Instrument/software terms", summary.instrument_or_software_terms),
        ("Parameter terms", summary.parameter_terms),
        ("Known traps", summary.known_traps),
        ("Uncertainty notes", summary.uncertainty_notes),
    ):
        if values:
            parts.append(label + ":\n" + "\n".join(f"- {value}" for value in values))
    return "\n".join(parts)
