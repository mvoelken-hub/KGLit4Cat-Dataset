from __future__ import annotations

from typing import Literal

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
    dataset_theme: str = Field(
        "",
        description="Short profile-independent description of the package context.",
    )
    summary: str = Field(
        "",
        description="Compact orientation text for later chunk extraction calls.",
    )
    file_roles: list[ExtractionOverviewFileRole] = Field(default_factory=list)
    likely_activities: list[str] = Field(default_factory=list)
    likely_entities: list[str] = Field(default_factory=list)
    likely_resources: list[str] = Field(default_factory=list)
    likely_methods: list[str] = Field(default_factory=list)
    instrument_or_device_names: list[str] = Field(default_factory=list)
    analytical_techniques: list[str] = Field(default_factory=list)
    sample_identifiers: list[str] = Field(default_factory=list)
    compound_names: list[str] = Field(default_factory=list)
    metadata_sources: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    parameter_attachment_guidance: list[str] = Field(default_factory=list)
    known_traps: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


class ExtractionOverviewFilePreview(BaseModel):
    rank: int = Field(..., ge=1)
    file_path: str
    byte_size: int | None = None
    first_lines: list[str] = Field(default_factory=list)


EXTRACTION_OVERVIEW_SYSTEM_PROMPT = """You are a scientific data archivist. Analyze research artifact bundles and extract structured context metadata.
Your output is guidance only. It is not evidence. Later extraction calls must still cite source_text from their current chunk.

Focus on:
- dataset theme and scientific/instrument context;
- precise instrument or device names, analytical techniques, sample identifiers, compound names, file roles, metadata provenance, keywords, and uncertainty;
- likely meaningful activities, entities, resources, methods, instruments, and software;
- concrete per-file guidance for how later chunk calls should interpret each ranked file;
- how low-level parameters should attach to meaningful objects;
- traps that would cause bad extraction objects.

Do not invent final metadata.
Do not mention files that are not present in the ranked files or per-file summaries.
Fill every field with concise evidence-grounded values. Use empty lists when no entities, relationships, metadata sources, or keywords can be identified.
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
Emphasize file roles, likely dataset context, parameter attachment rules, and traps to avoid.
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
        summary.model_dump_json() for summary in (file_summaries or [])
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
        "If the content shows parameter labels such as PLW1, PULPROG, SFO1, TD, D1, or NS, "
        "list them as parameter_terms and warn that they are settings/labels, not standalone scientific objects."
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
    if overview.dataset_theme:
        parts.append(f"Dataset theme: {overview.dataset_theme}")
    if overview.summary:
        parts.append(f"Summary: {overview.summary}")
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
        ("Metadata sources", overview.metadata_sources),
        ("Analytical techniques", overview.analytical_techniques),
        ("Instruments/devices", overview.instrument_or_device_names),
        ("Sample identifiers", overview.sample_identifiers),
        ("Compound names", overview.compound_names),
        ("Likely activities", overview.likely_activities),
        ("Likely entities", overview.likely_entities),
        ("Likely resources", overview.likely_resources),
        ("Likely methods", overview.likely_methods),
        ("Keywords", overview.keywords),
        ("Parameter attachment guidance", overview.parameter_attachment_guidance),
        ("Known traps", overview.known_traps),
        ("Uncertainty notes", overview.uncertainty_notes),
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
