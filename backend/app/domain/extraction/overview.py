from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.extraction.file_ranking import RankedFile


ExtractionOverviewStatus = Literal["structured", "unstructured_fallback", "failed"]
ExtractionFileSummaryStatus = Literal["summarized", "failed"]
InitialFileSummaryStatus = Literal["completed", "partial", "failed"]
ExtractionOverviewNodeKind = Literal["package", "directory", "file", "group"]
ExtractionOverviewRelation = Literal[
    "contains",
    "describes",
    "derives_from",
    "documents",
    "configures",
    "parameterizes",
    "generated_by",
    "related_to",
    "uncertain_relation",
]


class ExtractionOverviewNode(BaseModel):
    node_id: str = Field(
        ...,
        description="Stable graph node id. Use file:<path> for file nodes.",
    )
    label: str = Field(..., description="Short display label for the node.")
    kind: ExtractionOverviewNodeKind = "group"
    file_path: str | None = Field(
        default=None,
        description="Original package file path for file nodes only.",
    )
    rank: int | None = Field(
        default=None,
        ge=1,
        description="Ranked file position when the node represents a ranked file.",
    )
    summary: str = Field(
        "",
        description="Concise orientation note grounded in summaries/previews.",
    )


class ExtractionOverviewEdge(BaseModel):
    edge_id: str = Field(
        "",
        description="Stable edge id; backend may replace empty ids.",
    )
    source: str = Field(..., description="Source node_id.")
    target: str = Field(..., description="Target node_id.")
    relation: ExtractionOverviewRelation = "related_to"
    evidence: list[str] = Field(
        default_factory=list,
        description="Short phrases from summaries/previews supporting the relation.",
    )
    note: str = Field("", description="Concise relation-specific orientation note.")


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
    file_path: str
    status: ExtractionFileSummaryStatus = "summarized"
    data_format: str = Field(
        "",
        description="Obvious file format or syntax visible in the sampled content.",
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
    instrument_or_software_terms_and_settings: list[str] = Field(default_factory=list)
    quantitative_signals: list[str] = Field(default_factory=list)


class InitialFileSummaryDiagnosticRecord(BaseModel):
    file_path: str
    reason: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class InitialFileSummaryDiagnostics(BaseModel):
    records: list[InitialFileSummaryDiagnosticRecord] = Field(default_factory=list)


class InitialFileSummaryProgress(BaseModel):
    total_files: int = Field(default=0, ge=0)
    processed_files: int = Field(default=0, ge=0)
    summarized_files: int = Field(default=0, ge=0)
    skipped_files: int = Field(default=0, ge=0)
    failed_files: int = Field(default=0, ge=0)
    current_file_path: str | None = None


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
    nodes: list[ExtractionOverviewNode] = Field(
        default_factory=list,
        description="File-centered package graph nodes.",
    )
    edges: list[ExtractionOverviewEdge] = Field(
        default_factory=list,
        description="Controlled package graph relations.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Unresolved graph-level scope, identity, and relation uncertainties.",
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


class InitialOverviewPromptDiagnostic(BaseModel):
    status: Literal["structured_success", "structured_failure"] = "structured_success"
    prompt_budget: dict[str, Any] = Field(default_factory=dict)
    included_summary_paths: list[str] = Field(default_factory=list)
    dropped_summary_paths: list[str] = Field(default_factory=list)
    included_ranked_paths: list[str] = Field(default_factory=list)
    dropped_ranked_paths: list[str] = Field(default_factory=list)
    included_preview_paths: list[str] = Field(default_factory=list)
    dropped_preview_paths: list[str] = Field(default_factory=list)
    hard_truncated: bool = False
    error_type: str = ""
    message: str = ""
    last_error_type: str = ""
    last_error: str = ""
    failed_response_excerpt: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class ExtractionOverviewFilePreview(BaseModel):
    rank: int = Field(..., ge=1)
    file_path: str
    byte_size: int | None = None
    first_lines: list[str] = Field(default_factory=list)


EXTRACTION_OVERVIEW_SYSTEM_PROMPT = """You are a scientific data archivist creating a conservative file-centered package graph for heterogeneous scientific data.
Your output is guidance only. It is not evidence. Later extraction calls must still cite source_text from their current chunk.
Return raw JSON only. Never wrap the JSON in Markdown fences, code blocks, language labels, or prose.

Create graph nodes, controlled relation edges, and unresolved uncertainties:
- The backend has already seeded package, directory, file, conservative group nodes, real path-containment edges, and conservative summary-supported group edges.
- Do not create contains edges. Ranked order is prioritization metadata only; it is never containment evidence.
- First group files by shared package role before adding file-to-file relations.
- nodes: add inferred group nodes when summaries explicitly support shared roles such as dataset documentation, acquisition settings, processing settings, audit/provenance, instrument settings, method/program logic, raw data, processed data, derived results, or ambiguous supporting resources. Reuse seeded file/directory node IDs when connecting relations.
- edges: connect files to group nodes first using describes, documents, configures, parameterizes, generated_by, related_to, or uncertain_relation. Add group-to-group edges only when supported. File-to-file semantic edges are secondary.
- uncertainties: unresolved scope, identity, modality, provenance, raw/processed, or title/owner/origin ambiguities.

Keep the graph file-centered. Prefer files, directories, and generic groups such as raw data, processed data, acquisition settings, processing settings, audit/provenance, metadata, derived results, and quality notes.
Only create scientific/domain group nodes when summaries explicitly support the grouping. Do not invent final metadata, sample identity, method identity, instrument identity, creator, owner, or dataset title.
Do not turn file-local headers, local titles, audit labels, or resource titles into dataset-level identity. Represent them as file-local nodes/notes or uncertainties when useful.
When one explicit signal appears inconsistent with another, preserve the unresolved relation with uncertain_relation or uncertainties instead of selecting one interpretation.
When validated per-file summaries are present, do not claim that summaries are unavailable. If no group can be formed, explain which specific relation evidence is missing.

Do not invent final metadata.
Do not mention files that are not present in the ranked files or per-file summaries.
Every semantic edge must include evidence phrases from summaries or previews. Placeholder evidence such as ranked, listed, top file, or empty evidence is not useful; use uncertainties instead.
Fill every field with concise evidence-grounded values. Use empty lists when no additional semantic nodes, group-oriented edges, or uncertainties can be identified.
"""


EXTRACTION_FILE_SUMMARY_SYSTEM_PROMPT = """You are a scientific data archivist summarizing one file from a research artifact bundle.
Your output is guidance only. It is not extraction evidence for later chunk calls.

Summarize only facts visible in this file's sampled content, filename, or obvious syntax.
The explicit_purpose field is strict: fill it only when the sampled content or filename directly states the file's purpose. Otherwise leave it empty.
metadata_signals should absorb useful observable characteristics and file-local metadata cues without inventing dataset-level identity.
instrument_or_software_terms_and_settings should list concrete visible instrument, software, method, and setting terms when supported.
quantitative_signals should be coarse orientation only, such as explicit numeric settings, quantity labels, or visible units. Do not dump exhaustive parameter labels and do not create structured quantity facts.
Evidence fields must quote short snippets from this file only.
Do not output known traps, detected identifiers, uncertainty notes, exhaustive parameter terms, or structured quantitative attributes.
Do not invent dataset purpose, instrument names, file roles, sample identities, or software-project files.
"""


EXTRACTION_OVERVIEW_FALLBACK_SYSTEM_PROMPT = """You are a scientific data archivist. Write a compact plain-text file-relation orientation for chunk-level metadata extraction.
The text is guidance only and must not be treated as evidence. Keep it concise and profile-independent.
Emphasize file/package relations, raw-to-processed or parameter-to-resource links, provenance, and unresolved conflicts or limitations.
"""


def build_extraction_overview_prompt(
    *,
    data_package_name: str,
    ranked_files: list[RankedFile],
    file_previews: list[ExtractionOverviewFilePreview],
    file_summaries: list[ExtractionFileSummary] | None = None,
    seeded_overview: ExtractionOverview | None = None,
) -> str:
    return "".join(
        content
        for _, content in build_extraction_overview_prompt_components(
            data_package_name=data_package_name,
            ranked_files=ranked_files,
            file_summaries=file_summaries,
            file_previews=file_previews,
            seeded_overview=seeded_overview,
        )
    )


def build_extraction_overview_prompt_components(
    *,
    data_package_name: str,
    ranked_files: list[RankedFile],
    file_previews: list[ExtractionOverviewFilePreview],
    file_summaries: list[ExtractionFileSummary] | None = None,
    seeded_overview: ExtractionOverview | None = None,
) -> list[tuple[str, str]]:
    preview_json = ",\n".join(preview.model_dump_json() for preview in file_previews)
    ranked_json = ",\n".join(file.model_dump_json() for file in ranked_files)
    summary_json = ",\n".join(
        summary.model_dump_json(exclude_defaults=True) for summary in (file_summaries or [])
    )
    seed_json = seeded_overview.model_dump_json(exclude_defaults=True) if seeded_overview else "{}"
    summary_section = (
        "Validated per-file summaries JSON:\n"
        f"[{summary_json}]\n\n"
        if file_summaries
        else ""
    )
    return [
        (
            "intro_and_counts",
            "Analyze the research artifact archive and extract a graph-shaped ExtractionOverview. "
            "The backend has summarized text-extractable files, ranked summarized files, and seeded deterministic path containment.\n\n"
            f"Data package name: {data_package_name}\n"
            f"Ranked file count: {len(ranked_files)}\n"
            f"Per-file summary count: {len(file_summaries or [])}\n"
            f"Fallback preview file count: {len(file_previews)}\n\n",
        ),
        ("ranked_files_json", "Ranked files JSON:\n" f"[{ranked_json}]\n\n"),
        ("validated_summaries_json", summary_section),
        (
            "seeded_graph_json",
            "Backend-seeded package/directory/file/group graph JSON:\n"
            f"{seed_json}\n\n",
        ),
        (
            "fallback_previews_json",
            "Fallback raw file previews JSON:\n" f"[{preview_json}]\n\n",
        ),
        (
            "final_task_instructions",
            "Create an ExtractionOverview package graph that will orient later one-shot chunk extraction calls. "
            "Use the per-file summaries as the primary input and previews only as fallback context when summaries are absent. "
            "Treat this as package graph triage, not final scientific interpretation. "
            "Do not create contains edges: the backend will add real package/directory/file containment from paths. "
            "Ranked order means extraction priority only; it is not evidence that one file contains another. "
            "Group files first: create generic group nodes for shared package roles, then connect file nodes to those groups with semantic edges. "
            "Prefer file-to-group and group-to-group edges over file-to-file edges. "
            "When connecting to files, use file node IDs exactly as file:<file_path> from the ranked files JSON. "
            "When summaries are present, do not claim that per-file summaries are unavailable. "
            "Use uncertain_relation or uncertainties for weak or ambiguous links. "
            "Later extracted objects must still be supported "
            "by source_text from the current chunk only.",
        ),
    ]


def build_extraction_file_summary_prompt(
    *,
    data_package_name: str,
    file_path: str,
    byte_size: int | None,
    extracted_char_count: int,
    content_windows: list[ExtractionFileContentWindow],
) -> str:
    windows_json = ",\n".join(window.model_dump_json() for window in content_windows)
    return (
        "Summarize one file for later extraction orientation.\n\n"
        f"Data package name: {data_package_name}\n"
        f"File path: {file_path}\n"
        f"Byte size: {byte_size if byte_size is not None else 'unknown'}\n"
        f"Extracted character count: {extracted_char_count}\n\n"
        "Sampled content windows JSON:\n"
        f"[{windows_json}]\n\n"
        "Return an ExtractionFileSummary for this exact file_path. "
        "Use common metadata categories as orientation only, such as instrument settings, "
        "software settings, acquisition settings, processing settings, calibration or reference settings, "
        "sample conditions, identifiers, units, and quantity labels. "
        "These categories are examples only: do not copy them into the output. "
        "Use metadata_signals for concise file-local orientation, instrument_or_software_terms_and_settings for visible "
        "instrument/software/method/setting terms, and quantitative_signals only for coarse quantitative orientation."
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
    if overview.nodes:
        parts.append(
            "Overview graph nodes:\n"
            + "\n".join(
                f"- {node.node_id}: {node.label} [{node.kind}]"
                + (f" path={node.file_path}" if node.file_path else "")
                + (f" - {node.summary}" if node.summary else "")
                for node in overview.nodes[:12]
            )
        )
    if overview.edges:
        parts.append(
            "Overview graph relations:\n"
            + "\n".join(
                f"- {edge.source} -[{edge.relation}]-> {edge.target}"
                + (f" ({edge.note})" if edge.note else "")
                for edge in overview.edges[:16]
            )
        )
    if overview.uncertainties:
        parts.append(
            "Overview uncertainties:\n"
            + "\n".join(f"- {value}" for value in overview.uncertainties[:8])
        )
    return "\n\n".join(parts)


def file_summary_to_prompt_text(summary: ExtractionFileSummary | None) -> str:
    if summary is None:
        return ""
    parts = [
        f"File path: {summary.file_path}",
        f"Summary status: {summary.status}",
    ]
    if summary.data_format:
        parts.append(f"Data format: {summary.data_format}")
    if summary.explicit_purpose:
        parts.append(f"Explicit purpose: {summary.explicit_purpose}")
    for label, values in (
        ("Purpose evidence", summary.purpose_evidence),
        ("Metadata signals", summary.metadata_signals),
        (
            "Instrument/software terms and settings",
            summary.instrument_or_software_terms_and_settings,
        ),
        ("Quantitative signals", summary.quantitative_signals),
    ):
        if values:
            parts.append(label + ":\n" + "\n".join(f"- {value}" for value in values))
    return "\n".join(parts)
