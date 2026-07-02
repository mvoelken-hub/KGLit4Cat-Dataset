from typing import Any

from pydantic import BaseModel, Field


class FileContext(BaseModel):
    file_path: str = Field(..., description="Path to the file within the data package.")
    byte_size: int | None = Field(None, description="Size of the file in bytes, if available.")


class RankedFile(BaseModel):
    """
    Represents a file ranked for relevance to metadata extraction.
    The ranking is based on file path, name, extension, size, and other heuristics.
    """
    rank: int = Field(..., description="1-based rank of the file, with 1 being the most relevant.", ge=1)
    file_path: str = Field(..., description="Path to the file within the data package.")
    score: float | None = Field(
        default=None,
        description="Deterministic usefulness score for metadata extraction priority.",
        ge=0.0,
        le=1.0,
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Short deterministic reasons explaining the rank.",
    )


class FileRankingResult(BaseModel):
    files: list[RankedFile] = Field(
        default_factory=list,
        description="Ranked files, sorted from most to least relevant.",
    )

FILE_RANKING_SYSTEM_PROMPT = """
You rank files from a research data package for initial metadata extraction.

Return only JSON. Rank files by their likelihood of
containing dataset-level metadata, experimental context, sample descriptions,
instrument details, acquisition methods, processing notes, or relationships
between files.

Rules:
- Use only file_path values from the provided input.
- Prefer README files, metadata tables, manifests, reports, protocols,
  notebooks, scripts, and compact processed tables with descriptive names.
- Deprioritize raw binaries, large images, archives, caches, and files whose
  names suggest they contain only numeric measurements with no context.
- Do not invent files or inspect content that was not provided.
"""

def build_file_ranking_prompt(
    file_context_list: list[FileContext],
    dataset_name: str | None = None,
) -> str:
    
    if not file_context_list:
        return ""
    
    package_line = (
        f"Data package name: {dataset_name}\n"
        if dataset_name
        else ""
    )

    return (
        f"{package_line}"
        f"Rank up these {len(file_context_list)} files for initial context extraction.\n"
        "Input files JSON:\n"
        f"[{', '.join(file.model_dump_json(exclude_none=True) for file in file_context_list)}]"
    )


def fallback_file_ranking(
    files: list[FileContext],
) -> FileRankingResult:
    scored_files = [(file, _fallback_score(file)) for file in files]
    scored_files.sort(key=lambda x: (-x[1], x[0].file_path))
    return FileRankingResult(
        files=[
            RankedFile(
                rank=i + 1,
                file_path=file.file_path,
                score=score,
                reasons=_fallback_reasons(file),
            )
            for i, (file, score) in enumerate(scored_files)
        ]
    )


def rank_summarized_files(
    summaries: list[Any],
    *,
    file_contexts: dict[str, FileContext] | None = None,
) -> FileRankingResult:
    scored = [
        (summary, *_summary_score(summary, file_contexts.get(summary.file_path) if file_contexts else None))
        for summary in summaries
        if getattr(summary, "file_path", "")
    ]
    scored.sort(key=lambda item: (-item[1], getattr(item[0], "file_path", "")))
    return FileRankingResult(
        files=[
            RankedFile(
                rank=index + 1,
                file_path=summary.file_path,
                score=score,
                reasons=reasons,
            )
            for index, (summary, score, reasons) in enumerate(scored)
        ]
    )

def _fallback_score(file: FileContext) -> float:
    text = f"{file.file_path}".lower()
    extension = _file_extension(file).lower()
    score = 0.2

    if any(marker in text for marker in ("readme", "metadata", "manifest", "method", "protocol")):
        score += 0.45
    if any(marker in text for marker in ("sample", "instrument", "experiment", "report", "summary")):
        score += 0.25
    if extension in {".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".xlsx", ".xls", ".pdf"}:
        score += 0.2
    if extension in {".ipynb", ".py", ".r", ".m"}:
        score += 0.1
    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".zip", ".gz", ".raw", ".h5", ".hdf5", ".npy"}:
        score -= 0.25
    if file.byte_size is not None and file.byte_size > 100_000:
        score -= 0.1

    return round(max(0.0, min(1.0, score)), 2)


def _summary_score(summary: Any, file: FileContext | None) -> tuple[float, list[str]]:
    values = _summary_values(summary)
    text = " ".join(values).lower()
    score = 0.08
    reasons: list[str] = []

    if _contains_any(text, ("dataset description", "package description", "readme", "metadata", "manifest")):
        score += 0.38
        reasons.append("explicit dataset/package documentation")
    if _contains_any(text, ("method", "protocol", "program", "plan")):
        score += 0.28
        reasons.append("method or protocol orientation")
    if _contains_any(text, ("acquisition", "measurement settings", "experiment parameters")):
        score += 0.26
        reasons.append("acquisition settings")
    if _contains_any(text, ("processing", "processed", "post-processing", "process parameters")):
        score += 0.22
        reasons.append("processing settings")
    if _contains_any(text, ("instrument", "software", "device", "calibration", "reference")):
        score += 0.2
        reasons.append("instrument/software/settings terms")
    if _contains_any(text, ("sample", "specimen", "material", "condition", "solvent")):
        score += 0.16
        reasons.append("sample or condition context")
    if _contains_any(text, ("audit", "provenance", "history", "log")):
        score += 0.1
        reasons.append("audit/provenance context")

    if not any(values):
        score -= 0.12
        reasons.append("empty or low-signal summary")

    if file is not None:
        path_score = _fallback_score(file)
        score += (path_score - 0.2) * 0.25
        for reason in _fallback_reasons(file):
            if reason not in reasons:
                reasons.append(reason)

    score = round(max(0.0, min(1.0, score)), 3)
    if not reasons:
        reasons.append("stable fallback ordering")
    return score, reasons[:6]


def _summary_values(summary: Any) -> list[str]:
    values = [
        getattr(summary, "file_path", ""),
        getattr(summary, "data_format", ""),
        getattr(summary, "explicit_purpose", ""),
        getattr(summary, "information_summary", ""),
    ]
    return [value for value in values if value]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _fallback_reasons(file: FileContext) -> list[str]:
    text = f"{file.file_path}".lower()
    extension = _file_extension(file).lower()
    reasons: list[str] = []
    if any(marker in text for marker in ("readme", "metadata", "manifest", "method", "protocol")):
        reasons.append("metadata-oriented path")
    if any(marker in text for marker in ("sample", "instrument", "experiment", "report", "summary")):
        reasons.append("context-oriented path")
    if extension in {".md", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".xlsx", ".xls", ".pdf"}:
        reasons.append("text-like extension")
    if extension in {".ipynb", ".py", ".r", ".m"}:
        reasons.append("script/notebook extension")
    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".zip", ".gz", ".raw", ".h5", ".hdf5", ".npy"}:
        reasons.append("low-context or opaque extension")
    if file.byte_size is not None and file.byte_size > 100_000:
        reasons.append("large file penalty")
    return reasons or ["path/extension fallback"]

def _file_extension(file: FileContext) -> str:
    return "." + file.file_path.rsplit(".", 1)[-1] if "." in file.file_path else ""
