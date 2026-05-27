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

FileRankingResult = list[RankedFile]

FILE_RANKING_SYSTEM_PROMPT = """
You rank files from a research data package for initial metadata extraction.

Return only a FileRankingResult JSON object. Rank files by their likelihood of
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
    scored_files.sort(key=lambda x: x[1], reverse=True)
    return [
        RankedFile(rank=i + 1, file_path=file.file_path)
        for i, (file, score) in enumerate(scored_files)
    ]

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

def _file_extension(file: FileContext) -> str:
    return "." + file.file_path.rsplit(".", 1)[-1] if "." in file.file_path else ""
