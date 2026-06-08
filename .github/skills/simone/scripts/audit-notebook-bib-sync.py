#!/usr/bin/env python3
"""Audit whether known SIMONE thesis NotebookLM sources map to BibTeX entries."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


DEFAULT_REPO = Path(r"C:\Users\simcl\Documents\GitHub\Semantic Inference Module for Ontology-driven Node Extraction (SIMONE)")
DEFAULT_BIB = Path("docs/thesis/bibliography/references.bib")
DEFAULT_NOTEBOOKS = {
    "c1644a03-c921-48a5-96e8-3535ad62b5be": "Development of an LLM supported workflow for semantic metadata extraction from catalytic experiments",
    "e1ee729b-32f8-4a23-a17a-fb9ad07eac61": "LLMs for Scientific Semantic Data Extraction",
    "ecf48655-d9e6-4654-8dd8-69a8493674c3": "Catalysis, Ontologies, and Semantic Metadata",
    "75ad7b6d-1d53-4f43-a7c5-d1bf326411a7": "LLM-Based Metadata Extraction from Scientific Resources",
}
SCRIPT_DIR = Path(__file__).resolve().parent
WRAPPER = SCRIPT_DIR / "notebooklm-local.ps1"


KNOWN_SOURCE_TO_KEY = {
    "ChemDCAT-AP Documentation": "chemdcatapDocumentation",
    "DCAT-AP+ Documentation": "dcatapplusDocumentation",
    "NFDI4Cat_Whitepaper-2024-04-26.pdf": "nfdi4cat2024whitepaper",
    "Repo4Cat_How_to_Setup_Configure_and_Operate_a_Data_Repository_for_Catalysis-Related_Sciences.pdf": "kushnarenko2025repo4cat",
    "MTSR_2025_paper_1431-1.pdf": "stroemert2026chemdcat",
    "ChemCatChem - 2021 - Wulf - A Unified Research Data Infrastructure for Catalysis Research   Challenges and Concepts.pdf": "wulf2021unified",
    "ChemCatChem - 2025 - Mendes - Data as a Key Resource in Catalysis  A Community Account.pdf": "mendes2025data",
    "s13321-024-00807-2.pdf": "behr2024ontologies4cat",
    "s41524-025-01953-3_reference.pdf": "salim2026methane",
    "use-of-large-language-models-for-extracting-and-analyzing-data-from-heterogeneous-catalysis-literature.pdf": "walls2025catminer",
    "unleashing-the-power-of-knowledge-extraction-from-scientific-literature-in-catalysis.pdf": "zhang2022knowledge",
}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def latex_clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[{}]", "", value or "")).strip()


def field(entry: str, name: str) -> str:
    pattern = rf"\b{name}\s*=\s*([{{\"])(.*?)(?:\1|}})\s*,?\s*(?=\n\s*\w+\s*=|\n\s*}})"
    match = re.search(pattern, entry, flags=re.IGNORECASE | re.DOTALL)
    return latex_clean(match.group(2)) if match else ""


def parse_bib(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    for match in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        start = match.start()
        next_match = re.search(r"\n@", text[match.end():])
        end = match.end() + next_match.start() if next_match else len(text)
        body = text[start:end]
        key = match.group(2)
        entries[key] = {
            "key": key,
            "title": field(body, "title"),
            "title_norm": normalize(field(body, "title")),
            "doi": normalize_doi(field(body, "doi")),
            "eprint": normalize_arxiv(field(body, "eprint")) or normalize_arxiv(field(body, "url")),
            "url": field(body, "url"),
        }
    return entries


def normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value.replace("\\_", "_").rstrip(".,;)")


def normalize_arxiv(value: str) -> str:
    match = re.search(r"(?i)(?:arxiv:|/abs/)?(\d{4}\.\d{4,5})(?:v\d+)?", value or "")
    return match.group(1).lower() if match else ""


def doi_from_name(name: str) -> str:
    stem = Path(name).stem
    match = re.search(r"(?i)(10\.\d{4,9}[_./;()A-Z0-9-]+)", stem)
    if not match:
        return ""
    doi = match.group(1)
    if "/" not in doi and "_" in doi:
        doi = doi.replace("_", "/", 1)
    return normalize_doi(doi)


def notebook_sources(notebook_id: str) -> list[dict]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(WRAPPER),
        "source",
        "list",
        "-n",
        notebook_id,
        "--json",
    ]
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    payload = json.loads(proc.stdout)
    return payload.get("sources", payload if isinstance(payload, list) else [])


def source_title(source: dict) -> str:
    return (
        source.get("title")
        or source.get("display_title")
        or source.get("name")
        or source.get("source_title")
        or source.get("file_name")
        or source.get("filename")
        or ""
    )


def infer_key(title: str, entries: dict[str, dict]) -> tuple[str | None, str]:
    mapped = KNOWN_SOURCE_TO_KEY.get(title)
    if mapped:
        return mapped, "known-map"

    arxiv = normalize_arxiv(title)
    if arxiv:
        for key, entry in entries.items():
            if entry.get("eprint") == arxiv:
                return key, "arxiv"

    doi = doi_from_name(title)
    if doi:
        for key, entry in entries.items():
            if entry.get("doi") == doi:
                return key, "doi"

    title_norm = normalize(Path(title).stem)
    if title_norm:
        for key, entry in entries.items():
            entry_title = entry.get("title_norm", "")
            if entry_title and (title_norm == entry_title or title_norm in entry_title or entry_title in title_norm):
                return key, "title"

    return None, "unmatched"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("SIMONE_REPO"), help="SIMONE repo root")
    parser.add_argument("--bib", default=str(DEFAULT_BIB), help="BibTeX path relative to repo or absolute")
    parser.add_argument("--notebook", action="append", help="NotebookLM notebook ID; repeatable")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    repo = Path(args.repo) if args.repo else DEFAULT_REPO
    bib_path = Path(args.bib)
    if not bib_path.is_absolute():
        bib_path = repo / bib_path
    if not bib_path.exists():
        print(f"audit-notebook-bib-sync: BibTeX file not found: {bib_path}", file=sys.stderr)
        return 2

    entries = parse_bib(bib_path)
    notebooks = {nid: DEFAULT_NOTEBOOKS.get(nid, "") for nid in (args.notebook or DEFAULT_NOTEBOOKS.keys())}
    mappings: list[dict] = []
    per_notebook: list[dict] = []
    notebook_errors: list[dict] = []

    for notebook_id, notebook_title in notebooks.items():
        try:
            sources = notebook_sources(notebook_id)
        except Exception as exc:
            error_item = {
                "notebook_id": notebook_id,
                "notebook_title": notebook_title,
                "source_count": 0,
                "matched_count": 0,
                "error": str(exc),
            }
            per_notebook.append(error_item)
            notebook_errors.append(error_item)
            continue

        matched_count = 0
        for source in sources:
            title = source_title(source)
            key, method = infer_key(title, entries)
            exists = bool(key and key in entries)
            matched_count += 1 if exists else 0
            mappings.append({
                "notebook_id": notebook_id,
                "notebook_title": notebook_title,
                "source_id": source.get("source_id") or source.get("id"),
                "source_title": title,
                "status": source.get("status"),
                "bib_key": key,
                "match_method": method,
                "bib_entry_exists": exists,
            })
        per_notebook.append({
            "notebook_id": notebook_id,
            "notebook_title": notebook_title,
            "source_count": len(sources),
            "matched_count": matched_count,
        })

    missing = [item for item in mappings if not item["bib_entry_exists"]]
    report = {
        "bib_path": str(bib_path),
        "bib_entry_count": len(entries),
        "notebook_count": len(notebooks),
        "source_count": len(mappings),
        "matched_count": len(mappings) - len(missing),
        "missing_count": len(missing),
        "notebook_error_count": len(notebook_errors),
        "per_notebook": per_notebook,
        "mappings": mappings,
        "missing": missing,
        "notebook_errors": notebook_errors,
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"BibTeX entries: {report['bib_entry_count']}")
        print(f"NotebookLM sources: {report['source_count']}")
        print(f"Matched sources: {report['matched_count']}")
        print(f"Missing sources: {report['missing_count']}")
        print(f"Notebook access errors: {report['notebook_error_count']}")
        for item in per_notebook:
            if "error" in item:
                print(f"- {item['notebook_id']}: ERROR {item['error']}")
            else:
                print(f"- {item['notebook_id']}: {item['matched_count']}/{item['source_count']} matched")
        if missing:
            print("\nUnmatched sources:")
            for item in missing:
                print(f"- {item['source_title']} ({item['notebook_id']})")

    if notebook_errors:
        return 2
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
