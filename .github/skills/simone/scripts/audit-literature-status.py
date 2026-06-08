#!/usr/bin/env python3
"""Summarize SIMONE thesis literature status across BibTeX and NotebookLM.

Drive audit is intentionally optional for this repo-specific skill. If a Drive
audit script is later added and MATON_API_KEY is configured, this script can
delegate to it; otherwise it reports Drive status as unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_AUDIT = SCRIPT_DIR / "audit-notebook-bib-sync.py"
DRIVE_AUDIT = SCRIPT_DIR / "audit-drive-bib-sync.py"


def run_json(cmd: list[str]) -> tuple[int, dict | None, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout.strip():
        try:
            return proc.returncode, json.loads(proc.stdout), proc.stderr.strip()
        except json.JSONDecodeError:
            return proc.returncode, None, proc.stdout.strip() or proc.stderr.strip()
    return proc.returncode, None, proc.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("SIMONE_REPO"))
    parser.add_argument("--notebook", action="append", help="NotebookLM notebook ID; repeatable")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    notebook_cmd = [sys.executable, str(NOTEBOOK_AUDIT), "--json"]
    if args.repo:
        notebook_cmd.extend(["--repo", args.repo])
    for notebook_id in args.notebook or []:
        notebook_cmd.extend(["--notebook", notebook_id])

    notebook_code, notebook, notebook_error = run_json(notebook_cmd)

    drive = None
    drive_status = "unavailable"
    drive_error = ""
    if DRIVE_AUDIT.exists() and os.environ.get("MATON_API_KEY"):
        drive_code, drive, drive_error = run_json([sys.executable, str(DRIVE_AUDIT), "--json"])
        drive_status = "ok" if drive_code == 0 else "error"
    elif DRIVE_AUDIT.exists():
        drive_status = "skipped_missing_maton_api_key"

    report = {
        "notebook_status": "ok" if notebook_code == 0 else "issues",
        "notebook_error": notebook_error,
        "notebook": notebook,
        "drive_status": drive_status,
        "drive_error": drive_error,
        "drive": drive,
        "summary": {
            "bib_entry_count": notebook.get("bib_entry_count") if notebook else None,
            "notebook_count": notebook.get("notebook_count") if notebook else None,
            "notebook_source_count": notebook.get("source_count") if notebook else None,
            "notebook_missing_bib_count": notebook.get("missing_count") if notebook else None,
        },
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report["summary"]
        print(f"BibTeX entries: {summary['bib_entry_count']}")
        print(f"NotebookLM notebooks: {summary['notebook_count']}")
        print(f"NotebookLM sources: {summary['notebook_source_count']}")
        print(f"NotebookLM missing BibTeX: {summary['notebook_missing_bib_count']}")
        print(f"Drive audit: {drive_status}")
        if notebook_error:
            print(f"Notebook audit note: {notebook_error}")
        if drive_error:
            print(f"Drive audit note: {drive_error}")

    return 1 if notebook_code not in (0, None) else 0


if __name__ == "__main__":
    raise SystemExit(main())
