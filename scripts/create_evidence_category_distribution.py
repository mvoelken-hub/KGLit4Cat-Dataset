"""Recreate the raw-baseline evidence-category distribution figure.

Example:
    uv run --with matplotlib python scripts/create_evidence_category_distribution.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_EVIDENCE_DIR = Path(
    "backend/.runtime/output/2e38a6a6/evidence_notes/fixed_tokens/gemma3_12b-cloud"
)
DEFAULT_OUTPUT_STEM = Path(
    "docs/thesis/figures/evidence_category_distribution_raw_baseline"
)
DEFAULT_CATEGORY_ORDER = [
    "resource_signal",
    "instrument_signal",
    "measurement_signal",
    "surrounding_signal",
    "measurement_condition",
    "activity_signal",
    "software_signal",
    "method_signal",
]


def _load_notes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in (
            "notes",
            "portable_notes",
            "contextual_notes",
            "rejected_notes",
            "candidates",
            "filtered_notes",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find evidence-note list in {path}")


def _category(note: dict[str, Any]) -> str:
    if isinstance(note.get("category"), str):
        return note["category"]
    candidate = note.get("candidate")
    if isinstance(candidate, dict) and isinstance(candidate.get("category"), str):
        return candidate["category"]
    return "other"


def _counter(path: Path) -> Counter[str]:
    return Counter(_category(note) for note in _load_notes(path))


def _label(category: str) -> str:
    return category.replace("_", "\n")


def create_chart(evidence_dir: Path, output_stem: Path) -> None:
    portable = _counter(evidence_dir / "portable_evidence.json")
    contextual = _counter(evidence_dir / "contextual_evidence.json")
    rejected = _counter(evidence_dir / "rejected_evidence.json")

    categories = [
        category
        for category in DEFAULT_CATEGORY_ORDER
        if portable[category] or contextual[category] or rejected[category]
    ]
    all_categories = set(portable) | set(contextual) | set(rejected)
    remaining = sorted(all_categories - set(categories))
    categories.extend(remaining)

    x = list(range(len(categories)))
    width = 0.26

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.55), constrained_layout=True)
    ax.bar(
        [value - width for value in x],
        [portable[category] for category in categories],
        width,
        label="Portable notes",
        color="#4C78A8",
    )
    ax.bar(
        x,
        [contextual[category] for category in categories],
        width,
        label="Contextual notes",
        color="#72B7B2",
    )
    ax.bar(
        [value + width for value in x],
        [rejected[category] for category in categories],
        width,
        label="Rejected candidates",
        color="#E45756",
    )

    ax.set_ylabel("Number of evidence notes")
    ax.set_xlabel("Evidence category", labelpad=16)
    ax.tick_params(axis="x", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(category) for category in categories])
    ax.set_ylim(0, 46)
    ax.set_yticks(range(0, 46, 5))
    ax.grid(axis="y", color="#D4D4D4", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()
    create_chart(args.evidence_dir, args.output_stem)


if __name__ == "__main__":
    main()
