"""Run synthetic followed-fact vocabulary queries against the SIMONE API.

The queries mirror the vocabulary-query shape produced by the grounding stage,
but they are intentionally synthetic. They answer the diagnostic question:
which terms would the vocabulary service retrieve if the followed evaluation
facts had reached eligible profile fields?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_API_BASE = "http://127.0.0.1:8000/api/v1"


@dataclass(frozen=True)
class FollowedFactQuery:
    fact: str
    kind: str
    vocabulary_identifier: str
    rdf_type: str
    source_value: str
    semantic_context: str
    vector_query: str
    fulltext_query: str


FOLLOWED_FACT_QUERIES: list[FollowedFactQuery] = [
    FollowedFactQuery(
        fact="dataset identity",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="1H NMR spectral data",
        semantic_context="Dataset described as 1H NMR data from an NMR experiment.",
        vector_query="1H NMR spectrum",
        fulltext_query="1H NMR spectral data",
    ),
    FollowedFactQuery(
        fact="observed nucleus",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="Observed nucleus 1H",
        semantic_context="The acquisition metadata states that the observed nucleus is 1H.",
        vector_query="NMR observed nucleus",
        fulltext_query="observed nucleus 1H proton NMR",
    ),
    FollowedFactQuery(
        fact="solvent or reference information",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="CDCl3 solvent and shift reference",
        semantic_context="The NMR metadata states CDCl3 as solvent or shift reference.",
        vector_query="NMR solvent reference compound",
        fulltext_query="CDCl3 solvent shift reference chloroform",
    ),
    FollowedFactQuery(
        fact="pulse sequence",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="Pulse program zg30",
        semantic_context="The NMR experiment uses the zg30 pulse program.",
        vector_query="NMR pulse sequence",
        fulltext_query="pulse program zg30 NMR sequence",
    ),
    FollowedFactQuery(
        fact="instrument description",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="Bruker Avance NMR spectrometer",
        semantic_context="The measurement was acquired on a Bruker Avance NMR spectrometer.",
        vector_query="NMR spectrometer",
        fulltext_query="Bruker Avance NMR spectrometer",
    ),
    FollowedFactQuery(
        fact="observed resonance frequency quantity kind",
        kind="profile_has_quantity_type",
        vocabulary_identifier="http://qudt.org/vocab/quantitykind",
        rdf_type="qudt__QuantityKind",
        source_value="Observed resonance frequency",
        semantic_context="The acquisition metadata reports an observed 1H resonance frequency of approximately 500.13 MHz.",
        vector_query="frequency",
        fulltext_query="observed resonance frequency NMR",
    ),
    FollowedFactQuery(
        fact="observed resonance frequency unit",
        kind="profile_unit",
        vocabulary_identifier="http://qudt.org/vocab/unit",
        rdf_type="qudt__Unit",
        source_value="MHz",
        semantic_context="The observed resonance frequency is given in MHz.",
        vector_query="megahertz",
        fulltext_query="MHz frequency",
    ),
    FollowedFactQuery(
        fact="raw package resource",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="Bruker FID raw signal",
        semantic_context="The raw package contains the FID signal as the primary raw NMR data resource.",
        vector_query="free induction decay raw NMR data",
        fulltext_query="Bruker FID file",
    ),
    FollowedFactQuery(
        fact="derived package resource",
        kind="profile_type",
        vocabulary_identifier="nmrCV",
        rdf_type="owl__Class",
        source_value="JCAMP-DX NMR spectrum",
        semantic_context="The ChemSpectra export contains a derived JCAMP-DX NMR spectrum.",
        vector_query="NMR spectrum",
        fulltext_query="JCAMP-DX NMR spectrum",
    ),
]


def query_payload(fact_query: FollowedFactQuery, *, top_k: int, max_hops: int) -> dict[str, Any]:
    return {
        "rdf_type": fact_query.rdf_type,
        "vector_query": fact_query.vector_query,
        "fulltext_query": fact_query.fulltext_query,
        "traversal_direction": "undirected",
        "vector_top_k": top_k,
        "fulltext_top_k": top_k,
        "seed_top_k": min(top_k, 10),
        "max_hops": max_hops,
        "max_statements_per_seed": 12,
        "vector_weight": 1.0,
        "fulltext_weight": 1.0,
        "rrf_k": 60,
    }


def post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resource_label(result: dict[str, Any], uri: str) -> str:
    resource = (result.get("resources") or {}).get(uri) or {}
    props = resource.get("properties") or {}
    for key in ("rdfs__label", "skos__prefLabel", "dcterms__title", "title"):
        value = props.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if value:
            return str(value)
    return ""


def run(args: argparse.Namespace) -> int:
    api_base = args.api_base.rstrip("/")
    outputs: list[dict[str, Any]] = []

    for fact_query in FOLLOWED_FACT_QUERIES:
        encoded_identifier = quote(fact_query.vocabulary_identifier, safe="")
        url = f"{api_base}/semantic/vocabularies/query/{encoded_identifier}"
        payload = query_payload(fact_query, top_k=args.top_k, max_hops=args.max_hops)

        try:
            result = post_json(url, payload, timeout=args.timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[error] {fact_query.fact}: HTTP {exc.code} {detail}", file=sys.stderr)
            continue
        except URLError as exc:
            print(f"[error] {fact_query.fact}: {exc}", file=sys.stderr)
            return 2

        seeds = result.get("seeds") or []
        top_seeds = seeds[: args.print_top]
        print(f"\n{fact_query.fact}")
        print(f"  vocabulary: {fact_query.vocabulary_identifier}")
        print(f"  query: vector='{fact_query.vector_query}' | fulltext='{fact_query.fulltext_query}'")
        for rank, seed in enumerate(top_seeds, start=1):
            uri = seed.get("uri", "")
            label = resource_label(result, uri)
            score = seed.get("rrf_score")
            score_text = f"{score:.5f}" if isinstance(score, int | float) else str(score)
            label_text = f" | {label}" if label else ""
            print(f"  {rank}. {uri}{label_text} | rrf={score_text}")

        outputs.append(
            {
                "fact": fact_query.fact,
                "kind": fact_query.kind,
                "source_value": fact_query.source_value,
                "semantic_context": fact_query.semantic_context,
                "vocabulary_identifier": fact_query.vocabulary_identifier,
                "rdf_type": fact_query.rdf_type,
                "query": payload,
                "result": result,
            }
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {len(outputs)} query results to {args.output}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="SIMONE API base URL.")
    parser.add_argument("--top-k", type=int, default=12, help="Vector and fulltext top-k.")
    parser.add_argument("--max-hops", type=int, default=1, help="Vocabulary graph traversal hops.")
    parser.add_argument("--print-top", type=int, default=5, help="Number of top seeds to print per fact.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Optional JSON file for full results.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
