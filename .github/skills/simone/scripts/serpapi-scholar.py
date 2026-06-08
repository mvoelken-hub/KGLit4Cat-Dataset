#!/usr/bin/env python3
"""Search Google Scholar through SerpAPI and print compact JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def load_api_key() -> str:
    key = os.environ.get("SERPAPI_KEY")
    if key:
        return key.strip()

    key_path = Path(os.environ.get("SERPAPI_KEY_FILE", r"C:\Users\simcl\.config\serpapi\api_key"))
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def compact_result(item: dict) -> dict:
    cited_by = item.get("inline_links", {}).get("cited_by", {})
    versions = item.get("inline_links", {}).get("versions", {})
    publication = item.get("publication_info", {})
    return {
        "position": item.get("position"),
        "title": item.get("title"),
        "link": item.get("link"),
        "result_id": item.get("result_id"),
        "snippet": item.get("snippet"),
        "publication_summary": publication.get("summary"),
        "cited_by_count": cited_by.get("total"),
        "cited_by_link": cited_by.get("link"),
        "related_pages_link": item.get("inline_links", {}).get("related_pages_link"),
        "versions_count": versions.get("total"),
        "resources": item.get("resources", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Google Scholar query")
    parser.add_argument("--num", type=int, default=10, help="Number of results")
    parser.add_argument("--start", type=int, default=0, help="Result offset")
    parser.add_argument("--year-from", dest="year_from", help="Minimum publication year")
    parser.add_argument("--year-to", dest="year_to", help="Maximum publication year")
    parser.add_argument("--sort-by-date", action="store_true", help="Sort by date")
    parser.add_argument("--raw", action="store_true", help="Print raw SerpAPI JSON")
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("serpapi-scholar: set SERPAPI_KEY or SERPAPI_KEY_FILE", file=sys.stderr)
        return 2

    params = {
        "engine": "google_scholar",
        "q": args.query,
        "api_key": api_key,
        "num": str(args.num),
        "start": str(args.start),
    }
    if args.year_from:
        params["as_ylo"] = args.year_from
    if args.year_to:
        params["as_yhi"] = args.year_to
    if args.sort_by_date:
        params["scisbd"] = "1"

    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.load(response)
    except Exception as exc:
        print(f"serpapi-scholar: request failed: {exc}", file=sys.stderr)
        return 1

    if "error" in payload:
        print(f"serpapi-scholar: {payload['error']}", file=sys.stderr)
        return 1

    output = payload if args.raw else {
        "query": args.query,
        "search_metadata": {
            "status": payload.get("search_metadata", {}).get("status"),
            "total_time_taken": payload.get("search_metadata", {}).get("total_time_taken"),
        },
        "organic_results": [compact_result(item) for item in payload.get("organic_results", [])],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
