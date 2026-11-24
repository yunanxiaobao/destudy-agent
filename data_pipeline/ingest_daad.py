"""Fetch English-taught Master's programs across Germany from the DAAD
International Programmes public catalog and save to data/daad_programs.json.

Usage:
    python -m data_pipeline.ingest_daad             # fetch all English Master's programs
    python -m data_pipeline.ingest_daad --max 500   # cap total fetched
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import httpx

from settings import DATA_DIR, USER_AGENT

DAAD_API = "https://www2.daad.de/deutschland/studienangebote/international-programmes/api/solr/en/search.json"
DAAD_OUT = DATA_DIR / "daad_programs.json"


def fetch_daad_masters(max_items: int = 2000, page_size: int = 100, delay: float = 0.4) -> list[dict]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    courses: list[dict] = []
    offset = 0
    total = None
    with httpx.Client(headers=headers, timeout=20, follow_redirects=True) as client:
        while offset < max_items and (total is None or offset < total):
            params = [
                ("degree[]", "2"),  # Master's degree
                ("lang[]", "2"),    # English
                ("limit", str(page_size)),
                ("offset", str(offset)),
                ("display", "list"),
            ]
            r = client.get(DAAD_API, params=params)
            r.raise_for_status()
            data = r.json()
            if total is None:
                total = int(data.get("numResults", 0))
                print(f"DAAD catalog reports {total} English-taught Master's programs")
            batch = data.get("courses", [])
            if not batch:
                break
            courses.extend(batch)
            offset += len(batch)
            print(f"  fetched {len(courses)}/{min(total, max_items)}")
            time.sleep(delay)
    return courses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2000)
    args = ap.parse_args()

    courses = fetch_daad_masters(max_items=args.max)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "DAAD International Programmes",
        "count": len(courses),
        "courses": courses,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DAAD_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {len(courses)} programs -> {DAAD_OUT}")


if __name__ == "__main__":
    main()
