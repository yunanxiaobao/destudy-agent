"""LLM Structured Extraction pipeline: turn crawled official university pages
into validated Pydantic ProgramRecord entries and store them in SQLite.

Usage:
    python -m data_pipeline.extract_programs                      # extract all program sources not in programs.yaml
    python -m data_pipeline.extract_programs --only tum-cs rwth-cs
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import yaml

from data_pipeline.programs_db import ProgramRecord, build_db
from rag.models import get_llm
from settings import EXTRACTED_PROGRAMS_PATH, OPENAI_MODEL, PROGRAMS_PATH, RAW_DIR, ROOT, load_prompt

MAX_SOURCE_CHARS = 16000


def _collect_source_text(source_id: str, seeds: list[str] | None = None) -> tuple[str, str]:
    files = sorted((RAW_DIR / source_id).glob("*.json"))
    if not files:
        return "", ""
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    seed_set = {u.rstrip("/") for u in (seeds or [])}
    # Put seed program page(s) FIRST, then English pages, then shortest URLs
    docs.sort(key=lambda d: (0 if d["url"].rstrip("/") in seed_set else 1, 0 if d.get("lang") == "en" else 1, len(d["url"])))
    parts, total = [], 0
    for d in docs:
        block = f"### {d['title']}\nURL: {d['url']}\n{d['text']}\n"
        if total + len(block) > MAX_SOURCE_CHARS:
            block = block[: MAX_SOURCE_CHARS - total]
        parts.append(block)
        total += len(block)
        if total >= MAX_SOURCE_CHARS:
            break
    return "\n---\n".join(parts), docs[0]["url"]


def extract_one(src: dict) -> ProgramRecord | None:
    text, primary_url = _collect_source_text(src["id"], src.get("seeds"))
    if not text:
        print(f"  [skip] no raw docs for {src['id']}")
        return None
    resp = get_llm().chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": load_prompt("extract_program")},
            {
                "role": "user",
                "content": (
                    f"Target University: {src.get('university', '')}\n"
                    f"Target Program: {src.get('program', '')}\n\n"
                    f"Official crawled pages:\n\n{text}"
                ),
            },
        ],
    )
    raw = json.loads(resp.choices[0].message.content)
    raw["id"] = src["id"]
    raw["url"] = src["seeds"][0] if src.get("seeds") else primary_url
    raw["source_type"] = "llm_extracted"
    raw["verified_at"] = date.today().isoformat()
    # Clean null values from strict "null if not stated" extraction
    for k in ("city", "state", "field", "deadline_note", "portal", "semester_fee_note", "language_requirement", "gre_requirement", "notes"):
        if raw.get(k) is None:
            raw[k] = ""
    if raw.get("aps_required") is None:
        raw["aps_required"] = True
    if not raw.get("teaching_language"):
        raw["teaching_language"] = ["English"]
    if raw.get("intakes") is None:
        raw["intakes"] = []
    if raw.get("degree") is None:
        raw["degree"] = "M.Sc."
    if raw.get("gre_policy") not in ("mandatory", "optional", "recommended", "none", "unknown"):
        raw["gre_policy"] = "unknown"
    return ProgramRecord.model_validate(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="source ids to extract")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "data_pipeline" / "sources.yaml").read_text(encoding="utf-8"))
    curated_ids = set()
    if PROGRAMS_PATH.exists():
        curated_ids = {p["id"] for p in yaml.safe_load(PROGRAMS_PATH.read_text(encoding="utf-8")).get("programs", [])}

    existing: dict[str, dict] = {}
    if EXTRACTED_PROGRAMS_PATH.exists():
        for item in json.loads(EXTRACTED_PROGRAMS_PATH.read_text(encoding="utf-8")):
            existing[item["id"]] = item

    for src in cfg["sources"]:
        if src.get("category") != "program":
            continue
        if args.only:
            if src["id"] not in args.only:
                continue
        elif src["id"] in curated_ids:
            continue
        print(f"== extracting {src['id']} ({src.get('university')})")
        rec = extract_one(src)
        if rec:
            existing[rec.id] = rec.model_dump()
            print(
                f"   ok: WS={rec.deadline_ws} SS={rec.deadline_ss} tuition={rec.tuition_non_eu_eur} "
                f"IELTS={rec.ielts_min} TOEFL={rec.toefl_min} GRE={rec.gre_policy}"
            )

    EXTRACTED_PROGRAMS_PATH.write_text(
        json.dumps(list(existing.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(existing)} extracted records -> {EXTRACTED_PROGRAMS_PATH}")
    counts = build_db()
    print("rebuilt programs.db:", counts)


if __name__ == "__main__":
    main()
