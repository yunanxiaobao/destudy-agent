"""Pydantic schema + SQLite database for German Master's programs.

Merges three data layers (lowest -> highest precedence):
  1. DAAD International Programmes catalog (data/daad_programs.json, ~1,600 English M.Sc. programs)
  2. LLM Structured Extraction over crawled official university pages (data/extracted_programs.json)
  3. Hand-verified curated official entries (data/programs.yaml)

Usage:
    python -m data_pipeline.programs_db   # rebuild data/programs.db and print stats
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from settings import DAAD_PATH, EXTRACTED_PROGRAMS_PATH, PROGRAMS_DB_PATH, PROGRAMS_PATH

SourceType = Literal["curated_official", "llm_extracted", "daad_catalog"]
GrePolicy = Literal["mandatory", "optional", "recommended", "none", "unknown"]


class ProgramRecord(BaseModel):
    id: str
    university: str
    name: str
    city: str = ""
    state: str = ""
    field: str = ""
    degree: str = "M.Sc."
    ects: int | None = None
    duration_semesters: int | None = None
    teaching_language: list[str] = Field(default_factory=lambda: ["English"])
    intakes: list[str] = Field(default_factory=list)
    deadline_ws: str | None = None
    deadline_ss: str | None = None
    deadline_note: str = ""
    portal: str = ""
    tuition_non_eu_eur: int | None = None
    semester_fee_note: str = ""
    ielts_min: float | None = None
    toefl_min: int | None = None
    language_requirement: str = ""
    gre_policy: GrePolicy = "unknown"
    gre_requirement: str = ""
    aps_required: bool = True
    notes: str = ""
    url: str = ""
    source_type: SourceType = "daad_catalog"
    verified_at: str = ""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS programs (
    id TEXT PRIMARY KEY,
    university TEXT NOT NULL,
    name TEXT NOT NULL,
    city TEXT DEFAULT '',
    state TEXT DEFAULT '',
    field TEXT DEFAULT '',
    degree TEXT DEFAULT 'M.Sc.',
    ects INTEGER,
    duration_semesters INTEGER,
    teaching_language TEXT DEFAULT '[]',
    intakes TEXT DEFAULT '[]',
    deadline_ws TEXT,
    deadline_ss TEXT,
    deadline_note TEXT DEFAULT '',
    portal TEXT DEFAULT '',
    tuition_non_eu_eur INTEGER,
    semester_fee_note TEXT DEFAULT '',
    ielts_min REAL,
    toefl_min INTEGER,
    language_requirement TEXT DEFAULT '',
    gre_policy TEXT DEFAULT 'unknown',
    gre_requirement TEXT DEFAULT '',
    aps_required INTEGER DEFAULT 1,
    notes TEXT DEFAULT '',
    url TEXT DEFAULT '',
    source_type TEXT DEFAULT 'daad_catalog',
    verified_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_programs_uni ON programs(university);
CREATE INDEX IF NOT EXISTS idx_programs_city ON programs(city);
CREATE INDEX IF NOT EXISTS idx_programs_tuition ON programs(tuition_non_eu_eur);
CREATE INDEX IF NOT EXISTS idx_programs_gre ON programs(gre_policy);
CREATE INDEX IF NOT EXISTS idx_programs_ielts ON programs(ielts_min);
CREATE INDEX IF NOT EXISTS idx_programs_source ON programs(source_type);
"""


def _parse_tuition(raw: str | None) -> int | None:
    if not raw:
        return None
    low = raw.lower()
    if "no tuition" in low or "none" in low:
        return 0
    m = re.search(r"(\d[\d,\.]*)\s*(?:eur|€)", raw, re.I)
    if m:
        num = re.sub(r"[,\.]", "", m.group(1))
        if num.isdigit():
            return int(num)
    return None


def _parse_intakes(raw: str | None) -> list[str]:
    if not raw:
        return []
    low = raw.lower()
    out = []
    if "winter" in low:
        out.append("WS")
    if "summer" in low:
        out.append("SS")
    return out


def _parse_semesters(raw: str | None) -> int | None:
    if not raw:
        return None
    m = re.search(r"(\d+)\s*semester", raw, re.I)
    return int(m.group(1)) if m else None


def daad_course_to_record(c: dict, fetched_at: str) -> ProgramRecord:
    link = c.get("link") or ""
    url = f"https://www2.daad.de{link}" if link.startswith("/") else link
    tuition_str = c.get("tuitionFees") or ""
    cost_str = c.get("costString") or ""
    lang_en = c.get("languageLevelEnglish") or ""
    lang_de = c.get("languageLevelGerman") or ""
    lang_parts = [p for p in (f"English: {lang_en}" if lang_en else "", f"German: {lang_de}" if lang_de else "") if p]
    return ProgramRecord(
        id=f"daad-{c['id']}",
        university=(c.get("academy") or "").strip(),
        name=(c.get("courseName") or "").strip(),
        city=(c.get("city") or "").strip(),
        field=(c.get("subject") or "").strip(),
        degree="M.Sc." if "science" in (c.get("courseName") or "").lower() else "Master",
        duration_semesters=_parse_semesters(c.get("programmeDuration")),
        teaching_language=c.get("languages") or ["English"],
        intakes=_parse_intakes(c.get("beginning")),
        deadline_note=(c.get("applicationDeadline") or "").strip(),
        tuition_non_eu_eur=_parse_tuition(tuition_str or cost_str),
        semester_fee_note=" / ".join(p for p in (tuition_str, cost_str) if p),
        language_requirement="; ".join(lang_parts),
        url=url,
        source_type="daad_catalog",
        verified_at=fetched_at[:10],
    )


def yaml_item_to_record(p: dict) -> ProgramRecord:
    dls = p.get("deadlines") or {}
    lang_req = p.get("language_requirement", "")
    ielts_m = re.search(r"IELTS[^\d]*(\d(?:\.\d)?)", lang_req, re.I)
    toefl_m = re.search(r"TOEFL[^\d]*(\d{2,3})", lang_req, re.I)
    gre_txt = (p.get("gre_requirement") or "").lower()
    if "mandatory" in gre_txt:
        gre_policy: GrePolicy = "mandatory"
    elif "optional" in gre_txt:
        gre_policy = "optional"
    elif "recommend" in gre_txt:
        gre_policy = "recommended"
    elif "not required" in gre_txt or "none" in gre_txt:
        gre_policy = "none"
    else:
        gre_policy = "unknown"
    return ProgramRecord(
        id=p["id"],
        university=p["university"],
        name=p["name"],
        city=p.get("city", ""),
        state=p.get("state", ""),
        field=p.get("field", ""),
        degree=p.get("degree", "M.Sc."),
        ects=p.get("ects"),
        duration_semesters=p.get("duration_semesters"),
        teaching_language=p.get("teaching_language") or ["English"],
        intakes=p.get("intakes") or [],
        deadline_ws=dls.get("WS"),
        deadline_ss=dls.get("SS"),
        deadline_note=(p.get("deadline_note") or "").strip(),
        portal=(p.get("portal") or "").strip(),
        tuition_non_eu_eur=p.get("tuition_non_eu_eur_per_semester"),
        semester_fee_note=(p.get("semester_fee_note") or "").strip(),
        ielts_min=float(ielts_m.group(1)) if ielts_m else None,
        toefl_min=int(toefl_m.group(1)) if toefl_m else None,
        language_requirement=lang_req.strip(),
        gre_policy=gre_policy,
        gre_requirement=(p.get("gre_requirement") or "").strip(),
        aps_required=bool(p.get("aps_required", True)),
        notes=(p.get("notes") or "").strip(),
        url=p.get("url", ""),
        source_type="curated_official",
        verified_at=str(p.get("verified_at", "")),
    )


def _upsert(conn: sqlite3.Connection, rec: ProgramRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO programs (
            id, university, name, city, state, field, degree, ects, duration_semesters,
            teaching_language, intakes, deadline_ws, deadline_ss, deadline_note, portal,
            tuition_non_eu_eur, semester_fee_note, ielts_min, toefl_min, language_requirement,
            gre_policy, gre_requirement, aps_required, notes, url, source_type, verified_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            rec.id, rec.university, rec.name, rec.city, rec.state, rec.field, rec.degree,
            rec.ects, rec.duration_semesters,
            json.dumps(rec.teaching_language, ensure_ascii=False),
            json.dumps(rec.intakes, ensure_ascii=False),
            rec.deadline_ws, rec.deadline_ss, rec.deadline_note, rec.portal,
            rec.tuition_non_eu_eur, rec.semester_fee_note, rec.ielts_min, rec.toefl_min,
            rec.language_requirement, rec.gre_policy, rec.gre_requirement,
            1 if rec.aps_required else 0, rec.notes, rec.url, rec.source_type, rec.verified_at,
        ),
    )


def build_db(db_path: Path = PROGRAMS_DB_PATH) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)

    counts = {"daad_catalog": 0, "llm_extracted": 0, "curated_official": 0}

    # Layer 1: DAAD catalog
    if DAAD_PATH.exists():
        raw = json.loads(DAAD_PATH.read_text(encoding="utf-8"))
        fetched_at = raw.get("fetched_at", "")
        for c in raw.get("courses", []):
            _upsert(conn, daad_course_to_record(c, fetched_at))
            counts["daad_catalog"] += 1

    # Layer 2: LLM-extracted official pages
    if EXTRACTED_PROGRAMS_PATH.exists():
        extracted = json.loads(EXTRACTED_PROGRAMS_PATH.read_text(encoding="utf-8"))
        for item in extracted:
            rec = ProgramRecord.model_validate({**item, "source_type": "llm_extracted"})
            _upsert(conn, rec)
            counts["llm_extracted"] += 1

    # Layer 3: Curated official YAML (highest precedence)
    if PROGRAMS_PATH.exists():
        curated = yaml.safe_load(PROGRAMS_PATH.read_text(encoding="utf-8")).get("programs", [])
        for p in curated:
            _upsert(conn, yaml_item_to_record(p))
            counts["curated_official"] += 1

    conn.commit()
    conn.close()
    return counts


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["teaching_language"] = json.loads(d["teaching_language"] or "[]")
    d["intakes"] = json.loads(d["intakes"] or "[]")
    d["aps_required"] = bool(d["aps_required"])
    d["deadlines"] = {k: v for k, v in (("WS", d["deadline_ws"]), ("SS", d["deadline_ss"])) if v}
    return d


def query_programs(
    keyword: str | None = None,
    university: str | None = None,
    city: str | None = None,
    language: str | None = None,
    intake: str | None = None,
    max_tuition_eur: int | None = None,
    max_ielts: float | None = None,
    gre_policy: str | None = None,
    source_type: str | None = None,
    limit: int = 20,
    db_path: Path = PROGRAMS_DB_PATH,
) -> list[dict]:
    if not db_path.exists():
        build_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    clauses, params = [], []
    if keyword:
        terms = [t.strip().lower() for t in re.split(r"\s+OR\s+|[,|/]", keyword, flags=re.I) if t.strip()]
        if terms:
            or_parts = []
            for t in terms:
                like = f"%{t}%"
                or_parts.append("(LOWER(name) LIKE ? OR LOWER(field) LIKE ? OR LOWER(university) LIKE ? OR LOWER(notes) LIKE ?)")
                params.extend([like, like, like, like])
            clauses.append("(" + " OR ".join(or_parts) + ")")
    if university:
        clauses.append("LOWER(university) LIKE ?")
        params.append(f"%{university.lower()}%")
    if city:
        clauses.append("LOWER(city) LIKE ?")
        params.append(f"%{city.lower()}%")
    if language:
        clauses.append("LOWER(teaching_language) LIKE ?")
        params.append(f"%{language.lower()}%")
    if intake:
        clauses.append("LOWER(intakes) LIKE ?")
        params.append(f"%{intake.lower()}%")
    if max_tuition_eur is not None:
        clauses.append("tuition_non_eu_eur IS NOT NULL AND tuition_non_eu_eur <= ?")
        params.append(int(max_tuition_eur))
    if max_ielts is not None:
        clauses.append("ielts_min IS NOT NULL AND ielts_min <= ?")
        params.append(float(max_ielts))
    if gre_policy:
        gp = gre_policy.lower()
        if gp in ("no_gre", "not_required", "none_or_optional"):
            clauses.append("gre_policy IN ('none', 'optional', 'recommended')")
        else:
            clauses.append("LOWER(gre_policy) = ?")
            params.append(gp)
    if source_type:
        clauses.append("source_type = ?")
        params.append(source_type)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = (
        "ORDER BY CASE source_type "
        "WHEN 'curated_official' THEN 0 "
        "WHEN 'llm_extracted' THEN 1 "
        "ELSE 2 END, university, name"
    )
    sql = f"SELECT * FROM programs {where} {order} LIMIT ?"
    params.append(max(1, min(limit, 100)))
    rows = [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows


def get_program_by_id(program_id: str, db_path: Path = PROGRAMS_DB_PATH) -> dict | None:
    if not db_path.exists():
        build_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM programs WHERE id = ?", (program_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def db_stats(db_path: Path = PROGRAMS_DB_PATH) -> dict:
    if not db_path.exists():
        build_db(db_path)
    conn = sqlite3.connect(db_path)
    total = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
    by_src = dict(conn.execute("SELECT source_type, COUNT(*) FROM programs GROUP BY source_type").fetchall())
    free = conn.execute("SELECT COUNT(*) FROM programs WHERE tuition_non_eu_eur = 0").fetchone()[0]
    cs_ai = conn.execute(
        "SELECT COUNT(*) FROM programs WHERE LOWER(name) LIKE '%computer%' OR LOWER(name) LIKE '%informat%' "
        "OR LOWER(name) LIKE '%artificial intelligence%' OR LOWER(name) LIKE '%data science%' "
        "OR LOWER(name) LIKE '%machine learning%' OR LOWER(field) LIKE '%computer%'"
    ).fetchone()[0]
    unis = conn.execute("SELECT COUNT(DISTINCT university) FROM programs").fetchone()[0]
    conn.close()
    return {
        "total_programs": total,
        "distinct_universities": unis,
        "tuition_free_programs": free,
        "cs_ai_ds_programs": cs_ai,
        "by_source": by_src,
    }


if __name__ == "__main__":
    counts = build_db()
    print("Rebuilt programs.db:", counts)
    print("Stats:", json.dumps(db_stats(), ensure_ascii=False, indent=2))
