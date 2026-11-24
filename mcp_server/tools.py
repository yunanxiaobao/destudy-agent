"""Pure, deterministic tool implementations (no MCP dependency -> easy to unit test)."""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import yaml

from data_pipeline.programs_db import db_stats, get_program_by_id, query_programs
from settings import FACTS_PATH


@lru_cache(maxsize=1)
def load_facts() -> dict:
    return yaml.safe_load(FACTS_PATH.read_text(encoding="utf-8"))


def load_programs(limit: int = 100) -> list[dict]:
    return query_programs(limit=limit)


# ---------------------------------------------------------------- grades
def convert_grade(score: float, max_score: float = 100, min_pass: float = 60) -> dict:
    """Modified Bavarian Formula: N = 1 + 3 * (Nmax - Nd) / (Nmax - Nmin)."""
    if max_score <= min_pass:
        raise ValueError("max_score must be greater than min_pass")
    raw = 1 + 3 * (max_score - score) / (max_score - min_pass)
    german = min(max(raw, 1.0), 5.0)
    german = int(german * 10 + 1e-9) / 10  # German grades are truncated, not rounded
    if german <= 1.5:
        label = "sehr gut (very good)"
    elif german <= 2.5:
        label = "gut (good)"
    elif german <= 3.5:
        label = "befriedigend (satisfactory)"
    elif german <= 4.0:
        label = "ausreichend (sufficient)"
    else:
        label = "nicht bestanden (fail)"
    return {
        "german_grade": german,
        "label": label,
        "formula": f"1 + 3 × ({max_score} − {score}) / ({max_score} − {min_pass}) = {raw:.3f}",
        "note": "German scale: 1.0 best, 4.0 lowest pass. Universities / uni-assist may use their own conversion.",
    }


# ---------------------------------------------------------------- money
def calc_blocked_account(months: int = 12) -> dict:
    ba = load_facts()["blocked_account"]
    return {
        "months": months,
        "monthly_eur": ba["monthly_eur"],
        "total_eur": ba["monthly_eur"] * months,
        "as_of": ba["as_of"],
        "source": ba["source"],
        "note": ba["note"].strip(),
    }


# ---------------------------------------------------------------- programs (SQLite)
def search_programs(
    keyword: str | None = None,
    university: str | None = None,
    city: str | None = None,
    language: str | None = None,
    intake: str | None = None,
    max_tuition_eur: int | None = None,
    max_ielts: float | None = None,
    gre_policy: str | None = None,
    limit: int = 15,
) -> list[dict]:
    """Filter the SQLite German Master's program database (1,600+ DAAD + official pages)."""
    return query_programs(
        keyword=keyword,
        university=university,
        city=city,
        language=language,
        intake=intake,
        max_tuition_eur=max_tuition_eur,
        max_ielts=max_ielts,
        gre_policy=gre_policy,
        limit=limit,
    )


def get_program(program_id: str) -> dict | None:
    return get_program_by_id(program_id)


def get_program_stats() -> dict:
    return db_stats()


# ---------------------------------------------------------------- timeline
def _intake_start(intake: str) -> date:
    """'WS2027' -> 2027-10-01, 'SS2027' -> 2027-04-01."""
    intake = intake.upper().replace(" ", "")
    year = int(intake[-4:])
    return date(year, 10, 1) if intake.startswith("WS") else date(year, 4, 1)


def _parse_deadline(s: str, intake_start: date) -> date | None:
    """Program deadlines are stored as 'MM-DD' relative to the intake year (the
    deadline falls before the intake start, possibly in the previous year)."""
    try:
        m, d = map(int, s.split("-"))
    except (ValueError, AttributeError):
        return None
    dl = date(intake_start.year, m, d)
    return dl if dl < intake_start else date(intake_start.year - 1, m, d)


def build_timeline(
    intake: str,
    program_ids: list[str] | None = None,
    has_aps: bool = False,
    has_language_score: bool = False,
    today: date | None = None,
) -> dict:
    today = today or date.today()
    start = _intake_start(intake)
    w = load_facts()["durations_weeks"]

    deadlines = []
    for pid in program_ids or []:
        p = get_program(pid)
        if not p:
            continue
        dl_str = (p.get("deadlines") or {}).get(intake[:2].upper())
        dl = _parse_deadline(dl_str, start) if dl_str else None
        if dl:
            deadlines.append({"program": f"{p['university']} – {p['name']}", "deadline": dl, "note": p.get("deadline_note", "")})
    deadlines.sort(key=lambda d: d["deadline"])
    # default: assume applications close ~ 5 months before semester start
    first_deadline = deadlines[0]["deadline"] if deadlines else start - timedelta(weeks=22)

    m: list[dict] = []

    def add(when: date, task: str, why: str = ""):
        m.append({"date": when, "task": task, "why": why})

    apply_ready = first_deadline - timedelta(weeks=w["uni_assist_vpd"] + 2)
    if not has_aps:
        add(apply_ready - timedelta(weeks=w["aps"]), "提交 APS 审核材料", "大多数中国学生申请德国大学前必须持有 APS 证书；处理时间较长")
    if not has_language_score:
        add(apply_ready - timedelta(weeks=w["language_test"]), "报名并参加语言考试（IELTS/TOEFL 或德语）", "需在申请前拿到成绩")
    add(apply_ready, "准备申请材料：CV、动机信、成绩单、课程描述、推荐信", "需在 uni-assist/VPD 处理之前准备齐全")
    add(first_deadline - timedelta(weeks=w["uni_assist_vpd"]), "通过 uni-assist / 学校门户提交申请（如需 VPD 请更早提交）", "uni-assist 处理需要数周")
    for d in deadlines:
        add(d["deadline"], f"申请截止：{d['program']}", d["note"])
    admit = first_deadline + timedelta(weeks=8)
    add(admit, "预计收到录取结果（因学校而异）")
    add(admit, f"开通自保金账户（Sperrkonto）并存入资金，约需 {w['blocked_account']} 周")
    add(admit + timedelta(weeks=1), "购买入境医疗保险 / 预约签证", f"签证预约可能需要等待约 {w['visa_appointment']} 周")
    add(admit + timedelta(weeks=1 + w["visa_appointment"]), "递交签证申请", f"处理时间约 {w['visa_processing']} 周")
    add(start - timedelta(weeks=w["housing_search"]), "开始找房（宿舍 Studierendenwerk / WG）")
    add(start - timedelta(weeks=2), "抵达德国，办理注册（Immatrikulation）")
    add(start, f"{intake} 学期开始")

    m.sort(key=lambda x: x["date"])
    for x in m:
        x["status"] = "已过期 ⚠️" if x["date"] < today else ""
        x["date"] = x["date"].isoformat()
    return {
        "intake": intake,
        "semester_start": start.isoformat(),
        "milestones": m,
        "disclaimer": "时间为规划估算，请以学校官网和使馆的最新信息为准。",
    }
