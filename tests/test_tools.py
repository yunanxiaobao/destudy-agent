from datetime import date

import pytest

from mcp_server import tools


def test_convert_grade_chinese_scale():
    r = tools.convert_grade(85, 100, 60)
    # 1 + 3 * 15 / 40 = 2.125 -> truncated 2.1
    assert r["german_grade"] == 2.1
    assert "gut" in r["label"]


def test_convert_grade_bounds():
    assert tools.convert_grade(100, 100, 60)["german_grade"] == 1.0
    assert tools.convert_grade(60, 100, 60)["german_grade"] == 4.0
    with pytest.raises(ValueError):
        tools.convert_grade(50, 60, 60)


def test_blocked_account_total():
    r = tools.calc_blocked_account(12)
    assert r["total_eur"] == r["monthly_eur"] * 12


def test_timeline_sorted_and_ends_with_semester():
    t = tools.build_timeline("WS2027", today=date(2026, 11, 22))
    dates = [m["date"] for m in t["milestones"]]
    assert dates == sorted(dates)
    assert t["semester_start"] == "2027-10-01"
    assert dates[-1] == "2027-10-01"


def test_deadline_rolls_into_previous_year():
    # SS2027 starts 2027-04-01; a 11-30 deadline belongs to 2026
    assert tools._parse_deadline("11-30", date(2027, 4, 1)) == date(2026, 11, 30)
    assert tools._parse_deadline("05-31", date(2027, 10, 1)) == date(2027, 5, 31)


def test_search_programs_and_timeline_with_real_programs():
    all_progs = tools.search_programs()
    assert len(all_progs) >= 4
    freiburg = tools.search_programs(university="Freiburg", keyword="Computer Science")
    assert freiburg and freiburg[0]["id"] == "freiburg-cs"
    assert len(tools.search_programs(language="English", intake="WS")) >= 4

    t = tools.build_timeline("WS2027", program_ids=["tud-cs", "kit-cs", "freiburg-cs", "tuda-cs"], today=date(2026, 11, 23))
    tasks = [m["task"] for m in t["milestones"]]
    assert any("TU Dresden" in task for task in tasks)
    assert any("KIT" in task for task in tasks)
    assert any("Freiburg" in task for task in tasks)
    assert any("TU Darmstadt" in task for task in tasks)
