"""The grade: a person's verdict on the door's call, and which way it moves the cut."""

from __future__ import annotations

from agent.grades import (
    Grade,
    Verdict,
    advice,
    append_grade,
    grade_path,
    load_grades,
    tally,
)


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "grades.jsonl"
    append_grade("r1", Verdict.right, note="clear enough", path=path)
    loaded = load_grades(path)
    assert loaded["r1"].verdict is Verdict.right
    assert loaded["r1"].note == "clear enough"


def test_a_string_verdict_is_accepted(tmp_path):
    assert append_grade("r1", "too_strict", path=tmp_path / "g.jsonl").verdict is Verdict.too_strict


def test_an_unknown_verdict_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        append_grade("r1", "maybe", path=tmp_path / "g.jsonl")


def test_the_latest_grade_wins_but_history_is_kept(tmp_path):
    path = tmp_path / "grades.jsonl"
    append_grade("r1", Verdict.too_lenient, path=path)
    append_grade("r1", Verdict.right, path=path)
    assert load_grades(path)["r1"].verdict is Verdict.right
    assert sum(1 for _ in path.read_text().splitlines()) == 2


def test_a_truncated_line_is_skipped(tmp_path):
    path = tmp_path / "grades.jsonl"
    append_grade("r1", Verdict.right, path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "r2", "verdict": "to')
    assert set(load_grades(path)) == {"r1"}


def test_a_missing_file_is_empty(tmp_path):
    assert load_grades(tmp_path / "none.jsonl") == {}


def test_grade_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_GRADE_PATH", str(tmp_path / "x.jsonl"))
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path / "ignored"))
    assert grade_path() == tmp_path / "x.jsonl"


def test_direction_says_which_way_the_cut_moves(tmp_path):
    assert append_grade("a", "too_strict", path=tmp_path / "g").direction == "lower the cut (ask fewer)"
    assert append_grade("b", "too_lenient", path=tmp_path / "g").direction == "raise the cut (ask more)"
    assert append_grade("c", "right", path=tmp_path / "g").direction == ""


def test_tally_counts_verdicts():
    grades = [Grade("a", Verdict.right), Grade("b", Verdict.right), Grade("c", Verdict.too_lenient)]
    counts = tally(grades)
    assert counts[Verdict.right] == 2
    assert counts[Verdict.too_lenient] == 1


def test_advice_names_the_direction_the_grades_point():
    assert "lower the cut" in advice([Grade("a", Verdict.too_strict), Grade("b", Verdict.too_strict)])
    assert "raise the cut" in advice([Grade("a", Verdict.too_lenient)])


def test_advice_is_empty_until_a_verdict_points_somewhere():
    assert "no verdict" in advice([])
    assert "no verdict" in advice([Grade("a", Verdict.right), Grade("b", Verdict.unclear)])


def test_evenly_wrong_is_not_a_threshold_problem():
    """Equal false accepts and false rejects means the cut is not the lever."""
    even = [Grade("a", Verdict.too_strict), Grade("b", Verdict.too_lenient)]
    assert "not the lever" in advice(even)
