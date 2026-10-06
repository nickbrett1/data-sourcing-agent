"""Tests for the gate-log report — the calibration view over the shadow corpus."""

from __future__ import annotations

import json

from agent.gate import D1, D2, D3, D5, GateAction, GateDecision, GatePolicy
from agent.gate_report import load_records, main, render, summarise
from agent.gatelog import build_records
from agent.jev import NoulAnswer, ScoreAnswer


def _log(tmp_path, answers, decision, *, request_id="r1", **kw):
    rows = build_records("state", answers, decision, request_id=request_id, **kw)
    path = tmp_path / "gate-log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _answers(d1=0.99, d2=0.95, d3=3):
    return {
        D1: NoulAnswer(noul=d1),
        D2: NoulAnswer(noul=d2),
        D3: ScoreAnswer(score=float(d3)),
    }


def test_a_request_counts_once_though_it_has_a_row_per_question(tmp_path):
    path = _log(tmp_path, _answers(), GateDecision(action=GateAction.proceed))
    report = summarise(load_records(path))
    assert report.requests == 1
    assert report.records == 3
    assert report.computed_actions["proceed"] == 1


def test_distributions_split_noul_from_score(tmp_path):
    path = _log(tmp_path, _answers(d2=0.4), GateDecision(action=GateAction.ask_clarifying))
    report = summarise(load_records(path))
    assert report.questions[D2].median() == 0.4
    assert report.questions[D3].levels["exact"] == 1


def test_would_stop_rate_is_the_size_of_the_change(tmp_path):
    for i in range(4):  # 4 requests would proceed
        _log(tmp_path, _answers(), GateDecision(action=GateAction.proceed), request_id=f"ok{i}")
    path = _log(tmp_path, _answers(), GateDecision(action=GateAction.reject), request_id="bad")
    report = summarise(load_records(path))
    assert report.requests == 5
    assert report.would_stop == 1
    assert report.would_stop_rate == 0.2


def test_failed_answers_are_counted_as_no_opinion_not_values(tmp_path):
    answers = {D2: NoulAnswer(failed=True)}
    path = _log(tmp_path, answers, GateDecision(action=GateAction.proceed))
    stat = summarise(load_records(path)).questions[D2]
    assert stat.total == 1 and stat.failed == 1 and stat.usable == 0


def test_load_records_skips_a_truncated_last_line(tmp_path):
    path = _log(tmp_path, _answers(), GateDecision(action=GateAction.proceed))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "half-writ')  # a crash mid-append
    assert len(load_records(path)) == 3


def test_the_report_says_the_cuts_are_not_calibrated_yet(tmp_path):
    path = _log(tmp_path, _answers(), GateDecision(action=GateAction.proceed))
    text = render(summarise(load_records(path)))
    assert "NONE yet" in text
    assert "escalate the top N%" in text
    assert "D2's label is the one NOT to trust" in text


def test_the_day_one_lever_is_priced_from_the_logged_scores(tmp_path):
    for i in range(100):
        path = _log(tmp_path, _answers(d2=i / 100), GateDecision(action=GateAction.proceed), request_id=f"r{i}")
    text = render(summarise(load_records(path)))
    # With 100 scores the top 10% cut is inside the observed range.
    assert "top   10%" in text
    assert "D2 scores available: 100" in text


def test_holdbacks_are_reported(tmp_path):
    path = _log(
        tmp_path,
        _answers(),
        GateDecision(action=GateAction.reject),
        action_taken="proceed",
        holdback=True,
    )
    assert summarise(load_records(path)).holdbacks == 1


def test_main_returns_nonzero_with_no_log(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "absent.jsonl"))
    assert main([]) == 1


def test_main_renders_a_report_over_a_log(tmp_path, capsys):
    path = _log(tmp_path, _answers(), GateDecision(action=GateAction.proceed))
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "Gate shadow log" in out
    assert "requests logged : 1" in out


def test_a_deeper_cut_moves_the_would_stop_banner(tmp_path):
    """The policy is an input: a cut below the observed answer changes nothing but a
    cut above it would. Pin only that the report reflects the policy it is handed."""
    path = _log(tmp_path, _answers(d2=0.5), GateDecision(action=GateAction.ask_clarifying))
    report = summarise(load_records(path), GatePolicy(d2_cut=0.9))
    assert report.policy.d2_cut == 0.9


def test_d5_is_summarised_like_any_other_noul(tmp_path):
    path = _log(tmp_path, {D5: NoulAnswer(noul=0.3)}, GateDecision(action=GateAction.proceed))
    assert summarise(load_records(path)).questions[D5].median() == 0.3
