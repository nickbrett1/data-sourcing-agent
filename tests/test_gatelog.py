"""Tests for the gate's shadow log."""

from __future__ import annotations

import json

from agent.gate import D1, D2, D3, GateAction, GateDecision
from agent.gatelog import append_records, build_records, gate_log_path, new_request_id
from agent.jev import NoulAnswer


def _answers() -> dict:
    return {D1: NoulAnswer(noul=0.98), D2: NoulAnswer(noul=0.3)}


def test_build_records_makes_one_row_per_question_with_the_action():
    decision = GateDecision(action=GateAction.ask_clarifying, reasons=("d2 low",))
    rows = build_records("state text", _answers(), decision, request_id="req1")
    assert {r["question_id"] for r in rows} == {D1, D2}
    assert all(r["action_taken"] == "ask_clarifying" for r in rows)
    assert all(r["request_id"] == "req1" for r in rows)
    assert all(r["downstream_outcome"] is None for r in rows)  # filled later
    d2 = next(r for r in rows if r["question_id"] == D2)
    assert d2["primitive"] == "noul"
    assert d2["probability"] == 0.3
    # Same state text -> same hash, so identical requests group.
    assert d2["state_hash"] == build_records("state text", _answers(), decision)[0]["state_hash"]


def test_append_records_writes_jsonl(tmp_path):
    rows = build_records("s", _answers(), GateDecision(action=GateAction.proceed))
    path = tmp_path / "gate-log.jsonl"
    assert append_records(rows, path) == 2
    append_records(rows, path)  # appends, does not truncate
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["question_id"] == D1


def test_gate_log_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "x.jsonl"))
    assert gate_log_path() == tmp_path / "x.jsonl"


def test_new_request_id_is_unique_and_usable_as_the_record_id():
    """One id per turn joins the checkpoints; two calls must not collide."""
    a, b = new_request_id(), new_request_id()
    assert a != b
    rows = build_records("s", _answers(), GateDecision(action=GateAction.proceed), request_id=a)
    assert {r["request_id"] for r in rows} == {a}


def test_d3_threshold_is_recorded_as_its_real_cut_not_nan():
    """D3 *is* thresholded (`decide` requires level >= close), so the log must
    record that cut — a sentinel `NaN` was both wrong and invalid JSON."""
    from agent.gate import D3_CLOSE
    from agent.gatelog import default_thresholds

    assert default_thresholds(d1=0.5, d2=0.91, d5=0.5)[D3] == float(D3_CLOSE)


def test_the_log_is_strict_json_even_if_a_non_finite_float_sneaks_in(tmp_path):
    """A `NaN`/`Infinity` token is not JSON; the writer must emit `null` instead."""
    rows = build_records("s", _answers(), GateDecision(action=GateAction.proceed))
    rows[0]["answer"] = float("nan")
    rows[1]["answer"] = float("inf")
    path = tmp_path / "gate-log.jsonl"
    append_records(rows, path)

    # `parse_constant` only fires on NaN/Infinity, so a clean parse proves none.
    parsed = [
        json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(c)))
        for line in path.read_text().splitlines()
    ]
    assert parsed[0]["answer"] is None
    assert parsed[1]["answer"] is None
