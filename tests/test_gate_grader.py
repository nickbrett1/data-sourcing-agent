"""The grading MCP: the same loop the UI serves, as tools an agent can call."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from agent import gate_grader
from agent.grader import create_app
from agent.grades import load_grades
from agent.samples import append_sample


def _wire(tmp_path, monkeypatch, *, graded: bool = False):
    log = tmp_path / "gate-log.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "request_id": "r1", "question_id": "d2_specification_sufficient",
            "answer": 0.4, "failed": False, "action_computed": "ask_clarifying", "holdback": False,
        }) + "\n")
    append_sample("r1", "one day of SPY options", "dataset: OPRA.PILLAR", path=tmp_path / "samples.jsonl")
    if graded:
        from agent.grades import append_grade

        append_grade("r1", "right", path=tmp_path / "grades.jsonl")
    monkeypatch.setenv("GATE_LOG_PATH", str(log))
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "samples.jsonl"))
    monkeypatch.setenv("GATE_GRADE_PATH", str(tmp_path / "grades.jsonl"))


def test_next_request_returns_the_first_ungraded(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    item = gate_grader.next_request()
    assert item["request_id"] == "r1"
    assert item["ticket"] == "dataset: OPRA.PILLAR"


def test_next_request_is_none_when_everything_is_graded(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, graded=True)
    assert gate_grader.next_request() is None


def test_queue_hides_graded_items_by_default(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, graded=True)
    assert gate_grader.queue() == []
    assert len(gate_grader.queue(only_ungraded=False)) == 1


def test_record_grade_writes_and_names_the_direction(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    result = gate_grader.record_grade("r1", "too_strict", note="window was fine")
    assert result["direction"] == "lower the cut (ask fewer)"
    assert load_grades(tmp_path / "grades.jsonl")["r1"].verdict.value == "too_strict"


def test_record_grade_rejects_an_unknown_verdict(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        gate_grader.record_grade("r1", "meh")


def test_status_matches_the_http_summary(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch)
    http = TestClient(create_app()).get("/api/summary").json()
    assert gate_grader.status() == http


def test_grades_tool_returns_the_history(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, graded=True)
    assert gate_grader.grades()["r1"]["verdict"] == "right"


def test_the_server_exposes_the_tools():
    assert gate_grader.mcp.name == "data-sourcing-gate-grader"


def test_the_cli_defaults_to_stdio(monkeypatch):
    calls = []
    monkeypatch.setattr(gate_grader.mcp, "run", lambda **kw: calls.append(kw))
    assert gate_grader.main([]) == 0
    assert calls == [{"transport": "stdio"}]


def test_the_cli_can_serve_streamable_http_for_a_hub(monkeypatch):
    """A hub that cannot spawn a process reaches the same tools over a URL."""
    calls = []
    monkeypatch.setattr(gate_grader.mcp, "run", lambda **kw: calls.append(kw))
    assert gate_grader.main(["--transport", "streamable-http", "--port", "9001"]) == 0
    assert calls == [{"transport": "streamable-http", "host": "0.0.0.0", "port": 9001, "stateless_http": True}]


def test_the_legend_tool_spells_out_the_codes():
    """`legend` answers 'what does D2 mean?' without a doc lookup."""
    from agent.gate import D1, D2, D3, D5

    legend = gate_grader.legend()
    assert set(legend) == {D1, D2, D3, D5}
    assert legend[D2].startswith("Specification sufficient")
