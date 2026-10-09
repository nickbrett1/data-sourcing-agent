"""The grading UI: the queue it builds and the one write it makes."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from agent.grader import build_queue, create_app
from agent.grades import load_grades
from agent.samples import append_sample


def _sample(path, rid, text="one day of SPY options", ticket="dataset: OPRA.PILLAR"):
    append_sample(rid, text, ticket, path=path)


def _record(rid, *, action="ask_clarifying", holdback=False, failed=False):
    return {
        "request_id": rid,
        "question_id": "d2_specification_sufficient",
        "primitive": "noul",
        "answer": 0.4,
        "failed": failed,
        "action_computed": action,
        "holdback": holdback,
    }


def _wire(tmp_path, monkeypatch, records):
    log = tmp_path / "gate-log.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    monkeypatch.setenv("GATE_LOG_PATH", str(log))
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "samples.jsonl"))
    monkeypatch.setenv("GATE_GRADE_PATH", str(tmp_path / "grades.jsonl"))


def test_the_queue_carries_everything_a_human_grades(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1", holdback=True)])
    _sample(tmp_path / "samples.jsonl", "r1")
    item = build_queue()[0]
    assert item["text"] == "one day of SPY options"
    assert item["ticket"].startswith("dataset:")
    assert item["door"] == "ask_clarifying"
    assert item["holdback"] is True
    assert item["jev"][0]["q"] == "d2_specification_sufficient"
    assert item["grade"] is None


def test_a_sample_without_a_log_row_is_not_gradeable(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r2")  # a half-written turn
    assert build_queue() == []


def test_an_existing_grade_comes_back_with_the_item(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r1")
    from agent.grades import append_grade

    append_grade("r1", "too_strict", note="template was fine", path=tmp_path / "grades.jsonl")
    item = build_queue()[0]
    assert item["grade"] == "too_strict"
    assert item["note"] == "template was fine"


def test_the_queue_is_oldest_first(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1"), _record("r2")])
    _sample(tmp_path / "samples.jsonl", "r2")
    _sample(tmp_path / "samples.jsonl", "r1")
    # r2's sample was written first; order follows the sample ts.
    assert [item["request_id"] for item in build_queue()] == ["r2", "r1"]


def test_the_index_page_serves_the_ui(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [])
    response = TestClient(create_app()).get("/")
    assert response.status_code == 200
    assert "Grade the door" in response.text


def test_the_page_is_mobile_ready_and_coerces_a_numeric_answer(tmp_path, monkeypatch):
    """Two bugs the UI shipped with, pinned so they cannot come back.

    * No `<meta name="viewport">` -> a phone renders the page at desktop width and
      scales the whole thing down, which is why the fonts came out tiny.
    * `answer` is a NUMBER in the log (a Noul probability or a D3 score), but the
      page rendered it through an esc() that called `.replace()` on it. That throws
      a TypeError inside render(), so the first gradeable request left the page
      stuck on "Loading..." -- invisible in an empty-queue test, which is why it
      reached the NAS.
    """
    _wire(tmp_path, monkeypatch, [])
    page = TestClient(create_app()).get("/").text
    assert 'name="viewport"' in page
    # The answer is coerced/formatted, never handed to `.replace()` raw.
    assert "String(s ??" in page
    assert "esc(a.answer)" not in page
    # The footer reserves its own space (flex column) instead of floating over the
    # content -- a fixed footer covered the last lines of the ticket on a phone.
    assert "position: fixed" not in page
    assert "flex-direction: column" in page


def test_posting_a_grade_writes_it(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r1")
    client = TestClient(create_app())
    response = client.post("/api/grade", json={"request_id": "r1", "verdict": "right", "note": "ok"})
    assert response.status_code == 200
    assert load_grades(tmp_path / "grades.jsonl")["r1"].verdict.value == "right"
    # And the queue reflects it on the next read — the resume state.
    assert client.get("/api/queue").json()[0]["grade"] == "right"


def test_an_unknown_verdict_is_a_bad_request(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r1")
    response = TestClient(create_app()).post("/api/grade", json={"request_id": "r1", "verdict": "meh"})
    assert response.status_code == 400


def test_a_grade_without_a_request_id_is_a_bad_request(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [])
    response = TestClient(create_app()).post("/api/grade", json={"verdict": "right"})
    assert response.status_code == 400


def test_each_jev_answer_carries_a_plain_english_means(tmp_path, monkeypatch):
    """A grader should not have to remember what `d2_specification_sufficient` means."""
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r1")
    entry = build_queue()[0]["jev"][0]
    assert entry["means"].startswith("Specification sufficient")
    assert "d2_specification_sufficient" not in entry["means"]
