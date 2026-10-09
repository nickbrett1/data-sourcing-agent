"""The grading UI: the queue it builds and the one write it makes."""

from __future__ import annotations

import json

from starlette.testclient import TestClient

from agent.grader import build_queue, create_app
from agent.grades import load_grades
from agent.samples import append_sample


def _sample(path, rid, text="one day of SPY options", ticket="dataset: OPRA.PILLAR"):
    append_sample(rid, text, ticket, path=path)


def _record(rid, *, action="ask_clarifying", holdback=False, failed=False, answer=0.4, threshold=0.91):
    return {
        "request_id": rid,
        "question_id": "d2_specification_sufficient",
        "primitive": "noul",
        "answer": answer,
        "failed": failed,
        "threshold_at_time": threshold,
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
    # The verdicts are explained on screen, so `unclear` does not read as a riddle.
    assert 'id="legend"' in page
    assert "doesn't settle it" in page


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


def test_an_answer_carries_the_cut_that_was_in_force(tmp_path, monkeypatch):
    """So the UI can show *why* the door stopped: this answer fell below this cut."""
    _wire(tmp_path, monkeypatch, [_record("r1", answer=0.8, threshold=0.91)])
    _sample(tmp_path / "samples.jsonl", "r1")
    entry = build_queue()[0]["jev"][0]
    assert entry["answer"] == 0.8
    assert entry["threshold"] == 0.91


def test_the_page_explains_which_answer_breached_its_cut(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, [_record("r1")])
    _sample(tmp_path / "samples.jsonl", "r1")
    page = TestClient(create_app()).get("/").text
    # The breach is computed from answer < threshold in the page, not left to the eye.
    assert "a.answer < a.threshold" in page
    assert "class=\"breach\"" in page


def test_grading_the_last_item_is_not_a_dead_stop(tmp_path, monkeypatch):
    """Grading the end of the queue must not leave the screen frozen: the page
    wraps to any still-ungraded item and shows an explicit all-graded state, and
    re-tapping an unchanged verdict is a no-op rather than a duplicate write."""
    _wire(tmp_path, monkeypatch, [])
    _sample(tmp_path / "samples.jsonl", "r1")
    page = TestClient(create_app()).get("/").text
    assert "queue.findIndex((x, j) => j > i && !x.grade)" in page      # forward
    assert "const anywhere = queue.findIndex(x => !x.grade)" in page   # then wrap
    assert "all ${queue.length} graded · nothing left" in page         # explicit done
    assert "your verdict:" in page                                     # visible feedback
    assert "q.grade === v" in page                                     # no duplicate writes


def test_the_done_state_leads_with_the_result_not_a_request(tmp_path, monkeypatch):
    """Finished = summary, not a request you can accidentally re-grade. The grade
    controls come back only when the grader asks to review."""
    _wire(tmp_path, monkeypatch, [])
    _sample(tmp_path / "samples.jsonl", "r1")
    page = TestClient(create_app()).get("/").text
    assert 'id="controls"' in page and 'id="donebar"' in page
    assert "document.getElementById('controls').hidden = done && !review" in page
    assert 'id="summary"' in page
    assert "All ${n} requests graded." in page
    assert "review / re-grade" in page
    assert "const isDone = () => queue.length > 0 && queue.every(x => x.grade)" in page


def test_a_legacy_nan_answer_does_not_500_the_queue(tmp_path, monkeypatch):
    """An old log row can carry a bare `NaN` token, which Python reads back as a
    non-finite float. Starlette's JSONResponse (allow_nan=False) rejects that, so
    one bad legacy row would 500 the whole endpoint — the reader must normalise."""
    _wire(tmp_path, monkeypatch, [])
    log = tmp_path / "gate-log.jsonl"
    log.write_text(json.dumps(_record("r1", answer=float("nan"))) + "\n", encoding="utf-8")
    _sample(tmp_path / "samples.jsonl", "r1")
    response = TestClient(create_app()).get("/api/queue")
    assert response.status_code == 200
    assert response.json()[0]["jev"][0]["answer"] is None
