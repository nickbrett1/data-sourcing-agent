"""The grading sample: text + ticket, joined to the gate log by request id."""

from __future__ import annotations

import json

from agent.samples import (
    Sample,
    append_sample,
    join,
    load_samples,
    render_one,
    sample_path,
)


def _record(request_id: str, **kw) -> dict:
    base = {"request_id": request_id, "question_id": "d2_specification_sufficient",
            "answer": 0.4, "failed": False, "action_computed": "ask_clarifying",
            "holdback": False}
    base.update(kw)
    return base


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "gate-samples.jsonl"
    append_sample("r1", "one day of SPY options", "state: draft\n", path=path)
    loaded = load_samples(path)
    assert loaded["r1"].text == "one day of SPY options"
    assert loaded["r1"].ticket == "state: draft\n"


def test_the_latest_sample_for_an_id_wins(tmp_path):
    path = tmp_path / "gate-samples.jsonl"
    append_sample("r1", "first", "t1", path=path)
    append_sample("r1", "second", "t2", path=path)
    assert load_samples(path)["r1"].text == "second"


def test_a_truncated_line_is_skipped(tmp_path):
    path = tmp_path / "gate-samples.jsonl"
    append_sample("r1", "ok", "t", path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "r2", "text": "half')
    assert set(load_samples(path)) == {"r1"}


def test_a_missing_file_is_empty(tmp_path):
    assert load_samples(tmp_path / "none.jsonl") == {}


def test_sample_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "x.jsonl"))
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path / "ignored"))
    assert sample_path() == tmp_path / "x.jsonl"


def test_join_pairs_a_sample_with_its_rows_only(tmp_path):
    samples = {"r1": Sample("r1", "text", "ticket"), "orphan": Sample("orphan", "t", "t")}
    pairs = join(samples, [_record("r1")])
    assert len(pairs) == 1
    assert pairs[0][0].request_id == "r1"


def test_render_one_shows_the_four_things_a_human_grades(tmp_path):
    out = render_one(Sample("r1", "one day of SPY options", "dataset: OPRA.PILLAR"), [_record("r1")])
    assert "one day of SPY options" in out
    assert "dataset: OPRA.PILLAR" in out
    assert "d2_specification_sufficient" in out  # Jev's score
    assert "DOOR SAID: ask_clarifying" in out


def test_render_one_marks_a_failed_question_as_no_opinion():
    out = render_one(Sample("r1", "t", "t"), [_record("r1", failed=True)])
    assert "FAILED (no opinion)" in out


def test_render_one_names_a_holdback_admission():
    out = render_one(Sample("r1", "t", "t"), [_record("r1", holdback=True)])
    assert "holdback: admitted" in out


def test_main_reports_an_empty_queue(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "none.jsonl"))
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "none-log.jsonl"))
    from agent import samples

    assert samples.main([]) == 1
    assert "no gradeable samples" in capsys.readouterr().out


def test_main_renders_the_queue(tmp_path, monkeypatch, capsys):
    sample_file = tmp_path / "gate-samples.jsonl"
    log_file = tmp_path / "gate-log.jsonl"
    append_sample("r1", "one day of SPY options", "dataset: OPRA.PILLAR", path=sample_file)
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_record("r1")) + "\n")
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(sample_file))
    monkeypatch.setenv("GATE_LOG_PATH", str(log_file))
    from agent import samples

    assert samples.main([]) == 0
    out = capsys.readouterr().out
    assert "one day of SPY options" in out
    assert "DOOR SAID: ask_clarifying" in out
