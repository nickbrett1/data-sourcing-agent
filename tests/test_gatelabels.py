"""Tests for the delayed label — `downstream_outcome`, filled after the fact."""

from __future__ import annotations

import pytest

from agent.gatelabels import (
    Label,
    Outcome,
    append_label,
    apply_labels,
    gate_label_path,
    load_labels,
)


def test_only_real_outcomes_carry_a_value():
    assert Outcome.approved.label_value == 1.0
    assert Outcome.rejected.label_value == 0.0
    assert Outcome.reran.label_value == 0.0
    # The agent's own labels carry no signal: drafted is not "good".
    assert Outcome.drafted.label_value is None
    assert Outcome.run_failed.label_value is None
    assert Outcome.door_stopped.label_value is None


def test_append_and_load_round_trip(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label("r1", Outcome.drafted, path=path)
    labels = load_labels(path)
    assert labels["r1"].outcome == "drafted"
    assert labels["r1"].source == "agent"


def test_the_latest_label_wins_but_history_is_kept(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label("r1", Outcome.drafted, path=path)
    append_label("r1", Outcome.approved, source="human", path=path)
    assert load_labels(path)["r1"].outcome == "approved"
    assert len(path.read_text().splitlines()) == 2  # both events on disk


def test_unknown_outcome_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_label("r1", "banana", path=tmp_path / "l.jsonl")


def test_a_truncated_label_line_is_skipped(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label("r1", Outcome.drafted, path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"request_id": "half')
    assert set(load_labels(path)) == {"r1"}


def test_apply_labels_stamps_matching_records_only(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_label("r1", Outcome.approved, source="human", path=path)
    records = [
        {"request_id": "r1", "question_id": "d2"},
        {"request_id": "r2", "question_id": "d2"},
    ]
    assert apply_labels(records, load_labels(path)) == 1
    assert records[0]["downstream_outcome"] == "approved"
    assert records[0]["outcome_source"] == "human"
    assert "downstream_outcome" not in records[1]


def test_label_path_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LABEL_PATH", str(tmp_path / "explicit.jsonl"))
    assert gate_label_path() == tmp_path / "explicit.jsonl"
    monkeypatch.delenv("GATE_LABEL_PATH")
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path))
    assert gate_label_path() == tmp_path / "gate-labels.jsonl"


def test_load_labels_of_a_missing_file_is_empty(tmp_path):
    assert load_labels(tmp_path / "nope.jsonl") == {}
    assert Label.__dataclass_fields__  # the shape is used by load_labels
