"""The rolling cutoff: "escalate the top N%" frozen into a number."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agent import cutoff
from agent.cutoff import Cutoff, arm, d2_scores, read, resolve_policy, write
from agent.gate import (
    D1,
    D2,
    D2_CUT_PLACEHOLDER,
    DEFAULT_POLICY,
    GateAction,
    escalation_cutoff,
)

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)


def _d2(value: float, *, failed: bool = False, request_id: str = "r") -> dict:
    return {"request_id": request_id, "question_id": D2, "primitive": "noul",
            "answer": value, "probability": value, "failed": failed}


# --- arming -------------------------------------------------------------------


def test_a_thin_log_does_not_arm_a_cutoff():
    cut = arm([0.1, 0.2, 0.3], rate=0.10, min_n=50, now=NOW)
    assert cut.armed is False
    assert cut.n == 3
    assert "need 50" in cut.reason
    # It still reports the provisional cut, so the log has something to stamp.
    assert cut.value == D2_CUT_PLACEHOLDER


def test_an_armed_cutoff_is_the_top_rate_of_the_window():
    scores = [i / 100 for i in range(100)]
    cut = arm(scores, rate=0.10, window=200, min_n=50, now=NOW)
    assert cut.armed is True
    assert cut.n == 100
    assert cut.value == escalation_cutoff(scores, 0.10, side="low")


def test_the_window_uses_trailing_scores_only():
    older = [0.0] * 100  # would drag a rate over the whole history down
    recent = [0.9 + i / 1000 for i in range(100)]
    cut = arm(older + recent, rate=0.10, window=100, min_n=50, now=NOW)
    assert cut.n == 100
    assert cut.value == escalation_cutoff(recent, 0.10, side="low")


# --- reading scores out of the log -------------------------------------------


def test_d2_scores_take_the_noul_answer_and_skip_other_questions():
    records = [
        _d2(0.4),
        {"question_id": D1, "primitive": "noul", "answer": 0.99, "failed": False},
        _d2(0.7, failed=True),  # a Jev outage is "no opinion", not a value
        _d2(0.5),
    ]
    assert d2_scores(records) == [0.4, 0.5]


def test_d2_scores_ignore_a_non_numeric_answer():
    assert d2_scores([{"question_id": D2, "failed": False, "answer": None}]) == []


# --- the frozen file ----------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    frozen = Cutoff(value=0.83, rate=0.10, window=200, n=120, armed=True,
                    computed_at=NOW.isoformat(), reason="top 10% of the trailing 120")
    write(frozen, path)
    assert read(path) == frozen


def test_a_missing_cutoff_file_is_not_an_error(tmp_path):
    assert read(tmp_path / "nope.json") is None


def test_a_corrupt_cutoff_file_reads_as_absent(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    path.write_text("{not json", encoding="utf-8")
    assert read(path) is None


def test_refresh_with_no_log_writes_an_unarmed_cutoff(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    frozen = cutoff.refresh([], path=path, now=NOW)
    assert frozen.armed is False
    assert read(path) is not None
    assert json.loads(path.read_text(encoding="utf-8"))["armed"] is False


def test_refresh_over_a_full_window_arms(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    records = [_d2(i / 100) for i in range(80)]
    frozen = cutoff.refresh(records, rate=0.10, window=200, min_n=50, path=path, now=NOW)
    assert frozen.armed is True
    assert read(path).value == frozen.value


def test_a_freeze_records_how_many_grades_it_consumed(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    grades = {"a": object(), "b": object(), "c": object()}
    frozen = cutoff.refresh([], path=path, grades=grades, now=NOW)
    assert frozen.watermark == 3
    assert read(path).watermark == 3


def test_an_old_cutoff_file_without_a_watermark_reads_as_zero(tmp_path):
    """A file written before the watermark existed consumes nothing — the safe default."""
    import json

    path = tmp_path / "gate-cutoff.json"
    path.write_text(json.dumps({"value": 0.6, "armed": True}), encoding="utf-8")
    assert read(path).watermark == 0


# --- the switch ---------------------------------------------------------------


def test_without_the_env_var_the_policy_stays_observe_only(tmp_path):
    assert resolve_policy(environ={}, path=tmp_path / "none.json") == DEFAULT_POLICY
    assert resolve_policy(environ={}, path=tmp_path / "none.json").enforce is False


def test_the_switch_goes_on_at_the_provisional_cut_with_no_measured_rate(tmp_path):
    policy = resolve_policy(environ={"GATE_ENFORCE": "1"}, path=tmp_path / "none.json")
    assert policy.enforce is True
    assert policy.d2_cut == D2_CUT_PLACEHOLDER
    # Only the ask-path is routed; a computed reject is still observed.
    assert policy.enforce_actions == frozenset({GateAction.ask_clarifying})


def test_an_armed_cutoff_refines_the_cut_the_switch_uses(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    write(Cutoff(value=0.72, rate=0.10, window=200, n=90, armed=True, reason="armed"), path)
    policy = resolve_policy(environ={"GATE_ENFORCE": "1"}, path=path)
    assert policy.enforce is True
    assert policy.d2_cut == 0.72
    assert policy.enforce_actions == frozenset({GateAction.ask_clarifying})


def test_an_unarmed_cutoff_does_not_refine_the_cut(tmp_path):
    path = tmp_path / "gate-cutoff.json"
    write(Cutoff(value=0.99, rate=0.10, window=200, n=3, armed=False, reason="thin"), path)
    policy = resolve_policy(environ={"GATE_ENFORCE": "1"}, path=path)
    assert policy.d2_cut == D2_CUT_PLACEHOLDER


def test_the_cli_reports_what_it_froze(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(cutoff.CUTOFF_ENV, str(tmp_path / "gate-cutoff.json"))
    assert cutoff.main(["--rate", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "NOT armed" in out


def test_if_due_does_nothing_when_the_bucket_is_not_full(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(cutoff.DEFAULT_DIR_ENV, str(tmp_path))
    assert cutoff.main(["--if-due"]) == 0
    assert "no re-freeze due" in capsys.readouterr().out
    # And it did not write a cutoff: an early freeze would bump the watermark.
    assert not (tmp_path / "gate-cutoff.json").exists()


def test_if_due_freezes_when_the_bucket_is_full(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(cutoff.DEFAULT_DIR_ENV, str(tmp_path))
    # A log worth arming over, and enough samples to fill the bucket.
    from agent.gate import D2
    from agent.gatelog import append_records
    from agent.samples import append_sample

    records = []
    for i in range(60):
        records.append(
            {"request_id": f"r{i}", "question_id": D2, "answer": 0.5 + i / 1000, "failed": False}
        )
        append_sample(f"r{i}", "text", "ticket")
    append_records(records)
    assert cutoff.main(["--if-due"]) == 0
    assert (tmp_path / "gate-cutoff.json").exists()
