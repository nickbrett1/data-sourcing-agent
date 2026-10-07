"""The status payload: is the door in sync — cut fresh, grades agree, grading kept up."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent.cutoff import Cutoff
from agent.grades import Grade, Verdict
from agent.samples import Sample
from agent.summary import build_summary, main

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _samples(n: int) -> dict[str, Sample]:
    return {f"r{i}": Sample(f"r{i}", "text", "ticket") for i in range(n)}


def _grades(**counts: int) -> dict[str, Grade]:
    grades: dict[str, Grade] = {}
    i = 0
    for verdict, count in counts.items():
        for _ in range(count):
            grades[f"g{i}"] = Grade(f"g{i}", Verdict(verdict))
            i += 1
    return grades


def _cut(hours_old: float, *, armed: bool = True, watermark: int = 10**9) -> Cutoff:
    # A large default watermark means "a freeze already consumed everything" — the
    # in-sync baseline. Tests that want a due re-freeze set it explicitly.
    return Cutoff(
        value=0.6, rate=0.1, window=200, n=120, armed=armed, watermark=watermark,
        computed_at=(NOW - timedelta(hours=hours_old)).isoformat(), reason="x",
    )


def test_fresh_cut_and_balanced_grades_is_in_sync():
    summary = build_summary(
        grades=_grades(right=30, too_strict=2, too_lenient=1),
        samples=_samples(40),
        cut=_cut(2),
        now=NOW,
    )
    assert summary["status"] == "ok"
    assert summary["issue"] == "in sync"


def test_too_few_grades_is_a_warning_not_a_verdict():
    summary = build_summary(grades=_grades(right=3), samples=_samples(40), cut=_cut(2), now=NOW)
    assert summary["status"] == "warn"
    assert "need 20" in summary["issue"]


def test_a_full_bucket_asks_for_a_refreeze():
    """New grades unused by the last freeze fill the bucket and trigger a freeze."""
    summary = build_summary(
        grades=_grades(right=30), samples=_samples(30), cut=_cut(2, watermark=5), now=NOW
    )
    assert any("re-freeze due" in issue for issue in summary["issues"])
    assert summary["unused_graded"] == 25
    assert summary["actionable"] == 25


def test_the_ungraded_slab_counts_toward_the_bucket():
    """Grading the backlog is what fills the next bucket, so it is part of the sum."""
    summary = build_summary(
        grades=_grades(right=10), samples=_samples(60), cut=_cut(2, watermark=10), now=NOW
    )
    assert any("grading due" in issue for issue in summary["issues"])
    assert summary["unused_graded"] == 0
    assert summary["actionable"] == 50
    # Nothing graded is unused, so there is nothing for a freeze to consume.
    assert summary["refreeze_due"] is False


def test_a_quiet_period_with_a_consumed_freeze_stays_in_sync():
    """A frozen cut is still correct while nothing new has arrived — time alone is not a trigger."""
    summary = build_summary(
        grades=_grades(right=38), samples=_samples(38), cut=_cut(500, watermark=38), now=NOW
    )
    assert summary["actionable"] == 0
    assert summary["status"] == "ok"


def test_time_is_a_backstop_only_when_there_is_material():
    summary = build_summary(
        grades=_grades(right=30), samples=_samples(30), cut=_cut(200, watermark=20), now=NOW
    )
    assert summary["unused_graded"] == 10  # under the bucket...
    assert summary["actionable"] == 10
    assert any("re-freeze overdue" in issue for issue in summary["issues"])
    assert summary["refreeze_due"] is True


def test_no_armed_cut_is_called_out():
    summary = build_summary(grades=_grades(right=30), samples=_samples(40), cut=None, now=NOW)
    assert any("provisional" in issue for issue in summary["issues"])
    assert summary["cut_armed"] is False


def test_grades_point_the_direction_the_cut_should_move():
    summary = build_summary(
        grades=_grades(right=20, too_strict=14, too_lenient=2),
        samples=_samples(40),
        cut=_cut(2),
        now=NOW,
    )
    assert any("lower the cut" in issue for issue in summary["issues"])


def test_a_small_imbalance_does_not_move_the_cut():
    """Below the margin it is noise, and a status that cries wolf gets ignored."""
    summary = build_summary(
        grades=_grades(right=40, too_strict=12, too_lenient=10),
        samples=_samples(60),
        cut=_cut(2),
        now=NOW,
    )
    assert not any("the cut" in issue for issue in summary["issues"])


def test_an_ungraded_backlog_reads_as_a_full_bucket():
    """A big ungraded pile is the same signal: enough material for a re-freeze."""
    summary = build_summary(
        grades=_grades(right=5), samples=_samples(200), cut=_cut(2, watermark=5), now=NOW
    )
    assert any("grading due" in issue for issue in summary["issues"])
    assert summary["ungraded"] == 195
    assert summary["coverage_pct"] == 2.5


def test_check_exits_nonzero_only_when_out_of_sync(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path))
    assert main(["--check"]) == 1  # nothing graded, no armed cut
    assert "warn" in capsys.readouterr().out


def test_json_flag_emits_the_payload(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path))
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] in {"ok", "warn"}
    assert "issues" in payload
