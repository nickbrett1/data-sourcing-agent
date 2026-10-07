"""The one read surface: is the door in sync with what we know?

Three questions, one payload — because Homepage, Dagu, and an agent all want the
same answer, and none of them should re-derive it:

* **Is the cut fresh?** The enforced D2 cut is frozen from a window of scores and
  re-frozen on a cadence. A cut older than that window is enforcing a number the
  log no longer supports.
* **Do the grades agree with the cut?** If the door is `too_strict` far more often
  than `too_lenient`, the cut is wrong *in a direction* — and that is the signal to
  move it, not a number to discover.
* **Is grading keeping up?** An ungraded backlog means the second answer is stale
  even when the first is fresh.

This is a *status*, not a metric series: it is computed on demand from the streams
that already exist. That keeps it cheap and keeps the cardinality rule intact
(`D5TaBez3KoKmRdrXSuHhqK`) — no per-request label ever reaches a consumer.

Streams: the grade stream and the sample stream (`agent/grades.py`,
`agent/samples.py`), the shadow log (`agent/gatelog.py`), and the frozen cut
(`agent/cutoff.py`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from agent.cutoff import Cutoff
from agent.cutoff import read as read_cutoff
from agent.grades import Grade, Verdict, load_grades
from agent.samples import Sample, load_samples

# --- when call it "out of sync" ----------------------------------------------
# Deliberately few, deliberately loose: a status that cries wolf gets ignored, and
# the point is to catch a *trend*, not every wobble. All overridable in tests.
MIN_GRADES = 20  # below this the grades cannot say anything; not a warning
MISMATCH_MARGIN = 5  # and at least this many one way...
MISMATCH_FRACTION = 0.20  # ...and this share of the graded set
CUT_STALE_HOURS = 26.0  # a daily re-freeze leaves a day of slack
BACKLOG_FRACTION = 0.50  # more than half ungraded...
BACKLOG_MIN = 20  # ...and at least this many, is a backlog


def _cut_age_hours(cut: Cutoff | None, now: datetime) -> float | None:
    if cut is None or not cut.computed_at:
        return None
    try:
        computed = datetime.fromisoformat(cut.computed_at)
    except ValueError:
        return None
    if computed.tzinfo is None:
        computed = computed.replace(tzinfo=UTC)
    return (now - computed).total_seconds() / 3600.0


def build_summary(
    *,
    grades: Mapping[str, Grade] | None = None,
    samples: Mapping[str, Sample] | None = None,
    cut: Cutoff | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The status payload. Every input injectable so the logic is testable alone."""
    now = now or datetime.now(UTC)
    grades = load_grades() if grades is None else grades
    samples = load_samples() if samples is None else samples
    cut = read_cutoff() if cut is None else cut

    graded = len(grades)
    total = len(samples)
    ungraded = max(0, total - graded)
    coverage = graded / total if total else 0.0

    counts = {v: 0 for v in Verdict}
    for grade in grades.values():
        counts[grade.verdict] += 1
    strict, lenient = counts[Verdict.too_strict], counts[Verdict.too_lenient]

    issues: list[str] = []
    if graded < MIN_GRADES:
        issues.append(f"only {graded} graded; need {MIN_GRADES} before the grades speak")

    # (1) the cut is older than its refresh cadence.
    age = _cut_age_hours(cut, now)
    if cut is None or not cut.armed:
        issues.append("no armed cutoff; enforcing the provisional probability cut")
    elif age is not None and age > CUT_STALE_HOURS:
        issues.append(f"cut is {age:.0f}h old (window {CUT_STALE_HOURS:.0f}h); re-freeze due")

    # (2) the grades name a direction the cut has not moved in.
    imbalance = abs(strict - lenient)
    if graded >= MIN_GRADES and imbalance >= MISMATCH_MARGIN and imbalance / graded >= MISMATCH_FRACTION:
        way = "lower" if strict > lenient else "raise"
        issues.append(f"grades say {way} the cut ({strict} too_strict vs {lenient} too_lenient)")

    # (3) grading is behind, so (2) is stale even if the count looks decisive.
    if ungraded >= BACKLOG_MIN and coverage < (1 - BACKLOG_FRACTION):
        issues.append(f"grading backlog: {ungraded} of {total} ungraded ({coverage:.0%} done)")

    return {
        "status": "ok" if not issues else "warn",
        "graded": graded,
        "total": total,
        "ungraded": ungraded,
        "coverage_pct": round(coverage * 100, 1),
        "right": counts[Verdict.right],
        "too_strict": strict,
        "too_lenient": lenient,
        "unclear": counts[Verdict.unclear],
        "cut_value": cut.value if cut else None,
        "cut_armed": bool(cut and cut.armed),
        "cut_age_hours": round(age, 1) if age is not None else None,
        "issues": issues,
        "issue": issues[0] if issues else "in sync",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agent.summary [--json]` — print the status; `--check` to exit on drift.

    `--check` is the one Dagu needs: exit non-zero (with the reason on stdout) when
    anything is out of sync, so the scheduler's own alerting does the alerting and
    this stays a plain command.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Is the gate in sync with what we know?")
    parser.add_argument("--json", action="store_true", help="machine-readable payload (Homepage)")
    parser.add_argument("--check", action="store_true", help="exit non-zero when out of sync")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = build_summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{summary['status']}: {summary['issue']}")
        for issue in summary["issues"][1:]:
            print(f"  - {issue}")
    if args.check and summary["status"] != "ok":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
