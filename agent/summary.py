"""The one read surface: is the door in sync with what we know?

Three questions, one payload — because Homepage, Dagu, and an agent all want the
same answer, and none of them should re-derive it:

* **Is the cut fresh?** A freeze consumes the graded entries it has seen. The
  question is not "how long ago" but "is there enough new material to make a new
  freeze worth doing" — `actionable = unused_graded + ungraded`. Alert when that
  reaches the bucket. A frozen cut is still correct while nothing new has arrived,
  so time is only a long backstop, never the trigger.
* **Do the grades agree with the cut?** If the door is `too_strict` far more often
  than `too_lenient`, the cut is wrong *in a direction* — and that is the signal to
  move it, not a number to discover.
* **Is grading keeping up?** The ungraded slab is part of `actionable`: grading it
  is exactly what fills the next bucket, so the alert is always satisfiable by work
  a human can actually do.

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

from agent.cutoff import DEFAULT_MIN_N, Cutoff, logged_scores
from agent.cutoff import read as read_cutoff
from agent.grades import Grade, Verdict, load_grades
from agent.samples import Sample, load_samples

# --- when call it "out of sync" ----------------------------------------------
# Deliberately few, deliberately loose: a status that cries wolf gets ignored, and
# the point is to catch a *trend*, not every wobble. All overridable in tests.
MIN_GRADES = 20  # below this the grades cannot say anything; not a warning
MISMATCH_MARGIN = 5  # and at least this many one way...
MISMATCH_FRACTION = 0.20  # ...and this share of the graded set
BUCKET_SIZE = 20  # enough new material to make a re-freeze worth doing
BACKSTOP_HOURS = 168.0  # a week — time alone never triggers; only time *with* material


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
    scores: Sequence[float] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The status payload. Every input injectable so the logic is testable alone."""
    now = now or datetime.now(UTC)
    grades = load_grades() if grades is None else grades
    samples = load_samples() if samples is None else samples
    cut = read_cutoff() if cut is None else cut
    scores = logged_scores() if scores is None else scores

    graded = len(grades)
    total = len(samples)
    ungraded = max(0, total - graded)
    coverage = graded / total if total else 0.0

    # The bucket: what a freeze has not yet consumed. A freeze records how many
    # grades it had seen (`watermark`); anything graded since is *unused*, and the
    # ungraded slab is material a person could still convert. Their sum is what the
    # next freeze would have to work with.
    watermark = cut.watermark if cut else 0
    unused_graded = max(0, graded - watermark)
    actionable = unused_graded + ungraded

    counts = {v: 0 for v in Verdict}
    for grade in grades.values():
        counts[grade.verdict] += 1
    strict, lenient = counts[Verdict.too_strict], counts[Verdict.too_lenient]

    issues: list[str] = []
    if graded < MIN_GRADES:
        issues.append(f"only {graded} graded; need {MIN_GRADES} before the grades speak")

    # (1) the bucket is full — there is a bucket's worth of material to work with.
    # It is always *satisfiable*: anything ungraded can be graded, and grading is
    # what fills the next freeze.
    age = _cut_age_hours(cut, now)
    bucket_full = actionable >= BUCKET_SIZE
    # A thin trickle can stay under the bucket for a long time. Time alone is not a
    # reason (a quiet period needs no new number), but time *plus* grades nobody has
    # frozen on yet is.
    overdue = age is not None and age > BACKSTOP_HOURS and unused_graded > 0
    unarmed = cut is None or not cut.armed
    # The first freeze is driven by the log, not the grades: it replaces the
    # provisional cut with the measured one as soon as a window's worth of scores
    # exists. Later freezes consume *grades* — with none unused there is nothing to
    # freeze — so the scheduler action is narrower than the alert: warn on the sum,
    # act only when there is a graded entry it would actually consume.
    if unarmed:
        refreeze_due = len(scores) >= DEFAULT_MIN_N
    else:
        refreeze_due = unused_graded > 0 and (bucket_full or overdue)

    if unarmed:
        issues.append("no armed cutoff; enforcing the provisional probability cut")
    elif bucket_full:
        if unused_graded >= BUCKET_SIZE:
            issues.append(
                f"re-freeze due: {unused_graded} graded since the last freeze "
                f"(+{ungraded} ungraded) >= bucket {BUCKET_SIZE}"
            )
        else:
            issues.append(
                f"grading due: {ungraded} ungraded + {unused_graded} unused "
                f">= bucket {BUCKET_SIZE}; grade them to fill the next freeze"
            )
    elif overdue:
        issues.append(f"re-freeze overdue: cut is {age:.0f}h old with {unused_graded} graded unconsumed")

    # (2) the grades name a direction the cut has not moved in.
    imbalance = abs(strict - lenient)
    if graded >= MIN_GRADES and imbalance >= MISMATCH_MARGIN and imbalance / graded >= MISMATCH_FRACTION:
        way = "lower" if strict > lenient else "raise"
        issues.append(f"grades say {way} the cut ({strict} too_strict vs {lenient} too_lenient)")

    return {
        "status": "ok" if not issues else "warn",
        "graded": graded,
        "total": total,
        "ungraded": ungraded,
        "coverage_pct": round(coverage * 100, 1),
        "watermark": watermark,
        "unused_graded": unused_graded,
        "actionable": actionable,
        "bucket_size": BUCKET_SIZE,
        "right": counts[Verdict.right],
        "too_strict": strict,
        "too_lenient": lenient,
        "unclear": counts[Verdict.unclear],
        "scores": len(scores),
        "cut_value": cut.value if cut else None,
        "cut_armed": bool(cut and cut.armed),
        "cut_age_hours": round(age, 1) if age is not None else None,
        "refreeze_due": refreeze_due,
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
