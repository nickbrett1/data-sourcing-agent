"""Read the gate's shadow log and report what it says — the calibration view.

The gate writes records (gate memo §3) but writes nothing *about* them. This module
is the reader: it turns `gate-log.jsonl` into the numbers the enforcement decisions
actually turn on, so those decisions are made from the corpus rather than from
memory. It is read-only and model-free — it opens the log, counts, and prints.

What it answers, and why each matters:

* **Volume and failures** — how many requests, over what window, and how often Jev
  gave no opinion. A gate whose availability is unmeasured is a gate you are
  trusting blind.
* **Score distributions** — per question, the spread of probabilities and scores.
  This is the shape a probability threshold would cut.
* **The day-one lever** — `escalation_cutoff` at a range of rates: "with review
  capacity for the top N%, this is the score that escalates." This is settable with
  **no labels at all** (data-acquisition §8 #3a), so the report computes it first.
* **What would have stopped** — the rate at which the door's composition would
  refuse or clarify, at the thresholds in force. This is the size of the change
  enforcement makes, before making it.
* **Labels** — how much of the log carries a `downstream_outcome`. It is `None`
  today, and that absence is the finding: the probability cuts stay provisional
  until this is non-zero (gate memo §5). The report says so rather than printing a
  calibrated-looking number that is not calibrated.

The one honest caveat it cannot fix is §6.5.1: even once labels arrive, they are
clean for D5 ("was this ticket worth buying") and **not** for D2 (a wrong axis is
legal, priced, and approved — nothing distinguishes it). The report flags D2's
label as the one not to trust.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent.gate import (
    D1,
    D2,
    D3,
    D3_CLOSE,
    D3_LEVELS,
    D5,
    DEFAULT_POLICY,
    GatePolicy,
    escalation_cutoff,
)
from agent.gatelabels import apply_labels, load_labels
from agent.gatelog import gate_log_path
from agent.grades import Grade, Verdict, advice, load_grades
from agent.grades import tally as grade_tally

# The rates the report prices the day-one lever at: "if we can review the top N%".
REVIEW_RATES = (0.01, 0.02, 0.05, 0.10, 0.20)

# Questions whose answer is a probability (thresholdable) vs a grade (ordered).
NOUL_QUESTIONS = (D1, D2, D5)


def load_records(path: Path | None = None) -> list[dict]:
    """Read the JSONL log, skipping unreadable lines rather than dying on one.

    A shadow log is appended by a best-effort writer, so the last line can be
    truncated by a crash. Skipping it (with a count) is the right failure: a report
    over 9,999 good rows should not be withheld because row 10,000 was half-written.
    """
    target = path or gate_log_path()
    if not target.exists():
        return []
    records: list[dict] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


@dataclass
class QuestionStats:
    """Per-question summary: how many, how many failed, and the distribution."""

    question_id: str
    total: int = 0
    failed: int = 0
    values: list[float] = field(default_factory=list)
    levels: Counter = field(default_factory=Counter)

    @property
    def usable(self) -> int:
        return len(self.values)

    @property
    def fail_rate(self) -> float:
        return self.failed / self.total if self.total else 0.0

    def quantile(self, q: float) -> float | None:
        if not self.values:
            return None
        if len(self.values) == 1:
            return self.values[0]
        return statistics.quantiles(self.values, n=100)[int(q * 100) - 1]

    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None


def _question_stats(records: list[dict]) -> dict[str, QuestionStats]:
    stats: dict[str, QuestionStats] = {}
    for record in records:
        qid = record.get("question_id")
        if qid is None:
            continue
        stat = stats.setdefault(qid, QuestionStats(question_id=qid))
        stat.total += 1
        if record.get("failed"):
            stat.failed += 1
            continue
        if qid == D3:
            answer = record.get("answer")
            if answer is not None:
                # Jev's Score is the *expected level* — a continuous value in
                # [0, len-1], not an index. Keep the exact value for the
                # distribution and bucket by the nearest level (round, not
                # truncate: `int(0.67) == 0` would file a near-`plausible` answer
                # under `none`). Clamp, so a stray out-of-range score is still
                # counted rather than dropped under a string key.
                expected = float(answer)
                stat.values.append(expected)
                index = round(expected)
                stat.levels[D3_LEVELS[index] if 0 <= index < len(D3_LEVELS) else str(index)] += 1
        else:
            probability = record.get("probability")
            if probability is not None:
                stat.values.append(float(probability))
    return stats


@dataclass
class GateReport:
    """Everything the corpus says, assembled — then rendered by `render`."""

    requests: int
    records: int
    first_ts: str | None
    last_ts: str | None
    questions: dict[str, QuestionStats]
    computed_actions: Counter
    holdbacks: int
    labelled: int
    policy: GatePolicy
    outcomes: Counter = field(default_factory=Counter)
    # region -> Counter(outcome or "unlabelled"). The "was Jev wrong?" view: the
    # outcome means a different thing in each region, and `stopped` is confounded.
    regions: dict[str, Counter] = field(default_factory=dict)
    grades: dict[str, Grade] = field(default_factory=dict)

    @property
    def would_stop(self) -> int:
        """Requests whose computed action is not `proceed` — the door's workload."""
        return sum(
            count for action, count in self.computed_actions.items() if action != "proceed"
        )

    @property
    def would_stop_rate(self) -> float:
        return self.would_stop / self.requests if self.requests else 0.0


def _region(computed: str, holdback: bool) -> str:
    """Which of the three parts of the score space a request landed in.

    The three parts are the whole answer to "was Jev wrong?" — because the label
    means a different thing in each, and one of them has no label at all:

    * `let_through`   — Jev said pass. A `rejected`/`reran` downstream is a Jev
                        false-accept, measured (but §6.5.1: not cleanly for D2).
    * `would_stop_admitted` — Jev said stop, the holdback admitted it anyway. An
                        `approved` downstream is a Jev false-reject, MEASURED — the
                        only place a false reject is ever visible (§6.6).
    * `stopped`       — Jev said stop and the door routed it. The outcome is
                        confounded: the request then got the clarifying question, so
                        a later `approved` cannot say whether the stop was right.
    """
    if computed == "proceed":
        return "let_through"
    return "would_stop_admitted" if holdback else "stopped"


def summarise(records: list[dict], policy: GatePolicy = DEFAULT_POLICY) -> GateReport:
    """Fold the raw records into the report — one pass, no model, deterministic."""
    requests = {r.get("request_id") for r in records if r.get("request_id")}
    # The per-request facts (computed action, holdback, outcome) are recorded on
    # every row, so gather them once per request rather than four times.
    per_request: dict[str, dict] = {}
    for record in records:
        rid = record.get("request_id")
        if rid is None:
            continue
        facts = per_request.setdefault(rid, {"action": "proceed", "holdback": False, "outcome": None})
        facts["action"] = record.get("action_computed", "proceed")
        facts["holdback"] = facts["holdback"] or bool(record.get("holdback"))
        if record.get("downstream_outcome"):
            facts["outcome"] = record["downstream_outcome"]
    actions: Counter = Counter()
    holdbacks = 0
    outcomes: Counter = Counter()
    regions: dict[str, Counter] = {}
    for facts in per_request.values():
        actions[facts["action"]] += 1
        if facts["holdback"]:
            holdbacks += 1
        if facts["outcome"]:
            outcomes[facts["outcome"]] += 1
        regions.setdefault(_region(facts["action"], facts["holdback"]), Counter())[
            facts["outcome"] or "unlabelled"
        ] += 1
    timestamps = sorted(r["ts"] for r in records if r.get("ts"))
    return GateReport(
        requests=len(requests),
        records=len(records),
        first_ts=timestamps[0] if timestamps else None,
        last_ts=timestamps[-1] if timestamps else None,
        questions=_question_stats(records),
        computed_actions=actions,
        holdbacks=holdbacks,
        outcomes=outcomes,
        regions=regions,
        grades=load_grades(),
        labelled=sum(
            1
            for r in records
            if r.get("downstream_outcome") is not None and r.get("question_id") in (D2, D5)
        ),
        policy=policy,
    )


def _fmt(value: float | None, places: int = 3) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def render(report: GateReport) -> str:
    """The report as text, ordered by what the decisions depend on."""
    lines: list[str] = []
    add = lines.append

    add("Gate shadow log — calibration report")
    add("=" * 38)
    add("")
    add(f"requests logged : {report.requests}")
    add(f"records         : {report.records}")
    add(f"window          : {report.first_ts or '—'} .. {report.last_ts or '—'}")
    add(f"enforced        : {report.policy.enforce}")
    add("")

    # Jev availability comes first: every other number is conditioned on the gate
    # having answered, so its failure rate is the context for all of them.
    add("Jev availability (a failure is 'no opinion', not an error)")
    add("-" * 52)
    if report.questions:
        for qid, stat in sorted(report.questions.items()):
            add(f"  {qid:<28} n={stat.total:<5} failed={stat.failed:<4} ({stat.fail_rate:.1%})")
    else:
        add("  (no records)")
    add("")

    add("Distributions")
    add("-" * 52)
    for qid, stat in sorted(report.questions.items()):
        if qid == D3:
            spread = ", ".join(f"{level}={count}" for level, count in stat.levels.most_common())
            add(f"  {qid:<28} {spread or '(none)'}")
        else:
            add(
                f"  {qid:<28} median={_fmt(stat.median())} "
                f"p10={_fmt(stat.quantile(0.10))} p90={_fmt(stat.quantile(0.90))} "
                f"n={stat.usable}"
            )
    add("")

    # The day-one lever: settable with NO labels (§8 #3a). Reported before the
    # probability view on purpose — this is the threshold that is actually usable.
    add("Day-one lever — escalate the top N% (no labels required, §8 #3a)")
    add("-" * 58)
    d2 = report.questions.get(D2)
    if d2 and d2.values:
        add(f"  D2 scores available: {d2.usable}")
        for rate in REVIEW_RATES:
            cut = escalation_cutoff(d2.values, rate, side="low")
            add(f"    top {rate:>5.0%}  ->  escalate when D2 < {_fmt(cut, 3)}")
    else:
        add("  (no D2 scores yet)")
    add("")

    add("What would have stopped (computed action, current thresholds)")
    add("-" * 58)
    if report.computed_actions:
        for action, count in report.computed_actions.most_common():
            add(f"  {action:<16} {count:>6}  ({count / report.requests:.1%})")
        add(f"  would-stop rate  {report.would_stop_rate:.1%} of {report.requests} requests")
    else:
        add("  (no requests)")
    add("")

    add("Labels (the delayed `downstream_outcome`)")
    add("-" * 58)
    add(f"  labelled D2/D5 records: {report.labelled} of {report.records}")
    if report.outcomes:
        tally = ", ".join(f"{name}={count}" for name, count in report.outcomes.most_common())
        add(f"  outcomes: {tally}")
    if report.labelled == 0:
        add("  NONE yet — the probability cuts below are provisional, not calibrated.")
        add("  The `escalate the top N%` lever above is the only threshold that is usable now.")
    add("  D2's label is the one NOT to trust (§6.5.1): a wrong axis is legal, priced,")
    add("  and approved; nothing in the log distinguishes it. D5's label is clean.")
    add("")

    if report.holdbacks:
        add(f"Holdback admissions: {report.holdbacks} (sampled rejected region, §6.6)")
        add("")

    add("Was Jev wrong? (outcome by region — the only place a false reject is visible)")
    add("-" * 74)
    if not report.regions:
        add("  (no requests)")
    for region, interpretation in (
        ("let_through", "Jev said pass -> a `rejected`/`reran` here is a false ACCEPT"),
        ("would_stop_admitted", "Jev said stop, holdback admitted -> an `approved` here is a false REJECT"),
        ("stopped", "Jev said stop and it routed -> outcome confounded by the clarifying question"),
    ):
        counts = report.regions.get(region)
        if not counts:
            continue
        total = sum(counts.values())
        tally = ", ".join(f"{name}={count}" for name, count in counts.most_common())
        add(f"  {region:<22} n={total:<5} {tally}")
        add(f"  {'':<22} {interpretation}")
    add("")
    add("  A region with only `unlabelled` is the honest state: nothing has been")
    add("  reviewed yet. D2's false accepts stay invisible even once it is (§6.5.1).")
    add("")

    add("Your grades (a verdict on the door's call, from the grading UI)")
    add("-" * 64)
    if not report.grades:
        add("  none yet — `python -m agent.grader`, then `docker exec` it or open localhost:8801.")
    else:
        counts = grade_tally(report.grades)
        line = ", ".join(f"{v.value}={counts[v]}" for v in Verdict)
        add(f"  {len(report.grades)} graded: {line}")
        add(f"  => {advice(report.grades)}")
    add("")

    return "\n".join(lines)


def _policy_from_args(args: argparse.Namespace) -> GatePolicy:
    return GatePolicy(
        enforce=args.enforce,
        d1_cut=args.d1_cut,
        d2_cut=args.d2_cut,
        d5_cut=args.d5_cut,
        d3_min=D3_LEVELS.index(args.d3_min),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: `python -m agent.gate_report [path]` (defaults to `$GATE_LOG_PATH`)."""
    parser = argparse.ArgumentParser(description="Report over the gate shadow log.")
    parser.add_argument("path", nargs="?", type=Path, default=None, help="Path to gate-log.jsonl")
    parser.add_argument("--d1-cut", type=float, default=DEFAULT_POLICY.d1_cut)
    parser.add_argument("--d2-cut", type=float, default=DEFAULT_POLICY.d2_cut)
    parser.add_argument("--d5-cut", type=float, default=DEFAULT_POLICY.d5_cut)
    parser.add_argument("--d3-min", choices=list(D3_LEVELS), default=D3_LEVELS[D3_CLOSE])
    parser.add_argument("--enforce", action="store_true", help="Report as if enforcing.")
    args = parser.parse_args(argv)

    records = load_records(args.path)
    if not records:
        print("No gate log found (set $GATE_LOG_PATH or pass a path).", file=sys.stderr)
        return 1
    # Join the delayed labels so the report sees outcomes, not just decisions.
    apply_labels(records, load_labels())
    print(render(summarise(records, _policy_from_args(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
