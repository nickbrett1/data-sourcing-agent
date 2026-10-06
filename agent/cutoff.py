"""The rolling cutoff — turn "escalate the top N%" into a frozen number.

A rate is a quantity over a *population*, not a request (data-acquisition §8 #3a):
with `N` = 10% and 200 scores the top 20 escalate, and that is a statement about
the twenty, not about any one row. So enforcing on a rate takes three things the
gate cannot do per-request:

* **compute** the boundary over a window of logged scores (`escalation_cutoff`),
* **freeze** it, so every request in a period is judged against the same number,
* **re-freeze** on a cadence — daily here — because the population moves.

This module keeps that number and its provenance (`rate`, `window`, `n`, when it
was computed) in one small file. The gate then reads `d2_cut` from it and never
has to know a rate exists.

Arithmetic over the log. No model, no Jev, no network.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from agent.gate import (
    D2,
    D2_CUT_PLACEHOLDER,
    DEFAULT_POLICY,
    GateAction,
    GatePolicy,
    escalation_cutoff,
)

# --- where the frozen number lives -------------------------------------------

CUTOFF_ENV = "GATE_CUTOFF_PATH"
DEFAULT_DIR_ENV = "AGENT_STATE_DIR"
ENFORCE_ENV = "GATE_ENFORCE"

# --- the starting configuration ----------------------------------------------
# `DEFAULT_WINDOW` / `DEFAULT_MIN_N` are the stability knobs: a cut computed from
# twenty scores swings on a single request, so the rate does not apply until the
# window has a floor's worth. `DEFAULT_RATE` is the review-capacity knob — the
# fraction of requests we are willing to look at, priced at 1/2/5/10/20% in
# `gate_report`. 10% is a handful a day and one line to change.
DEFAULT_WINDOW = 200
DEFAULT_MIN_N = 50
DEFAULT_RATE = 0.10

# The cautious start: route the ask-path only. A computed `reject` (D1 "not in
# remit") stays observed until the log says a hard stop earns its keep.
START_ACTIONS = frozenset({GateAction.ask_clarifying})


def cutoff_path() -> Path:
    """Where the frozen cutoff lives: explicit env, else the agent state dir."""
    explicit = os.environ.get(CUTOFF_ENV)
    if explicit:
        return Path(explicit)
    return Path(os.environ.get(DEFAULT_DIR_ENV, ".")) / "gate-cutoff.json"


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- the frozen number --------------------------------------------------------


@dataclass(frozen=True)
class Cutoff:
    """A frozen threshold, plus the working that produced it.

    `armed` is the honest bit: a cutoff computed from too few scores is still
    written down (so the report and the log can show it) but must not be used to
    route. `n` and `window` say what it was measured over; `computed_at` says when
    to distrust it.
    """

    value: float
    rate: float = DEFAULT_RATE
    window: int = DEFAULT_WINDOW
    n: int = 0
    armed: bool = False
    computed_at: str = ""
    reason: str = ""


def arm(
    scores: Sequence[float],
    rate: float = DEFAULT_RATE,
    *,
    window: int = DEFAULT_WINDOW,
    min_n: int = DEFAULT_MIN_N,
    now: datetime | None = None,
) -> Cutoff:
    """Compute a cutoff from the trailing `window` scores, or report why not.

    The floor is what makes this safe to run daily: a thin log yields an unarmed
    cutoff rather than a confident number off three requests.
    """
    recent = list(scores)[-window:]
    stamp = (now or datetime.now(UTC)).isoformat()
    if len(recent) < min_n:
        return Cutoff(
            value=D2_CUT_PLACEHOLDER,
            rate=rate,
            window=window,
            n=len(recent),
            armed=False,
            computed_at=stamp,
            reason=f"only {len(recent)} D2 scores; need {min_n} to arm",
        )
    return Cutoff(
        value=escalation_cutoff(recent, rate),
        rate=rate,
        window=window,
        n=len(recent),
        armed=True,
        computed_at=stamp,
        reason=f"top {rate:.0%} of the trailing {len(recent)} D2 scores",
    )


def d2_scores(records: Iterable[Mapping[str, object]]) -> list[float]:
    """The D2 probabilities in the log, in order, as a list of floats.

    D2 is a Noul, so its `answer` *is* its probability. Failed answers are not
    values — they are "no opinion", and including them would let a Jev outage
    move the cutoff.
    """
    scores: list[float] = []
    for record in records:
        if record.get("question_id") != D2 or record.get("failed"):
            continue
        value = record.get("answer")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append(float(value))
    return scores


# --- read / write the frozen file --------------------------------------------


def write(cutoff: Cutoff, path: Path | None = None) -> Path:
    """Persist a cutoff. The file is small and human-readable on purpose."""
    target = path or cutoff_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(cutoff), indent=2) + "\n", encoding="utf-8")
    return target


def read(path: Path | None = None) -> Cutoff | None:
    """Load the frozen cutoff, or None if there is not a usable one yet.

    A missing or corrupt file is not an error: it means "no rate has been
    measured", and the caller falls back to the provisional cut.
    """
    target = path or cutoff_path()
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return Cutoff(**payload)
    except (json.JSONDecodeError, TypeError):
        return None


def refresh(
    records: Iterable[Mapping[str, object]] | None = None,
    *,
    rate: float = DEFAULT_RATE,
    window: int = DEFAULT_WINDOW,
    min_n: int = DEFAULT_MIN_N,
    path: Path | None = None,
    now: datetime | None = None,
) -> Cutoff:
    """Re-freeze the cutoff from the log. The daily cron's one call.

    Reads the D2 scores, arms a cutoff over the trailing window, writes it, and
    returns it. Safe to run before the log exists — it arms nothing and says so.
    """
    if records is None:
        from agent.gatelog import gate_log_path

        records = _read_jsonl(gate_log_path())
    frozen = arm(d2_scores(records), rate, window=window, min_n=min_n, now=now)
    write(frozen, path)
    return frozen


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# --- the switch ---------------------------------------------------------------


def resolve_policy(
    base: GatePolicy = DEFAULT_POLICY,
    *,
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> GatePolicy:
    """The policy in force at start-up: enforcement if asked, refined by the cutoff.

    One env var flips the switch (`GATE_ENFORCE`), and the frozen cutoff refines
    the number it flips to. Two independent things on purpose:

    * **no cutoff yet** — enforce at the provisional probability cut, the declared
      0.91 that the log already stamps as `threshold_at_time`. This is how the
      switch goes on *today*, before there is a population to measure.
    * **an armed cutoff** — enforce at the measured rate boundary instead. The
      probability cut stays in the log as the refinement it was.

    Either way the enforced set starts as the ask-path only (`START_ACTIONS`), so
    a computed `reject` is still observed, never routed.
    """
    env = environ if environ is not None else os.environ
    if not _truthy(env.get(ENFORCE_ENV, "")):
        return base
    frozen = read(path)
    if frozen is not None and frozen.armed:
        return replace(base, enforce=True, enforce_actions=START_ACTIONS, d2_cut=frozen.value)
    return replace(base, enforce=True, enforce_actions=START_ACTIONS)


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agent.cutoff [--rate N] [--window W] [--min-n K]` — re-freeze and print."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-freeze the rolling D2 cutoff from the gate log.")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="fraction to escalate (default 0.10)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="trailing scores to use")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="scores needed to arm")
    args = parser.parse_args(argv)

    frozen = refresh(rate=args.rate, window=args.window, min_n=args.min_n)
    state = "armed" if frozen.armed else "NOT armed"
    print(f"cutoff {frozen.value:.4f} ({state}, n={frozen.n}, rate={frozen.rate:.0%}) — {frozen.reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
