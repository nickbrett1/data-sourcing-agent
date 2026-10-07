"""The gate's delayed label — `downstream_outcome`, filled in after the fact.

The shadow log records what the gate *decided*; it cannot record what *became* of the
request, because at decision time that has not happened yet. The gate memo §3 calls
this field the delayed label, and it is the whole reason the shadow corpus is worth
keeping: without it the log is a distribution of opinions with no ground truth, and
the probability cuts stay provisional forever (gate memo §5).

So labels are written **later, by something else, into a separate stream**. Two
reasons for a separate file rather than rewriting the log in place:

* the measurement stream stays immutable — it is audit evidence, and evidence you
  edit is not evidence;
* the log is append-only by contract (a crash loses at most the last line), and an
  in-place update pass would give that up.

## What can be labelled, and by whom

Not every outcome is known to the same observer, so `source` records who said so:

* **the agent, at the end of a turn** — `drafted` (a ticket was produced) or
  `run_failed` (the turn produced nothing). It knows this with certainty and needs no
  human. It does **not** know whether the ticket was any *good*.
* **outside, later** — `approved` / `rejected` (a human promoted or killed the
  ticket) or `reran` (the user came back with a changed request). These are the ones
  that carry signal, and none of them is visible from inside the turn.

`label_value` is the bridge to calibration: `approved` is 1, `rejected`/`reran` are
0, and everything else is **`None` — not a 0**. A drafted-but-unreviewed ticket is
not evidence against the gate; treating it as one is how a calibration set silently
fills with noise.

## The D2 caveat is structural (§6.5.1)

These labels are close to **D5** ("was this priced ticket worth buying") and *not* to
**D2**. A guessed axis is legal, gets priced, and can be approved — nothing in an
approve/reject distinguishes it. So this filler makes D5 calibratable and leaves D2
where it was: needing a re-run signal or a seed eval set, not a better log.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

LABEL_ENV = "GATE_LABEL_PATH"
DEFAULT_DIR_ENV = "AGENT_STATE_DIR"

#: Outcomes mapped to a calibration value. Anything not here is "unknown", which is
#: deliberately distinct from "bad" — see the module docstring.
_LABEL_VALUES = {"approved": 1.0, "rejected": 0.0, "reran": 0.0}


class Outcome(StrEnum):
    """What became of a request. A closed vocabulary, so a typo cannot enter the log."""

    drafted = "drafted"
    run_failed = "run_failed"
    door_stopped = "door_stopped"
    approved = "approved"
    rejected = "rejected"
    reran = "reran"

    @property
    def label_value(self) -> float | None:
        """1.0 good, 0.0 bad, or None when this outcome is not yet a label."""
        return _LABEL_VALUES.get(self.value)


def gate_label_path() -> Path:
    """Where labels live: `$GATE_LABEL_PATH`, else `$AGENT_STATE_DIR/gate-labels.jsonl`."""
    explicit = os.environ.get(LABEL_ENV)
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get(DEFAULT_DIR_ENV)
    base = Path(state_dir) if state_dir else Path.cwd()
    return base / "gate-labels.jsonl"


@dataclass(frozen=True)
class Label:
    """One label event. The latest event for a request wins; history is kept."""

    request_id: str
    outcome: str
    source: str = "agent"
    note: str | None = None
    ts: str = ""


def append_label(
    request_id: str,
    outcome: Outcome | str,
    *,
    source: str = "agent",
    note: str | None = None,
    path: Path | None = None,
) -> Label:
    """Append one label event. Best-effort at the call site — a label must not break a turn."""
    value = outcome.value if isinstance(outcome, Outcome) else str(outcome)
    if value not in Outcome._value2member_map_:
        raise ValueError(f"unknown outcome {value!r}; expected one of {list(Outcome)}")
    label = Label(
        request_id=request_id,
        outcome=value,
        source=source,
        note=note,
        ts=datetime.now(UTC).isoformat(),
    )
    target = path or gate_label_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(label), separators=(",", ":")) + "\n")
    return label


def load_labels(path: Path | None = None) -> dict[str, Label]:
    """The latest label per request id. Later events supersede earlier ones.

    A request can legitimately be labelled twice — `drafted` at the end of the turn,
    `approved` when the human promotes it — and the later, more informative label
    should win. Earlier events stay on disk as history.
    """
    target = path or gate_label_path()
    if not target.exists():
        return {}
    labels: dict[str, Label] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # a truncated last line from a crash
            rid = payload.get("request_id")
            if rid:
                labels[rid] = Label(**{k: payload.get(k) for k in Label.__dataclass_fields__})
    return labels


def apply_labels(records: Iterable[dict], labels: dict[str, Label]) -> int:
    """Stamp `downstream_outcome` onto matching records. Returns how many were labelled.

    Also writes `outcome_source`, so a reader can tell the agent's own `drafted`
    (no signal) from a human's `approved` (signal) without knowing the vocabulary.
    """
    updated = 0
    for record in records:
        label = labels.get(record.get("request_id"))
        if label is None:
            continue
        record["downstream_outcome"] = label.outcome
        record["outcome_source"] = label.source
        updated += 1
    return updated
