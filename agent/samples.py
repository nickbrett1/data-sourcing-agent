"""The grading sample — request text, and the ticket the agent produced, keyed by id.

The shadow log (`gatelog`) records what Jev *answered* and what the door *did*. It
does not record what the agent *produced*. Without the artifact, a stopped request
has nothing to grade: "Jev said stop" is not, on its own, right or wrong. Pair the
request text with the rendered ticket and it is — a human can read, side by side,
whether the reading Jev called insufficient actually produced a sensible request.

Two streams, joined on `request_id`, because they are two different things:

* `gate-log.jsonl` — the **measurement**: scores, cuts, actions. High cardinality,
  never sampled.
* `gate-samples.jsonl` — the **evidence**: the text and the artifact a human scores.
  This is the one that makes the stop region gradeable at all.

Storing the ticket spends nothing: a ticket is a draft, and the agent cannot buy.
The download needs a human; this is the artifact that human was always going to
look at, kept so the look becomes a grade.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

SAMPLE_ENV = "GATE_SAMPLE_PATH"
DEFAULT_DIR_ENV = "AGENT_STATE_DIR"


def sample_path() -> Path:
    """Where the grading samples live: explicit env, else the agent state dir."""
    explicit = os.environ.get(SAMPLE_ENV)
    if explicit:
        return Path(explicit)
    return Path(os.environ.get(DEFAULT_DIR_ENV, ".")) / "gate-samples.jsonl"


@dataclass(frozen=True)
class Sample:
    """One request's evidence: what came in, what the agent made of it.

    The Jev answers are *not* duplicated here — they are in the gate log under the
    same `request_id`. This stream carries only what that stream cannot: the prose
    and the artifact.
    """

    request_id: str
    text: str
    ticket: str
    ts: str = ""


def append_sample(
    request_id: str,
    text: str,
    ticket: str,
    *,
    path: Path | None = None,
    now: datetime | None = None,
) -> Sample:
    """Record one request's text and rendered ticket. Best-effort, append-only.

    Callable from anywhere the artifact exists — including a turn the door would
    have stopped, once enforcement makes one: the sample must not disappear just
    because the routing did.
    """
    sample = Sample(
        request_id=request_id,
        text=text,
        ticket=ticket,
        ts=(now or datetime.now(UTC)).isoformat(),
    )
    target = path or sample_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample)) + "\n")
    return sample


def load_samples(path: Path | None = None) -> dict[str, Sample]:
    """Read the sample stream, later entries winning, a truncated last line skipped.

    Same failure shape as the log: an append-only writer can be cut off mid-line,
    and a grader over 500 good samples should not die on the 501st.
    """
    target = path or sample_path()
    if not target.exists():
        return {}
    samples: dict[str, Sample] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sample = Sample(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            samples[sample.request_id] = sample
    return samples


def join(
    samples: Mapping[str, Sample],
    records: Iterable[Mapping[str, object]],
) -> list[tuple[Sample, list[dict]]]:
    """Pair each sample with its gate-log rows, for a human to grade.

    Returns only samples that have rows, so a half-written turn (text logged, gate
    never reached) does not appear as a gradeable request with no verdict.
    """
    by_request: dict[str, list[dict]] = {}
    for record in records:
        rid = record.get("request_id")
        if isinstance(rid, str):
            by_request.setdefault(rid, []).append(dict(record))
    return [(sample, by_request[rid]) for rid, sample in samples.items() if rid in by_request]


def _verdict(records: list[dict]) -> str:
    """The door's verdict for one request, from its first gate-log row."""
    if not records:
        return "?"
    computed = records[0].get("action_computed", "proceed")
    if records[0].get("holdback"):
        return f"{computed} (holdback: admitted)"
    return str(computed)


def render_one(sample: Sample, records: list[dict]) -> str:
    """One request, as a human grades it: text, Jev's scores, the ticket, the verdict.

    Ordered so the comparison is a single vertical read — what came in, what Jev
    thought, what the agent produced, what the door did — rather than four columns
    the eye has to chase sideways.
    """
    lines = [f"── {sample.request_id} " + "─" * 40, "", "REQUEST", "", sample.text.strip(), ""]
    lines += ["JEV", ""]
    for record in records:
        if record.get("failed"):
            lines.append(f"  {record.get('question_id'):<28} FAILED (no opinion)")
        else:
            lines.append(f"  {record.get('question_id'):<28} {record.get('answer')}")
    lines += ["", f"DOOR SAID: {_verdict(records)}", "", "TICKET", "", sample.ticket.strip(), ""]
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    """`python -m agent.samples` — the grading queue: text, Jev, ticket, verdict."""
    from agent.gate_report import load_records

    records = load_records()
    pairs = join(load_samples(), records)
    if not pairs:
        print("no gradeable samples yet (need both a gate-log row and a sample).")
        return 1
    pairs.sort(key=lambda pair: pair[0].ts)
    for sample, rows in pairs:
        print(render_one(sample, rows))
    print(f"— {len(pairs)} request(s). Grade each, then record it with agent.gatelabels.")
    return 0
