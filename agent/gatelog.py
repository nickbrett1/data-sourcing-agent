"""The gate's shadow log — the observe-only traffic, retained (gate memo §3).

While the gate enforces nothing, the *only* thing it produces of value is the log:
one record per question per request, in the shape calibration will later need.
Without this, observing produces nothing to calibrate against — the whole point of
the shadow phase (gate memo §5) is that it accumulates a corpus.

A record carries the gate memo §3 fields: `request_id`, `question_id`, `primitive`,
`state_hash`, `answer`, `probability`, `confidence`, `threshold_at_time`,
`action_taken`, `downstream_outcome`. `downstream_outcome` is written `None` now
and filled by a later pass — it is the delayed label, and a record without it is
still a record (we can find it again by `request_id`).

Storage is **JSONL appended to one file** under `AGENT_STATE_DIR`: append-only,
one line per record, so a crash mid-write loses at most the last line and no
reader needs a schema migration. Not a database, deliberately — the log is a
stream to be replayed, not a thing to be queried in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

from agent.gate import D1, D2, D3, D5, GateDecision
from agent.jev import Answer, ChoiceAnswer, NoulAnswer

LOG_ENV = "GATE_LOG_PATH"
DEFAULT_DIR_ENV = "AGENT_STATE_DIR"


def gate_log_path() -> Path:
    """Where the log lives: `$GATE_LOG_PATH`, else `$AGENT_STATE_DIR/gate-log.jsonl`."""
    explicit = os.environ.get(LOG_ENV)
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get(DEFAULT_DIR_ENV)
    base = Path(state_dir) if state_dir else Path.cwd()
    return base / "gate-log.jsonl"


def _state_hash(state_text: str) -> str:
    """A stable fingerprint of the state, so identical requests group together."""
    return hashlib.sha256(state_text.encode("utf-8")).hexdigest()[:16]


def _primitive(answer: Answer) -> str:
    if isinstance(answer, NoulAnswer):
        return "noul"
    if isinstance(answer, ChoiceAnswer):
        return "choice"
    return "score"


def _answer_fields(answer: Answer) -> tuple[object, float | None, float | None]:
    """(answer, probability, confidence) for the record, per answer kind."""
    if isinstance(answer, NoulAnswer):
        return answer.noul, answer.noul, None
    if isinstance(answer, ChoiceAnswer):
        p = answer.probabilities.get(answer.choice or "", None)
        return answer.choice, p, answer.confidence
    return answer.score, None, answer.confidence


def build_records(
    state_text: str,
    answers: dict[str, Answer],
    decision: GateDecision,
    *,
    thresholds: dict[str, float] | None = None,
    request_id: str | None = None,
) -> list[dict]:
    """One record per question, plus the composed action on each.

    Every question carries `action_taken`, so the log can be read either per
    question (calibration) or per request (what the door did) without a join.
    """
    rid = request_id or uuid.uuid4().hex
    thresholds = thresholds or {}
    digest = _state_hash(state_text)
    records: list[dict] = []
    for qid, answer in answers.items():
        value, probability, confidence = _answer_fields(answer)
        records.append(
            {
                "request_id": rid,
                "question_id": qid,
                "primitive": _primitive(answer),
                "state_hash": digest,
                "answer": value,
                "probability": probability,
                "confidence": confidence,
                "failed": answer.failed,
                "threshold_at_time": thresholds.get(qid),
                "action_taken": decision.action.value,
                "downstream_outcome": None,
            }
        )
    return records


def append_records(records: Iterable[dict], path: Path | None = None) -> int:
    """Append records as JSONL. Returns how many were written.

    Best-effort by contract: the caller wraps this, because a log that cannot be
    written must never fail the turn it is observing (the gate is fail-open).
    """
    rows: Sequence[dict] = list(records)
    if not rows:
        return 0
    target = path or gate_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


def default_thresholds(d1: float, d2: float, d5: float) -> dict[str, float]:
    """The cuts in force, recorded so a later reader knows what produced a call."""
    return {D1: d1, D2: d2, D5: d5, D3: float("nan")}  # D3 is graded, not thresholded
