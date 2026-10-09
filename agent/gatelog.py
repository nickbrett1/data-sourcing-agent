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

Three fields were added for enforcement (§6.6): `action_computed` (the verdict,
before policy), `holdback` (was this a would-be stop the holdback admitted?), and
`state` (the raw state, retained only for would-be stops, so the rejected region
can be audited once the door starts stopping things). `action_taken` is the
*effective* action — `proceed` in observe-only — so it never overstates what the
door did.

Storage is **JSONL appended to one file** under `AGENT_STATE_DIR`: append-only,
one line per record, so a crash mid-write loses at most the last line and no
reader needs a schema migration. Not a database, deliberately — the log is a
stream to be replayed, not a thing to be queried in place.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from agent.gate import D1, D2, D3, D3_CLOSE, D5, GateDecision
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


def new_request_id() -> str:
    """A fresh request id, so the checkpoints of one turn share one `request_id`.

    The gate now runs at two checkpoints (front door, then post-price for D5), and
    their records must join. Minting the id once per turn and handing it to both is
    what makes that a join rather than a guess (gate memo §3).
    """
    return uuid.uuid4().hex


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
    action_taken: str | None = None,
    holdback: bool = False,
    retain_state: bool = False,
) -> list[dict]:
    """One record per question, plus the composed action on each.

    Every question carries `action_taken`, so the log can be read either per
    question (calibration) or per request (what the door did) without a join.

    `action_taken` is the *effective* action (what the door did); it defaults to the
    computed one, which is what observe-only means. When enforcement is on, the two
    differ for a holdback admission — `action_computed` keeps the verdict so the
    calibrated score distribution is not lost to a "proceed" that was really a stop.

    `retain_state` writes the raw `state` under `state` (alongside the always-present
    `state_hash`) for anything the door would stop, per §6.6: a hash cannot be
    audited, and the rejected region is the one that disappears when enforcement
    begins.
    """
    rid = request_id or uuid.uuid4().hex
    thresholds = thresholds or {}
    digest = _state_hash(state_text)
    taken = action_taken or decision.action.value
    ts = datetime.now(UTC).isoformat()
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
                "action_taken": taken,
                "action_computed": decision.action.value,
                "holdback": holdback,
                "downstream_outcome": None,
                "ts": ts,
            }
        )
    if retain_state:
        # The raw state rides on the first record only: it is per-request, not
        # per-question, and repeating it per row would balloon the log.
        for record in records:
            record["state"] = state_text if record is records[0] else None
    return records


def _finite(value: object) -> object:
    """Map non-finite floats to `None`: JSON has no NaN/Infinity, and a strict
    reader (JS `JSON.parse`, a parquet importer) rejects a bare `NaN` token."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_row(row: dict) -> dict:
    """A record with every non-finite float (top-level or nested) replaced by None.

    `append_records` guards *new* writes, but a log written before that guard can
    carry a bare `NaN`/`Infinity` token — which Python's `json.loads` accepts and
    a strict reader (Starlette's `JSONResponse`, JS `JSON.parse`, parquet) rejects.
    Readers normalise through here so a legacy row cannot 500 a whole endpoint.
    """

    def clean(value: object) -> object:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return {k: clean(v) for k, v in row.items()}


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
            clean = {k: _finite(v) for k, v in row.items()}
            # `allow_nan=False` is a guard, not the fix: `_finite` has already
            # removed every non-finite float, so this only fires if a new one is
            # introduced — a loud failure beats silently writing invalid JSON.
            handle.write(json.dumps(clean, separators=(",", ":"), allow_nan=False) + "\n")
    return len(rows)


def default_thresholds(
    d1: float, d2: float, d5: float, d3: int = D3_CLOSE
) -> dict[str, float]:
    """The cuts in force, recorded so a later reader knows what produced a call.

    D3 *is* thresholded — `decide` requires its chosen level at least `close` — so
    the log records that cut (`d3_min`, the `close` index) rather than a sentinel.
    A level score is compared against this index, so the threshold is the index.
    """
    return {D1: d1, D2: d2, D3: float(d3), D5: d5}
