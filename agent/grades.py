"""The human grade — a verdict on the door's call, one per request.

The shadow log measures what Jev said. The sample keeps what the agent produced.
Neither says whether the door *should* have done what it did. Only a person can,
reading `text | Jev | ticket`, and that reading is the label the threshold moves on.

The grade is on the **door's verdict**, not on the ticket and not on a single
question, because the threshold is a property of the door. Three real answers, one
of which is honest about not knowing:

* `right`        — the door did the correct thing.
* `too_strict`   — it stopped (asked/rejected) a request that was actually fine.
                   Jev over-cautious. → **lower the cut**: ask fewer.
* `too_lenient`  — it let through a request that was bad. Jev under-cautious.
                   → **raise the cut**: ask more.
* `unclear`      — the artifact does not settle it. Not a grade; it is why we say so.

`too_strict` is the false reject and `too_lenient` the false accept — the same two
errors the region table reports, here read off the artifact instead of the outcome.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

GRADE_ENV = "GATE_GRADE_PATH"
DEFAULT_DIR_ENV = "AGENT_STATE_DIR"


class Verdict(StrEnum):
    """What a person decided about the door's call."""

    right = "right"
    too_strict = "too_strict"
    too_lenient = "too_lenient"
    unclear = "unclear"


# Which way each verdict pushes the cut. The point of grading: `too_strict` means
# Jev stopped things it should not have, so the cut comes down; `too_lenient` means
# it let things through, so the cut goes up. `right` and `unclear` move nothing.
_DIRECTION = {
    Verdict.too_strict: "lower the cut (ask fewer)",
    Verdict.too_lenient: "raise the cut (ask more)",
}


def grade_path() -> Path:
    """Where grades live: explicit env, else the agent state dir."""
    explicit = os.environ.get(GRADE_ENV)
    if explicit:
        return Path(explicit)
    return Path(os.environ.get(DEFAULT_DIR_ENV, ".")) / "gate-grades.jsonl"


@dataclass(frozen=True)
class Grade:
    """A person's verdict on one request, with an optional note for the audit trail."""

    request_id: str
    verdict: Verdict
    note: str = ""
    ts: str = ""

    @property
    def direction(self) -> str:
        """Which way this verdict pushes the threshold — empty for `right`/`unclear`."""
        return _DIRECTION.get(self.verdict, "")


def append_grade(
    request_id: str,
    verdict: Verdict | str,
    *,
    note: str = "",
    path: Path | None = None,
    now: datetime | None = None,
) -> Grade:
    """Record one verdict. Append-only, like every other stream here — history is kept.

    Accepts the raw string so the HTTP handler does not have to coerce, but rejects
    anything outside the vocabulary rather than storing a value no reader expects.
    """
    try:
        resolved = Verdict(verdict)
    except ValueError as exc:
        raise ValueError(f"unknown verdict {verdict!r}; expected one of {[v.value for v in Verdict]}") from exc
    grade = Grade(
        request_id=request_id,
        verdict=resolved,
        note=note,
        ts=(now or datetime.now(UTC)).isoformat(),
    )
    target = path or grade_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(grade)) + "\n")
    return grade


def load_grades(path: Path | None = None) -> dict[str, Grade]:
    """Read the grade stream, latest verdict per request winning.

    Same failure shape as the other streams: skip a truncated line rather than lose
    the file, and let a re-grade supersede the first without deleting it.
    """
    target = path or grade_path()
    if not target.exists():
        return {}
    grades: dict[str, Grade] = {}
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                grade = Grade(**json.loads(line))
                grade = Grade(grade.request_id, Verdict(grade.verdict), grade.note, grade.ts)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            grades[grade.request_id] = grade
    return grades


def tally(grades: Mapping[str, Grade] | Iterable[Grade]) -> Counter:
    """How many of each verdict — the input to the move-the-cut decision."""
    values = grades.values() if isinstance(grades, Mapping) else grades
    return Counter(grade.verdict for grade in values)


def advice(grades: Mapping[str, Grade] | Iterable[Grade]) -> str:
    """One line: which way the grades say to move the cut, if they say anything.

    Deliberately does not pick a magnitude. The counts tell you the direction and
    whether the door is biased; the size of the move is a judgement the counts
    inform but do not make.
    """
    counts = tally(grades)
    strict, lenient = counts[Verdict.too_strict], counts[Verdict.too_lenient]
    if not strict and not lenient:
        return "no verdict points anywhere yet."
    if strict > lenient:
        return f"too strict on {strict} vs {lenient} — {_DIRECTION[Verdict.too_strict]}."
    if lenient > strict:
        return f"too lenient on {lenient} vs {strict} — {_DIRECTION[Verdict.too_lenient]}."
    return f"evenly wrong ({strict} each way) — the cut is not the lever; the questions are."
