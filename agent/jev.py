"""Jev — the TypeSafe System One decision client (memo `jev-integration-v1`).

**This is the only module that talks to Jev.** It owns three things and nothing
else:

* the **typed question vocabulary** — `Noul`, `Choice`, `Score`, mirroring the
  API's three question kinds;
* the **typed answer vocabulary** — one answer model per question kind, so a
  caller reads a number, not a dict it has to trust;
* the **HTTP call** — one POST carrying *all* questions, because the API
  evaluates them in parallel in a single round trip. Asking five questions in
  five calls would throw that away.

The credential never appears here. Jev is reached through the LiteLLM proxy's
native `/typesafe/{path}` route, which injects the upstream key itself; this
client authenticates with the same `LITELLM_API_KEY` it uses for model calls
(memo `jev-litellm-decision-v1`). The specialist therefore holds no
`TYPESAFE_API_KEY`.

Fail-open is a property of the *type*, not a policy bolted on: every call
returns an answer for every question asked, and a failure is an answer with
`failed=True` rather than an exception. A caller that forgets to check `failed`
gets a neutral value, not a crash — and the gate treats "no opinion" as "do not
escalate" (memo `jev-integration-v1` §4).

Boundary rule (memo §2.4): Jev is a model. It may live here and be *called
from* `agent/gate.py`. It is **never** imported by `agent/validator.py` — that
module's whole reason for existing is to hold no model at all. A test enforces
this by reading the import graph, so the rule is checked rather than asserted.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

# The proxy route. The `/v1/` matters: LiteLLM's `/typesafe/{endpoint:path}`
# route strips its prefix and forwards the REST verbatim, so `/typesafe/v1/
# systemone` reaches `api.typesafe.ai/v1/systemone`. Dropping the `/v1/` would
# send the upstream a bare `/systemone` and 404 (memo `jev-litellm-wiring-v1`).
SYSTEMONE_PATH = "/typesafe/v1/systemone"

DEFAULT_BASE_URL = "http://litellm:4000"
DEFAULT_MODEL = "jev-latest"
# Jev answers in ~50 ms; 2 s is already 40x headroom. A gate must not hang the
# turn it is advising, so the timeout is deliberately short (memo §4).
DEFAULT_TIMEOUT = 2.0

BASE_URL_ENV = "LITELLM_BASE_URL"
API_KEY_ENV = "LITELLM_API_KEY"


# --- the question vocabulary -------------------------------------------------


class Noul(BaseModel):
    """A yes/no proposition; the answer is P(statement is true)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: str = Field(
        description="The statement to judge, e.g. `The request is specific enough to price`."
    )


class Choice(BaseModel):
    """One value out of a set the caller supplies. The answer cannot leave it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: str = Field(description="What is being chosen, e.g. `Which action to take`.")
    criteria: dict[str, str] = Field(
        description="Option key -> what that option means. Jev may only return one of these keys."
    )


class Score(BaseModel):
    """A rating against ordered levels the caller supplies."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: str = Field(description="What is being rated.")
    criteria: list[str] = Field(
        description="Ordered levels, lowest first; the index is the score."
    )


Question = Noul | Choice | Score


# --- the answer vocabulary ---------------------------------------------------


class NoulAnswer(BaseModel):
    """P(statement). `failed` marks a call that produced no opinion."""

    noul: float | None = None
    failed: bool = False


class ChoiceAnswer(BaseModel):
    """The chosen key and the distribution over the keys that were supplied."""

    choice: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    failed: bool = False


class ScoreAnswer(BaseModel):
    """The chosen level (its index), the legend it was chosen against, the spread."""

    score: float | None = None
    legend: dict[int, str] = Field(default_factory=dict)
    probabilities: dict[int, float] = Field(default_factory=dict)
    confidence: float | None = None
    failed: bool = False


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def _failed(question: Question) -> Answer:
    """The fail-open answer for a question: the right shape, no opinion."""
    if isinstance(question, Noul):
        return NoulAnswer(failed=True)
    if isinstance(question, Choice):
        return ChoiceAnswer(failed=True)
    return ScoreAnswer(failed=True)


class JevClient:
    """One client, one call, fail-open.

    Construct with `from_env()` in the agent; construct directly with a stub
    transport in tests. The injectable client is the same seam `DatabentoValidator`
    uses, for the same reason: the truth the tests pin is the parsing and the
    fail-open behaviour, not a remote server's availability.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._api_key = api_key
        self._client = client

    @classmethod
    def from_env(cls, **kwargs: object) -> JevClient:
        """Build from the environment, the only way the agent constructs one."""
        return cls(
            base_url=os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL),
            api_key=os.environ.get(API_KEY_ENV, ""),
            **kwargs,  # type: ignore[arg-type]
        )

    def _http(self) -> httpx.AsyncClient:
        """The real client, built lazily so tests can inject their own."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def ask(
        self,
        state: str,
        questions: dict[str, Question],
    ) -> dict[str, Answer]:
        """Ask every question in one call; never raise.

        On any failure every question comes back `failed=True`. The caller must
        check `failed` rather than assume a number is present (memo §4).
        """
        if not questions:
            return {}
        payload = {
            "state": state,
            "model": self._model,
            "questions": {
                qid: q.model_dump(exclude_none=True) for qid, q in questions.items()
            },
        }
        try:
            response = await self._http().post(SYSTEMONE_PATH, json=payload)
        except httpx.HTTPError as exc:
            return self._all_failed(questions, f"transport: {exc!r}")
        if response.status_code >= 400:
            return self._all_failed(
                questions, f"HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError:
            return self._all_failed(questions, "response body was not JSON")
        answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(answers, dict):
            return self._all_failed(questions, "response had no `answers` object")
        return {qid: _answer_from(q, answers.get(qid)) for qid, q in questions.items()}

    def _all_failed(
        self, questions: dict[str, Question], reason: str
    ) -> dict[str, Answer]:
        """Fail open: log once, answer every question with no opinion."""
        log.warning("jev call failed open (%d questions): %s", len(questions), reason)
        return {qid: _failed(q) for qid, q in questions.items()}


def _answer_from(question: Question, raw: object) -> Answer:
    """Turn one raw answer into its typed form, or fail that answer open.

    A missing or malformed answer for a *single* question fails only that
    question — the others in the same call are still usable.
    """
    if not isinstance(raw, dict):
        return _failed(question)
    try:
        if isinstance(question, Noul):
            if raw.get("noul") is None:
                return _failed(question)
            return NoulAnswer(noul=float(raw["noul"]))
        if isinstance(question, Choice):
            if raw.get("choice") is None:
                return _failed(question)
            return ChoiceAnswer(
                choice=str(raw["choice"]),
                probabilities=_float_map(raw.get("probabilities")),
                confidence=_opt_float(raw.get("confidence")),
            )
        if raw.get("score") is None:
            return _failed(question)
        return ScoreAnswer(
            score=float(raw["score"]),
            legend={int(k): str(v) for k, v in (raw.get("legend") or {}).items()},
            probabilities={int(k): float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=_opt_float(raw.get("confidence")),
        )
    except (TypeError, ValueError):
        return _failed(question)


def _float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def _opt_float(raw: object) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
