"""The front-door gate — Jev advises, deterministic code decides (memo §6.5).

The gate asks Jev **four** questions about one request and composes their answers
into **one action**, in code:

* **D1** — in-remit / legitimacy (Noul)
* **D2** — specification sufficiency (Noul) — the load-bearing one
* **D3** — dataset/schema fit (Score)
* **D5** — cost proportionality (Noul, fires only when spend is material)

D1/D2/D3 are answerable at the front door, before anything is priced. **D5 is
not** — "is this spend proportionate?" needs a number — so it is asked at a
*second* checkpoint, `ask_cost_gate`, once the Validator has produced its
deterministic estimate. One turn, two checkpoints, one `request_id`.

**D4 is not asked.** The gate memo listed "what should the front desk do?" as a
fifth question, but its own criteria are a pure function of D1-D3, so asking Jev
for it would put a classifier in the loop doing arithmetic on its own upstream
probabilities — the exact failure §6.5 exists to prevent. D4 is therefore this
module's *return value* (memo `jev-integration-v1` §3.1, decided 2026-10-06):

    action = reject          if D1 says no
           = ask_clarifying  if D3 is below `close`, or D2 is below its cut
           = proceed         otherwise

Because a failed answer is "no opinion" (`failed=True`), a Jev outage makes every
Noul failed, no threshold is breached, and the action falls out as `proceed` —
**fail-open by construction**, not by a special case.

## The threshold is a rate first, a probability later (memo §8 #3)

A usable day-one cut needs no shadow data. `escalation_cutoff()` turns "escalate
the top N%" into a number using only the scores in hand and a review capacity N —
assuming only that Jev's score is monotone in risk. The probability cut is a
*provisional placeholder* until a log of outcomes calibrates it; calibration makes
the threshold *interpretable*, it does not make it *usable*.

## Observe-only by default

`GatePolicy.enforce` is False. The gate computes and returns its action but a
caller that has not opted into enforcement ignores it — the whole point of the
shadow phase (gate memo §5). Enforcement also biases the calibration set (§6.6),
which is why switching it on is a deliberate act with a holdback slice, not a
config default.

**Enforcement is built, and one flag away.** `enforce()` is the single place that
turns a computed decision into what happens, and the machinery the switch needs is
already here: the **holdback slice** (`is_holdback`, a deterministic draw on the
request id) admits a fraction of would-be-stops so the rejected region stays
observable, and `should_retain_state` keeps the raw `state` for anything the door
would stop. Flipping `GatePolicy(enforce=True)` is the entire act of enabling it;
`holdback_rate` and its seed, and the retention policy, are already wired. See
§6.6 of the data-acquisition memo for *why* both must be in place at the moment of
enforcement rather than added after.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel
from pydantic_ai import RunContext, Tool
from pydantic_ai.toolsets import FunctionToolset

from agent.jev import (
    Answer,
    Choice,
    ChoiceAnswer,
    JevClient,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
)

# --- question ids ------------------------------------------------------------
# Stable strings: they are the keys of the log row and the keys of the answer
# map, so renaming one silently orphans history. Treat them as wire format.

D1 = "d1_in_remit"
D2 = "d2_specification_sufficient"
D3 = "d3_dataset_fit"
D5 = "d5_cost_proportional"

# D3 levels, ordered lowest-to-highest so the Score index is "how well it fits".
# Jev returns the index into this list; `D3_CLOSE` is the "close" level's index,
# and the composition requires at least it (gate memo §2, D3/D4 criteria).
D3_LEVELS = ["none", "plausible", "close", "exact"]
D3_EXACT = D3_LEVELS.index("exact")
D3_CLOSE = D3_LEVELS.index("close")

# The D2 probability cut is a PLACEHOLDER. The original 0.91 was a pure cost-ratio
# number (10:1 false-accept:false-reject -> 10/11, gate memo §2), chosen before any
# log existed. Measured against the front door, it sits at the TOP of the
# distribution a fully-specified request actually gets: known-good requests score
# ~0.82-0.92 on D2 while vague ones score ~0.02-0.05, so 0.91 false-rejected ~5 in
# 6 good requests -- a wall, not a filter. Reset to 0.5: roughly a 1:1 cost ratio
# (a false accept and a false reject are close in cost here -- the deterministic
# Validator catches illegal guesses and every ticket still needs a human sign-off),
# comfortably below the good cluster even allowing for D2's ~0.1 run-to-run spread,
# and far above every vague request seen. Still a placeholder: the measured rate
# boundary replaces it once the log arms (agent/cutoff.py).
D2_CUT_PLACEHOLDER = 0.5
D1_CUT_PLACEHOLDER = 0.5
D5_CUT_PLACEHOLDER = 0.5
# The holdback fraction (§6.6): ~1–5%. Exact value open (§8 #8); inert until enforced.
HOLDBACK_RATE_PLACEHOLDER = 0.02

# Grader-facing legend. The log stores question *codes*
# (`d2_specification_sufficient`); a human grading a row should not have to hold
# that mapping in their head, so the grader (UI and MCP) shows this prose beside
# each answer. One line each: what it measures, and which way the answer cuts.
QUESTION_LEGEND: dict[str, str] = {
    D1: "In remit: is this a legitimate Databento historical-data request at all? "
    "High = yes; a low answer is a computed reject.",
    D2: "Specification sufficient: could a priced ticket be produced without guessing "
    "dataset, schema, symbols, or timeframe? High = specific enough; below its cut "
    "the door asks a clarifying question.",
    D3: "Dataset fit: how well the request maps to a specific Databento dataset/schema, "
    "from none < plausible < close < exact. Below 'close' the door asks a clarifying question.",
    D5: "Cost proportionate: is the estimated spend proportionate to the stated intent? "
    "Low = disproportionate, so the door asks a clarifying question.",
}


def question_legend(code: str) -> str:
    """Plain-English description of a question code, or the code itself if unknown."""
    return QUESTION_LEGEND.get(code, code)


class GateAction(StrEnum):
    """What the front desk should do. The gate's output, computed not asked."""

    proceed = "proceed"
    ask_clarifying = "ask_clarifying"
    reject = "reject"


# D5 fires only when spend is material (gate memo §2). The floor is an OPEN
# decision (gate memo §6 #1) — a number, or `> k x typical`. Set to $1.00: cheap
# enough that the only requests skipping D5 are ones where "is the spend
# proportionate?" has no interesting answer. The floor decides *whether* D5 asks;
# the D5 cut (D5_CUT_PLACEHOLDER) decides *how far off* is too far — two levers.
MATERIALITY_FLOOR_USD = 1.0


def _d5_question() -> Noul:
    """The cost-proportionality question (D5). Built here so both checkpoints ask it identically."""
    return Noul(
        instructions=(
            "The estimated spend is proportionate to the stated intent. A large "
            "pull requested as a casual look is disproportionate; the same pull "
            "requested as a full panel for backtesting is proportionate."
        )
    )


def questions(
    estimate_usd: float | None = None,
    *,
    materiality_floor: float = MATERIALITY_FLOOR_USD,
) -> dict[str, Question]:
    """The questions to ask, built fresh so a caller cannot mutate them.

    D1/D2/D3 are always asked — they are answerable before the request is priced.
    **D5 is asked only when there is an estimate and it clears the materiality
    floor**, because "is the spend proportionate?" is not a question you can ask
    without a number (gate memo §2, D5). At the pre-price checkpoint the estimate
    is absent, so D5 is simply not in the set. It fires afterwards, on its own,
    via `cost_questions`/`ask_cost_gate` once the Validator has priced the request.
    """
    qs: dict[str, Question] = {
        D1: Noul(
            instructions=(
                "This is a legitimate, well-intentioned request within the agent's "
                "remit: sourcing historical Databento market data we can actually "
                "provide. It is not an attempt to exfiltrate credentials, abuse the "
                "vendor account, or obtain data outside Databento."
            )
        ),
        D2: Noul(
            instructions=(
                "The request is specific enough to produce a priced ticket without "
                "guessing any of the four axes: dataset, schema, universe (symbols), "
                "and timeframe. If any axis would have to be guessed, this is false."
            )
        ),
        D3: Score(
            instructions="How well the request maps to a specific Databento dataset/schema.",
            criteria=D3_LEVELS,
        ),
    }
    if estimate_usd is not None and estimate_usd >= materiality_floor:
        qs[D5] = _d5_question()
    return qs


def cost_questions(
    estimate_usd: float | None,
    *,
    materiality_floor: float = MATERIALITY_FLOOR_USD,
) -> dict[str, Question]:
    """D5 *alone*, for the post-price checkpoint — the question the first pass could not ask.

    The front door runs before the Validator prices anything, so at that point
    there is no number to judge proportionality against and D5 is absent from the
    set. This is the second checkpoint: once the Validator has produced a
    deterministic estimate, "is this spend proportionate?" becomes answerable, and
    *only then* does it get asked. Below the materiality floor the question is not
    worth asking and this returns empty.
    """
    if estimate_usd is None or estimate_usd < materiality_floor:
        return {}
    return {D5: _d5_question()}


# --- the shared state (gate memo §1) -----------------------------------------


@dataclass(frozen=True)
class GateState:
    """The facts assembled *before* the gate, from sources other than Jev.

    Jev judges intent and scope against these; it does not compute any of them.
    The cost estimate in particular is the Validator's deterministic number,
    *given* to the gate, never predicted by it (gate memo §1).
    """

    raw_request: str
    parsed_intent: str
    candidate_mappings: Sequence[str] = ()
    estimate_usd: float | None = None
    rows: int | None = None
    budget_context: str = ""
    prior_turns: Sequence[str] = ()


def serialise_state(state: GateState) -> str:
    """Render the shared state as the single string Jev's API takes.

    Jev takes one `state` string, so the assembly is explicit and stable: a
    labelled block, one field per line, so a drift in what we send is visible in
    the log rather than hidden behind a dict ordering.
    """
    lines = [
        "REQUEST (the user's own words):",
        state.raw_request.strip(),
        "",
        "PARSED INTENT (dataset/schema/universe/timeframe as understood):",
        state.parsed_intent.strip() or "(not parsed)",
    ]
    if state.candidate_mappings:
        lines += ["", "CANDIDATE MAPPINGS (from discovery):"]
        lines += [f"- {m}" for m in state.candidate_mappings]
    lines += ["", "DETERMINISTIC ESTIMATE (from the Validator, not Jev):"]
    if state.estimate_usd is None:
        lines.append("(no cost estimate available)")
    else:
        rows = "unknown" if state.rows is None else f"{state.rows:,}"
        lines.append(f"estimated_usd={state.estimate_usd} rows={rows}")
    if state.budget_context:
        lines += ["", "BUDGET CONTEXT:", state.budget_context.strip()]
    if state.prior_turns:
        lines += ["", "PRIOR TURNS (same conversation):"]
        lines += [f"- {t}" for t in state.prior_turns]
    return "\n".join(lines)


# --- the decision ------------------------------------------------------------


@dataclass(frozen=True)
class GatePolicy:
    """The cuts and whether they are enforced.

    `enforce=False` is the default and the only safe default: the gate computes
    its action but the caller must opt in to acting on it (gate memo §5).
    """

    enforce: bool = False
    # Which computed actions the door is allowed to route on. Default: all of them,
    # so a bare `enforce=True` is complete. The cautious start is narrower — the
    # ask-path only, because a wrong clarify costs one question while a wrong reject
    # bounces a real request with no human touch. A computed action outside this set
    # is admitted (proceed) and kept in `computed` for the log, exactly like a holdback.
    enforce_actions: frozenset[GateAction] = field(
        default_factory=lambda: frozenset(GateAction)
    )
    d1_cut: float = D1_CUT_PLACEHOLDER
    d2_cut: float = D2_CUT_PLACEHOLDER
    d5_cut: float = D5_CUT_PLACEHOLDER
    d3_min: int = D3_CLOSE
    # The holdback slice (§6.6): the fraction of would-be-stops admitted anyway, so
    # the rejected region stays observable once enforcement begins. Inert while
    # `enforce` is False; the fraction itself is open (§8 #8). `holdback_seed` lets
    # the slice be re-drawn for a new calibration period without changing request ids.
    holdback_rate: float = HOLDBACK_RATE_PLACEHOLDER
    holdback_seed: str = ""


# A module-level singleton so the default is not a call in a signature (B008),
# and so "the default policy" is one object rather than one per call site.
DEFAULT_POLICY = GatePolicy()


@dataclass(frozen=True)
class GateDecision:
    """The action plus the working that produced it — logged, per gate memo §3."""

    action: GateAction
    reasons: tuple[str, ...] = ()
    # True when Jev answered none of the questions in a way the gate could use.
    abstained: bool = False


def _noul(answer: Answer | None) -> NoulAnswer | None:
    return answer if isinstance(answer, NoulAnswer) and not answer.failed else None


def _score(answer: Answer | None) -> ScoreAnswer | None:
    return answer if isinstance(answer, ScoreAnswer) and not answer.failed else None


def decide(
    answers: dict[str, Answer],
    policy: GatePolicy = DEFAULT_POLICY,
) -> GateDecision:
    """Compose D1/D2/D3/D5 into one action. Deterministic; no model here.

    A failed (or absent) answer is "no opinion": it cannot breach a threshold, so
    an outage yields `proceed`. That is fail-open, and it is deliberate (memo §4).
    """
    d1 = _noul(answers.get(D1))
    d2 = _noul(answers.get(D2))
    d3 = _score(answers.get(D3))

    if d1 is None and d2 is None and d3 is None:
        return GateDecision(
            action=GateAction.proceed,
            reasons=("Jev gave no usable answer on any question; the gate abstains.",),
            abstained=True,
        )

    reasons: list[str] = []
    if d1 is not None and d1.noul is not None and d1.noul < policy.d1_cut:
        reasons.append(f"{D1}={d1.noul:.2f} < {policy.d1_cut:.2f}: not in remit.")
        return GateDecision(action=GateAction.reject, reasons=tuple(reasons))

    if d3 is not None and d3.score is not None and d3.score < policy.d3_min:
        # Jev returns a Score as the *expected level* over `D3_LEVELS` (a continuous
        # value in [0, 3]), not a discrete index — verify by the returned
        # `probabilities`, whose cell-wise expectation equals `score`. So report the
        # number, not `D3_LEVELS[int(score)]`, which truncates 1.65 to "plausible"
        # and contradicts the "below close" call the comparison just made.
        reasons.append(
            f"{D3}={d3.score:.2f} (expected level) < required close={D3_CLOSE}: "
            "fit too uncertain."
        )
    if d2 is not None and d2.noul is not None and d2.noul < policy.d2_cut:
        reasons.append(
            f"{D2}={d2.noul:.2f} < {policy.d2_cut:.2f} (provisional cut): "
            "specification not sufficient."
        )
    if reasons:
        return GateDecision(action=GateAction.ask_clarifying, reasons=tuple(reasons))

    reasons.append("D1 in remit, D2 at or above cut, D3 at least close.")
    return GateDecision(action=GateAction.proceed, reasons=tuple(reasons))


def decide_cost(
    answers: dict[str, Answer],
    policy: GatePolicy = DEFAULT_POLICY,
) -> GateDecision:
    """Compose **D5 alone** into an action, for the post-price checkpoint.

    A separate composition from `decide` on purpose: at this checkpoint D1/D2/D3
    are not in play — they were judged at the front door against the *unpriced*
    request — so running them through `decide` would make every cost-check look
    like an abstention (all three absent). Cost proportionality has exactly one
    question and one cut, and this composes that.

    Same fail-open shape as `decide`: a failed or absent D5 is "no opinion", it
    cannot breach the cut, and the action is `proceed` (memo §4). Disproportionate
    spend asks for clarification — narrow it or confirm — it does not reject.
    """
    d5 = _noul(answers.get(D5))
    if d5 is None or d5.noul is None:
        return GateDecision(
            action=GateAction.proceed,
            reasons=("D5 gave no usable answer on cost proportionality; the gate abstains.",),
            abstained=True,
        )
    if d5.noul < policy.d5_cut:
        return GateDecision(
            action=GateAction.ask_clarifying,
            reasons=(
                f"{D5}={d5.noul:.2f} < {policy.d5_cut:.2f}: spend disproportionate to intent.",
            ),
        )
    return GateDecision(
        action=GateAction.proceed,
        reasons=(f"{D5}={d5.noul:.2f} >= {policy.d5_cut:.2f}: spend proportionate.",),
    )


def effective_action(
    decision: GateDecision,
    policy: GatePolicy = DEFAULT_POLICY,
    *,
    request_id: str | None = None,
) -> GateAction:
    """What a caller should actually *do* — the enforcement decision's action.

    Kept as the narrow façade over `enforce()` so the common call site ("what do I
    do?") does not have to know about holdback. `request_id` is optional because a
    caller that only wants the observe-only answer does not need one; when
    enforcement is on and a request id is supplied, the holdback slice can admit a
    would-be-stop as `proceed`.
    """
    return enforce(decision, request_id, policy).action


# --- enforcement: the holdback slice (§6.6, §8 #8) --------------------------
#
# Enforcement biases the calibration set: the moment the door stops a request, that
# request stops producing a `downstream_outcome`, so recalibration then runs on the
# set that *survived* the door and a false reject — whose counterfactual was deleted
# at the door — becomes unmeasurable (data-acquisition memo §6.6, the "one defect
# enforcement introduces"). Two mitigations, both to be in place *at* the moment of
# enforcement rather than added after:
#
#   1. hold back a random slice that passes the door regardless of score, and
#   2. retain the raw `state` (not just its hash) for anything the door would stop.
#
# The holdback draw is a *function of the request id*, not a coin flip at call time,
# because the same turn may evaluate its action more than once (the post-price
# checkpoint re-reads it) and the two evaluations must agree. Determinism by
# construction, so replaying a request id gives the same admission decision.

def holdback_draw(request_id: str, *, seed: str = "") -> float:
    """A deterministic uniform draw in `[0, 1)` for a request id.

    SHA-256 of `seed:request_id`, first eight bytes as a fraction. Stable across
    processes and restarts — no RNG state to persist — which is what lets the
    holdback be reproduced for a later audit rather than trusted.
    """
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def is_holdback(request_id: str, rate: float, *, seed: str = "") -> bool:
    """Is this request in the holdback slice? A rate of 0 never admits; 1 always."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    return holdback_draw(request_id, seed=seed) < rate


@dataclass(frozen=True)
class EnforcementOutcome:
    """What the door did, and why — the record for the enforcement-bias analysis."""

    action: GateAction
    computed: GateAction
    holdback: bool = False
    reason: str = ""

    @property
    def enforce_differed(self) -> bool:
        """True when the door did something other than the computed action."""
        return self.action is not self.computed


def enforce(
    decision: GateDecision,
    request_id: str | None = None,
    policy: GatePolicy = DEFAULT_POLICY,
) -> EnforcementOutcome:
    """Turn a computed decision into what actually happens — the one place that knows.

    Three cases, in order:

    * **Observe-only** (`policy.enforce` False) — the action is `proceed` regardless;
      the computed verdict survives in `computed` for the log (gate memo §5).
    * **Holdback** — when enforcing and the computed action would stop the request,
      the holdback slice admits it as `proceed`. This is the sampled rejected region
      (§6.6): the only way a false reject is ever measured.
    * **Enforced** — otherwise the computed action stands.

    `request_id` being `None` while enforcing disables holdback for that call rather
    than guessing an id — an un-attributable draw is not a draw.
    """
    if not policy.enforce:
        return EnforcementOutcome(
            GateAction.proceed, decision.action, reason="observe-only; nothing routed on."
        )
    if decision.action is GateAction.proceed:
        return EnforcementOutcome(GateAction.proceed, decision.action, reason="computed action is proceed.")
    if decision.action not in policy.enforce_actions:
        return EnforcementOutcome(
            GateAction.proceed,
            decision.action,
            reason=f"{decision.action.value} is not in the enforced set; observed only.",
        )
    if request_id is not None and is_holdback(
        request_id, policy.holdback_rate, seed=policy.holdback_seed
    ):
        return EnforcementOutcome(
            GateAction.proceed,
            decision.action,
            holdback=True,
            reason="holdback slice: admitted regardless of score, to keep the rejected region observable.",
        )
    return EnforcementOutcome(decision.action, decision.action, reason="enforced.")


def should_retain_state(decision: GateDecision) -> bool:
    """Should the raw `state` be kept, not just its hash?

    Yes for anything the door would stop (`reject`/`ask_clarifying`). A hash cannot
    be audited after the fact; a stored state lets the rejected region be hand-read
    later without re-running the gate against live credit (data-acquisition §6.6).
    Kept true even in observe-only, because that is exactly when the would-be-stops
    accumulate the corpus enforcement will be judged on.
    """
    return decision.action is not GateAction.proceed


# --- the day-one lever: a rate, not a probability (§8 #3a) -------------------


def escalation_cutoff(scores: Sequence[float], rate: float, *, side: str = "top") -> float:
    """The score at the top-`rate` boundary — "escalate the top N%" as a number.

    Needs no labels: it assumes only that Jev's score is monotone in risk and
    that `rate` (the fraction to review) comes from review capacity. With `rate`
    of 0.1 and 100 scores, the top 10 escalate. This is the threshold that is
    settable on day one; the probability cut is the one that waits for the log.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate!r}")
    if not scores:
        raise ValueError("cannot compute a cutoff from no scores")
    ordered = sorted(scores)
    if rate <= 0.0:  # nothing escalates: unreachable by any score
        return float("-inf") if side == "low" else float("inf")
    if rate >= 1.0:  # everything escalates
        return float("inf") if side == "low" else ordered[0]
    # An order statistic, not an interpolated quantile: `k` is how many to
    # escalate, and the cutoff is the k-th score from that end. `side="top"` is
    # for a risk score (high = risky, e.g. `decide` stops when `score >= cut`);
    # `side="low"` is for D2, where LOW = risky and `decide` stops when
    # `score < cut` — so the boundary must come off the bottom, or "escalate the
    # top N%" would stop ~(1-N) of requests instead of N.
    k = max(1, round(rate * len(ordered)))
    return ordered[k - 1] if side == "low" else ordered[len(ordered) - k]


def median_score(scores: Sequence[float]) -> float:
    """A small convenience for the log: the centre of a batch of scores."""
    return statistics.median(scores)


# --- the typed tool functions (memo §2.3) ------------------------------------


class GateAssessment(BaseModel):
    """The typed answer a tool call returns to the model.

    Every field is a typed answer or the composed action — there is no free-text
    field, so a model reading this cannot confuse itself about what the gate said.
    """

    d1_in_remit: NoulAnswer | None = None
    d2_specification_sufficient: NoulAnswer | None = None
    d3_dataset_fit: ScoreAnswer | None = None
    d5_cost_proportional: NoulAnswer | None = None
    action: GateAction | None = None
    reasons: list[str] = []


async def ask_gate(
    client: JevClient,
    state: GateState,
    policy: GatePolicy = DEFAULT_POLICY,
) -> tuple[dict[str, Answer], GateDecision]:
    """Run the four questions in one Jev call, compose the action, return both.

    Both halves are returned because the log needs the *answers* (for
    calibration) and the *action* (for the record) — gate memo §3.
    """
    answers = await client.ask(serialise_state(state), questions(state.estimate_usd))
    return answers, decide(answers, policy)


async def ask_cost_gate(
    client: JevClient,
    state: GateState,
    estimate_usd: float | None,
    policy: GatePolicy = DEFAULT_POLICY,
) -> tuple[dict[str, Answer], GateDecision]:
    """Run D5 alone once the Validator has priced the request, and compose it.

    The second checkpoint. `estimate_usd` is the Validator's deterministic number,
    typically read back off the ticket it produced. Returns `({}, decision)` without
    a call when the estimate is below the materiality floor — there is nothing to
    ask, and the record still carries an action ("proceed, below the floor") so the
    log shows the checkpoint ran.

    Fail-open like `ask_gate`: a Jev outage yields a failed D5, which `decide_cost`
    reads as "no opinion" and passes.
    """
    qs = cost_questions(estimate_usd)
    if not qs:
        return {}, GateDecision(
            action=GateAction.proceed,
            reasons=(f"estimate below materiality floor (${MATERIALITY_FLOOR_USD:.2f}); D5 not asked.",),
        )
    answers = await client.ask(serialise_state(state), qs)
    return answers, decide_cost(answers, policy)


def toolset(
    client: JevClient,
    state_of: Callable[..., GateState],
    policy: GatePolicy = DEFAULT_POLICY,
):
    """A Pydantic AI toolset exposing the gate as typed tool functions.

    `assess_request` returns the whole `GateAssessment` from **one** Jev call —
    one tool, not one per question, to preserve the API's parallel evaluation.
    `choose_legal_value` is the repair-loop Choice (§6.3 row 5): the model
    supplies the legal set the API's 422 named and Jev ranks them, so the model
    cannot return a value outside that set.

    This toolset is advisory: the front-door gate runs deterministically and does
    not depend on the model choosing to call it (memo §5 #3).
    """

    async def assess_request(ctx: RunContext) -> GateAssessment:
        """Assess the current request against the front-door question set."""
        answers, decision = await ask_gate(client, state_of(ctx), policy)
        return GateAssessment(
            d1_in_remit=_noul(answers.get(D1)),
            d2_specification_sufficient=_noul(answers.get(D2)),
            d3_dataset_fit=_score(answers.get(D3)),
            d5_cost_proportional=_noul(answers.get(D5)),
            action=decision.action,
            reasons=list(decision.reasons),
        )

    async def choose_legal_value(
        ctx: RunContext, instruction: str, legal_values: list[str]
    ) -> ChoiceAnswer:
        """Choose among `legal_values` (the set the API named) for a defect."""
        question: dict[str, Question] = {
            "choice": Choice(
                instructions=instruction,
                criteria={v: v for v in legal_values},
            )
        }
        answers = await client.ask(state_of(ctx).raw_request, question)
        answer = answers.get("choice")
        return answer if isinstance(answer, ChoiceAnswer) else ChoiceAnswer(failed=True)

    return FunctionToolset(
        tools=[
            Tool(assess_request, takes_ctx=True, name="assess_request"),
            Tool(choose_legal_value, takes_ctx=True, name="choose_legal_value"),
        ]
    )


def record(decision: GateDecision) -> str:
    """A compact JSON line for the shadow log — the action and its reasons."""
    return json.dumps({"action": decision.action.value, "reasons": list(decision.reasons)})
