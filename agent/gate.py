"""The front-door gate — Jev advises, deterministic code decides (memo §6.5).

The gate asks Jev **four** questions about one request and composes their answers
into **one action**, in code:

* **D1** — in-remit / legitimacy (Noul)
* **D2** — specification sufficiency (Noul) — the load-bearing one
* **D3** — dataset/schema fit (Score)
* **D5** — cost proportionality (Noul, fires only when spend is material)

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
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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

# The D2 probability cut is a PLACEHOLDER (10:1 false-accept:false-reject -> 0.91,
# gate memo §2). Nothing calibrates it yet. In observe-only it is inert; for
# enforcement the rate (`escalation_cutoff`) is the day-one lever, and this number
# becomes the log's `threshold_at_time` until calibration earns a real one.
D2_CUT_PLACEHOLDER = 0.91
D1_CUT_PLACEHOLDER = 0.5
D5_CUT_PLACEHOLDER = 0.5


class GateAction(StrEnum):
    """What the front desk should do. The gate's output, computed not asked."""

    proceed = "proceed"
    ask_clarifying = "ask_clarifying"
    reject = "reject"


def questions() -> dict[str, Question]:
    """The four questions, built fresh each call so a caller cannot mutate them."""
    return {
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
        D5: Noul(
            instructions=(
                "The estimated spend is proportionate to the stated intent. A large "
                "pull requested as a casual look is disproportionate; the same pull "
                "requested as a full panel for backtesting is proportionate."
            )
        ),
    }


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
    d1_cut: float = D1_CUT_PLACEHOLDER
    d2_cut: float = D2_CUT_PLACEHOLDER
    d5_cut: float = D5_CUT_PLACEHOLDER
    d3_min: int = D3_CLOSE


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
        level = D3_LEVELS[int(d3.score)]
        reasons.append(f"{D3}={level} < required close: fit too uncertain.")
    if d2 is not None and d2.noul is not None and d2.noul < policy.d2_cut:
        reasons.append(
            f"{D2}={d2.noul:.2f} < {policy.d2_cut:.2f} (provisional cut): "
            "specification not sufficient."
        )
    if reasons:
        return GateDecision(action=GateAction.ask_clarifying, reasons=tuple(reasons))

    reasons.append("D1 in remit, D2 at or above cut, D3 at least close.")
    return GateDecision(action=GateAction.proceed, reasons=tuple(reasons))


def effective_action(
    decision: GateDecision, policy: GatePolicy = DEFAULT_POLICY
) -> GateAction:
    """What a caller should actually *do*: the computed action only when enforcing.

    This is where observe-only lives. The gate always *computes* an action and the
    log always records it, but until `policy.enforce` is true the effective action
    is `proceed` — nothing is routed on. Keeping that a function (rather than an
    `if` scattered at call sites) means "are we enforcing?" has exactly one
    home, and the distinction between *computed* and *enforced* is impossible to
    lose track of.
    """
    return decision.action if policy.enforce else GateAction.proceed


# --- the day-one lever: a rate, not a probability (§8 #3a) -------------------


def escalation_cutoff(scores: Sequence[float], rate: float) -> float:
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
    if rate <= 0.0:
        return float("inf")  # nothing escalates
    if rate >= 1.0:
        return ordered[0]  # everything escalates
    # An order statistic, not an interpolated quantile: `k` is how many to
    # escalate, and the cutoff is the k-th largest. So exactly the top `k`
    # scores are at or above it — which is what "escalate the top N%" means.
    k = max(1, round(rate * len(ordered)))
    return ordered[len(ordered) - k]


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
    answers = await client.ask(serialise_state(state), questions())
    return answers, decide(answers, policy)


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
