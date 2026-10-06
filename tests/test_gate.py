"""Tests for the front-door gate.

Two things are pinned: the *composition* (D1/D2/D3/D5 -> one action, in code) and
the *fail-open-by-construction* property (a Jev outage yields `proceed`, because a
"no opinion" answer cannot breach a cut). The composition is pure, so most tests
build answer dicts directly; only the end-to-end one touches a transport.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from agent.gate import (
    D1,
    D2,
    D3,
    D3_CLOSE,
    D3_EXACT,
    D3_LEVELS,
    D5,
    DEFAULT_POLICY,
    GateAction,
    GatePolicy,
    GateState,
    ask_cost_gate,
    ask_gate,
    cost_questions,
    decide,
    decide_cost,
    effective_action,
    escalation_cutoff,
    questions,
    serialise_state,
    toolset,
)
from agent.jev import JevClient, NoulAnswer, ScoreAnswer


def _answers(d1: float, d2: float, d3: int = D3_EXACT, d5: float = 0.9) -> dict:
    return {
        D1: NoulAnswer(noul=d1),
        D2: NoulAnswer(noul=d2),
        D3: ScoreAnswer(score=float(d3), legend={i: lv for i, lv in enumerate(D3_LEVELS)}),
        D5: NoulAnswer(noul=d5),
    }


def test_the_question_set_excludes_d4_and_d5_is_conditional():
    # D4 is computed, never asked (memo §3.1). Without an estimate, D5 cannot fire.
    qs = questions()
    assert set(qs) == {D1, D2, D3}
    assert "d4" not in " ".join(qs)
    assert qs[D1].type == "noul"
    assert qs[D3].type == "score"
    assert qs[D3].criteria == D3_LEVELS
    # D5 appears only when the estimate clears the materiality floor.
    assert D5 not in questions(estimate_usd=1.0)
    assert D5 in questions(estimate_usd=12.74)


def test_serialise_state_carries_the_deterministic_estimate():
    text = serialise_state(
        GateState(
            raw_request="whole SPY chain",
            parsed_intent="OPRA.PILLAR / ohlcv-1d / SPY.OPT / no window",
            candidate_mappings=["OPRA.PILLAR ohlcv-1d"],
            estimate_usd=1.22,
            rows=1100,
            budget_context="no stated budget",
        )
    )
    assert "whole SPY chain" in text
    assert "estimated_usd=1.22 rows=1,100" in text
    assert "OPRA.PILLAR ohlcv-1d" in text
    assert "no stated budget" in text


def test_decide_rejects_when_d1_is_below_cut():
    d = decide(_answers(d1=0.1, d2=0.99), GatePolicy())
    assert d.action is GateAction.reject


def test_decide_asks_clarifying_when_d2_is_below_cut():
    d = decide(_answers(d1=0.99, d2=0.35))
    assert d.action is GateAction.ask_clarifying
    assert any(D2 in r for r in d.reasons)


def test_decide_asks_clarifying_when_d3_is_below_close():
    d = decide(_answers(d1=0.99, d2=0.99, d3=D3_LEVELS.index("plausible")))
    assert d.action is GateAction.ask_clarifying
    assert any(D3 in r for r in d.reasons)


def test_decide_proceeds_when_all_pass():
    d = decide(_answers(d1=0.98, d2=0.95, d3=D3_CLOSE))
    assert d.action is GateAction.proceed
    assert d.abstained is False


def test_decide_fails_open_when_every_answer_is_failed():
    failed = {D1: NoulAnswer(failed=True), D2: NoulAnswer(failed=True)}
    d = decide(failed)
    assert d.action is GateAction.proceed
    assert d.abstained is True


def test_decide_treats_a_missing_question_as_no_opinion():
    # Only D3 present and passing -> no cut breached -> proceed (fail-open).
    d = decide({D3: ScoreAnswer(score=float(D3_EXACT))})
    assert d.action is GateAction.proceed


def test_observe_only_never_routes_on_the_computed_action():
    computed = decide(_answers(d1=0.1, d2=0.99))  # would reject
    assert computed.action is GateAction.reject
    assert effective_action(computed, DEFAULT_POLICY) is GateAction.proceed
    enforcing = GatePolicy(enforce=True)
    assert effective_action(computed, enforcing) is GateAction.reject


def test_escalation_cutoff_selects_the_top_n_percent():
    scores = [float(i) for i in range(100)]  # 0..99
    # Rate 0.1 -> the top 10 are scores >= 90.
    assert escalation_cutoff(scores, 0.1) == pytest.approx(90.0, abs=0.6)


def test_escalation_cutoff_edge_rates():
    scores = [1.0, 2.0, 3.0]
    assert escalation_cutoff(scores, 0.0) == float("inf")  # none escalate
    assert escalation_cutoff(scores, 1.0) == 1.0  # all escalate
    with pytest.raises(ValueError):
        escalation_cutoff(scores, 1.5)
    with pytest.raises(ValueError):
        escalation_cutoff([], 0.5)


def test_ask_gate_runs_one_call_and_returns_answers_and_action():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        # One request. Without an estimate the set is D1/D2/D3 (D5 needs a number).
        assert set(body["questions"]) == {D1, D2, D3}
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    D1: {"type": "noul", "noul": 0.98},
                    D2: {"type": "noul", "noul": 0.30},
                    D3: {"type": "score", "score": float(D3_EXACT)},
                    D5: {"type": "noul", "noul": 0.9},
                },
            },
        )

    client = JevClient(
        base_url="http://litellm:4000",
        api_key="k",
        client=httpx.AsyncClient(base_url="http://litellm:4000", transport=httpx.MockTransport(handler)),
    )
    answers, decision = asyncio.run(ask_gate(client, GateState(raw_request="x", parsed_intent="y")))
    assert calls["n"] == 1
    assert decision.action is GateAction.ask_clarifying  # D2 below cut
    assert answers[D1].noul == pytest.approx(0.98)


def test_cost_questions_fire_only_at_or_above_the_materiality_floor():
    """D5 is absent below the floor and is the *only* question above it — the
    post-price checkpoint asks nothing else, since D1/D2/D3 were judged upfront."""
    assert cost_questions(None) == {}
    assert cost_questions(0.0) == {}
    assert cost_questions(4.99) == {}
    assert set(cost_questions(5.0)) == {D5}
    assert set(cost_questions(12.74)) == {D5}
    assert cost_questions(12.74)[D5].type == "noul"


def test_decide_cost_asks_clarifying_when_spend_is_disproportionate():
    decision = decide_cost({D5: NoulAnswer(noul=0.20)})
    assert decision.action is GateAction.ask_clarifying
    assert not decision.abstained


def test_decide_cost_proceeds_when_spend_is_proportionate():
    decision = decide_cost({D5: NoulAnswer(noul=0.95)})
    assert decision.action is GateAction.proceed


def test_decide_cost_fails_open_on_a_failed_answer():
    """A Jev outage at the cost checkpoint must not block a priced ticket."""
    decision = decide_cost({D5: NoulAnswer(failed=True)})
    assert decision.action is GateAction.proceed
    assert decision.abstained is True
    assert decide_cost({}).action is GateAction.proceed


def test_decide_cost_is_observe_only_until_enforced():
    decision = decide_cost({D5: NoulAnswer(noul=0.01)})
    assert effective_action(decision) is GateAction.proceed
    enforced = effective_action(decision, GatePolicy(enforce=True))
    assert enforced is GateAction.ask_clarifying


def test_ask_cost_gate_asks_only_d5_and_composes_it():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        assert set(body["questions"]) == {D5}  # only the cost question
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {D5: {"type": "noul", "noul": 0.10}},
            },
        )

    client = JevClient(
        base_url="http://litellm:4000",
        api_key="k",
        client=httpx.AsyncClient(base_url="http://litellm:4000", transport=httpx.MockTransport(handler)),
    )
    state = GateState(raw_request="x", parsed_intent="y", estimate_usd=50.0)
    answers, decision = asyncio.run(ask_cost_gate(client, state, 50.0))
    assert calls["n"] == 1
    assert decision.action is GateAction.ask_clarifying
    assert answers[D5].noul == pytest.approx(0.10)


def test_ask_cost_gate_makes_no_call_below_the_floor():
    """Below the floor there is no question to ask; the checkpoint still answers."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        calls["n"] += 1
        return httpx.Response(200, json={"answers": {}})

    client = JevClient(
        base_url="http://litellm:4000",
        api_key="k",
        client=httpx.AsyncClient(base_url="http://litellm:4000", transport=httpx.MockTransport(handler)),
    )
    answers, decision = asyncio.run(
        ask_cost_gate(client, GateState(raw_request="x", parsed_intent="y"), 1.0)
    )
    assert calls["n"] == 0
    assert answers == {}
    assert decision.action is GateAction.proceed


def test_toolset_exposes_the_two_typed_tools():
    """The toolset builds and names its tools; the gate does not depend on it."""
    ts = toolset(
        JevClient(base_url="http://litellm:4000", api_key="k"),
        lambda ctx: GateState(raw_request="x", parsed_intent="y"),
    )
    assert set(ts.tools) == {"assess_request", "choose_legal_value"}


def test_validator_imports_no_model():
    """The boundary rule (memo §2.4): no model, no Jev, in the Validator."""
    source = Path(__file__).resolve().parent.parent / "agent" / "validator.py"
    imports = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("jev", "typesafe", "pydantic_ai", "agent.gate", "agent.model")
    offenders = [line for line in imports if any(f in line for f in forbidden)]
    assert not offenders, f"validator.py must hold no model, but imports: {offenders}"
