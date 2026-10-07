"""Tests for the two-checkpoint gate wiring in the executor.

The gate now runs twice per turn: at the front door (D1/D2/D3, before pricing) and
after the Validator has priced the request (D5 alone). These tests drive both
checkpoints with fakes — no model, no Jev, no network — and pin the two properties
that make the second checkpoint safe and useful:

* it fires D5 *only* once there is a material estimate, and asks nothing else;
* both checkpoints write to the shadow log under **one** `request_id`, so the turn
  joins; and the whole thing stays observe-only (nothing routes on the computed
  action).
"""

from __future__ import annotations

import asyncio
import json

from agent import main
from agent.gate import D1, D2, D3, D5, GateAction, GatePolicy
from agent.interpret import ParsedIntent
from agent.jev import NoulAnswer, ScoreAnswer
from agent.samples import load_samples
from agent.ticket import (
    CostProposal,
    RequestSpec,
    Ticket,
    TicketCost,
    render_ticket_yaml,
)


class _Interpreter:
    async def run(self, _text: str):
        return type(
            "R",
            (),
            {
                "output": ParsedIntent(
                    understood="daily SPY option bars",
                    dataset="OPRA.PILLAR",
                    schema="ohlcv-1d",
                    symbols=["SPY.OPT"],
                )
            },
        )()


class _Jev:
    """Records the question sets it is asked, and answers them plausibly."""

    def __init__(self):
        self.question_sets: list[set[str]] = []

    async def ask(self, _state: str, questions: dict):
        self.question_sets.append(set(questions))
        out = {}
        for qid in questions:
            if qid == D1:
                out[qid] = NoulAnswer(noul=0.99)
            elif qid == D2:
                out[qid] = NoulAnswer(noul=0.97)
            elif qid == D3:
                out[qid] = ScoreAnswer(score=3.0)
            elif qid == D5:
                out[qid] = NoulAnswer(noul=0.20)
        return out


def _ticket(estimate: float) -> Ticket:
    return Ticket(
        state="draft",
        group=None,
        request=RequestSpec(
            dataset="OPRA.PILLAR",
            schema="ohlcv-1d",
            symbols=["SPY.OPT"],
            stype_in="raw_symbol",
            start="2024-01-01",
            end="2024-01-31",
        ),
        cost=TicketCost(estimate_usd=estimate, max_usd=100.0, actual_usd=None),
        why="test",
    )


def _executor(interpreter, jev) -> main.TicketAgentExecutor:
    return main.TicketAgentExecutor(object(), interpreter=interpreter, jev=jev)


def test_the_second_checkpoint_asks_only_d5_when_the_estimate_is_material(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    jev = _Jev()
    ex = _executor(_Interpreter(), jev)
    rid = "rid-join"

    intent, _ = asyncio.run(ex._assess("daily SPY option bars for January", rid))
    asyncio.run(ex._assess_cost("daily SPY option bars for January", intent, _ticket(50.0), rid))

    # Front door asked D1/D2/D3; the cost checkpoint asked D5 and nothing else.
    assert len(jev.question_sets) == 2
    assert jev.question_sets[0] == {D1, D2, D3}
    assert jev.question_sets[1] == {D5}

    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    # One request id spans both checkpoints, so the turn's records join.
    assert {r["request_id"] for r in rows} == {rid}
    by_q = {r["question_id"]: r for r in rows}
    assert set(by_q) == {D1, D2, D3, D5}
    # D5 was disproportionate (0.20 < 0.5) but we are observe-only: the *computed*
    # action is ask_clarifying, the *taken* action is proceed, and nothing routed.
    assert by_q[D5]["action_computed"] == "ask_clarifying"
    assert by_q[D5]["action_taken"] == "proceed"
    assert by_q[D5]["holdback"] is False
    assert by_q[D5]["probability"] == 0.20
    assert by_q[D5]["threshold_at_time"] == 0.5


def test_the_second_checkpoint_is_skipped_below_the_materiality_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    jev = _Jev()
    ex = _executor(_Interpreter(), jev)
    rid = "rid-cheap"

    intent, _ = asyncio.run(ex._assess("a tiny pull", rid))
    asyncio.run(ex._assess_cost("a tiny pull", intent, _ticket(0.01), rid))

    # Only the front-door call happened; no D5 call for an immaterial estimate.
    assert jev.question_sets == [{D1, D2, D3}]
    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    assert D5 not in {r["question_id"] for r in rows}


def test_the_cost_checkpoint_survives_an_interpreter_failure(tmp_path, monkeypatch):
    """No reading is not a reason to skip judging a priced request."""
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    jev = _Jev()
    ex = _executor(_Interpreter(), jev)
    rid = "rid-no-read"

    asyncio.run(ex._assess_cost("a pull with no reading", None, _ticket(50.0), rid))

    assert jev.question_sets == [{D5}]
    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    assert {r["question_id"] for r in rows} == {D5}


def test_the_gate_path_never_changes_the_rendered_ticket(tmp_path, monkeypatch):
    """Observe-only, proven on the artefact: the same ticket renders identically."""
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    ticket = _ticket(50.0)
    before = render_ticket_yaml(ticket)

    ex = _executor(_Interpreter(), _Jev())
    rid = "rid-inert"
    intent, _ = asyncio.run(ex._assess("daily SPY option bars", rid))
    asyncio.run(ex._assess_cost("daily SPY option bars", intent, ticket, rid))

    assert render_ticket_yaml(ticket) == before


def test_a_jev_failure_at_the_cost_checkpoint_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))

    class _Boom:
        async def ask(self, *_a, **_k):
            raise RuntimeError("jev is down")

    ex = _executor(_Interpreter(), _Boom())
    rid = "rid-boom"
    intent, _ = asyncio.run(ex._assess("daily SPY option bars", rid))
    # Must not raise: the checkpoint is best-effort.
    asyncio.run(ex._assess_cost("daily SPY option bars", intent, _ticket(50.0), rid))


def test_cost_proposals_still_render_a_ticket():
    """Guard the fixture itself: a Ticket is a Ticket."""
    assert "estimate_usd" in render_ticket_yaml(_ticket(1.0))
    assert CostProposal(max_usd=1.0).max_usd == 1.0


class _RejectingJev(_Jev):
    """Jev that would stop the request at the door (D1 = no)."""

    async def ask(self, _state: str, questions: dict):
        self.question_sets.append(set(questions))
        return {qid: NoulAnswer(noul=0.01) for qid in questions}


def test_would_be_stops_retain_their_raw_state(tmp_path, monkeypatch):
    """§6.6: a hash cannot be audited, so keep the raw state for the rejected region."""
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    ex = _executor(_Interpreter(), _RejectingJev())
    rid = "rid-stop"

    asyncio.run(ex._assess("do something out of remit", rid))

    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    assert rows and all(r["action_computed"] == "reject" for r in rows)
    assert all(r["action_taken"] == "proceed" for r in rows)  # observe-only
    # The raw state rides on the first row only.
    retained = [r.get("state") for r in rows if r.get("state")]
    assert len(retained) == 1 and "out of remit" in retained[0]


def test_nothing_is_retained_for_a_request_that_would_proceed(tmp_path, monkeypatch):
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    ex = _executor(_Interpreter(), _Jev())
    asyncio.run(ex._assess("daily SPY option bars", "rid-clean"))

    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    assert all(r.get("state") is None for r in rows)


def test_flipping_enforce_is_the_only_thing_that_changes_a_turn(tmp_path, monkeypatch):
    """The holdback admits a would-be stop; without it, the door stops the turn."""
    monkeypatch.setenv("GATE_LOG_PATH", str(tmp_path / "gate-log.jsonl"))
    ex = _executor(_Interpreter(), _RejectingJev())
    rid = "rid-flip"

    # Observe-only: the outcome action is proceed, so the turn is unchanged.
    _, outcome = asyncio.run(ex._assess("out of remit", rid))
    assert main._enforced_reply(outcome) is None

    # Enforcing with no holdback: the door stops the turn with the reject message.
    monkeypatch.setattr(main, "DEFAULT_POLICY", GatePolicy(enforce=True, holdback_rate=0.0))
    _, outcome = asyncio.run(ex._assess("out of remit", rid + "-2"))
    assert outcome.action is GateAction.reject
    assert main._enforced_reply(outcome).startswith("I can't take this request")

    # Enforcing *with* the holdback admitting everything: the turn proceeds anyway.
    monkeypatch.setattr(main, "DEFAULT_POLICY", GatePolicy(enforce=True, holdback_rate=1.0))
    _, outcome = asyncio.run(ex._assess("out of remit", rid + "-3"))
    assert outcome.action is GateAction.proceed
    assert outcome.holdback is True
    assert main._enforced_reply(outcome) is None

    rows = [json.loads(line) for line in (tmp_path / "gate-log.jsonl").read_text().splitlines()]
    holdback_rows = [r for r in rows if r["holdback"]]
    assert holdback_rows and all(r["action_computed"] == "reject" for r in holdback_rows)


class _FakeAgent:
    """A pydantic-agent stand-in: returns a Ticket, or raises, on demand."""

    def __init__(self, ticket=None, boom=False):
        self._ticket, self._boom = ticket, boom

    async def run(self, _prompt):
        if self._boom:
            raise RuntimeError("model down")
        return type("R", (), {"output": self._ticket})()


def test_a_stopped_turn_still_keeps_the_artifact_it_would_have_made(tmp_path, monkeypatch):
    """The door stops the turn; it must not stop the evidence (§grading-sample-v1)."""
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "samples.jsonl"))
    ex = _executor(_Interpreter(), _Jev())
    ex._agent = _FakeAgent(_ticket(1.0))
    asyncio.run(ex._shadow_sample("one day of SPY options", None, "rid-shadow"))
    samples = load_samples(tmp_path / "samples.jsonl")
    assert samples["rid-shadow"].text == "one day of SPY options"
    assert samples["rid-shadow"].ticket.startswith("state:")


def test_a_shadow_generation_failure_never_raises(tmp_path, monkeypatch):
    """Best-effort: the turn is already being stopped on purpose; a sample failure
    must not become a failure of the turn."""
    monkeypatch.setenv("GATE_SAMPLE_PATH", str(tmp_path / "samples.jsonl"))
    ex = _executor(_Interpreter(), _Jev())
    ex._agent = _FakeAgent(boom=True)
    asyncio.run(ex._shadow_sample("text", None, "rid-x"))  # must not raise
    assert load_samples(tmp_path / "samples.jsonl") == {}
