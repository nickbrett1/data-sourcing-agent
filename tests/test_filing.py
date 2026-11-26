"""Filing happens after validation, from the ticket — never from the model.

The property under test is an *order*: a request row can only exist because a
`Ticket` that `validate_agent_result` priced and accepted produced its body. The
model's toolset has `submit_request` filtered out precisely so the two cannot be
reordered at run time, so these tests pin both halves — the mapping and the
filter — plus the refusal path, which must name its reason rather than fail
silently.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import pytest

from agent.filing import (
    FILED_BY,
    MCP_URL,
    ON_FAILURE,
    PLAN_TYPE,
    SUBMIT_TOOL,
    VENDOR,
    FilingRefused,
    file_ticket,
    plan_body,
)
from agent.ticket import RequestSpec, StypeIn, Ticket, TicketCost, render_ticket_yaml


def _ticket(**overrides) -> Ticket:
    fields = dict(
        request=RequestSpec(
            dataset="GLBX.MDP3",
            schema="ohlcv-1m",
            symbols=["ES.c.0"],
            stype_in=StypeIn.continuous,
            start=date(2026, 11, 19),
            end=date(2026, 11, 26),
        ),
        cost=TicketCost(estimate_usd=0.954, max_usd=1.0),
        why="One week of continuous ES minute bars to test a session-boundary bug.",
    )
    fields.update(overrides)
    return Ticket(**fields)


class _FakeClient:
    """Stands in for `mcp.Client`: records the call, returns a canned answer."""

    calls: list[tuple[str, dict]] = []
    answer: object = None

    def __init__(self, url: str):
        self.url = url
        _FakeClient.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def call_tool(self, name, arguments):
        _FakeClient.calls.append((name, arguments))
        return _FakeClient.answer


def _result(**structured):
    return SimpleNamespace(
        structured_content=structured,
        content=[],
        is_error=None,
        isError=None,
    )


def test_the_body_is_the_tickets_own_fields_and_not_the_models():
    """`plan_body` maps the validated ticket, whole — nothing is re-derived here.

    The two fields the ticket has no home for are the ones this pins: `vendor`
    (a property of the seller, not of the question) and `verify.on_failure`
    (a policy about bad bytes, which the model is never asked about).
    """
    body = plan_body(_ticket())

    assert body == {
        "kind": "plan",
        "plan_type": PLAN_TYPE,
        "params": {
            "vendor": VENDOR,
            "dataset": "GLBX.MDP3",
            "schema": "ohlcv-1m",
            "symbols": ["ES.c.0"],
            "stype_in": "continuous",
            "start": "2026-11-19",
            "end": "2026-11-26",
        },
        "verify": {"on_failure": ON_FAILURE},
        "max_usd": 1.0,
        "why": "One week of continuous ES minute bars to test a session-boundary bug.",
    }


def test_the_ceiling_filed_is_the_ceiling_that_was_validated():
    """The Validator compares the estimate against `cost.max_usd`; the body must
    carry that same number, or the ceiling a human approves is not the one that
    was checked."""
    ticket = _ticket(cost=TicketCost(estimate_usd=4.2, max_usd=5.0))

    assert plan_body(ticket)["max_usd"] == ticket.cost.max_usd


def test_the_body_never_carries_a_state():
    """`Ticket` pins `state=draft`; the plan body must not forward it — data-download
    writes the state of a filed request, and a body that named one would be this
    agent claiming a policy field."""
    assert "state" not in plan_body(_ticket())


def test_filing_names_the_agent_and_keys_on_the_turn(monkeypatch):
    """Provenance: `filed_by` is this agent, the idempotency key is the turn.

    Two filings of one turn are one request; the trail says who vouched.
    """
    _FakeClient.answer = _result(request_id="req_abc")
    monkeypatch.setattr("agent.filing.Client", _FakeClient)

    filed = asyncio.run(file_ticket(_ticket(), turn_id="turn123"))

    assert filed.request_id == "req_abc"
    name, arguments = _FakeClient.calls[0]
    assert name == SUBMIT_TOOL
    assert arguments["filed_by"] == FILED_BY
    assert arguments["idempotency_key"] == "turn123"
    assert arguments["body"] == plan_body(_ticket())


def test_a_filing_without_a_turn_id_omits_the_idempotency_key(monkeypatch):
    """The key is optional upstream; sending an empty one would collide every
    turn's filing with every other turn's."""
    _FakeClient.answer = _result(request_id="req_abc")
    monkeypatch.setattr("agent.filing.Client", _FakeClient)

    asyncio.run(file_ticket(_ticket()))

    assert "idempotency_key" not in _FakeClient.calls[0][1]


def test_a_refusal_is_raised_with_the_runtimes_own_words(monkeypatch):
    """A refused body must fail the turn loudly and readably.

    data-download writes a sentence naming the offending field, which is the one
    thing a human can act on; summarising it here would lose it.
    """
    refusal = (
        "plan_not_valid: the catalogue the vendor describes this dataset with "
        "does not carry schema 'ohlcv-1m'"
    )
    _FakeClient.answer = SimpleNamespace(
        structured_content=None,
        content=[SimpleNamespace(text=refusal)],
        is_error=True,
        isError=True,
    )
    monkeypatch.setattr("agent.filing.Client", _FakeClient)

    with pytest.raises(FilingRefused) as raised:
        asyncio.run(file_ticket(_ticket(), turn_id="turn123"))

    assert refusal in str(raised.value)


def test_a_filing_that_cannot_name_its_row_is_not_a_filing(monkeypatch):
    """A 202 with no id is treated as a failure: rendering a ticket that claims
    to have been filed, but cannot say which row, is the worse outcome."""
    _FakeClient.answer = _result()
    monkeypatch.setattr("agent.filing.Client", _FakeClient)

    with pytest.raises(FilingRefused):
        asyncio.run(file_ticket(_ticket(), turn_id="turn123"))


def test_the_rendered_ticket_names_the_row_it_became():
    """The artifact a human reads should point at what actually exists — and a
    ticket that was never filed must not pretend to."""
    assert "request_id:" not in render_ticket_yaml(_ticket())

    rendered = render_ticket_yaml(_ticket(request_id="req_abc"))

    assert "request_id: req_abc" in rendered
    assert rendered.startswith("state: draft\n")


def test_the_model_cannot_file():
    """The write is off the model's toolset, which is what makes the order real.

    Without this the model can file mid-loop — before the Validator has priced
    the request, and again after each `ModelRetry` — leaving rows nothing
    validated. Every other tool stays available: the model needs discovery, the
    price and the book to construct a good draft.
    """
    from agent.main import _model_may_not_file

    assert _model_may_not_file(None, SimpleNamespace(name=SUBMIT_TOOL)) is False
    for allowed in (
        "data-download-quote",
        "data-download-list_plan_types",
        "data-download-list_requests",
        "databento-discovery-resolve_symbols",
    ):
        assert _model_may_not_file(None, SimpleNamespace(name=allowed)) is True, allowed


def test_every_toolset_the_agent_holds_is_filtered():
    """A second, unfiltered toolset would silently restore the write."""
    from agent.main import AGENT_TOOLSETS

    assert AGENT_TOOLSETS
    for toolset in AGENT_TOOLSETS:
        assert type(toolset).__name__ == "FilteredToolset", type(toolset).__name__


def test_the_filing_url_is_the_group_that_carries_the_write():
    """mcphub's one `data` group: discovery, the book and the write in one
    toolset. Filing against a group without `submit_request` would fail at run
    time with a tool-not-found that looks like a bug in this module."""
    assert MCP_URL.endswith("/mcp/data")
