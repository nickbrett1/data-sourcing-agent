"""Tests for the agent package — the code the image actually runs.

These live in `tests/` (the configured pytest scope) and import the top-level
`agent` package via `pythonpath = ["."]`. They deliberately do **not** touch the
network: the Validator's HTTP client is injected with an `httpx.MockTransport`,
so the tests pin the handling of the API's truth without depending on the API.

The one thing they cannot prove is that the live API still answers the way the
fixtures assume; the handover memo §5.1 records the live probe that established
the fixtures, and a change there is a change to the oracle, not to this code.
"""

from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest
from pydantic import ValidationError

from agent.contract import validate_agent_result
from agent.ticket import (
    CostProposal,
    RequestSpec,
    Ticket,
    TicketCost,
    TicketProposal,
    render_ticket_yaml,
)
from agent.validator import DatabentoRequestError, DatabentoValidator

# --- fixtures -----------------------------------------------------------------


def _request(**overrides: object) -> RequestSpec:
    base = {
        "dataset": "OPRA.PILLAR",
        "schema": "ohlcv-1d",
        "symbols": ["SPY.OPT"],
        "stype_in": "parent",
        "start": date(2026, 10, 1),
        "end": date(2026, 10, 2),
    }
    base.update(overrides)
    return RequestSpec(**base)  # type: ignore[arg-type]


def _proposal(summary_max_usd: float = 2.0) -> TicketProposal:
    return TicketProposal(
        group="opra-shape-probe",
        request=_request(),
        cost=CostProposal(max_usd=summary_max_usd),
        why="Sample one day of the SPY chain to test the expiry-keyed table shape.",
    )


def _validator(handler) -> DatabentoValidator:
    client = httpx.Client(
        base_url="https://hist.databento.com",
        transport=httpx.MockTransport(handler),
    )
    return DatabentoValidator("test-key", client=client)


# --- the ticket model ---------------------------------------------------------


def test_ticket_proposal_cannot_express_state_or_estimate():
    """The model's own type must not contain the fields it may not write.

    `state` and `estimate_usd` are policy and oracle fields. If they are absent
    from the proposal schema, the model cannot invent them — the constraint is
    structural, not a prompt instruction.
    """
    fields = set(TicketProposal.model_fields)
    assert "state" not in fields
    assert "estimate_usd" not in fields
    assert fields == {"group", "request", "cost", "why"}


def test_request_window_must_be_ordered():
    with pytest.raises(ValidationError):
        _request(start=date(2026, 10, 2), end=date(2026, 10, 1))


def test_request_rejects_illegal_stype():
    with pytest.raises(ValidationError):
        _request(stype_in="not_a_stype")


def test_ticket_yaml_renders_the_contract():
    ticket = Ticket(
        request=_request(),
        cost=TicketCost(estimate_usd=1.2197, max_usd=2.0),
        why="probe the table shape",
    )
    rendered = render_ticket_yaml(ticket)
    assert "state: draft" in rendered
    assert "dataset: OPRA.PILLAR" in rendered
    assert "stype_in: parent" in rendered
    assert "estimate_usd: 1.2197" in rendered
    assert "actual_usd: None" in rendered


# --- the Validator, against the API's own truth -------------------------------


def test_price_request_returns_the_api_number():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["symbols"] == "SPY.OPT"
        assert request.url.params["stype_in"] == "parent"
        # The live endpoint answers with a bare float, not JSON.
        return httpx.Response(200, text="1.219747960567")

    assert _validator(handler).price_request(_request()) == 1.219747960567


def test_validate_request_raises_the_api_refusal_verbatim():
    body = {
        "detail": {
            "case": "symbology_invalid_request",
            "message": "Unable to process symbology with parameters: `stype_in=parent`.",
            "status_code": 422,
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    with pytest.raises(DatabentoRequestError) as exc:
        _validator(handler).validate_request(_request())
    assert exc.value.case == "symbology_invalid_request"
    assert exc.value.status_code == 422
    assert "stype_in=parent" in exc.value.message


def test_validator_refuses_to_build_without_a_key():
    with pytest.raises(ValueError):
        DatabentoValidator("")


# --- the Validator layer, as the agent wires it -------------------------------


def test_contract_returns_a_draft_ticket_costed_by_the_api(monkeypatch):
    validator = _validator(lambda request: httpx.Response(200, text="1.219747960567"))
    monkeypatch.setattr("agent.contract._validator", validator)

    ticket = asyncio.run(validate_agent_result(None, _proposal()))  # type: ignore[arg-type]

    assert isinstance(ticket, Ticket)
    assert ticket.state == "draft"
    assert ticket.cost.estimate_usd == 1.2197
    assert ticket.cost.actual_usd is None


def test_contract_retries_with_the_apis_own_words(monkeypatch):
    body = {
        "detail": {
            "case": "dataset_schema_not_supported",
            "message": "Schema `mbp-1` is not supported on dataset `OPRA.PILLAR`.",
            "status_code": 422,
        }
    }
    validator = _validator(lambda request: httpx.Response(422, json=body))
    monkeypatch.setattr("agent.contract._validator", validator)

    from pydantic_ai import ModelRetry

    with pytest.raises(ModelRetry) as exc:
        asyncio.run(validate_agent_result(None, _proposal()))  # type: ignore[arg-type]
    assert "dataset_schema_not_supported" in str(exc.value)
    assert "OPRA.PILLAR" in str(exc.value)


def test_contract_narrows_rather_than_raising_the_ceiling(monkeypatch):
    """Over-ceiling must repair by narrowing, and must forbid raising max_usd."""
    validator = _validator(lambda request: httpx.Response(200, text="99.0"))
    monkeypatch.setattr("agent.contract._validator", validator)

    from pydantic_ai import ModelRetry

    with pytest.raises(ModelRetry) as exc:
        asyncio.run(validate_agent_result(None, _proposal(summary_max_usd=2.0)))  # type: ignore[arg-type]
    message = str(exc.value)
    assert "exceeds the ceiling" in message
    assert "may not raise `max_usd`" in message


def test_contract_rejects_a_blank_why_without_calling_the_api(monkeypatch):
    def explode(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("the API must not be called for a blank why")

    monkeypatch.setattr("agent.contract._validator", _validator(explode))

    from pydantic_ai import ModelRetry

    proposal = _proposal()
    proposal.why = "   "
    with pytest.raises(ModelRetry):
        asyncio.run(validate_agent_result(None, proposal))  # type: ignore[arg-type]


# --- the runtime wiring -------------------------------------------------------


def test_main_uses_prompted_output_not_tool_output():
    """deepseek-v4-flash refuses forced tool_choice; the output mode is load-bearing.

    A regression to the default (ToolOutput) would 400 on every call. The
    handover memo §5.1 is the evidence.
    """
    from pydantic_ai import PromptedOutput

    from agent import main

    assert isinstance(main.agent.output_type, PromptedOutput)
    assert main.agent.output_type.outputs is TicketProposal


def test_card_url_is_a_dial_address_not_a_bind_address():
    from agent import main

    assert main.CARD_URL != "http://0.0.0.0:8700"
    assert "0.0.0.0" not in main.CARD_URL
