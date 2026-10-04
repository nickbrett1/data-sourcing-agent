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


def _stub_agent():
    """A stand-in for the Pydantic AI agent: runs offline, returns a Ticket."""
    from types import SimpleNamespace

    class _Stub:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, prompt):  # noqa: ARG002
            return SimpleNamespace(
                output=Ticket(
                    request=_request(),
                    cost=TicketCost(estimate_usd=1.2197, max_usd=2.0),
                    why="probe the table shape",
                )
            )

    return _Stub()


def _app_with_stub():
    from agent import main

    stub = _stub_agent()
    return main.create_app(
        agent_executor=main.TicketAgentExecutor(stub), agent_instance=stub
    )


def test_served_card_declares_the_wire_the_server_speaks():
    """The card's interface version must match the SDK server's wire.

    The server is the official a2a-sdk, whose native JSON-RPC dialect is 1.0
    (`SendMessage`). An a2a-sdk client — which is what the LiteLLM gateway uses
    to invoke us — reads this field to choose its transport. The card must
    advertise the version the server actually speaks, and it must be the same
    version the container registers with LiteLLM.
    """
    from starlette.testclient import TestClient

    from agent import main

    with TestClient(_app_with_stub()) as client:
        card = client.get("/.well-known/agent-card.json").json()

    jsonrpc = [i for i in card["supportedInterfaces"] if i["protocolBinding"] == "JSONRPC"]
    assert jsonrpc, "the card must advertise a JSONRPC interface"
    assert all(i["protocolVersion"] == "1.0" for i in jsonrpc)
    assert main.PROTOCOL_VERSION == "1.0"


def test_native_send_message_returns_a_completed_task():
    """The a2a-sdk 1.0 request the gateway sends must return a task, not an error.

    This is the transport the gateway selects once the card advertises 1.0: a
    `SendMessage` with the `A2A-Version: 1.0` header. Success is a JSON-RPC
    `result` carrying a completed task whose artifact holds the rendered ticket.
    """
    from a2a.helpers.proto_helpers import new_text_part
    from a2a.types.a2a_pb2 import Message, Role, SendMessageRequest
    from google.protobuf.json_format import MessageToDict
    from starlette.testclient import TestClient

    params = MessageToDict(
        SendMessageRequest(
            message=Message(
                message_id="m1",
                role=Role.ROLE_USER,
                parts=[new_text_part("One day of the SPY options chain.")],
            )
        )
    )
    body = {"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": params}

    with TestClient(_app_with_stub()) as client:
        response = client.post("/", json=body, headers={"A2A-Version": "1.0"})

    assert response.status_code == 200
    task = response.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    texts = [p["text"] for a in task["artifacts"] for p in a["parts"] if "text" in p]
    assert any("state: draft" in t for t in texts)


def test_v03_message_send_gateway_shape_is_accepted():
    """The legacy `message/send` payload the gateway currently sends must work.

    The v0.3 adapter is enabled on the same endpoint, so a caller that reads the
    card as 0.3 — including the partial `configuration` (`{blocking: true}`
    without `acceptedOutputModes`) that broke FastA2A — is still served.
    """
    from a2a.compat.v0_3 import types as v03
    from starlette.testclient import TestClient

    params = v03.MessageSendParams(
        message=v03.Message(
            message_id="m1",
            role=v03.Role.user,
            parts=[v03.Part(root=v03.TextPart(text="One day of the SPY chain."))],
        ),
        configuration=v03.MessageSendConfiguration(blocking=True),
    )
    body = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "message/send",
        "params": params.model_dump(by_alias=True, exclude_none=True),
    }

    with TestClient(_app_with_stub()) as client:
        response = client.post("/", json=body)

    assert response.status_code == 200
    # A result, not an error: the gateway's failure mode was HTTP 500 / error.
    assert "result" in response.json()


def test_request_headers_reach_the_litellm_forwarder():
    """Inbound `x-litellm-*` headers must reach `current_litellm_headers()`.

    The gateway stamps its calls with trace/agent headers, and the agent's own
    model calls must carry them (see agent/headers.py). The a2a-sdk executor no
    longer receives them through FastA2A's JSON-RPC metadata round-trip, so this
    pins that the SDK's `call_context.state['headers']` path still feeds the
    forwarder.
    """
    from types import SimpleNamespace

    from a2a.helpers.proto_helpers import new_text_part
    from a2a.types.a2a_pb2 import Message, Role, SendMessageRequest
    from google.protobuf.json_format import MessageToDict
    from starlette.testclient import TestClient

    from agent import main
    from agent.headers import current_litellm_headers

    seen: dict[str, str] = {}

    class _RecordingAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):  # noqa: ANN002
            return False

        async def run(self, prompt):  # noqa: ARG002
            seen.update(current_litellm_headers())
            return SimpleNamespace(
                output=Ticket(
                    request=_request(),
                    cost=TicketCost(estimate_usd=1.2197, max_usd=2.0),
                    why="probe the table shape",
                )
            )

    stub = _RecordingAgent()
    app = main.create_app(
        agent_executor=main.TicketAgentExecutor(stub), agent_instance=stub
    )
    params = MessageToDict(
        SendMessageRequest(
            message=Message(
                message_id="m1",
                role=Role.ROLE_USER,
                parts=[new_text_part("One day of the SPY chain.")],
            )
        )
    )
    body = {"jsonrpc": "2.0", "id": "1", "method": "SendMessage", "params": params}

    with TestClient(app) as client:
        response = client.post(
            "/",
            json=body,
            headers={"A2A-Version": "1.0", "X-LiteLLM-Trace-Id": "trace-abc"},
        )

    assert response.status_code == 200
    assert seen.get("x-litellm-trace-id") == "trace-abc"
