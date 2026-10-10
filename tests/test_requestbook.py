"""Tests for the write path — a ticket becomes a file in the book's inbox/.

Design memo `data-request-book-v1` §4. These pin the two structural properties
the design rests on — the draft lands *only* under the inbox it is given, and it
is `state: draft` — plus the naming (§4.2 with the `request_id` join token) and
the no-clobber write. The last test drives the real executor over the A2A wire,
so the wiring in `agent/main.py` is covered, not just this module.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from agent.requestbook import DEFAULT_INBOX, draft_name, inbox_dir, write_draft
from agent.ticket import RequestSpec, Ticket, TicketCost, render_ticket_yaml


def _ticket(**overrides: object) -> Ticket:
    base: dict[str, object] = {
        "request": RequestSpec(
            dataset="OPRA.PILLAR",
            schema="ohlcv-1d",
            symbols=["SPY.OPT"],
            stype_in="parent",
            start=date(2026, 10, 1),
            end=date(2026, 10, 2),
        ),
        "cost": TicketCost(estimate_usd=1.2197, max_usd=2.0),
        "why": "probe the table shape",
    }
    base.update(overrides)
    return Ticket(**base)  # type: ignore[arg-type]


_ON = datetime(2026, 10, 10, tzinfo=UTC)


# --- the module ---------------------------------------------------------------


def test_draft_name_follows_the_memo_pattern_and_carries_the_request_id():
    name = draft_name(_ticket(), request_id="abc123", on=_ON.date())
    assert name == (
        "2026-10-10-OPRA.PILLAR-ohlcv-1d-"
        "SPY.OPT-2026-10-01-2026-10-02-abc123.yaml"
    )


def test_inbox_dir_defaults_to_the_container_mount(monkeypatch):
    monkeypatch.delenv("REQUESTS_INBOX_DIR", raising=False)
    assert inbox_dir() == Path(DEFAULT_INBOX)
    monkeypatch.setenv("REQUESTS_INBOX_DIR", "/tmp/somewhere-else")
    assert inbox_dir() == Path("/tmp/somewhere-else")


def test_write_draft_lands_under_the_inbox_and_is_a_draft(tmp_path):
    inbox = tmp_path / "inbox"
    path = write_draft(_ticket(), request_id="rid", inbox=inbox, now=_ON)
    assert path.parent == inbox
    assert path.read_text() == render_ticket_yaml(_ticket())
    assert "state: draft" in path.read_text()


def test_write_draft_never_clobbers_an_existing_file(tmp_path):
    first = write_draft(_ticket(), request_id="rid", inbox=tmp_path, now=_ON)
    second = write_draft(_ticket(), request_id="rid", inbox=tmp_path, now=_ON)
    assert first != second
    assert first.exists() and second.exists()


def test_a_symbol_cannot_become_a_path_separator(tmp_path):
    """Sanitising is the thing that keeps the write inside the inbox it was given."""
    ticket = _ticket(
        request=RequestSpec(
            dataset="GLBX.MDP3",
            schema="trades",
            symbols=["A/B"],
            stype_in="raw_symbol",
            start=date(2026, 1, 1),
            end=date(2026, 1, 2),
        )
    )
    path = write_draft(ticket, request_id="r", inbox=tmp_path, now=_ON)
    assert path.parent == tmp_path


# --- the wiring: a real turn files a real draft -------------------------------


def _stub_agent():
    """A stand-in for the Pydantic AI agent: runs offline, returns a Ticket."""
    from types import SimpleNamespace

    class _Stub:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, prompt):  # noqa: ARG002
            return SimpleNamespace(output=_ticket())

    return _Stub()


def test_a_completed_turn_files_the_draft_into_the_inbox(tmp_path, monkeypatch):
    from a2a.helpers.proto_helpers import new_text_part
    from a2a.types.a2a_pb2 import Message, Role, SendMessageRequest
    from google.protobuf.json_format import MessageToDict
    from starlette.testclient import TestClient

    from agent import main

    inbox = tmp_path / "inbox"
    monkeypatch.setenv("REQUESTS_INBOX_DIR", str(inbox))
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path / "state"))

    stub = _stub_agent()
    app = main.create_app(
        agent_executor=main.TicketAgentExecutor(stub), agent_instance=stub
    )
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

    with TestClient(app) as client:
        response = client.post("/", json=body, headers={"A2A-Version": "1.0"})

    assert response.status_code == 200
    drafts = list(inbox.glob("*.yaml"))
    assert len(drafts) == 1
    assert "state: draft" in drafts[0].read_text()
