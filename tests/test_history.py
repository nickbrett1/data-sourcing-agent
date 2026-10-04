"""Tests for durable session history (`agent/history.py`) and its roost wiring.

The store under test is real: every test drives an actual
`a2a.server.tasks.DatabaseTaskStore` over a real SQLite file, and the wire tests
drive a real loopback WebSocket - the same path production uses. Nothing here is
a mock, because the thing being added is precisely "does the transcript the A2A
server already keeps reach the roost UI, and does it survive a restart".

The contracts pinned here are the ones roost's History panel reads (verified in
`nickbrett1/roost`): `history.sessions` -> `{sessions:[{sessionId,name,createdAt,
updatedAt,messageCount}]}`; `history.messages` -> `{messages:[{index,role,text,
createdAt}],nextCursor}`; `history.search` -> `{matches:[{sessionId,name,matches}]}`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import websockets
from a2a.helpers.proto_helpers import new_task_from_user_message
from a2a.server.context import ServerCallContext
from a2a.server.tasks import DatabaseTaskStore
from a2a.types.a2a_pb2 import Message, Role
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.testclient import TestClient

from agent import main
from agent.history import TaskHistory, build_task_store, task_store_path
from agent.roost import RoostClient, RoostConfig
from agent.ticket import RequestSpec, StypeIn, Ticket, TicketCost

# --- harness ------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _real_store(path):
    """A real DatabaseTaskStore over a real SQLite file at `path`."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        yield DatabaseTaskStore(engine=engine)
    finally:
        await engine.dispose()


async def _save_turn(
    store,
    *,
    context_id,
    task_id,
    user_text,
    agent_text=None,
    when=None,
):
    """Save one turn the way the A2A pipeline does: a user message, an artifact."""
    message = Message(message_id=f"m-{task_id}", role=Role.ROLE_USER, context_id=context_id)
    message.parts.add().text = user_text
    task = new_task_from_user_message(message)
    task.id = task_id
    task.context_id = context_id
    if when is not None:
        task.status.timestamp.FromDatetime(when)
    if agent_text is not None:
        artifact = task.artifacts.add()
        artifact.name = "ticket"
        artifact.parts.add().text = agent_text
    await store.save(task, ServerCallContext())


def _ticket() -> Ticket:
    return Ticket(
        request=RequestSpec(
            dataset="OPRA.PILLAR",
            schema="ohlcv-1d",
            symbols=["SPY.OPT"],
            stype_in=StypeIn.parent,
            start=datetime(2026, 10, 1, tzinfo=UTC).date(),
            end=datetime(2026, 10, 2, tzinfo=UTC).date(),
        ),
        cost=TicketCost(estimate_usd=1.22, max_usd=2.0),
        why="sample",
    )


# --- the store and its metadata views -----------------------------------------


def test_sessions_are_grouped_by_context_newest_first(tmp_path):
    older = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    newer = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            await _save_turn(
                store, context_id="ctx-A", task_id="a1", user_text="first request", when=older
            )
            await _save_turn(
                store, context_id="ctx-A", task_id="a2", user_text="follow-up", when=newer
            )
            await _save_turn(
                store, context_id="ctx-B", task_id="b1", user_text="another session",
                when=newer + timedelta(hours=1),
            )
            return await TaskHistory(store).sessions()

    sessions = asyncio.run(run())
    assert [s["sessionId"] for s in sessions] == ["ctx-B", "ctx-A"]
    ctx_a = sessions[1]
    # Title is the FIRST user message, turnCount the number of turns, and the
    # session spans its turns in time.
    assert ctx_a["name"] == "first request"
    assert ctx_a["turnCount"] == 2
    assert ctx_a["createdAt"] == older.isoformat()
    assert ctx_a["updatedAt"] == newer.isoformat()
    assert ctx_a["retained"] is True


def test_a_long_first_message_is_truncated_for_the_title(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            await _save_turn(
                store, context_id="ctx-A", task_id="a1", user_text="x" * 200
            )
            return (await TaskHistory(store).sessions())[0]

    session = asyncio.run(run())
    assert len(session["name"]) <= 80
    assert session["name"].endswith("\u2026")


def test_messages_are_the_user_prompt_and_the_agent_answer(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            when = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
            await _save_turn(
                store, context_id="ctx-A", task_id="a1",
                user_text="buy SPY options data", agent_text="state: draft\nwhy: sample",
                when=when,
            )
            return await TaskHistory(store).messages("ctx-A")

    page = asyncio.run(run())
    roles = [m["role"] for m in page["messages"]]
    assert roles == ["user", "assistant"]
    assert page["messages"][0]["text"] == "buy SPY options data"
    assert "state: draft" in page["messages"][1]["text"]
    assert [m["index"] for m in page["messages"]] == [0, 1]
    assert page["nextCursor"] is None
    assert page["messages"][0]["createdAt"] == datetime(2026, 10, 1, 9, 0, tzinfo=UTC).isoformat()


def test_messages_paginate_forward_with_an_index_cursor(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            when = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
            for i in range(4):
                await _save_turn(
                    store, context_id="ctx-A", task_id=f"a{i}", user_text=f"prompt {i}",
                    agent_text=f"answer {i}", when=when + timedelta(minutes=i),
                )
            history = TaskHistory(store)
            first = await history.messages("ctx-A", limit=3)
            second = await history.messages("ctx-A", cursor=first["nextCursor"], limit=3)
            return first, second

    first, second = asyncio.run(run())
    assert len(first["messages"]) == 3
    assert first["nextCursor"] == "3"
    assert [m["index"] for m in second["messages"]] == [3, 4, 5]
    assert second["nextCursor"] == "6"  # 8 messages total (4 turns x 2)
    assert second["messages"][0]["text"] == "answer 1"


def test_search_matches_message_text_grouped_by_session(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            when = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
            await _save_turn(
                store, context_id="ctx-A", task_id="a1", user_text="the SPY options chain",
                agent_text="ticket for SPY", when=when,
            )
            await _save_turn(
                store, context_id="ctx-B", task_id="b1", user_text="futures data", when=when,
            )
            history = TaskHistory(store)
            return await history.search("spy"), await history.search("nomatch"), await history.search("")

    found, none, empty = asyncio.run(run())
    assert [m["sessionId"] for m in found["matches"]] == ["ctx-A"]
    assert len(found["matches"][0]["matches"]) == 2  # user prompt + agent answer
    assert none["matches"] == []
    assert empty["matches"] == []


def test_status_reports_the_sessions_present(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            await _save_turn(store, context_id="ctx-A", task_id="a1", user_text="x")
            await _save_turn(store, context_id="ctx-B", task_id="b1", user_text="y")
            return await TaskHistory(store).status()

    assert asyncio.run(run()) == {"count": 2, "retained": 2}


# --- durability and retention --------------------------------------------------


def test_history_survives_closing_and_reopening_the_database(tmp_path):
    """The whole point of the store swap: a recreate must not lose the transcript."""
    db = tmp_path / "tasks.db"

    async def write():
        async with _real_store(db) as store:
            await _save_turn(store, context_id="ctx-A", task_id="a1", user_text="persist me")

    async def read():
        async with _real_store(db) as store:
            return await TaskHistory(store).sessions()

    asyncio.run(write())
    sessions = asyncio.run(read())
    assert [s["sessionId"] for s in sessions] == ["ctx-A"]
    assert sessions[0]["name"] == "persist me"


def test_retention_sweep_drops_the_oldest_sessions_beyond_the_count(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            base = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
            for i in range(4):
                await _save_turn(
                    store, context_id=f"ctx-{i}", task_id=f"t{i}", user_text=f"s{i}",
                    when=base + timedelta(hours=i),
                )
            history = TaskHistory(store, max_sessions=2, max_age_days=3650)
            removed = await history.retention_sweep()
            remaining = [s["sessionId"] for s in await history.sessions()]
            return removed, remaining

    removed, remaining = asyncio.run(run())
    assert removed == 2  # the two oldest contexts, one task each
    assert remaining == ["ctx-3", "ctx-2"]


def test_retention_sweep_drops_sessions_older_than_the_age_bound(tmp_path):
    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            now = datetime.now(UTC)
            await _save_turn(
                store, context_id="old", task_id="o1", user_text="stale",
                when=now - timedelta(days=40),
            )
            await _save_turn(
                store, context_id="new", task_id="n1", user_text="fresh", when=now,
            )
            history = TaskHistory(store, max_sessions=1000, max_age_days=30)
            removed = await history.retention_sweep()
            return removed, [s["sessionId"] for s in await history.sessions()]

    removed, remaining = asyncio.run(run())
    assert removed == 1
    assert remaining == ["new"]


# --- wiring: a real turn lands in the durable store ----------------------------


class _StubAgent:
    """A minimal stand-in for the Pydantic AI agent that returns a ticket."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, prompt):  # noqa: ARG002
        class _Result:
            output = _ticket()

        return _Result()


def test_a_turn_through_the_app_is_readable_as_history(tmp_path):
    """End to end: send a real A2A message, then read it back as a session.

    This is the claim the whole change rests on - the transcript roost reads is
    the A2A server's own state, not a second store.
    """
    db = tmp_path / "tasks.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db}")
    store = DatabaseTaskStore(engine=engine)

    stub = _StubAgent()
    app = main.create_app(
        agent_executor=main.TicketAgentExecutor(stub, main.RoostBridge()),
        agent_instance=stub,
        task_store=store,
    )
    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "user",
                "parts": [{"kind": "text", "text": "buy SPY options data"}],
            },
        },
    }
    with TestClient(app) as client:
        assert client.post("/", json=body).status_code == 200

    async def read():
        history = TaskHistory(store)
        sessions = await history.sessions()
        session_id = sessions[0]["sessionId"]
        page = await history.messages(session_id)
        await engine.dispose()
        return sessions, page

    sessions, page = asyncio.run(read())
    assert len(sessions) == 1
    assert sessions[0]["name"] == "buy SPY options data"
    assert sessions[0]["turnCount"] == 1
    roles = [m["role"] for m in page["messages"]]
    assert roles == ["user", "assistant"]
    assert "state: draft" in page["messages"][1]["text"]


# --- the wire: real loopback WebSocket ----------------------------------------


@contextlib.asynccontextmanager
async def _roost_server(handler):
    server = await websockets.serve(handler, "127.0.0.1", 0)
    try:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        server.close()
        await server.wait_closed()


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def test_history_methods_answer_over_the_wire(tmp_path):
    frames: list[dict] = []

    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            when = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
            await _save_turn(
                store, context_id="ctx-A", task_id="a1", user_text="buy SPY options data",
                agent_text="state: draft", when=when,
            )
            history = TaskHistory(store)

            async def handler(ws):
                frames.append(json.loads(await ws.recv()))  # hello
                await ws.send(json.dumps(
                    {"type": "request", "id": "s", "method": "history.sessions", "params": {"limit": 5}}
                ))
                frames.append(json.loads(await ws.recv()))
                await ws.send(json.dumps(
                    {"type": "request", "id": "m", "method": "history.messages", "params": {"id": "ctx-A"}}
                ))
                frames.append(json.loads(await ws.recv()))
                await ws.send(json.dumps(
                    {"type": "request", "id": "q", "method": "history.search", "params": {"q": "spy"}}
                ))
                frames.append(json.loads(await ws.recv()))
                await ws.send(json.dumps(
                    {"type": "request", "id": "l", "method": "sessions.list", "params": {}}
                ))
                frames.append(json.loads(await ws.recv()))
                await ws.close()

            async with _roost_server(handler) as url:
                client = RoostClient(
                    RoostConfig(hub_url=url, agent_id="t"),
                    history=history,
                    backoff_initial=0.05,
                    backoff_max=0.05,
                )
                client.start()
                await _wait_until(lambda: len(frames) >= 5)
                await client.stop()

    asyncio.run(run())
    by_id = {f["id"]: f for f in frames[1:]}
    assert by_id["s"]["ok"] is True
    assert by_id["s"]["body"]["sessions"][0]["sessionId"] == "ctx-A"
    assert by_id["s"]["body"]["sessions"][0]["name"] == "buy SPY options data"
    assert by_id["m"]["body"]["messages"][0]["text"] == "buy SPY options data"
    assert by_id["m"]["body"]["nextCursor"] is None
    assert by_id["q"]["body"]["matches"][0]["sessionId"] == "ctx-A"
    assert by_id["l"]["body"]["count"] == 1


def test_history_survives_a_client_reconnect(tmp_path):
    """The hub drops sockets as normal; a reconnect must still see history."""
    responses: list[dict] = []
    connections = 0

    async def run():
        async with _real_store(tmp_path / "tasks.db") as store:
            await _save_turn(store, context_id="ctx-A", task_id="a1", user_text="kept")

            async def handler(ws):
                nonlocal connections
                connections += 1
                await ws.recv()  # hello
                await ws.send(json.dumps(
                    {"type": "request", "id": f"r{connections}", "method": "history.sessions"}
                ))
                responses.append(json.loads(await ws.recv()))
                await ws.close()  # the hub's deliberate drop -> client reconnects

            async with _roost_server(handler) as url:
                client = RoostClient(
                    RoostConfig(hub_url=url, agent_id="t"),
                    history=TaskHistory(store),
                    backoff_initial=0.05,
                    backoff_max=0.05,
                )
                client.start()
                await _wait_until(lambda: connections >= 2, timeout=5.0)
                await client.stop()

    asyncio.run(run())
    assert connections >= 2
    # Both answers are the same: history lives in the store, not the connection.
    assert all(r["body"]["sessions"][0]["sessionId"] == "ctx-A" for r in responses)


# --- store construction --------------------------------------------------------


def test_task_store_path_follows_agent_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path))
    assert task_store_path("data-sourcing-agent") == tmp_path / "a2a_tasks.db"


def test_build_task_store_returns_none_instead_of_raising_on_a_bad_path(monkeypatch, tmp_path):
    """Fail-open: an unusable state path must not take the agent down."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("AGENT_STATE_DIR", str(blocker / "sub"))
    assert build_task_store("data-sourcing-agent") is None
