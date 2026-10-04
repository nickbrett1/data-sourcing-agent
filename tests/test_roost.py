"""Tests for the roost fleet client (`agent/roost.py`).

These are the tests that make the roost client provable rather than merely
plausible: the roost wire has one other implementation, in Rust, and the hub
*carries unknown frames silently* - so a client that emits a wrong `hello` or a
wrong `status.get` body fails with no error anywhere. Every frame shape and
protocol rule this client claims is pinned here.

They drive a real local WebSocket server (`websockets.serve` on loopback), not a
mock, so the test exercises the same `websockets` send/receive path production
does. Nothing here touches the real hub or the network beyond loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest
import websockets

from agent.roost import (
    CAPABILITIES,
    PROTOCOL_VERSION,
    RoostBridge,
    RoostClient,
    RoostConfig,
)

# --- harness ------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _roost_server(handler):
    """A real loopback WebSocket server on an ephemeral port."""
    server = await websockets.serve(handler, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


def _client(url: str, **kwargs) -> RoostClient:
    config = RoostConfig(hub_url=url, agent_id="test-agent")
    return RoostClient(config, **kwargs)


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


# --- configuration ------------------------------------------------------------


def test_config_is_none_without_a_hub_url(monkeypatch):
    """No `ROOST_HUB_URL` means no client: the address is not guessable."""
    monkeypatch.delenv("ROOST_HUB_URL", raising=False)
    assert RoostConfig.from_env(agent_name="data-sourcing-agent") is None


def test_config_reads_hub_id_and_named_token(monkeypatch):
    monkeypatch.setenv("ROOST_HUB_URL", "ws://roost:3000/agent/ws")
    monkeypatch.setenv("ROOST_AGENT_ID", "data-sourcing-agent-roosttest")
    monkeypatch.setenv("ROOST_TOKEN_ENV", "MY_ROOST_TOKEN")
    monkeypatch.setenv("MY_ROOST_TOKEN", "s3cret")

    config = RoostConfig.from_env(agent_name="data-sourcing-agent")

    assert config is not None
    assert config.hub_url == "ws://roost:3000/agent/ws"
    assert config.agent_id == "data-sourcing-agent-roosttest"
    assert config.token == "s3cret"
    assert config.token_env == "MY_ROOST_TOKEN"


def test_config_defaults_the_agent_id_to_the_agent_name(monkeypatch):
    monkeypatch.setenv("ROOST_HUB_URL", "ws://roost:3000/agent/ws")
    monkeypatch.delenv("ROOST_AGENT_ID", raising=False)
    config = RoostConfig.from_env(agent_name="data-sourcing-agent")
    assert config is not None
    assert config.agent_id == "data-sourcing-agent"
    assert config.token is None  # no token env → anonymous dial


def test_config_can_be_disabled_without_unsetting_the_url(monkeypatch):
    monkeypatch.setenv("ROOST_HUB_URL", "ws://roost:3000/agent/ws")
    monkeypatch.setenv("ROOST_ENABLED", "false")
    assert RoostConfig.from_env(agent_name="data-sourcing-agent") is None


def test_a_bearer_token_is_sent_only_when_configured():
    """The dial carries `Authorization: Bearer <token>` iff a token is set."""
    seen: dict[str, str] = {}

    async def handler(ws):
        seen.update({k.lower(): v for k, v in ws.request.headers.items()})
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            config = RoostConfig(hub_url=url, agent_id="t", token="tok")
            client = RoostClient(config, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: "authorization" in seen)
            await client.stop()

    asyncio.run(run())
    assert seen["authorization"] == "Bearer tok"


# --- bootId and the hello frame ----------------------------------------------


def test_boot_id_is_fresh_per_process_and_stable_within_one():
    a = RoostClient(RoostConfig(hub_url="ws://x/agent/ws", agent_id="t"))
    b = RoostClient(RoostConfig(hub_url="ws://x/agent/ws", agent_id="t"))
    assert a.boot_id and b.boot_id
    assert a.boot_id != b.boot_id
    assert a.boot_id == a.boot_id  # stable for the life of the client


def test_hello_frame_has_the_protocol_shape():
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: frames)
            await client.stop()

    asyncio.run(run())
    hello = frames[0]
    assert hello["type"] == "hello"
    assert hello["agentId"] == "test-agent"
    assert hello["protocolVersion"] == PROTOCOL_VERSION == 1
    assert hello["kind"] == "pydantic-agent"
    assert hello["bootId"]  # required; freshness is pinned by the next test
    assert "startedAt" in hello
    assert hello["skills"] == ["ask"]
    # Presence, status, AND session history: this agent now answers `history.*`
    # from the A2A server's own task store, so it claims `sessions` - the same
    # token the `a2a-goose` agents advertise, so roost's UI history logic applies.
    assert hello["capabilities"] == CAPABILITIES == ["activity", "status", "sessions"]
    assert "sessions" in hello["capabilities"]
    # `seq` belongs to activity frames, not hello.
    assert "seq" not in hello


def test_hello_boot_id_matches_the_client_property():
    frames: list[dict] = []
    holder: dict[str, RoostClient] = {}

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            holder["client"] = client
            client.start()
            await _wait_until(lambda: frames)
            await client.stop()

    asyncio.run(run())
    assert frames[0]["bootId"] == holder["client"].boot_id


# --- status.get ---------------------------------------------------------------


def test_status_get_is_answered_with_a_parseable_body():
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))  # hello
        await ws.send(json.dumps({"type": "request", "id": "r1", "method": "status.get", "params": {}}))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: len(frames) >= 2)
            await client.stop()

    asyncio.run(run())
    response = frames[1]
    assert response["type"] == "response"
    assert response["id"] == "r1"
    assert response["ok"] is True
    body = response["body"]
    # The exact shape the fleet view reads field-by-field.
    assert body["activity"]["enabled"] is True
    assert body["acp"]["state"] in {"ready", "busy"}
    assert body["acp"]["inFlight"] == 0
    assert body["sessions"] == {"count": 0, "retained": 0}
    assert body["registry"] == {"registered": True}


def test_in_flight_turn_is_reflected_but_never_blocks_the_reply():
    """A status poll must be answered even while a turn is 'in progress'.

    The reply is produced by the read loop, which never awaits a model call, so
    the hub's ~15 s poll cannot time out - the failure that would drop the
    tunnel after three missed intervals.
    """
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "request", "id": "r1", "method": "status.get"}))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client._state.turn_started()  # noqa: SLF001 - the bridge bumps this in prod
            client.start()
            await _wait_until(lambda: len(frames) >= 2)
            await client.stop()

    asyncio.run(run())
    body = frames[1]["body"]
    assert body["acp"]["inFlight"] == 1
    assert body["acp"]["state"] == "busy"


def test_an_unclaimed_method_is_refused_honestly():
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "request", "id": "r2", "method": "logs.tail"}))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: len(frames) >= 2)
            await client.stop()

    asyncio.run(run())
    response = frames[1]
    assert response["ok"] is False
    assert "logs.tail" in response["error"]


def test_a_claimed_history_method_refuses_when_no_store_is_configured():
    """Fail-open, and honest: no store means an explicit refusal, not a lie.

    A client with no history provider still *advertises* `sessions` (the store is
    built at app startup), but before/without one a claimed method must answer
    `ok:false` rather than an empty transcript.
    """
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "request", "id": "r9", "method": "history.sessions"}))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: len(frames) >= 2)
            await client.stop()

    asyncio.run(run())
    response = frames[1]
    assert response["ok"] is False
    assert "history.sessions" in response["error"]


def test_an_unknown_frame_type_is_ignored_not_fatal():
    """Protocol rule 1: version skew is normal; unknown frames are ignored."""
    frames: list[dict] = []

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))
        await ws.send(json.dumps({"type": "some_future_frame", "anything": 1}))
        await ws.send(json.dumps({"type": "request", "id": "r3", "method": "status.get"}))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: len(frames) >= 2)
            await client.stop()

    asyncio.run(run())
    # The unknown frame did not stop the client from answering the next request.
    assert frames[1]["id"] == "r3"
    assert frames[1]["ok"] is True


# --- activity -----------------------------------------------------------------


def test_activity_frames_are_boot_stamped_and_monotonic():
    frames: list[dict] = []
    holder: dict[str, RoostClient] = {}

    async def handler(ws):
        frames.append(json.loads(await ws.recv()))  # hello
        # Wait for two activity frames from the client side.
        frames.append(json.loads(await ws.recv()))
        frames.append(json.loads(await ws.recv()))
        await ws.close()

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            holder["client"] = client
            client.start()
            await _wait_until(lambda: client.connected)
            await client.emit_activity("turn_started", taskId="t1")
            await client.emit_activity("finished", taskId="t1")
            await _wait_until(lambda: len(frames) >= 3)
            await client.stop()

    asyncio.run(run())
    first, second = frames[1], frames[2]
    assert first["type"] == "activity"
    assert first["event"] == {"type": "turn_started", "taskId": "t1"}
    assert second["event"] == {"type": "finished", "taskId": "t1"}
    assert first["seq"] == 0
    assert second["seq"] == 1
    assert first["bootId"] == second["bootId"] == holder["client"].boot_id


# --- fail-open and reconnect --------------------------------------------------


def test_a_dead_hub_does_not_raise_and_stop_is_clean():
    """Fail-open: an unreachable hub must not surface as an agent failure."""

    async def run() -> None:
        # Nothing listens on this port; connecting fails every time.
        config = RoostConfig(hub_url="ws://127.0.0.1:1/agent/ws", agent_id="t")
        client = RoostClient(config, backoff_initial=0.05, backoff_max=0.1)
        client.start()
        await asyncio.sleep(0.3)  # several failed attempts would have raised by now
        assert client.connected is False
        await client.stop()  # must not hang or raise

    asyncio.run(run())


def test_the_client_reconnects_after_the_hub_drops_the_socket():
    """The hub drops sockets to signal a dead tunnel; reconnect is normal."""
    connections = 0
    ready = asyncio.Event()

    async def handler(ws):
        nonlocal connections
        connections += 1
        if connections >= 2:
            ready.set()
        await ws.close()  # the hub's deliberate drop

    async def run() -> None:
        async with _roost_server(handler) as url:
            client = _client(url, backoff_initial=0.05, backoff_max=0.05)
            client.start()
            await _wait_until(lambda: connections >= 2, timeout=5.0)
            await client.stop()

    asyncio.run(run())
    assert connections >= 2


# --- the bridge ---------------------------------------------------------------


def test_bridge_is_a_noop_when_roost_is_unconfigured(monkeypatch):
    monkeypatch.delenv("ROOST_HUB_URL", raising=False)

    async def run() -> None:
        bridge = RoostBridge.from_env(agent_name="data-sourcing-agent")
        assert bridge.enabled is False
        bridge.turn_started()
        assert bridge.state.in_flight == 1
        bridge.turn_finished()
        assert bridge.state.in_flight == 0
        await bridge.start()  # no-op
        await bridge.emit("turn_started")  # no-op
        await bridge.stop()  # no-op

    asyncio.run(run())


def test_bridge_builds_a_client_when_configured(monkeypatch):
    monkeypatch.setenv("ROOST_HUB_URL", "ws://roost:3000/agent/ws")
    bridge = RoostBridge.from_env(agent_name="data-sourcing-agent")
    assert bridge.enabled is True


@pytest.mark.parametrize("enabled", ["false", "0", "no", "off"])
def test_bridge_is_disabled_by_roost_enabled(monkeypatch, enabled):
    monkeypatch.setenv("ROOST_HUB_URL", "ws://roost:3000/agent/ws")
    monkeypatch.setenv("ROOST_ENABLED", enabled)
    assert RoostBridge.from_env(agent_name="data-sourcing-agent").enabled is False


# --- wiring into the A2A app --------------------------------------------------


class _RecordingBridge(RoostBridge):
    """A bridge that records the lifecycle calls the app makes on it."""

    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.stopped = 0
        self.events: list[tuple[str, int]] = []

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def emit(self, event_type: str, **fields: object) -> None:
        self.events.append((event_type, self.state.in_flight))


def test_create_app_runs_the_bridge_across_the_app_lifespan():
    """Startup dials (and shutdown stops) the hub alongside the A2A server.

    The turn hooks also fire: a failed turn still brackets `turn_started` /
    `finished`, so `in_flight` never leaks and the feed is not silently skipped
    on the error path.
    """
    from starlette.testclient import TestClient

    from agent import main

    class _StubAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, prompt):  # noqa: ARG002
            raise RuntimeError("boom")

    bridge = _RecordingBridge()
    stub = _StubAgent()
    app = main.create_app(
        agent_executor=main.TicketAgentExecutor(stub, bridge),
        agent_instance=stub,
        roost_bridge=bridge,
    )

    body = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "m1",
                "role": "user",
                "parts": [{"kind": "text", "text": "hi"}],
            },
        },
    }
    with TestClient(app) as client:
        assert bridge.started == 1
        response = client.post("/", json=body)
        assert response.status_code == 200
    assert bridge.stopped == 1

    assert [e for e, _ in bridge.events] == ["turn_started", "finished"]
    assert bridge.state.in_flight == 0
