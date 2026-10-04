"""roost fleet client for the runtime A2A agent (hand-written, not generated).

roost is the fleet's mission-control hub (`nickbrett1/roost`). Every agent dials
*out* to it and holds one long-lived WebSocket open at `/agent/ws`; the hub is a
router, not a store. This module is the client that makes the **runtime**
Pydantic AI agent visible in roost's fleet view.

Why a bespoke client instead of reusing a config block
------------------------------------------------------
The dev agent (`<repo>-dev`) appears in roost through `a2a-goose`'s own `hub:`
client - but that is an `a2a-goose` process. The runtime agent is a Pydantic AI
/ A2A server and is not one, so it needs its own small client. This is that
client, and it is the first Python implementation of roost protocol v1 (the only
other implementation is Rust, inside `a2a-goose`). The wire is transcribed from
`roost/src/protocol.rs` and `roost/src/fake_agent.rs`.

The wire, in brief
------------------
Frames are JSON objects dispatched on a `type` tag, camelCase fields:

  agent -> hub   hello | activity | response
  hub -> agent   request | command

`hello` is the first frame after the tunnel opens and carries a **fresh
`bootId` per process start**. Ordering is `(agentId, bootId, seq)`, and `seq`
resets with the boot - so a `bootId` that survives a restart is a correctness
bug, not a detail. The hub polls `status.get` about every 15 s and drops the
socket after three missed intervals, so a long model call must never block the
reply: the read loop here is its own task and never awaits the model.

What this agent claims, and what it refuses to
----------------------------------------------
It advertises `capabilities: ["activity", "status"]` and deliberately **not**
`sessions` / `history`. A Pydantic AI agent has no goose `sessions.db`, so
answering `history.*` would mean inventing a second transcript store and a second
schema - exactly what roost's "router, not a store" principle exists to avoid.
Claim only what you can serve.

Fail-open by construction
-------------------------
A roost outage must never stop the agent serving. Connecting happens in a
background task, every failure is logged and retried with backoff, and the hub
dropping the socket is the *normal* path (it is how the hub signals a dead
tunnel), not an error.

Configuration is entirely environmental. There is no in-code hub default: roost
runs on a different Docker network from an agent on `ai_proxy`, and even `nas`
(the name that works from a devcontainer) does not resolve inside that bridge
network. A guessed default is a silent misconfiguration, so with no
`ROOST_HUB_URL` the client is simply not built. See agent/README.md for what to
set and the candidate addresses.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import websockets

# The roost wire version this client speaks.
PROTOCOL_VERSION = 1

# The agent's declared identity in the fleet. `kind` is freeform to the wire, but
# every live agent today declares "a2a-goose"; a Pydantic AI agent declares a
# distinct value so the fleet can tell the two species apart.
DEFAULT_KIND = "pydantic-agent"
AGENT_VERSION = "0.1.0"

# Capabilities are agent-declared and vary across the fleet. This agent can
# serve presence, a real (if coarse) activity feed, and liveness status - and
# nothing else. It does NOT claim sessions/history: it has no transcript store.
CAPABILITIES = ["activity", "status"]
SKILLS = ["ask"]

# Environment variables. The hub address has NO default (see the module
# docstring); the token is named, never inlined, and read only when the hub
# enforces tokens.
HUB_URL_ENV = "ROOST_HUB_URL"
AGENT_ID_ENV = "ROOST_AGENT_ID"
TOKEN_ENV_ENV = "ROOST_TOKEN_ENV"
ENABLED_ENV = "ROOST_ENABLED"
DEFAULT_TOKEN_ENV = "ROOST_AGENT_TOKEN"

# Reconnect backoff. The hub drops sockets deliberately, so a fast first retry
# after a drop is right; a genuinely unreachable hub backs off to the cap.
BACKOFF_INITIAL_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0


@dataclass(frozen=True)
class RoostConfig:
    """Everything the client needs, resolved from the environment."""

    hub_url: str
    agent_id: str
    token: str | None = None
    token_env: str = DEFAULT_TOKEN_ENV
    kind: str = DEFAULT_KIND
    agent_version: str = AGENT_VERSION

    @classmethod
    def from_env(cls, *, agent_name: str) -> RoostConfig | None:
        """Resolve the config, or None when roost is not configured.

        None is the honest answer when `ROOST_HUB_URL` is unset: there is no hub
        address to guess, and a client that dials a guessed address is worse than
        no client at all. `ROOST_ENABLED=false` also disables it, so a deployment
        can keep the variable set but turn the client off.
        """
        hub_url = os.environ.get(HUB_URL_ENV, "").strip()
        if not hub_url:
            return None
        if os.environ.get(ENABLED_ENV, "true").strip().lower() in {"0", "false", "no", "off"}:
            return None
        agent_id = os.environ.get(AGENT_ID_ENV, "").strip() or agent_name
        token_env = os.environ.get(TOKEN_ENV_ENV, "").strip() or DEFAULT_TOKEN_ENV
        token = os.environ.get(token_env, "").strip() or None
        return cls(hub_url=hub_url, agent_id=agent_id, token=token, token_env=token_env)


@dataclass
class RoostState:
    """The live bits of agent state a `status.get` reports.

    Kept deliberately tiny: the executor bumps `in_flight` around a turn, and the
    status reply reads it. Nothing here awaits anything, which is what keeps the
    status answer from ever queuing behind a model call.
    """

    in_flight: int = 0

    def turn_started(self) -> None:
        self.in_flight += 1

    def turn_finished(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class RoostClient:
    """One long-lived, reconnecting WebSocket to a roost hub.

    `connect` is injectable so tests can drive a local server (or a fake) without
    reaching the real hub. `boot_id` is injectable for the same reason, but the
    production path always mints a fresh one in `__init__`, which is what makes
    "a new process is a new stream" true by construction.
    """

    def __init__(
        self,
        config: RoostConfig,
        *,
        state: RoostState | None = None,
        connect: Callable[..., Any] | None = None,
        boot_id: str | None = None,
        started_at: str | None = None,
        clock: Callable[[], str] | None = None,
        backoff_initial: float = BACKOFF_INITIAL_SECONDS,
        backoff_max: float = BACKOFF_MAX_SECONDS,
    ) -> None:
        self._config = config
        self._state = state or RoostState()
        self._connect = connect or websockets.connect
        self._clock = clock or _utc_now_iso
        # Fresh per process: a bootId that survives a restart corrupts the hub's
        # (agentId, bootId, seq) ordering. Never read this from disk/env.
        self._boot_id = boot_id or str(uuid.uuid4())
        self._started_at = started_at or self._clock()
        # seq is monotonic within a boot and continues across a reconnect (same
        # bootId); it is never reset except by a new process.
        self._seq = 0
        self._ws: Any | None = None
        self._connected = False
        self._send_lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max

    @property
    def boot_id(self) -> str:
        return self._boot_id

    @property
    def connected(self) -> bool:
        return self._connected

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Dial in the background. Returns immediately; failure is not fatal."""
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        """Signal the loop to stop and close the socket. Safe to call twice."""
        self._stopped.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await self._close_socket()

    async def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        self._connected = False
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _run_forever(self) -> None:
        """Connect, serve, drop, back off, repeat - forever, never raising."""
        backoff = self._backoff_initial
        while not self._stopped.is_set():
            try:
                await self._session()
                # The session ran (we were connected) and then the socket ended.
                # That is the hub's normal drop, so retry quickly.
                backoff = self._backoff_initial
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[roost] could not reach {self._config.hub_url}: {exc!r} "
                    f"(retrying in {backoff:.0f}s; the agent keeps serving)",
                    flush=True,
                )
                backoff = min(backoff * 2, self._backoff_max)
            if self._stopped.is_set():
                break
            await self._sleep(backoff)

    async def _sleep(self, seconds: float) -> None:
        """Wait `seconds`, unless stopped first."""
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=seconds)
        except TimeoutError:
            pass

    # -- one connection ----------------------------------------------------

    async def _session(self) -> None:
        headers = None
        if self._config.token:
            headers = {"Authorization": f"Bearer {self._config.token}"}
        async with self._connect(self._config.hub_url, additional_headers=headers) as ws:
            self._ws = ws
            await self._send_hello()
            self._connected = True
            print(
                f"[roost] connected to {self._config.hub_url} as "
                f"'{self._config.agent_id}' (kind={self._config.kind}, "
                f"bootId={self._boot_id})",
                flush=True,
            )
            async for raw in ws:
                await self._handle_frame(raw)
        # The context manager closed the socket; reflect that immediately.
        self._connected = False
        self._ws = None

    async def _send_hello(self) -> None:
        await self._send(
            {
                "type": "hello",
                "agentId": self._config.agent_id,
                "host": socket.gethostname(),
                "kind": self._config.kind,
                "agentVersion": self._config.agent_version,
                "protocolVersion": PROTOCOL_VERSION,
                "bootId": self._boot_id,
                "startedAt": self._started_at,
                "skills": list(SKILLS),
                "capabilities": list(CAPABILITIES),
            }
        )

    # -- hub -> agent frames ----------------------------------------------

    async def _handle_frame(self, raw: str | bytes) -> None:
        """Dispatch one hub frame.

        Rule 1 of the protocol: an unrecognised frame `type` is ignored, never
        fatal - version skew between hub and client is the normal state. But a
        *known* frame that is malformed is not swallowed; it is logged.
        """
        try:
            frame = json.loads(raw)
        except (TypeError, ValueError):
            print(f"[roost] ignoring unparseable frame: {raw!r}", flush=True)
            return
        if not isinstance(frame, dict):
            print(f"[roost] ignoring non-object frame: {raw!r}", flush=True)
            return

        frame_type = frame.get("type")
        if frame_type == "request":
            await self._handle_request(frame)
        elif frame_type == "command":
            await self._handle_command(frame)
        # Anything else (a future frame type) is deliberately ignored.

    async def _handle_request(self, frame: dict) -> None:
        request_id = frame.get("id")
        method = frame.get("method")
        if method == "status.get":
            await self._respond(request_id, ok=True, body=self._status_body())
        else:
            # We advertised only activity/status, so history.* and sessions.* are
            # answered honestly as unsupported rather than with an invented body.
            await self._respond(
                request_id,
                ok=False,
                error=f"unsupported method {method!r} (this agent advertises {CAPABILITIES})",
            )

    async def _handle_command(self, frame: dict) -> None:
        # M4 commands (reboot) are not built in the hub yet. Do not pretend.
        await self._respond(
            frame.get("id"),
            ok=False,
            error=f"unsupported action {frame.get('action')!r}",
        )

    async def _respond(
        self, request_id: Any, *, ok: bool, body: dict | None = None, error: str | None = None
    ) -> None:
        response: dict[str, Any] = {"type": "response", "id": request_id, "ok": ok}
        if body is not None:
            response["body"] = body
        if error is not None:
            response["error"] = error
        await self._send(response)

    def _status_body(self) -> dict:
        """The `status.get` body, in the shape the fleet view reads.

        Every field is cheap to compute here; nothing touches the model or the
        network, so this always answers within one poll interval.
        """
        busy = self._state.in_flight > 0
        return {
            "activity": {
                "enabled": True,
                "backlog": 0,
                "subscribers": 1 if self._connected else 0,
            },
            "acp": {
                "state": "busy" if busy else "ready",
                "inFlight": self._state.in_flight,
                "pid": os.getpid(),
                "restarts": 0,
            },
            # We do not claim `sessions`; the field is present because the fleet
            # view reads it field-by-field, but the count is honestly zero.
            "sessions": {"count": 0, "retained": 0},
            "registry": {"registered": True},
        }

    # -- agent -> hub activity --------------------------------------------

    async def emit_activity(self, event_type: str, **fields: Any) -> None:
        """Publish one activity event, best-effort.

        `event` is opaque JSON to the hub (it forwards it verbatim), so this
        speaks the coarse vocabulary the fleet's UI already folds:
        request_received / turn_started / answer / finished. A dropped frame is
        not buffered and not an error: activity is a feed, not a queue.
        """
        async with self._send_lock:
            ws = self._ws
            if ws is None or not self._connected:
                return
            frame = {
                "type": "activity",
                "bootId": self._boot_id,
                "seq": self._seq,
                "at": self._clock(),
                "event": {"type": event_type, **fields},
            }
            try:
                await ws.send(json.dumps(frame))
                self._seq += 1
            except Exception as exc:  # a feed must never break the agent
                print(f"[roost] dropping activity frame after send error: {exc!r}", flush=True)

    # -- low-level send ----------------------------------------------------

    async def _send(self, frame: dict) -> None:
        async with self._send_lock:
            ws = self._ws
            if ws is None:
                return
            await ws.send(json.dumps(frame))


class RoostBridge:
    """The seam between the A2A server and the roost client.

    The executor holds a bridge so it can (a) reflect turn activity in
    `status.get` and (b) emit a coarse activity feed, without knowing whether
    roost is configured at all. This is the fail-open shape: with no
    `ROOST_HUB_URL` the bridge is still a valid no-op object, so no caller has to
    branch on "is roost on?".
    """

    def __init__(self, client: RoostClient | None = None, state: RoostState | None = None) -> None:
        self.state = state or RoostState()
        self._client = client

    @classmethod
    def from_env(cls, *, agent_name: str) -> RoostBridge:
        config = RoostConfig.from_env(agent_name=agent_name)
        if config is None:
            return cls()
        state = RoostState()
        return cls(client=RoostClient(config, state=state), state=state)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def start(self) -> None:
        if self._client is not None:
            self._client.start()

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.stop()

    def turn_started(self) -> None:
        self.state.turn_started()

    def turn_finished(self) -> None:
        self.state.turn_finished()

    async def emit(self, event_type: str, **fields: Any) -> None:
        if self._client is not None:
            await self._client.emit_activity(event_type, **fields)
