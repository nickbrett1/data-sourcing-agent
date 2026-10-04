"""Forward LiteLLM context headers from the A2A request to the model calls.

When LiteLLM invokes this agent it sends:

    X-LiteLLM-Trace-Id   groups all LLM calls from one agent execution
    X-LiteLLM-Agent-Id   attributes spend to this agent

Forwarding those headers on the agent's own LLM calls back to LiteLLM is what
makes trace grouping and cost attribution work. Without it both features
silently do nothing - the calls look wired up and produce ungrouped,
unattributed traces.

FastA2A runs each task on a background worker, so the request's HTTP headers are
not in scope by the time the model is called. So:

  1. `LiteLLMHeaderMiddleware` copies the incoming `X-LiteLLM-*` headers into
     the JSON-RPC request `metadata` (which FastA2A carries to the worker);
  2. `HeaderForwardingWorker` reads them back out into a context variable;
  3. `HeaderForwardingClient` injects them on every outbound model request.
"""

from __future__ import annotations

import contextvars
import json

import httpx
from fasta2a.pydantic_ai import AgentWorker

_headers: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "litellm_headers", default={}
)

# The metadata key FastA2A carries from the HTTP request to the worker.
_METADATA_KEY = "litellmHeaders"


def current_litellm_headers() -> dict[str, str]:
    """The LiteLLM context headers for the task being run, if any."""
    return _headers.get()


def _with_metadata(body: bytes, forwarded: dict[str, str]) -> bytes:
    """Put `forwarded` under `params.metadata` in a JSON-RPC request body."""
    try:
        payload = json.loads(body or b"{}")
    except (TypeError, ValueError):
        return body
    params = payload.get("params")
    if not isinstance(params, dict):
        return body
    metadata = params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata[_METADATA_KEY] = forwarded
    params["metadata"] = metadata
    return json.dumps(payload).encode("utf-8")


class LiteLLMHeaderMiddleware:
    """ASGI middleware: carry `X-LiteLLM-*` headers into the request metadata."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        forwarded = {
            key: value for key, value in headers.items() if key.startswith("x-litellm-")
        }
        if not forwarded:
            await self.app(scope, receive, send)
            return

        body = b""
        more = True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)
        body = _with_metadata(body, forwarded)

        async def replay():
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)


class HeaderForwardingWorker(AgentWorker):
    """The FastA2A Pydantic AI worker, with the request's headers in scope."""

    async def run_task(self, params):
        metadata = params.get("metadata") or {}
        forwarded = metadata.get(_METADATA_KEY) or {}
        _headers.set(dict(forwarded))
        await super().run_task(params)


class HeaderForwardingClient(httpx.AsyncClient):
    """An HTTP client that adds the current task's LiteLLM headers."""

    async def send(self, request, **kwargs):
        for key, value in current_litellm_headers().items():
            request.headers.setdefault(key, value)
        return await super().send(request, **kwargs)
