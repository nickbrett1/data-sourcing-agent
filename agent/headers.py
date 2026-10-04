"""Forward LiteLLM context headers from the A2A request to the model calls.

When LiteLLM invokes this agent it sends:

    X-LiteLLM-Trace-Id   groups all LLM calls from one agent execution
    X-LiteLLM-Agent-Id   attributes spend to this agent

Forwarding those headers on the agent's own LLM calls back to LiteLLM is what
makes trace grouping and cost attribution work. Without it both features
silently do nothing - the calls look wired up and produce ungrouped,
unattributed traces.

Under FastA2A the task ran on a background broker with no access to the HTTP
request, so the headers had to be copied into the JSON-RPC `metadata` and read
back out by a custom worker. The a2a-sdk `AgentExecutor` runs with the request's
`ServerCallContext` in hand, and the SDK's default context builder records the
request headers in `call_context.state['headers']` - so the executor reads them
directly (`capture_litellm_headers`) and puts them in a context variable that
`HeaderForwardingClient` injects on every outbound model request:

  1. `TicketAgentExecutor.execute` calls `capture_litellm_headers` with the
     request headers from the call context;
  2. `HeaderForwardingClient` injects them on every outbound model request.
"""

from __future__ import annotations

import contextvars
from collections.abc import Mapping

import httpx

_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "litellm_headers", default=None
)


def current_litellm_headers() -> dict[str, str]:
    """The LiteLLM context headers for the task being run, if any."""
    return _headers.get() or {}


def capture_litellm_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Record the request's `X-LiteLLM-*` headers for the current task.

    Header names are lower-cased before matching: Starlette hands the SDK a
    lower-cased header mapping, but a caller (or a test) may not.
    """
    forwarded = {
        str(key).lower(): value
        for key, value in (headers or {}).items()
        if str(key).lower().startswith("x-litellm-")
    }
    _headers.set(forwarded)
    return forwarded


class HeaderForwardingClient(httpx.AsyncClient):
    """An HTTP client that adds the current task's LiteLLM headers."""

    async def send(self, request, **kwargs):
        for key, value in current_litellm_headers().items():
            request.headers.setdefault(key, value)
        return await super().send(request, **kwargs)
