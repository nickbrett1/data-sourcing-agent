"""Filing the validated ticket — the only place this agent creates a request row.

Why this is code and not a tool the model holds. `submit_request` exists on the
mcphub `data-sourcing` group's `data-download` server, and until this module
existed it was on the toolset the model reasons over, so the model could call it
at *any* moment of the loop — including before `validate_agent_result` had priced
the request, and including again after a `ModelRetry`. With an output retry
budget of three, "file, get told off, file again, get told off again, fail" is a
reachable path, and it leaves rows behind that nothing validated. The instruction
text said "file only after you have checked"; the architecture said "file
whenever you like".

So the tool is filtered off the model's toolset (see `AGENT_TOOLSETS` in
`agent/main.py`) and the filing happens here, from the `Ticket` that
`validate_agent_result` returned. The door becomes literal: a body reaches
data-download only if a priced, legitimate Ticket produced it, and a validation
failure files nothing at all rather than filing something and complaining after.

What this module therefore owns, and the model does not:

* **The mapping** between the two schemas — `Ticket` to a `/v1/requests` plan
  body. The two shapes look alike and are not alike (the plan needs a `vendor`
  the ticket has no field for; the ticket carries a `state` the plan must never
  set), so the conversion lives in one place rather than in a prompt.
* **`verify.on_failure`.** The model has no field for it and must not acquire
  one: it decides what happens to bytes that fail the vendor-manifest check,
  which is a policy about data integrity, not a preference about a download.
* **Provenance.** `filed_by` names this agent, so the book's trail distinguishes
  a filing that came through the door from one that did not, and
  `idempotency_key` is the turn id, so a repeated turn cannot file the same
  request twice.

The transport is MCP against mcphub, not `/v1` directly: the token stays in
mcphub's configuration, and this agent holds no data-download credential of its
own.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp import Client

from agent.ticket import Ticket

# Where the group lives. `data-sourcing`, not `data`: the shared read group does
# not carry the write at all, so this URL is the only one that can file.
MCP_URL = os.environ.get("DATA_SOURCING_MCP_URL", "http://mcphub:3000/mcp/data-sourcing")

# The tool as mcphub presents it: prefixed with its server name.
SUBMIT_TOOL = "data-download-submit_request"

# The one plan type this deployment ships, and the one vendor this agent speaks
# for. Both are constants rather than model output: a ticket that named a
# different plan type would be a ticket for a different conversation.
PLAN_TYPE = "market_data_window"
VENDOR = "Databento"

# What to do with bytes that fail the vendor-manifest check. `quarantine` keeps a
# bad artifact out of the way rather than committing it with a flag: this agent
# files on a person's behalf, and the one thing it must never do is leave
# questionable bytes sitting somewhere that looks like `landed`.
ON_FAILURE = "quarantine"

# Who to blame in the trail. The book records this on the row and in its
# `submitted` event; `mcp` (the default) would make a validated filing
# indistinguishable from any other caller of the tool.
FILED_BY = "data-sourcing-agent"


class FilingRefused(RuntimeError):
    """The runtime refused the body, or could not be reached. The text is its own.

    data-download writes a sentence with a code and, where it can, the legal
    values — "the catalogue the vendor describes this dataset with says which
    schemas it carries" and so on. That sentence is more useful to a human than
    anything this module could summarise, so it is carried through unchanged.
    """


def plan_body(ticket: Ticket) -> dict[str, Any]:
    """One `Ticket` as the `/v1/requests` plan body that expresses the same ask.

    `params` is field-for-field the ticket's `RequestSpec` plus the vendor, which
    is a property of the seller rather than of the question and so has no place
    in `RequestSpec`. Dates are written `YYYY-MM-DD`: the plan's window is
    absolute and half-open, and a `date` already means "midnight UTC" to both
    sides, so adding a time would be inventing precision nobody asked for.
    """
    request = ticket.request
    return {
        "kind": "plan",
        "plan_type": PLAN_TYPE,
        "params": {
            "vendor": VENDOR,
            "dataset": request.dataset,
            "schema": request.schema_name,
            "symbols": list(request.symbols),
            "stype_in": request.stype_in.value,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
        },
        "verify": {"on_failure": ON_FAILURE},
        "max_usd": ticket.cost.max_usd,
        "why": ticket.why,
    }


def _row_id(result: Any) -> str:
    """The filed request's id, from the tool's structured answer.

    Fails loudly rather than returning `None`: a filing that cannot name its row
    is not a filing, and a ticket rendered without the id would read as though
    nothing had been created.
    """
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        request_id = structured.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
    # Some transports report the body as text only; parse it rather than give up
    # on a filing that did happen.
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            parsed = json.loads(text)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("request_id"), str):
            return parsed["request_id"]
    raise FilingRefused(f"the runtime accepted the request but returned no request_id: {result!r}")


async def file_ticket(ticket: Ticket, *, turn_id: str | None = None) -> Ticket:
    """File one validated ticket, returning it with the row's id attached.

    `turn_id` is the A2A turn's id (the gate's `request_id`), used as the
    idempotency key. It is the honest key: two filings of one turn are one
    request, and data-download answers a repeated key with a 409 naming the row
    that already exists instead of creating a second one.
    """
    body = plan_body(ticket)
    arguments: dict[str, Any] = {"body": body, "filed_by": FILED_BY}
    if turn_id:
        arguments["idempotency_key"] = turn_id
    try:
        async with Client(MCP_URL) as client:
            result = await client.call_tool(SUBMIT_TOOL, arguments)
    except Exception as exc:  # transport, handshake, or the tool's own failure
        raise FilingRefused(f"could not file the request: {exc}") from exc
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise FilingRefused(_error_text(result))
    return ticket.model_copy(update={"request_id": _row_id(result)})


def _error_text(result: Any) -> str:
    """The refusal's own words, out of whichever field the transport used."""
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return f"the runtime refused the request: {result!r}"
