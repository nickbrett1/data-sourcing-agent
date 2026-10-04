"""The agent's output contract and its Validator layer (memo `data-acquisition-agent-v1`).

`output_type` is a **Pydantic model, not free text and not YAML**: a typed model
is what buys validation and a bounded repair loop. The model produces a
`TicketProposal` (the ticket *minus* `state` and `estimate_usd`, which are not
the model's to write); `validate_agent_result` asks the Validator, and only then
returns a costed `Ticket`.

The repair loop is the design's Layer 3, and it is **bounded**: `retries={"output":
3}` in `agent/main.py`. On a defect the validator raises `ModelRetry` carrying
the API's own error text, the model proposes a fix, and the loop revalidates. An
unbounded repair loop against a paid API is how you re-purchase data by accident
(memo §6) — the budget is not to be loosened.

Two things this module refuses to do:

* it never lets the model be the oracle for validity or cost — it calls the
  Validator, which calls the API;
* it never lets the model raise `cost.max_usd`. When the estimate exceeds the
  ceiling, the repair instruction is to **narrow the request**, explicitly not to
  raise the ceiling.
"""

from __future__ import annotations

import asyncio

from pydantic_ai import ModelRetry, RunContext

from agent.ticket import Ticket, TicketCost, TicketProposal
from agent.validator import DatabentoRequestError, DatabentoValidator

# One validator per process. It holds an HTTP client; building it per request
# would leak connections and, worse, invite a per-request policy difference.
_validator: DatabentoValidator | None = None


def get_validator() -> DatabentoValidator:
    """The process's Validator, built from the environment on first use.

    Built lazily so importing this module (and so `agent.main`) does not require
    the Databento credential to be present — the import-time build is the CI
    smoke test's whole point, and it must not need secrets.
    """
    global _validator
    if _validator is None:
        _validator = DatabentoValidator.from_env()
    return _validator


async def validate_agent_result(
    _ctx: RunContext, output: TicketProposal
) -> Ticket:
    """Validate the model's proposal against the API, then cost it.

    Raises `ModelRetry(<the API's own text>)` on an illegal request so the model
    can repair it (bounded by `retries["output"]`). On success returns a `Ticket`
    with `state="draft"` and `estimate_usd` from the API. The retry budget's
    exhaustion is *not* swallowed: the run fails loudly, to a human.
    """
    if not output.why.strip():
        raise ModelRetry(
            "`why` is mandatory provenance and cannot be blank. State, in one "
            "sentence, why this download is worth its cost."
        )

    validator = get_validator()
    request = output.request
    try:
        # The API client is synchronous; run it off the event loop so a slow
        # oracle cannot stall the agent's other tasks.
        estimate_usd = await asyncio.to_thread(validator.price_request, request)
    except DatabentoRequestError as exc:
        # Pass the API's own case and message through unchanged. The message
        # frequently names the legal values (e.g. the schemas this dataset does
        # support), which is the runtime vocabulary the model must repair against.
        raise ModelRetry(
            f"Databento rejected the request: {exc}. Correct the offending field "
            "and propose the request again."
        ) from exc

    if estimate_usd > output.cost.max_usd:
        # Do NOT tell the model to raise `max_usd` - that is the one lever it may
        # never move (memo §6). Narrowing the request is the only permitted fix.
        raise ModelRetry(
            f"The request is legal but its estimated cost (${estimate_usd:.4f}) "
            f"exceeds the ceiling (${output.cost.max_usd:.2f}). Narrow the request "
            "(a shorter window, fewer symbols, a coarser schema). You may not "
            "raise `max_usd`."
        )

    return Ticket(
        state="draft",
        group=output.group,
        request=request,
        cost=TicketCost(
            estimate_usd=round(estimate_usd, 4),
            max_usd=output.cost.max_usd,
            actual_usd=None,
        ),
        why=output.why,
    )
