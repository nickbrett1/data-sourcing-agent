"""The download ticket — the agent's output type (memo `data-acquisition-agent-v1` §2).

The unit of work is a `requests/*.yaml` **ticket**: one file, one proposed
download, git history as the audit log. This module is the ticket's shape in
Python; the YAML is rendered from it in code (see `render_ticket_yaml`), because
asking the model for YAML buys nothing and loses the typing.

Three types, deliberately distinct:

* `RequestSpec` — mirrors a Databento `timeseries.get_range` call field for field.
* `TicketProposal` — **the only thing the model may produce.** It has no `state`
  and no `estimate_usd`: `state` is a policy field (the agent may only ever write
  `draft`) and `estimate_usd` is the API's truth, not the model's guess. Neither
  belongs in a model-authored type, so neither is in the schema the model sees.
* `Ticket` — the validated, costed artifact the Validator layer returns after it
  has asked the API. This is what gets rendered to YAML.

`max_usd` is a **ceiling**: the agent proposes it and may never raise it. The
Validator enforces `estimate_usd <= max_usd` and, when it does not hold, tells the
model to narrow the request — never to raise the ceiling (§6, "Do NOT").
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StypeIn(StrEnum):
    """The legal `stype_in` values. A closed set, so no value outside it exists.

    This is the field that broke slice #5 (memo §3): "the SPY options chain" does
    not determine whether it is `parent`, `raw_symbol`, `continuous` or
    `instrument_id`. Constraining the type is how the illegal value stops being
    expressible in the model's output — the API still has the final word.
    """

    raw_symbol = "raw_symbol"
    parent = "parent"
    continuous = "continuous"
    instrument_id = "instrument_id"


class RequestSpec(BaseModel):
    """A pre-formed `timeseries.get_range` call, field for field."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset: str = Field(description="Databento dataset id, e.g. `OPRA.PILLAR`.")
    # The field is `schema_name` only because a Python attribute named `schema`
    # shadows a deprecated pydantic method and warns; the wire name stays
    # `schema` via the alias, so the API parameter and the rendered ticket are
    # unchanged. `populate_by_name` accepts either spelling on the way in.
    schema_name: str = Field(alias="schema", description="Databento schema, e.g. `ohlcv-1d`.")
    symbols: Annotated[
        list[str],
        Field(min_length=1, description="Symbols as the intent wrote them; `ALL_SYMBOLS` allowed."),
    ]
    stype_in: StypeIn = Field(
        description="How `symbols` are to be read. Underdetermined by intent — do not guess.",
    )
    start: date = Field(description="Inclusive start of the request window (UTC).")
    end: date = Field(description="Exclusive end of the request window (UTC).")

    @model_validator(mode="after")
    def _window_is_ordered(self) -> RequestSpec:
        if self.end < self.start:
            raise ValueError(
                f"`end` ({self.end}) is before `start` ({self.start}); a request "
                "window cannot be empty."
            )
        return self


class CostProposal(BaseModel):
    """The cost block as the model may propose it: a ceiling, nothing else."""

    model_config = ConfigDict(extra="forbid")

    max_usd: Annotated[
        float,
        Field(gt=0, description="Ceiling the agent may not raise once set."),
    ]


class TicketProposal(BaseModel):
    """What the model may produce — no `state`, no `estimate_usd`.

    `estimated_usd` and `state` are filled by the Validator, not the model: one
    is the API's truth and the other is a policy the agent cannot hold. Keeping
    them out of this type means the model never sees them in the output schema
    and so cannot invent them.
    """

    model_config = ConfigDict(extra="forbid")

    group: str | None = Field(
        default=None,
        description="Optional bundle key shared by tickets that must be reviewed/bought together.",
    )
    request: RequestSpec
    cost: CostProposal
    why: Annotated[
        str,
        Field(
            min_length=1,
            description="Mandatory provenance: why this download is worth its cost.",
        ),
    ]


class TicketCost(BaseModel):
    """The cost block of a validated ticket."""

    estimate_usd: float | None = Field(
        default=None,
        description="From `metadata.get_cost` before the spend. The API's number, not the model's.",
    )
    max_usd: float = Field(description="Ceiling; the agent may not raise it.")
    actual_usd: float | None = Field(
        default=None,
        description="Reconciled after download; `null` until then.",
    )


class Ticket(BaseModel):
    """The validated, costed artifact. Rendered to `requests/*.yaml`.

    `state` is pinned to `draft`: `approved` is a human's to write and the agent
    may never write it — that is the entire safety property of the system (memo
    §6). The type is the enforcement.
    """

    model_config = ConfigDict(extra="forbid")

    state: Annotated[str, Field(pattern="^draft$")] = "draft"
    group: str | None = None
    request: RequestSpec
    cost: TicketCost
    why: str


def render_ticket_yaml(ticket: Ticket) -> str:
    """Render a validated ticket to the YAML that lands in `requests/`.

    Done in code, not by the model: the model produced a typed object, and the
    wire format is a presentation concern. `actual_usd` is always written as an
    explicit `null` so the post-download reconciliation has a field to fill.
    """
    request = ticket.request
    lines = [
        f"state: {ticket.state}",
    ]
    if ticket.group is not None:
        lines.append(f"group: {ticket.group}")
    lines += [
        "request:",
        f"  dataset: {request.dataset}",
        f"  schema: {request.schema_name}",
        "  symbols: [" + ", ".join(request.symbols) + "]",
        f"  stype_in: {request.stype_in.value}",
        f"  start: {request.start.isoformat()}",
        f"  end: {request.end.isoformat()}",
        "cost:",
        f"  estimate_usd: {ticket.cost.estimate_usd}",
        f"  max_usd: {ticket.cost.max_usd}",
        f"  actual_usd: {ticket.cost.actual_usd}",
        f"why: {_quote(ticket.why)}",
        "",
    ]
    return "\n".join(lines)


def _quote(text: str) -> str:
    """A YAML double-quoted scalar: escape backslash and quote, keep it one line."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'
