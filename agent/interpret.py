"""The interpretation step — prose in, a typed candidate intent out (memo §5).

This is the `interpret` state of the pipeline state machine, **pulled out of the
generative run into its own explicit step** (decided 2026-10-06). The reason is
the gate: the front-door gate (memo `jev-integration-v1`) judges the request
*before* the Validator prices anything, and to judge it it needs the things the
interpretation step produces — the understood axes and the discovery vocabulary —
as data the executor holds, not as tokens buried inside a model turn.

So the turn is now two explicit steps with the gate between them:

    interpret (generative, this module)  ->  gate (Jev, D1/D2/D3)  ->  validate+price

`ParsedIntent` is deliberately *best-effort and honest about its gaps*: an axis it
could not resolve is left `None` and named in `unclear_axes`. That is what makes
D2 answerable — "was any axis guessed?" is only a real question if the
interpretation says which axes it guessed, rather than silently filling them in.
Filling them in is exactly the failure D2 exists to catch (memo `gate-question-set-v1`
§2, D2; §6.5.1 of `data-acquisition-agent-v1`).

Jev does not do this job: parsing prose into slots is *generation*, which Jev
cannot do (memo §6.4). So this step is a generative model, and it is the one place
in the turn where an open-ended model is genuinely needed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.gate import GateState

# The four axes D2 judges. `ParsedIntent.unclear_axes` must be a subset of these,
# so the interpreter's own vocabulary matches the question D2 asks.
AXES = ("dataset", "schema", "universe", "timeframe")


class ParsedIntent(BaseModel):
    """The best-effort reading of a request, gaps included.

    Every field may be `None`/empty: the point is to *report* uncertainty, not to
    paper over it. `unclear_axes` names the axes that were inferred rather than
    stated — the signal D2 turns on.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    understood: str = Field(
        description="One sentence restating what the user is asking for, in the agent's terms."
    )
    dataset: str | None = Field(default=None, description="Databento dataset id, if stated or clearly implied.")
    schema_name: str | None = Field(
        default=None, alias="schema", description="Databento schema, if stated or clearly implied."
    )
    symbols: list[str] = Field(default_factory=list, description="Universe, as the user wrote it.")
    stype_in: str | None = Field(default=None, description="Symbology type, only if the user stated it.")
    start: str | None = Field(default=None, description="Window start (ISO date), if stated.")
    end: str | None = Field(default=None, description="Window end (ISO date), if stated.")
    unclear_axes: list[str] = Field(
        default_factory=list,
        description=(
            "Any of dataset/schema/universe/timeframe that had to be GUESSED rather "
            "than read from the request. Empty only when all four were stated or "
            "unambiguously resolvable. This is the honest-gaps field; under-reporting "
            "it defeats the gate."
        ),
    )
    candidate_mappings: list[str] = Field(
        default_factory=list,
        description=(
            "Dataset/schema candidates the discovery tools returned for this request "
            "(`dataset schema` pairs). What the request could map onto."
        ),
    )

    def render(self) -> str:
        """A stable text form for the gate's state — the understood request."""
        parts = [self.understood.strip() or "(nothing understood)"]
        axes = []
        for name, value in (
            ("dataset", self.dataset),
            ("schema", self.schema_name),
            ("universe", ", ".join(self.symbols) or None),
            ("timeframe", _window(self.start, self.end)),
        ):
            axes.append(f"{name}={value if value else 'UNKNOWN'}")
        parts.append("; ".join(axes))
        if self.unclear_axes:
            parts.append(f"guessed axes: {', '.join(self.unclear_axes)}")
        return " | ".join(parts)


def _window(start: str | None, end: str | None) -> str | None:
    if start and end:
        return f"{start}..{end}"
    if start or end:
        return f"{start or '?'}..{end or '?'}"
    return None


def to_gate_state(
    message: str,
    intent: ParsedIntent,
    *,
    estimate_usd: float | None = None,
    rows: int | None = None,
    budget_context: str = "",
    prior_turns: tuple[str, ...] = (),
) -> GateState:
    """Assemble the gate's shared state (gate memo §1) from an interpretation.

    `estimate_usd` is passed through only when the Validator has already priced
    the request; at the pre-price checkpoint it is `None`, and D5 simply does not
    fire (it is conditional on materiality — gate memo §2).
    """
    return GateState(
        raw_request=message,
        parsed_intent=intent.render(),
        candidate_mappings=tuple(intent.candidate_mappings),
        estimate_usd=estimate_usd,
        rows=rows,
        budget_context=budget_context,
        prior_turns=prior_turns,
    )


INTERPRETER_INSTRUCTIONS = """\
You read one message from a user who wants historical market data, and you report
what it asks for — honestly, including what it does not say.

You have discovery tools. Use them to find the datasets and schemas this request
could map onto, and put the `dataset schema` candidates you find in
`candidate_mappings`.

The rule that matters: for each of the four axes — dataset, schema, universe
(symbols), timeframe (dates) — say whether it was **stated or unambiguously
resolvable**, or whether you had to **guess**. Every axis you guessed goes in
`unclear_axes`. Return an axis as a value only when the request supports it; if
you are filling it in because something must go there, that is a guess and it
belongs in `unclear_axes`, not presented as a fact.

Do not invent a date window, a symbol, or a dataset to make the request look
complete. An incomplete request reported honestly is a good answer; a complete
request built on guesses is the failure this step exists to prevent.
"""


def build_interpreter(model, toolsets=None):
    """A Pydantic AI agent for the interpretation step, output-typed to `ParsedIntent`.

    Built here rather than in `main.py` so the interpretation contract (a typed,
    gap-honest `ParsedIntent`) lives with its tests. `model` is injected — the
    same gateway model the ticket run uses.
    """
    from pydantic_ai import Agent

    return Agent(
        model,
        instructions=INTERPRETER_INSTRUCTIONS,
        output_type=ParsedIntent,
        toolsets=toolsets or [],
    )
