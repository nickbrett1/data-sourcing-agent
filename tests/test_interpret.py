"""Tests for the interpretation step.

The behaviour that matters is *honesty about gaps*: an unclear axis must surface
as `UNKNOWN` and be named in `unclear_axes`, because that is the signal D2 turns
on. A silent fill-in is the failure the gate exists to catch.
"""

from __future__ import annotations

from agent.interpret import AXES, ParsedIntent, ticket_prompt, to_gate_state


def test_axes_vocabulary_matches_d2():
    assert set(AXES) == {"dataset", "schema", "universe", "timeframe"}


def test_render_names_each_unresolved_axis():
    intent = ParsedIntent(
        understood="the whole SPY options chain",
        schema_name="ohlcv-1d",
        symbols=["SPY.OPT"],
        unclear_axes=["timeframe"],
    )
    text = intent.render()
    assert "timeframe=UNKNOWN" in text
    assert "guessed axes: timeframe" in text
    assert "schema=ohlcv-1d" in text


def test_render_marks_all_four_axes_unknown_when_empty():
    assert ParsedIntent(understood="something").render().count("UNKNOWN") == 4


def test_render_formats_a_stated_window():
    intent = ParsedIntent(understood="x", start="2026-10-01", end="2026-10-02")
    assert "timeframe=2026-10-01..2026-10-02" in intent.render()


def test_to_gate_state_carries_the_discovery_mappings():
    intent = ParsedIntent(
        understood="u", candidate_mappings=["OPRA.PILLAR ohlcv-1d", "OPRA.PILLAR trades"]
    )
    state = to_gate_state("get SPY options", intent)
    assert state.raw_request == "get SPY options"
    assert state.candidate_mappings == ("OPRA.PILLAR ohlcv-1d", "OPRA.PILLAR trades")
    assert state.estimate_usd is None  # pre-price: D5 cannot fire


def test_ticket_prompt_hands_over_the_reading_so_it_is_not_rederived():
    intent = ParsedIntent(
        understood="the SPY options chain",
        schema_name="ohlcv-1d",
        unclear_axes=["timeframe"],
    )
    prompt = ticket_prompt("get the whole SPY chain", intent)
    assert "get the whole SPY chain" in prompt
    assert "timeframe=UNKNOWN" in prompt
    assert "rather than deriving" in prompt


def test_to_gate_state_passes_an_estimate_when_pricing_has_happened():
    state = to_gate_state(
        "m", ParsedIntent(understood="u"), estimate_usd=12.74, rows=4200
    )
    assert state.estimate_usd == 12.74
    assert state.rows == 4200
