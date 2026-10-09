"""Measure D2's distribution on known-good vs known-vague requests.

The question: is the 0.91 placeholder cut ABOVE the D2 that a fully-specified,
genuinely-good request gets? If so the ask-path is a wall, not a filter.

Runs the real interpreter (same gateway model) and the real Jev D2 question, the
same way the front door does, over a batch. Discovery toolset dropped (not
reachable here) - candidate_mappings empty, everything else identical.

    PYTHONPATH=. .venv/bin/python scripts/calibrate_d2.py
"""

from __future__ import annotations

import asyncio
import statistics

from agent.gate import D2, questions, serialise_state
from agent.interpret import build_interpreter, to_gate_state
from agent.jev import JevClient
from agent.model import build_model

GOOD = {
    "opra-pinned": (
        "dataset: OPRA.PILLAR\nschema: ohlcv-1d\nsymbols: SPY.OPT\n"
        "stype_in: parent\nstart: 2026-10-01\nend: 2026-10-02"
    ),
    "xnas-trades": (
        "dataset: XNAS.ITCH\nschema: trades\nsymbols: AAPL\nstype_in: raw_symbol\n"
        "start: 2026-09-01\nend: 2026-09-02"
    ),
    "glbx-continuous": (
        "dataset: GLBX.MDP3\nschema: ohlcv-1d\nsymbols: ES.FUT\n"
        "stype_in: continuous\nstart: 2026-06-01\nend: 2026-06-15"
    ),
    "equs-all": (
        "dataset: EQUS.MINI\nschema: ohlcv-1d\nsymbols: ALL_SYMBOLS\n"
        "stype_in: raw_symbol\nstart: 2026-06-01\nend: 2026-06-02"
    ),
    "xnas-mbp10": (
        "dataset: XNAS.ITCH\nschema: mbp-10\nsymbols: MSFT\nstype_in: raw_symbol\n"
        "start: 2026-01-02\nend: 2026-01-03"
    ),
    "opra-trades": (
        "dataset: OPRA.PILLAR\nschema: trades\nsymbols: SPY.OPT\n"
        "stype_in: parent\nstart: 2026-10-01\nend: 2026-10-02"
    ),
}

VAGUE = {
    "some-data": "I need some market data for research.",
    "stock-data": "Pull stock data please.",
    "backtest": "I want to backtest an intraday strategy on futures.",
    "last-month-options": "Can I get options data from last month?",
}


async def score(message: str, interpreter, jev) -> tuple[float | None, list[str]]:
    intent = (await interpreter.run(message)).output
    state = to_gate_state(message, intent)
    answers = await jev.ask(serialise_state(state), questions(estimate_usd=None))
    ans = answers.get(D2)
    value = None if ans is None or ans.failed else ans.noul
    return value, list(intent.unclear_axes)


async def main() -> None:
    model = build_model()
    interpreter = build_interpreter(model, toolsets=[])
    jev = JevClient.from_env()

    print("\nGOOD (fully specified) — D2 should be HIGH and clear 0.91")
    good_scores: list[float] = []
    for name, msg in GOOD.items():
        value, unclear = await score(msg, interpreter, jev)
        if value is not None:
            good_scores.append(value)
        print(f"  {name:<18} D2={value}  unclear_axes={unclear}")

    print("\nVAGUE — D2 should be LOW")
    vague_scores: list[float] = []
    for name, msg in VAGUE.items():
        value, unclear = await score(msg, interpreter, jev)
        if value is not None:
            vague_scores.append(value)
        print(f"  {name:<18} D2={value}  unclear_axes={unclear}")

    print("\n--- summary ---")
    if good_scores:
        print(
            f"good  (n={len(good_scores)}): min={min(good_scores):.2f} "
            f"median={statistics.median(good_scores):.2f} max={max(good_scores):.2f} "
            f"| clear 0.91: {sum(s >= 0.91 for s in good_scores)}/{len(good_scores)}"
        )
    if vague_scores:
        print(
            f"vague (n={len(vague_scores)}): min={min(vague_scores):.2f} "
            f"median={statistics.median(vague_scores):.2f} max={max(vague_scores):.2f}"
        )
    if good_scores and vague_scores:
        print(f"separation: good_median - vague_median = {statistics.median(good_scores) - statistics.median(vague_scores):+.2f}")

    await jev.aclose()


if __name__ == "__main__":
    asyncio.run(main())
