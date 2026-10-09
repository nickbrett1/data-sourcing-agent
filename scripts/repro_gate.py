"""Reproduce the front-door gate's decision for a request, locally.

Builds the real interpreter (same gateway model) and asks the real Jev the same
D1/D2/D3 questions the deployed front door asks, then composes the action with
agent.gate.decide. The discovery MCP toolset is dropped (not reachable from this
devcontainer), so candidate_mappings come back empty - everything else matches.

Edit MESSAGES to try other requests, then:

    PYTHONPATH=. .venv/bin/python scripts/repro_gate.py
"""

from __future__ import annotations

import asyncio

from agent.gate import D1, D2, D3, decide, questions, serialise_state
from agent.interpret import build_interpreter, to_gate_state
from agent.jev import JevClient
from agent.model import build_model

MESSAGES = [
    # Round 1 - dataset not named
    (
        "I want to study how the SPY options chain prices around the October 2026 "
        "monthly expiry. Pull one day of daily OHLCV bars for the full SPY options "
        "chain covering 2026-10-01 through 2026-10-02 (end exclusive). Keep the "
        "spend under $2.00."
    ),
    # Round 2 - every axis pinned
    (
        "dataset: OPRA.PILLAR (US equity options)\n"
        "schema: ohlcv-1d\n"
        "symbols: SPY.OPT\n"
        "stype_in: parent\n"
        "start: 2026-10-01\n"
        "end: 2026-10-02 (exclusive)\n"
        "cost.max_usd: 2.00"
    ),
]


async def main() -> None:
    model = build_model()
    interpreter = build_interpreter(model, toolsets=[])
    jev = JevClient.from_env()

    for i, message in enumerate(MESSAGES, 1):
        intent = (await interpreter.run(message)).output
        print(f"\n=== round {i} ===")
        print("PARSED INTENT:", intent.render())
        print("unclear_axes:", intent.unclear_axes)

        state = to_gate_state(message, intent)
        answers = await jev.ask(serialise_state(state), questions(estimate_usd=None))
        for qid in (D1, D2, D3):
            ans = answers.get(qid)
            print(f"  {qid}: {ans}")

        decision = decide(answers)
        print("=> action:", decision.action.value)
        for reason in decision.reasons:
            print("   -", reason)

    await jev.aclose()


if __name__ == "__main__":
    asyncio.run(main())
