"""The grading MCP — so an agent can do the grading loop, not just a browser.

Tools over the same streams the UI uses: read the queue (`next_request`/`queue`),
record a verdict (`record_grade`), read the status (`status`/`grades`), and look up
what a question code means (`legend`). Nothing here is new logic — it is the MCP skin
over `agent.grades`, `agent.grader`, and `agent.summary`, so the API, the UI, and an
agent all move through the same functions and cannot drift apart.

Each Jev answer is returned with a plain-English `means` (from `agent.gate`), so a
grader never has to hold the `d1_*`/`d2_*` codes in their head.

Run it over stdio (the client launches it):

    python -m agent.gate_grader

Deliberately stdio by default, not an HTTP port: the thing it grades is local, and a
stdio server inherits the caller's filesystem and permissions rather than needing
its own exposure decision (unlike the UI, which is bound to localhost for that
reason).

The exception is a *hub* that cannot spawn a local process — mcphub on the NAS
spawns stdio servers inside its own container, where this code and `/state` do not
exist. For that, the same server also speaks streamable-http, so it can be
registered by URL instead:

    python -m agent.gate_grader --transport streamable-http --host 0.0.0.0 --port 8802
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agent.gate import QUESTION_LEGEND
from agent.grader import build_queue
from agent.grades import Verdict, append_grade, load_grades
from agent.summary import build_summary

mcp = MCPServer(
    # Project-prefixed, so a hub carrying several MCPs does not have to guess
    # which project's gate this grades.
    name="data-sourcing-gate-grader",
    instructions=(
        "Grade the gate's decisions: read the queue of requests the door ruled on, "
        "judge each from the request text and the ticket it produced, and record a "
        "verdict. Verdicts are right / too_strict (a false reject -> lower the cut) / "
        "too_lenient (a false accept -> raise the cut) / unclear. The question codes "
        "(D1 remit, D2 specification sufficient, D3 dataset fit, D5 cost "
        "proportionate) are spelled out beside each answer in `means`; call `legend` "
        "for the full mapping."
    ),
)


@mcp.tool(
    description=(
        "The next ungraded request: its text, the ticket the door produced, the "
        "door's call, and each Jev answer. Every `jev` entry carries a plain-English "
        "`means` describing what the question asked (D1 remit, D2 specification, "
        "D3 dataset fit, D5 cost) so you need not remember the codes."
    )
)
def next_request() -> dict | None:
    """Return the first request with no verdict, or None when the queue is empty."""
    for item in build_queue():
        if not item["grade"]:
            return item
    return None


@mcp.tool(
    description=(
        "A page of requests to grade, oldest first; pass only_ungraded=false to "
        "include graded ones. Each `jev` entry includes a plain-English `means`."
    )
)
def queue(limit: int = 10, only_ungraded: bool = True) -> list[dict]:
    """The grading queue. Small by default: grading is per-request attention."""
    items = build_queue()
    if only_ungraded:
        items = [item for item in items if not item["grade"]]
    return items[: max(0, limit)]


@mcp.tool(
    description=(
        "What each question code means, in plain English — D1 remit, D2 "
        "specification sufficient, D3 dataset fit, D5 cost proportionate — and "
        "which way each answer cuts. Read this once if a row is unclear."
    )
)
def legend() -> dict:
    """The question-code legend: `d2_specification_sufficient` -> what it measures."""
    return dict(QUESTION_LEGEND)


@mcp.tool(description="Record a verdict on one request. verdict is right | too_strict | too_lenient | unclear.")
def record_grade(request_id: str, verdict: str, note: str = "") -> dict:
    """Write one verdict. Rejects a value outside the vocabulary rather than storing it."""
    saved = append_grade(request_id, verdict, note=note)
    return {"request_id": saved.request_id, "verdict": saved.verdict.value, "direction": saved.direction}


@mcp.tool(description="The verdict tally and whether the door is in sync: is the cut fresh, do the grades agree, is grading behind.")
def status() -> dict:
    """`agent.summary.build_summary` — the same payload Homepage and Dagu read."""
    return build_summary()


@mcp.tool(description="Every recorded verdict, keyed by request id — the full grade history, latest per request.")
def grades() -> dict:
    """The current grade per request (latest wins; the stream keeps history)."""
    return {
        rid: {"verdict": grade.verdict.value, "direction": grade.direction, "note": grade.note, "ts": grade.ts}
        for rid, grade in load_grades().items()
    }


def main(argv: list[str] | None = None) -> int:
    """`python -m agent.gate_grader [--transport stdio|streamable-http] [--host H] [--port P]`.

    stdio is the default (a client launches us). streamable-http is for a hub that
    cannot spawn a process — it reaches us by URL instead. Both paths run the exact
    same tools; only the wire changes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="The gate-grader MCP, over stdio or streamable-http.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (streamable-http only)")
    parser.add_argument("--port", type=int, default=8802, help="bind port (streamable-http only)")
    args = parser.parse_args(argv)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # stateless_http: each POST stands alone, so the hub needs no session pinning.
        mcp.run(transport="streamable-http", host=args.host, port=args.port, stateless_http=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["mcp", "next_request", "queue", "record_grade", "status", "grades", "legend", "Verdict"]
