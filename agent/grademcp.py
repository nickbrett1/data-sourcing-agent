"""The grading MCP — so an agent can do the grading loop, not just a browser.

Three tools over the same streams the UI uses: read the queue, record a verdict,
read the status. Nothing here is new logic — it is the MCP skin over
`agent.grades`, `agent.grader`, and `agent.summary`, so the API, the UI, and an agent
all move through the same functions and cannot drift apart.

Run it over stdio (the client launches it):

    python -m agent.grademcp

Deliberately stdio, not an HTTP port: the thing it grades is local, and a stdio
server inherits the caller's filesystem and permissions rather than needing its own
exposure decision (unlike the UI, which is bound to localhost for that reason).
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agent.grader import build_queue
from agent.grades import Verdict, append_grade, load_grades
from agent.summary import build_summary

mcp = MCPServer(
    name="gate-grader",
    instructions=(
        "Grade the gate's decisions: read the queue of requests the door ruled on, "
        "judge each from the request text and the ticket it produced, and record a "
        "verdict. Verdicts are right / too_strict (a false reject -> lower the cut) / "
        "too_lenient (a false accept -> raise the cut) / unclear."
    ),
)


@mcp.tool(description="The next ungraded request, with its Jev scores, the door's call, and the ticket.")
def next_request() -> dict | None:
    """Return the first request with no verdict, or None when the queue is empty."""
    for item in build_queue():
        if not item["grade"]:
            return item
    return None


@mcp.tool(description="A page of requests to grade, oldest first; pass only_ungraded=false to include graded ones.")
def queue(limit: int = 10, only_ungraded: bool = True) -> list[dict]:
    """The grading queue. Small by default: grading is per-request attention."""
    items = build_queue()
    if only_ungraded:
        items = [item for item in items if not item["grade"]]
    return items[: max(0, limit)]


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


def main() -> int:
    """Serve the tools over stdio."""
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["mcp", "next_request", "queue", "record_grade", "status", "grades", "Verdict"]
