"""The grading UI — one request at a time, keyboard-driven, resumable.

`python -m agent.grader` serves a single page that reads the sample + shadow log
(`agent/samples.py`, `agent/gatelog.py`) and writes verdicts to the grade stream
(`agent/grades.py`). It is a *workflow*, not a report: it shows one request, remembers
what is already graded, and ends in a write. The report tells you the state of the
door; this is how the state of the door gets decided.

Read-only over the logs; the only write is a grade. It runs on localhost by default
because grading is a single-person, on-the-spot activity — no auth is built, so do
not expose it on a shared network.
"""

from __future__ import annotations

from collections.abc import Iterable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from agent.gate import question_legend
from agent.gate_report import load_records
from agent.grades import append_grade, load_grades
from agent.samples import load_samples
from agent.summary import build_summary

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Grade the door</title>
<style>
  :root { color-scheme: dark; }
  html, body { height: 100%; }
  /* Full-height column: header and footer take their natural height, `main` takes
     the rest and scrolls. The footer therefore RESERVES its own space instead of
     floating over the content -- a fixed-position footer never does, so no
     padding-bottom can reliably clear it (it wraps to a different height on a
     narrow phone, which is why the buttons covered the ticket). */
  body { font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
         margin: 0; background: #101214; color: #d7dde3;
         display: flex; flex-direction: column; height: 100dvh; }
  header { background: #16191d; padding: 10px 18px; flex: 0 0 auto;
           border-bottom: 1px solid #262b31; display: flex; gap: 18px; align-items: center; }
  #bar { flex: 1; height: 6px; background: #262b31; border-radius: 3px; overflow: hidden; }
  #fill { height: 100%; width: 0; background: #3d9970; transition: width .15s; }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 0;
         flex: 1 1 auto; min-height: 0; overflow: auto; }
  section { padding: 16px 18px; }
  section + section { border-left: 1px solid #262b31; }
  h2 { font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
       color: #7d8896; margin: 0 0 8px; font-weight: 600; }
  pre { white-space: pre-wrap; word-break: break-word; margin: 0 0 20px; }
  .jev div { display: flex; justify-content: space-between; padding: 2px 0; max-width: 420px; }
  .fail { color: #b07a3f; }
  /* The row whose answer fell below the cut in force at the time -- the reason
     the door said what it said. Tinted, with the cut shown after the answer. */
  .breach { color: #e7d06e; }
  .breach .cut { color: #a89a5e; }
  .why { color: #e7d06e; margin-top: 6px; }
  .door { display: inline-block; padding: 2px 10px; border-radius: 3px; font-weight: 700; }
  .door.proceed { background: #1d3a2a; color: #6ee7a8; }
  .door.ask_clarifying { background: #3a3320; color: #e7d06e; }
  .door.reject { background: #3a2020; color: #e78b8b; }
  .holdback { color: #8aa1b8; font-style: italic; margin-left: 8px; }
  footer { background: #16191d; flex: 0 0 auto;
           border-top: 1px solid #262b31; padding: 10px 18px calc(10px + env(safe-area-inset-bottom));
           display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button { font: inherit; padding: 6px 14px; border-radius: 4px; border: 1px solid #39414b;
           background: #1d2228; color: #d7dde3; cursor: pointer; }
  button:hover { background: #262c34; }
  button[data-v="right"] { border-color: #2f6f4f; }
  button[data-v="too_strict"] { border-color: #6f5a2f; }
  button[data-v="too_lenient"] { border-color: #6f3030; }
  kbd { background: #262b31; padding: 1px 6px; border-radius: 3px; font-size: 12px; }
  #legend { flex: 1 1 100%; order: -2; margin: 0 0 6px; color: #7d8896;
            font-size: 12px; line-height: 1.5; }
  #legend b { color: #cfd6dd; font-weight: 600; }
  #note { flex: 1; min-width: 200px; font: inherit; background: #1d2228; color: inherit;
          border: 1px solid #39414b; border-radius: 4px; padding: 6px 10px; }
  #empty { padding: 40px; color: #7d8896; }
  /* Phone: one column, bigger type and tap targets, no keyboard hints. Without
     this (and the viewport meta above) the page renders at desktop width and
     every glyph comes out tiny. */
  @media (max-width: 760px) {
    body { font-size: 15.5px; }
    header { flex-wrap: wrap; row-gap: 6px; padding: 8px 12px; }
    #hint { display: none; }
    #bar { flex-basis: 100%; }
    main { grid-template-columns: 1fr; }
    section { padding: 14px; }
    section + section { border-left: 0; border-top: 1px solid #262b31; }
    h2 { font-size: 12px; margin-top: 12px; }
    .jev div { max-width: none; }
    footer { flex-wrap: wrap; gap: 8px; padding: 10px 12px; }
    #note { order: -1; flex: 1 1 100%; min-height: 44px; }
    #legend { order: -2; font-size: 13px; }
    button { flex: 1 1 calc(50% - 8px); min-height: 48px; font-size: 15px; }
  }
</style></head><body>
<header>
  <strong>Grade the door</strong>
  <span id="count">…</span>
  <div id="bar"><div id="fill"></div></div>
  <span id="hint">1 right · 2 too strict · 3 too lenient · 4 unclear · ← → move</span>
</header>
<main id="main"><div id="empty">Loading…</div></main>
<footer>
  <p id="legend"><b>right</b> the door did the right thing ·
     <b>too strict</b> it stopped a fine request (lower the cut) ·
     <b>too lenient</b> it let a bad one through (raise the cut) ·
     <b>unclear</b> the sample doesn't settle it (moves nothing)</p>
  <input id="note" placeholder="note (optional, saved with the verdict)">
  <button data-v="right">1 · right</button>
  <button data-v="too_strict">2 · too strict</button>
  <button data-v="too_lenient">3 · too lenient</button>
  <button data-v="unclear">4 · unclear</button>
</footer>
<script>
let queue = [], i = 0;
// `answer` in the log is a number (a Noul probability, or a D3 score), not a
// string -- so esc() must coerce before .replace(), or the whole render throws
// and the page never leaves "Loading...".
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt = s => (typeof s === 'number' ? Math.round(s * 1000) / 1000 : s);
const KEYS = { '1':'right', '2':'too_strict', '3':'too_lenient', '4':'unclear' };

async function load() {
  try {
    const res = await fetch('/api/queue');
    if (!res.ok) throw new Error('/api/queue -> HTTP ' + res.status);
    queue = await res.json();
  } catch (err) {
    document.getElementById('main').innerHTML =
      '<div id="empty">Could not load the queue: ' + esc(err && err.message || err) +
      '.<br>Is this the grader service, and is its /state volume mounted?</div>';
    return;
  }
  i = queue.findIndex(q => !q.grade);           // resume at the first ungraded
  if (i < 0) i = 0;
  render();
}
function render() {
  const graded = queue.filter(q => q.grade).length;
  document.getElementById('count').textContent = `${i + 1} / ${queue.length} · ${graded} graded`;
  document.getElementById('fill').style.width = queue.length ? (graded / queue.length * 100) + '%' : '0';
  const q = queue[i];
  const main = document.getElementById('main');
  if (!q) { main.innerHTML = '<div id="empty">Nothing to grade. Log some traffic first.</div>'; return; }
  // Readable question name (the part before the colon) with the full
  // explanation and the raw code on hover — the grader should not have to
  // remember what `d2_specification_sufficient` means.
  const qLabel = a => ((a.means || a.q || '').split(':')[0]).trim();
  const qTitle = a => (a.means ? a.q + ' — ' + a.means : a.q);
  // Every question is a minimum in `decide`: the door stops when an answer falls
  // BELOW the cut that was in force at the time. Mark those rows so the "why the
  // door said X" is on screen instead of something to reconstruct by hand.
  const breached = a => !a.failed && typeof a.answer === 'number'
      && typeof a.threshold === 'number' && a.answer < a.threshold;
  const jev = q.jev.map(a => a.failed
      ? `<div><span title="${esc(qTitle(a))}">${esc(qLabel(a))}</span><span class="fail">FAILED</span></div>`
      : `<div${breached(a) ? ' class="breach"' : ''}><span title="${esc(qTitle(a))}">${esc(qLabel(a))}</span><span>${esc(fmt(a.answer))}${breached(a) ? ` <span class="cut">(cut ${esc(fmt(a.threshold))} \u2717)</span>` : ''}</span></div>`).join('');
  const why = q.jev.filter(breached)
      .map(a => `${qLabel(a)} ${fmt(a.answer)} &lt; ${fmt(a.threshold)}`).join(' · ');
  const held = q.holdback ? '<span class="holdback">holdback: admitted</span>' : '';
  main.innerHTML = `
    <section>
      <h2>Request</h2><pre>${esc(q.text)}</pre>
      <h2>Jev</h2><div class="jev">${jev}</div>
      <h2>Door said</h2><div><span class="door ${esc(q.door)}">${esc(q.door)}</span>${held}</div>
      ${why ? `<div class="why">why: ${why}</div>` : ''}
    </section>
    <section>
      <h2>Ticket</h2><pre>${esc(q.ticket)}</pre>
    </section>`;
  document.getElementById('note').value = q.note || '';
  document.getElementById('count').textContent += q.grade ? ` · graded: ${q.grade}` : '';
}
async function grade(v) {
  const q = queue[i]; if (!q) return;
  const note = document.getElementById('note').value;
  await fetch('/api/grade', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request_id: q.request_id, verdict: v, note }) });
  q.grade = v; q.note = note;
  const next = queue.findIndex((x, j) => j > i && !x.grade);
  i = next < 0 ? Math.min(i + 1, queue.length - 1) : next;
  render();
}
document.addEventListener('keydown', e => {
  if (e.target.id === 'note' && e.key !== 'Escape' && !e.metaKey && !e.ctrlKey) return;
  if (KEYS[e.key]) { e.preventDefault(); grade(KEYS[e.key]); }
  else if (e.key === 'ArrowRight') { i = Math.min(i + 1, queue.length - 1); render(); }
  else if (e.key === 'ArrowLeft')  { i = Math.max(i - 1, 0); render(); }
});
for (const b of document.querySelectorAll('button[data-v]'))
  b.onclick = () => grade(b.dataset.v);
load();
</script></body></html>
"""


def build_queue(*, records: list[dict] | None = None) -> list[dict]:
    """The gradeable requests: text, Jev's answers, the door's call, the ticket, any grade.

    Ordered oldest-first, one item per request that has both a sample and a log row —
    a request with only one of the two is a half-written turn and is not gradeable.
    """
    records = load_records() if records is None else records
    samples = load_samples()
    grades = load_grades()

    by_request: dict[str, list[dict]] = {}
    for record in records:
        rid = record.get("request_id")
        if isinstance(rid, str):
            by_request.setdefault(rid, []).append(record)

    items: list[dict] = []
    for rid, sample in samples.items():
        rows = by_request.get(rid)
        if not rows:
            continue
        door = rows[0].get("action_computed", "proceed")
        held = bool(rows[0].get("holdback"))
        jev = [
            {
                "q": row.get("question_id"),
                "means": question_legend(row.get("question_id", "")),
                "answer": row.get("answer"),
                "threshold": row.get("threshold_at_time"),
                "failed": bool(row.get("failed")),
            }
            for row in rows
        ]
        grade = grades.get(rid)
        items.append(
            {
                "request_id": rid,
                "text": sample.text,
                "ticket": sample.ticket,
                "door": door,
                "holdback": held,
                "jev": jev,
                "ts": sample.ts,
                "grade": grade.verdict.value if grade else None,
                "note": grade.note if grade else "",
            }
        )
    items.sort(key=lambda item: item.get("ts") or "")
    return items


async def index(_request: Request) -> HTMLResponse:
    return HTMLResponse(PAGE)


async def queue(_request: Request) -> JSONResponse:
    return JSONResponse(build_queue())


async def grade(request: Request) -> JSONResponse:
    body = await request.json()
    request_id = body.get("request_id")
    verdict = body.get("verdict")
    if not request_id or not verdict:
        return JSONResponse({"error": "request_id and verdict are required"}, status_code=400)
    try:
        saved = append_grade(request_id, verdict, note=body.get("note", ""))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"request_id": saved.request_id, "verdict": saved.verdict.value})


async def summary(_request: Request) -> JSONResponse:
    """The read surface Homepage and Dagu consume: is the door in sync?"""
    return JSONResponse(build_summary())


def create_app() -> Starlette:
    """The grading server. Read-only over the logs; writes only grades."""
    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/queue", queue),
            Route("/api/grade", grade, methods=["POST"]),
            Route("/api/summary", summary),
            Route("/healthz", lambda _r: JSONResponse({"status": "ok"})),
        ]
    )


app = create_app()


def main(argv: Iterable[str] | None = None) -> int:
    """`python -m agent.grader [--port 8801]` — serve the grading UI on localhost."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Grade the gate's decisions, one request at a time.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default localhost)")
    parser.add_argument("--port", type=int, default=8801)
    args = parser.parse_args(list(argv) if argv is not None else None)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
