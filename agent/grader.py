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
  .verdict { margin-top: 10px; color: #7d8896; }
  .verdict b { color: #6ee7a8; }
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
  /* The legend is a one-line summary that expands on tap -- it is reference, not
     something to keep on screen, and open it ate a third of a phone display. */
  #legendbox { flex: 1 1 100%; order: -2; }
  #legendbox summary { color: #7d8896; font-size: 12px; cursor: pointer; }
  #legend { margin: 6px 0 0; color: #7d8896; font-size: 12px; line-height: 1.5; }
  #legend b { color: #cfd6dd; font-weight: 600; }
  #note { flex: 1; min-width: 200px; font: inherit; background: #1d2228; color: inherit;
          border: 1px solid #39414b; border-radius: 4px; padding: 6px 10px; }
  #controls, #donebar { display: flex; flex: 1 1 100%; gap: 10px; align-items: center; flex-wrap: wrap; }
  /* `hidden` must win over the flex rule, or the grade buttons stay on screen in
     the done state and invite accidental re-grades. */
  #controls[hidden], #donebar[hidden] { display: none; }
  /* Navigation, not a verdict: set apart from the four grade buttons. The id
     selectors also beat the phone rule that stretches every `button` to 50%. */
  #prev, #skip { flex: 0 0 auto; background: transparent; color: #9aa5b1; }
  #tally { color: #7d8896; }
  #summary ul { list-style: none; padding: 0; margin: 10px 0; }
  #summary li { padding: 3px 0; }
  #summary .hint { color: #7d8896; font-size: 13px; }
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
    #legendbox summary { font-size: 13px; }
    #legend { font-size: 13px; }
    button { flex: 1 1 calc(50% - 8px); min-height: 48px; font-size: 15px; }
  }
</style></head><body>
<header>
  <strong>Grade the door</strong>
  <span id="count">…</span>
  <div id="bar"><div id="fill"></div></div>
  <span id="hint">1 right · 2 too strict · 3 too lenient · 4 unclear · ← → / s skip</span>
</header>
<main id="main"><div id="empty">Loading…</div></main>
<footer>
  <details id="legendbox">
    <summary>what do the verdicts mean?</summary>
    <p id="legend"><b>right</b> the door did the right thing ·
       <b>too strict</b> it stopped a fine request (lower the cut) ·
       <b>too lenient</b> it let a bad one through (raise the cut) ·
       <b>unclear</b> the sample doesn't settle it (moves nothing)</p>
  </details>
  <div id="controls">
    <input id="note" placeholder="note (optional, saved with the verdict)">
    <button data-v="right">1 · right</button>
    <button data-v="too_strict">2 · too strict</button>
    <button data-v="too_lenient">3 · too lenient</button>
    <button data-v="unclear">4 · unclear</button>
    <button id="prev" type="button" title="previous item (no grade)">‹ prev</button>
    <button id="skip" type="button" title="next item (no grade)">skip ›</button>
  </div>
  <div id="donebar" hidden>
    <span id="tally"></span>
    <button id="review">review / re-grade →</button>
  </div>
</footer>
<script>
let queue = [], i = 0, review = false;
// The queue is done when nothing is left ungraded. Then the page leads with the
// result, not with a request: grade controls are hidden until the grader asks to
// review/re-grade (the history has value, but it is not the front door).
const isDone = () => queue.length > 0 && queue.every(x => x.grade);
const tally = () => {
  const t = { right: 0, too_strict: 0, too_lenient: 0, unclear: 0 };
  for (const x of queue) if (x.grade) t[x.grade] = (t[x.grade] || 0) + 1;
  return t;
};
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
  const done = isDone();
  document.getElementById('count').textContent = done
      ? `all ${queue.length} graded · nothing left`
      : `${i + 1} / ${queue.length} · ${graded} graded`;
  document.getElementById('fill').style.width = queue.length ? (graded / queue.length * 100) + '%' : '0';

  // Done = result-first. Show the grade controls only in review mode, so the
  // completed screen does not sit there inviting accidental re-grades.
  document.getElementById('controls').hidden = done && !review;
  document.getElementById('donebar').hidden = !done;
  document.getElementById('review').textContent = review ? '← back to summary' : 'review / re-grade →';

  const main = document.getElementById('main');
  if (done && !review) {
    const t = tally(), n = queue.length;
    main.innerHTML = `
      <section id="summary">
        <h2>Done</h2>
        <p>All ${n} requests graded.</p>
        <ul>
          <li><b>${t.right}</b> right</li>
          <li><b>${t.too_strict}</b> too strict — lower the cut</li>
          <li><b>${t.too_lenient}</b> too lenient — raise the cut</li>
          <li><b>${t.unclear}</b> unclear — moves nothing</li>
        </ul>
        <p class="hint">The history is kept in the log. Use “review / re-grade” to look back.</p>
      </section>`;
    document.getElementById('tally').textContent =
      `${t.right} right · ${t.too_strict} too strict · ${t.too_lenient} too lenient · ${t.unclear} unclear`;
    return;
  }

  const q = queue[i];
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
      <div class="verdict">your verdict: <b>${q.grade ? esc(q.grade) : 'not graded yet'}</b>${done && review ? ' · reviewing' : ''}</div>
    </section>
    <section>
      <h2>Ticket</h2><pre>${esc(q.ticket)}</pre>
    </section>`;
  document.getElementById('note').value = q.note || '';
}
async function grade(v) {
  const q = queue[i]; if (!q) return;
  const note = document.getElementById('note').value;
  // Re-tapping the same verdict on an already-graded item is a no-op: skip the
  // write (no duplicate history rows), just move on.
  if (q.grade === v && (q.note || '') === note) { advance(); return; }
  const res = await fetch('/api/grade', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request_id: q.request_id, verdict: v, note }) });
  if (!res.ok) { alert('Could not save the grade (HTTP ' + res.status + ').'); return; }
  q.grade = v; q.note = note;
  advance();
}
// Move without grading. While work remains, "next" means the next ungraded item
// (wrap if it is behind you); once everything is graded, it is simply the next
// item, so re-grade review can walk the whole history. `prev` is a plain step.
function skipItem() {
  if (!queue.length) return;
  if (!isDone()) {
    const after = queue.findIndex((x, j) => j > i && !x.grade);
    if (after >= 0) { i = after; render(); return; }
    const anywhere = queue.findIndex(x => !x.grade);
    if (anywhere >= 0) { i = anywhere; render(); return; }
  }
  i = (i + 1) % queue.length;
  render();
}
function prevItem() {
  if (!queue.length) return;
  i = (i - 1 + queue.length) % queue.length;
  render();
}
function advance() {
  // Next ungraded after here; failing that, wrap to any item still ungraded; else
  // stay put (render() then shows the "nothing left" state). Without the wrap,
  // grading the LAST item left the screen frozen at the end of the queue — which
  // reads as "nothing happened", so the same request got graded again and again.
  const after = queue.findIndex((x, j) => j > i && !x.grade);
  const anywhere = queue.findIndex(x => !x.grade);
  if (after >= 0) i = after;
  else if (anywhere >= 0) i = anywhere;
  render();
}
document.getElementById('review').onclick = () => {
  review = !review;
  if (review) i = 0;
  render();
};
document.getElementById('skip').onclick = skipItem;
document.getElementById('prev').onclick = prevItem;
document.addEventListener('keydown', e => {
  if (isDone() && !review) return;          // the summary screen takes no grade keys
  if (e.target.id === 'note' && e.key !== 'Escape' && !e.metaKey && !e.ctrlKey) return;
  if (KEYS[e.key]) { e.preventDefault(); grade(KEYS[e.key]); }
  else if (e.key === 'ArrowRight' || e.key === 's') { e.preventDefault(); skipItem(); }
  else if (e.key === 'ArrowLeft')                   { e.preventDefault(); prevItem(); }
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
