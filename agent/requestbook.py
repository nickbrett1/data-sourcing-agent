"""The request book — where a rendered ticket becomes a durable draft.

Design memo `data-request-book-v1` §4. Until this module existed the agent's
write path terminated in a chat artifact: `agent/ticket.py` rendered the YAML and
`agent/main.py` published it as an A2A artifact, but nothing wrote it down (§1).
This module is the write: a validated ticket lands as a file in the book's
`inbox/`, and nowhere else.

Two properties the design rests on, and both are structural rather than
instructed:

* **write scope** — the agent is handed the book's `inbox/` mount and nothing
  else (§4.1). This module can only ever write under the inbox it is given; it
  never reaches for `approved/`, `in_flight/` or `done/`, because it is not
  given them.
* **`state: draft`** — the ticket type pins `state` to `^draft$`
  (`agent/ticket.py`), so a file this module writes cannot be approvable by
  construction. `approved` stays a human's to write.

The book itself — the repo, the mount, the NAS autocommit — is §3; this module
is §4.

## Naming

§4.2 gives a *filer* the naming job
(`<YYYY-MM-DD>-<dataset>-<schema>-<slug>.yaml`) and lists naming among the three
things "the agent should not" own. We chose the no-filer option (§4.1): `inbox/`
*is* the book's `inbox/`, mounted directly, so the agent names its own file. The
pattern is kept, with the turn's `request_id` appended as a final token — that is
the join key an approval needs (§5: an approve/reject is *keyed by* `request_id`),
and without it a file in `inbox/` could not be tied back to the gate-log row a
human is grading. The date is the filing date (UTC), so `ls inbox/` sorts by when
drafts arrived.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime
from pathlib import Path

from agent.ticket import Ticket, render_ticket_yaml

#: The inbox the agent writes into. The container mounts the book's `inbox/`
#: here (`docker-compose.yml`, §4.1); the default is that mount. It is
#: overridable so a test — or a local run outside the container — can point it
#: somewhere harmless, and it is deliberately the *only* book path this module
#: knows: the rest of the book is not the agent's to touch.
INBOX_ENV = "REQUESTS_INBOX_DIR"
DEFAULT_INBOX = "/requests/inbox"

#: What survives into a filename token: letters, digits, dot, dash, underscore.
#: Everything else — a slash in a symbol, a space — collapses to a dash, so a
#: token can never become a path separator.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Bounded collision retries. Names carry a unique `request_id`, so a collision
#: is all but impossible; the loop is belt-and-braces against a clobber.
_MAX_COLLISIONS = 100


def inbox_dir() -> Path:
    """The inbox to write into: `REQUESTS_INBOX_DIR`, else the container mount."""
    return Path(os.environ.get(INBOX_ENV) or DEFAULT_INBOX)


def _token(text: str) -> str:
    """One path-safe filename token, or `-` if nothing survives sanitising."""
    return _UNSAFE.sub("-", text).strip("-.") or "-"


def draft_name(ticket: Ticket, *, request_id: str, on: date) -> str:
    """The filename for a ticket: `<date>-<dataset>-<schema>-<slug>-<request_id>.yaml`.

    `slug` is a short, readable hint at the symbols, for the human scanning the
    inbox; the file's content is the authority on everything it abbreviates.
    """
    request = ticket.request
    symbols = "-".join(request.symbols[:3]) if request.symbols else "symbols"
    slug = _token(f"{symbols}-{request.start.isoformat()}-{request.end.isoformat()}")
    stem = "-".join(
        (
            on.isoformat(),
            _token(request.dataset),
            _token(request.schema_name),
            slug,
            _token(request_id),
        )
    )
    return f"{stem}.yaml"


def write_draft(
    ticket: Ticket,
    *,
    request_id: str,
    inbox: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write one validated ticket into the book's `inbox/`; return the path.

    The write is **exclusive and non-destructive**: `O_EXCL` means an existing
    file is never overwritten — a re-run of the same request mints a new
    `request_id` and so a new file, rather than destroying the first draft.
    Nothing here moves a file between regions: promotion out of `inbox/` is a
    human's `rename()` (§3.2), not this module's.
    """
    target_dir = inbox if inbox is not None else inbox_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    name = draft_name(ticket, request_id=request_id, on=(now or datetime.now(UTC)).date())
    rendered = render_ticket_yaml(ticket)
    stem, suffix = name[:-5], name[-5:]  # split off the ".yaml" we just added
    for attempt in range(_MAX_COLLISIONS):
        candidate = target_dir / (name if attempt == 0 else f"{stem}-{attempt + 1}{suffix}")
        try:
            handle = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(rendered)
        return candidate
    raise FileExistsError(f"could not find a free draft name for {name!r}")
