"""Durable task history, read back out of the A2A server's own task store.

roost's History panel asks an agent to answer `history.*` about its own past
sessions. For a Pydantic AI agent there is no goose `sessions.db`, which is why
the first cut of the roost client advertised only `activity`/`status` and
refused `history.*` outright. That refusal was right *then* - there was no
store - but it was wrong as a permanent position, because **the A2A server
already keeps exactly this data**:

  * `contextId` groups tasks -> a **session**;
  * each `Task` in that context -> a **turn**;
  * `task.history` (and the task's artifacts) -> the **transcript**.

The transcript is therefore not invented, and no second schema is introduced:
this module reads the same state the A2A dispatcher writes. The only thing that
made the old position true was that the default `InMemoryTaskStore` discards
that state on restart - so this change persists it (see `build_task_store`).

Reading is done through the `TaskStore` interface (`list`), not by reaching into
the SQLite file, so the same code works over the durable store in production and
the in-memory store in a test. The one caveat the store imposes is that `list`
is owner-scoped: the A2A handler saves under the request's user (unauthenticated
-> owner `""`), and this module queries with a fresh `ServerCallContext()`
(the same default), so the owners line up. The client is not authenticated
today, so that is the whole story for now.

Fail-open by construction: `build_task_store` returns `None` rather than raising
when SQLAlchemy/the driver is unavailable, and the roost handlers answer
`ok:false` rather than crashing. A history problem must never stop the agent
serving.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.types.a2a_pb2 import ListTasksRequest, Role, Task

# `AGENT_STATE_DIR` is the same writable mount agent/register.py already uses for
# the remembered gateway id; reusing it means history survives a Watchtower
# recreate with no new environment variable and no new mount.
STATE_DIR_ENV = "AGENT_STATE_DIR"
TASKS_DB_FILENAME = "a2a_tasks.db"

# Retention: keep the newest N sessions, and nothing older than D days. Swept on
# startup so the SQLite file cannot grow without bound. `retained` in
# `status.get` reports what is actually still present after the sweep.
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_AGE_DAYS = 30

# How many tasks to request per page when draining the store. Larger than the
# SDK's 50 default so a normal agent's whole history is a couple of round-trips.
_TASK_PAGE_SIZE = 200
# Default page size for `history.messages`. The roost UI walks forward through
# cursors until it runs out; a generous default means a normal session is one
# page and only a genuinely long transcript paginates.
_MESSAGE_PAGE_SIZE = 500
_DEFAULT_SEARCH_LIMIT = 50
# A title is the first user message, clipped to something a list row can hold.
_TITLE_MAX = 80


def state_dir(agent_name: str) -> Path:
    """The writable state directory: `$AGENT_STATE_DIR`, else a per-agent default.

    Mirrors agent/register.py so the remembered id and the task database live
    beside each other, and one mount keeps both across a redeploy.
    """
    configured = os.environ.get(STATE_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".local" / "state" / "pydantic-agent" / agent_name


def task_store_path(agent_name: str) -> Path:
    """Where the SQLite task database lives."""
    return state_dir(agent_name) / TASKS_DB_FILENAME


def build_task_store(agent_name: str) -> TaskStore | None:
    """Build the durable SQLite task store, or `None` when it cannot be built.

    `None` is honest fail-open: the caller falls back to the in-memory store and
    the agent keeps serving. The alternative (raise) would turn a missing
    optional dependency into a dead agent.
    """
    try:
        from a2a.server.tasks import DatabaseTaskStore
        from sqlalchemy.ext.asyncio import create_async_engine
    except Exception as exc:  # pragma: no cover - depends on the install
        print(
            f"[history] SQLAlchemy is not installed, so task history will not "
            f"survive a restart (the agent keeps serving): {exc!r}",
            flush=True,
        )
        return None

    try:
        path = task_store_path(agent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        return DatabaseTaskStore(engine=engine)
    except Exception as exc:  # pragma: no cover - depends on the environment
        print(
            f"[history] could not open the task store at "
            f"{task_store_path(agent_name)}; history will be in-memory only "
            f"(the agent keeps serving): {exc!r}",
            flush=True,
        )
        return None


def _role_name(role: int) -> str:
    """The roost transcript role for an A2A message role."""
    if role == Role.ROLE_USER:
        return "user"
    if role == Role.ROLE_AGENT:
        return "assistant"
    return "unknown"


def _part_text(parts: Any) -> str:
    """Join the text of an A2A message/artifact's parts, skipping non-text parts."""
    return "\n".join(part.text for part in parts if part.text)


def _task_time(task: Task) -> datetime | None:
    """The task's status timestamp as aware UTC, or None when it has none.

    The A2A pipeline stamps every status update, so a task that came through the
    handler always has one; a directly-saved task may not.
    """
    if task.status.HasField("timestamp"):
        return task.status.timestamp.ToDatetime(tzinfo=UTC)
    return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _truncate(text: str) -> str:
    """Collapse whitespace and clip, for a session title."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TITLE_MAX:
        return collapsed
    return collapsed[: _TITLE_MAX - 1].rstrip() + "\u2026"


def _parse_cursor(cursor: Any) -> int:
    try:
        return max(0, int(cursor))
    except (TypeError, ValueError):
        return 0


def _coerce_limit(limit: Any, default: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


class TaskHistory:
    """Read-only views over a `TaskStore`, in the shapes roost's UI consumes.

    A "session" is a `contextId`; its "turns" are the tasks in that context; its
    transcript is every task's `history` messages followed by the rendered
    artifact (the agent's answer), in time order. The store is the source of
    truth; nothing here caches or duplicates it.
    """

    def __init__(
        self,
        store: TaskStore,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    ) -> None:
        self._store = store
        self._max_sessions = max_sessions
        self._max_age_days = max_age_days

    # -- reading -----------------------------------------------------------

    async def _all_tasks(self) -> list[Task]:
        """Every task in the store, draining the store's pagination.

        By default the store only ever lets a caller see tasks whose owner
        matches the call context; a fresh `ServerCallContext()` resolves to the
        same unauthenticated owner the handler saves under.
        """
        context = ServerCallContext()
        tasks: list[Task] = []
        token = ""
        while True:
            params = ListTasksRequest(page_size=_TASK_PAGE_SIZE)
            if token:
                params.page_token = token
            response = await self._store.list(params, context)
            tasks.extend(response.tasks)
            token = response.next_page_token
            if not token or not response.tasks:
                break
        return tasks

    def _group(self, tasks: list[Task]) -> list[tuple[str, list[Task]]]:
        """Group tasks by context, newest session first, turns oldest first."""
        groups: dict[str, list[Task]] = {}
        for task in tasks:
            groups.setdefault(task.context_id, []).append(task)

        ordered: list[tuple[str, list[Task]]] = []
        for context_id, turns in groups.items():
            turns.sort(key=lambda task: (_task_time(task) or _EPOCH, task.id))
            ordered.append((context_id, turns))
        ordered.sort(key=lambda item: self._last_activity(item[1]) or _EPOCH, reverse=True)
        return ordered

    @staticmethod
    def _last_activity(turns: list[Task]) -> datetime | None:
        times = [t for t in (_task_time(task) for task in turns) if t is not None]
        return max(times) if times else None

    @staticmethod
    def _first_activity(turns: list[Task]) -> datetime | None:
        times = [t for t in (_task_time(task) for task in turns) if t is not None]
        return min(times) if times else None

    @staticmethod
    def _messages(turns: list[Task]) -> list[dict]:
        """The ordered transcript for a session, with a stable global index.

        Each turn contributes its `history` messages, then the rendered artifact
        as the agent's answer (the artifact is what the turn *produced*; without
        it the transcript would be user prompts alone).
        """
        messages: list[dict] = []
        for task in turns:
            timestamp = _iso(_task_time(task))
            for message in task.history:
                messages.append(
                    {
                        "role": _role_name(message.role),
                        "text": _part_text(message.parts),
                        "createdAt": timestamp,
                    }
                )
            for artifact in task.artifacts:
                text = _part_text(artifact.parts)
                if text:
                    messages.append(
                        {"role": "assistant", "text": text, "createdAt": timestamp}
                    )
        for index, message in enumerate(messages):
            message["index"] = index
        return messages

    def _session_view(self, context_id: str, turns: list[Task]) -> dict:
        messages = self._messages(turns)
        first_user = next(
            (m["text"] for m in messages if m["role"] == "user" and m["text"]), ""
        )
        created = self._first_activity(turns)
        updated = self._last_activity(turns)
        return {
            # The fields the roost UI reads: sessionId, name, createdAt,
            # updatedAt, messageCount. `title`/`startedAt`/`lastActivity`/
            # `turnCount` are the same values under the names the design memo
            # used, kept so both vocabularies resolve.
            "sessionId": context_id,
            "contextId": context_id,
            "name": _truncate(first_user) or context_id,
            "title": _truncate(first_user) or context_id,
            "createdAt": _iso(created),
            "updatedAt": _iso(updated),
            "startedAt": _iso(created),
            "lastActivity": _iso(updated),
            "messageCount": len(messages),
            "turnCount": len(turns),
            "retained": True,
        }

    async def sessions(self) -> list[dict]:
        """Every session, newest first, with the metadata the UI lists."""
        tasks = await self._all_tasks()
        return [
            self._session_view(context_id, turns)
            for context_id, turns in self._group(tasks)
        ]

    async def session(self, session_id: str) -> dict | None:
        """One session's metadata, or None when the context is unknown."""
        tasks = await self._all_tasks()
        for context_id, turns in self._group(tasks):
            if context_id == session_id:
                return self._session_view(context_id, turns)
        return None

    async def messages(
        self, session_id: str, *, cursor: Any = None, limit: Any = None
    ) -> dict:
        """A page of a session's transcript, forward-only, matching roost's shape.

        The cursor is a message index (as the reference agent uses), so the view
        can grow a transcript downward without re-reading what it has.
        """
        tasks = await self._all_tasks()
        turns = next((t for cid, t in self._group(tasks) if cid == session_id), [])
        all_messages = self._messages(turns)
        start = _parse_cursor(cursor)
        size = _coerce_limit(limit, _MESSAGE_PAGE_SIZE)
        page = all_messages[start : start + size]
        end = start + len(page)
        next_cursor = str(end) if end < len(all_messages) else None
        return {"sessionId": session_id, "messages": page, "nextCursor": next_cursor}

    async def search(self, query: str, *, limit: int = _DEFAULT_SEARCH_LIMIT) -> dict:
        """Substring matches over stored message text, grouped by session."""
        needle = (query or "").strip().lower()
        if not needle:
            return {"matches": []}
        tasks = await self._all_tasks()
        matches: list[dict] = []
        for context_id, turns in self._group(tasks):
            hits = [
                message
                for message in self._messages(turns)
                if needle in (message["text"] or "").lower()
            ]
            if hits:
                view = self._session_view(context_id, turns)
                matches.append(
                    {"sessionId": context_id, "name": view["name"], "matches": hits}
                )
                if len(matches) >= limit:
                    break
        return {"matches": matches}

    async def status(self) -> dict:
        """The `status.get.sessions` block: what is actually present."""
        count = len(await self.sessions())
        return {"count": count, "retained": count}

    # -- retention ---------------------------------------------------------

    async def retention_sweep(self) -> int:
        """Drop sessions past the session-count or age bound. Returns tasks removed.

        Both bounds apply: the oldest sessions beyond `max_sessions` go, and so
        does anything whose last activity is older than `max_age_days`.
        """
        tasks = await self._all_tasks()
        grouped = self._group(tasks)
        cutoff = datetime.now(UTC) - timedelta(days=self._max_age_days)
        stale: list[str] = [
            context_id for context_id, _ in grouped[self._max_sessions :]
        ]
        for context_id, turns in grouped:
            last = self._last_activity(turns)
            if last is not None and last < cutoff:
                stale.append(context_id)

        removed = 0
        context = ServerCallContext()
        for context_id in set(stale):
            for task in next((t for cid, t in grouped if cid == context_id), []):
                try:
                    await self._store.delete(task.id, context)
                    removed += 1
                except Exception as exc:  # a failed delete must not fail startup
                    print(
                        f"[history] could not delete task {task.id!r} during the "
                        f"retention sweep: {exc!r}",
                        flush=True,
                    )
        if removed:
            print(
                f"[history] retention sweep removed {removed} task(s) "
                f"(keeping {self._max_sessions} sessions / {self._max_age_days} days)",
                flush=True,
            )
        return removed


# Aware UTC sentinel that sorts before any real timestamp.
_EPOCH = datetime.min.replace(tzinfo=UTC)
