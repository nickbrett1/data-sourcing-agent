"""Alert on drift: push the gate status to ntfy.

The drift check (`python -m agent.summary --check`) exits non-zero so a scheduler
*notices* — but a scheduler's failure state is only seen if you look. This turns
that failure into a push: one POST to ntfy carrying the first issue.

The topic is the address (a random suffix keeps a public topic unguessable), so it
is env-overridable and not buried in a YAML file:

* `NTFY_URL`   — default `https://ntfy.sh`
* `NTFY_TOPIC` — default `data-sourcing-jev-threshold-9831`

No dependency: the app image is a slim Python and needs no HTTP client for one POST.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Sequence

NTFY_URL_ENV = "NTFY_URL"
NTFY_TOPIC_ENV = "NTFY_TOPIC"
DEFAULT_URL = "https://ntfy.sh"
DEFAULT_TOPIC = "data-sourcing-jev-threshold-9831"


def notify(
    message: str,
    *,
    url: str | None = None,
    topic: str | None = None,
    title: str = "gate out of sync",
) -> str:
    """POST one message to ntfy. Returns the server's response body.

    A plain `urllib` POST: `Title`/`Tags` ride as headers, the body is the message.
    Raises on failure so a caller (Dagu) sees the alert itself die rather than
    silently swallow a drift.
    """
    base = (url or os.environ.get(NTFY_URL_ENV) or DEFAULT_URL).rstrip("/")
    target = topic or os.environ.get(NTFY_TOPIC_ENV) or DEFAULT_TOPIC
    request = urllib.request.Request(
        f"{base}/{target}",
        data=message.encode("utf-8"),
        headers={"Title": title, "Tags": "warning"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return response.read().decode("utf-8", "replace")


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m agent.notify` — push the current status if it is not in sync.

    This is what the Dagu drift job calls from `handlerOn.failure`: the check has
    already failed, so the summary is read once more and its reason is sent.
    """
    from agent.summary import build_summary

    summary = build_summary()
    if summary["status"] == "ok":
        print("in sync; no alert sent")
        return 0
    body = "\n".join(summary["issues"])
    notify(body)
    print(f"alerted: {summary['issue']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
