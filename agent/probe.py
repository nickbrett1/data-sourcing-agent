"""Registration-vs-reachability probe: is every registered agent actually dialable?

The defect this exists for is the recurring "a green signal that proves
nothing". An agent registers a card with LiteLLM and the registration succeeds -
but the *URL on that card* is one the caller cannot dial. It has happened twice
in this fleet:

* `CARD_URL = http://0.0.0.0:8700` - a **bind** address. LiteLLM reads it and
  dials itself. Registration is green; every invocation fails.
* a container-internal name that does not resolve from the caller's network.

Both are invisible to the two checks we already have:

* the **served-card wire test** (`tests/test_a2a_wire.py`) dials the endpoint
  in-process through a `TestClient`, so it never uses the advertised URL - and a
  `CARD_URL` of `0.0.0.0` still serves a card fine.
* the **startup guard** (`resolve_card_url`) refuses a loopback/wildcard value
  syntactically - correct, but it is a *syntax* check on one string, and it
  cannot see a name that resolves from here and not from there, nor whether the
  card that answers is the one that was registered.

Reachability is a property of a **pair** (address, vantage): a name that resolves
on `ai_proxy` need not resolve on the NAS host, and vice versa. So this probe
does the one thing neither check can: it reads the registry, then **dials each
advertised URL from wherever the probe is running**, and compares the card that
answers against what was registered.

Run it from a container that shares the caller's network - from inside the agent
itself is the representative vantage, because the caller (the LiteLLM proxy)
reaches the fleet over the same `ai_proxy` network:

    docker compose run --rm app python -m agent.probe

It exits non-zero if any agent is undialable, unreachable, or serving a card
that disagrees with its registration, so it can gate a deploy. In a deploy,
scope it to the agent being deployed — `python -m agent.probe --agent <name>` —
so an unrelated agent's outage cannot fail this one's release, while a name that
is not registered at all still fails it.

What "dialable" proves: the well-known card and the JSON-RPC endpoint are the
same host and port, so a successful card fetch is proof the caller can reach the
transport; a failed one is proof it cannot. This is deliberately *not* a full
A2A turn - a turn costs a model run, and reachability is a host:port question.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import sys
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

# The A2A standard well-known path, served by `create_agent_card_routes` on the
# same host:port as the JSON-RPC endpoint (agent/main.py).
CARD_PATH = "/.well-known/agent-card.json"

DEFAULT_BASE_URL = "http://litellm:4000"
DEFAULT_TIMEOUT = 5.0

# The registry read gets its own, more generous timeout. It is a control-plane
# call to the proxy, whose *first* request after a restart can exceed the short
# per-card timeout (observed: a cold `GET /v1/agents` ReadTimeout at 5 s while a
# warm one answered in milliseconds). A deploy gate that flakes on a cold proxy
# is worse than useless, so the registry read is slower and retried once, while
# the per-card dials stay short — reachability is still a quick yes/no.
REGISTRY_TIMEOUT = 15.0
REGISTRY_ATTEMPTS = 2

# Verdicts, worst first. `OK` is the only good one.
OK = "OK"
MISMATCH = "MISMATCH"
UNREACHABLE = "UNREACHABLE"
UNDIALABLE = "UNDIALABLE"
VERDICT_ORDER = {UNDIALABLE: 0, UNREACHABLE: 1, MISMATCH: 2, OK: 3}


@dataclass(frozen=True)
class Registration:
    """What the registry says about one agent."""

    agent_id: str | None
    name: str
    url: str
    protocol_version: str | None


@dataclass
class Finding:
    """One agent's verdict, with the evidence for it."""

    registration: Registration
    verdict: str
    detail: str
    latency_ms: float | None = None
    served: dict | None = None

    @property
    def ok(self) -> bool:
        return self.verdict == OK


def parse_registry(payload: object) -> list[Registration]:
    """The agents in a `GET /v1/agents` body.

    Both `{"agents": [...]}` and a bare list have been seen on the live proxy
    (agent/register.py notes the same), so the roster is read either way rather
    than guessed at. A row without a `url` is kept: an empty URL is itself a
    finding (`UNDIALABLE`), not a reason to drop the agent from the report.
    """
    rows = payload.get("agents", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []

    registrations: list[Registration] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        card = row.get("agent_card_params")
        card = card if isinstance(card, dict) else {}
        registrations.append(
            Registration(
                agent_id=row.get("agent_id"),
                name=str(row.get("agent_name") or card.get("name") or ""),
                url=str(card.get("url") or ""),
                protocol_version=card.get("protocolVersion"),
            )
        )
    return registrations


def undialable_reason(url: str) -> str | None:
    """Why a caller on the network cannot dial this URL, or None if it can.

    The syntactic half of the defect: a wildcard *bind* address or a loopback
    address is one that a caller resolves to **itself**, so the registration is
    green and every invocation reaches the wrong host. This mirrors
    `resolve_card_url`'s refusal in the generator - deliberately, because the
    probe must name the defect the guard prevents, not merely trust that the
    guard ran.
    """
    if not url or not url.strip():
        return "empty url"

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} is not http(s)"

    host = parsed.hostname
    if not host:
        return "url has no host"
    if host.lower() == "localhost":
        return f"loopback name {host!r} - the caller dials itself"

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None  # a name: resolution is decided by actually dialling it
    if ip.is_unspecified:
        return (
            f"wildcard bind address {host!r} - a caller dialling it reaches "
            "itself; register the name the caller resolves instead"
        )
    if ip.is_loopback:
        return f"loopback address {host!r} - the caller dials itself"
    return None


def _served_interface_url(card: dict) -> str | None:
    """The `url` of the served card's first JSON-RPC interface, if any.

    The served JSON is camelCase (`supportedInterfaces[].url`) because the SDK
    serialises the proto that way; a card that ships proto field names instead
    is read too, rather than reported as url-less.
    """
    interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces")
    if isinstance(interfaces, list):
        for interface in interfaces:
            if isinstance(interface, dict) and interface.get("url"):
                return str(interface["url"])
    url = card.get("url")
    return str(url) if url else None


def _served_protocol_version(card: dict) -> str | None:
    """The `protocolVersion` of the served card's first interface, if present."""
    interfaces = card.get("supportedInterfaces") or card.get("supported_interfaces")
    if isinstance(interfaces, list):
        for interface in interfaces:
            if isinstance(interface, dict) and interface.get("protocolVersion"):
                return str(interface["protocolVersion"])
    version = card.get("protocolVersion")
    return str(version) if version else None


def compare(
    registration: Registration,
    *,
    served: dict | None = None,
    error: str | None = None,
    latency_ms: float | None = None,
) -> Finding:
    """Turn one registration and what happened when we dialled it into a verdict.

    Kept pure (no I/O) so the classification is testable without a network, and
    so the *reasons* are the thing being pinned: `UNDIALABLE` (syntactically
    undialable), `UNREACHABLE` (dialled and did not answer), `MISMATCH` (answered,
    but the card disagrees with the registration), `OK`.
    """
    reason = undialable_reason(registration.url)
    if reason:
        return Finding(registration, UNDIALABLE, reason)

    if error is not None:
        return Finding(registration, UNREACHABLE, error, latency_ms)

    card = served or {}
    problems: list[str] = []

    served_name = card.get("name")
    if served_name and registration.name and served_name != registration.name:
        problems.append(
            f"served name {served_name!r} != registered {registration.name!r}"
        )

    served_version = _served_protocol_version(card)
    if served_version and registration.protocol_version:
        if served_version != registration.protocol_version:
            problems.append(
                f"served protocolVersion {served_version!r} != registered "
                f"{registration.protocol_version!r}"
            )

    served_url = _served_interface_url(card)
    if served_url and served_url.rstrip("/") != registration.url.rstrip("/"):
        problems.append(f"served url {served_url!r} != registered {registration.url!r}")

    if problems:
        return Finding(registration, MISMATCH, "; ".join(problems), latency_ms, card)
    return Finding(
        registration,
        OK,
        "reachable; the served card agrees with the registration",
        latency_ms,
        card,
    )


async def _dial(
    client: httpx.AsyncClient, registration: Registration, card_path: str
) -> Finding:
    """Fetch the advertised card from its own host:port and classify the result."""
    if undialable_reason(registration.url):
        # Do not even try: dialling 0.0.0.0 reaches *this* process, which would
        # answer OK and hide the defect. Report it as the syntax it is.
        return compare(registration)

    url = registration.url.rstrip("/") + card_path
    start = time.monotonic()
    try:
        response = await client.get(url)
    except Exception as exc:
        return compare(
            registration,
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    latency_ms = (time.monotonic() - start) * 1000
    if response.status_code >= 400:
        return compare(
            registration,
            error=f"HTTP {response.status_code} from {url}",
            latency_ms=latency_ms,
        )
    try:
        card = response.json()
    except ValueError:
        return compare(
            registration,
            error=f"the response from {url} is not JSON",
            latency_ms=latency_ms,
        )
    return compare(
        registration,
        served=card if isinstance(card, dict) else {},
        latency_ms=latency_ms,
    )


async def probe(
    *,
    base_url: str | None = None,
    key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
    card_path: str = CARD_PATH,
    only: Sequence[str] | None = None,
) -> list[Finding]:
    """Read the registry, dial every advertised URL, and report per agent.

    `transport` is an injection point for tests: the probe talks to the network
    and the tests must not, so the whole registry-and-dial flow is driven through
    an `httpx.MockTransport` (the same pattern as the Validator's tests).

    `only` restricts the probe to the named agents, so a deploy can gate on its
    **own** reachability without failing because an unrelated agent is down. A
    name in `only` that is not in the registry is not silently ignored - the
    caller (see `main`) reports it as a failure, because "my agent is not
    registered" is exactly what a deploy gate is for.
    """
    base_url = base_url or os.environ.get("LITELLM_BASE_URL", DEFAULT_BASE_URL)
    if key is None:
        # The listing is filtered by the calling key's owner (see
        # agent/register.py), so the master key is the one that sees the whole
        # fleet; a virtual key is accepted as a fallback for a scoped probe.
        key = os.environ.get("LITELLM_MASTER_KEY") or os.environ.get(
            "LITELLM_API_KEY", ""
        )
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        response = await _read_registry(
            client, f"{base_url.rstrip('/')}/v1/agents", headers
        )
        registrations = parse_registry(response.json())
        if only:
            wanted = set(only)
            registrations = [reg for reg in registrations if reg.name in wanted]
        return [await _dial(client, reg, card_path) for reg in registrations]


async def _read_registry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    attempts: int = REGISTRY_ATTEMPTS,
) -> httpx.Response:
    """Read the registry, retrying a cold proxy's slow first response.

    Retried because the failure being avoided is transient (a proxy that has not
    served a request yet), and re-raised when it is not, so `main` still reports
    an unreadable registry distinctly (exit 2) rather than as a fleet finding.
    """
    last: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            response = await client.get(url, headers=headers, timeout=REGISTRY_TIMEOUT)
            response.raise_for_status()
            return response
        except Exception as exc:
            last = exc
    assert last is not None  # attempts >= 1, so a failure set it
    raise last


def _format_report(findings: list[Finding]) -> str:
    """A fixed-width table, worst verdict first, then the failing details."""
    ordered = sorted(
        findings,
        key=lambda f: (VERDICT_ORDER.get(f.verdict, 99), f.registration.name),
    )
    lines: list[str] = []
    name_width = max((len(f.registration.name) for f in ordered), default=4)
    name_width = max(name_width, len("AGENT"))
    lines.append(f"{'AGENT':<{name_width}}  {'VERDICT':<11}  URL")
    for finding in ordered:
        url = finding.registration.url or "(none)"
        lines.append(
            f"{finding.registration.name:<{name_width}}  {finding.verdict:<11}  {url}"
        )

    failing = [f for f in ordered if not f.ok]
    if failing:
        lines.append("")
        for finding in failing:
            latency = (
                f" ({finding.latency_ms:.0f} ms)"
                if finding.latency_ms is not None
                else ""
            )
            lines.append(f"  {finding.registration.name}: {finding.detail}{latency}")
    return "\n".join(lines)


async def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.probe",
        description=(
            "Probe every agent in the LiteLLM registry: dial its advertised card "
            "URL from here and compare it to the registration. Run from a "
            "container on the caller's network (ai_proxy)."
        ),
    )
    parser.add_argument(
        "--base-url", default=None, help="LiteLLM base (default: $LITELLM_BASE_URL)"
    )
    parser.add_argument(
        "--key", default=None, help="master key (default: $LITELLM_MASTER_KEY)"
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--agent",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "only probe this agent (repeatable). Use in a deploy to gate on this "
            "agent's own reachability without failing on an unrelated agent's "
            "outage; a name that is not registered is reported as a failure."
        ),
    )
    args = parser.parse_args(argv)

    try:
        findings = await probe(
            base_url=args.base_url,
            key=args.key,
            timeout=args.timeout,
            only=args.agent,
        )
    except Exception as exc:
        print(f"probe failed to read the registry: {exc!r}", file=sys.stderr)
        return 2

    seen = {finding.registration.name for finding in findings}
    missing = [name for name in (args.agent or []) if name not in seen]

    if not findings and not missing:
        print("the registry is empty - nothing to probe")
        return 0

    if findings:
        print(_format_report(findings))
    for name in missing:
        print(f"  {name}: not present in the registry", file=sys.stderr)

    failing = [finding for finding in findings if not finding.ok]
    if failing or missing:
        if failing:
            print(
                f"\n{len(failing)}/{len(findings)} agents are not dialable/consistent "
                "from this vantage.",
                file=sys.stderr,
            )
        if missing:
            print(
                f"\n{len(missing)} requested agent(s) are not registered at all.",
                file=sys.stderr,
            )
        return 1
    print(
        f"\nall {len(findings)} agents are dialable and consistent from this vantage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
