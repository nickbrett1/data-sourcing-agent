"""Tests for the registration-vs-reachability probe.

The probe's whole value is the *classification*, and its whole risk is the
network. So the tests pin the classification on constructed inputs (`compare`,
`undialable_reason`, `parse_registry`) and drive the full registry-and-dial flow
through an `httpx.MockTransport` - the same pattern the Validator's tests use, and
for the same reason: the truth being pinned is ours, not a remote server's.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent.probe import (
    MISMATCH,
    OK,
    UNDIALABLE,
    UNREACHABLE,
    Registration,
    compare,
    main,
    parse_registry,
    probe,
    undialable_reason,
)


def _reg(
    url: str = "http://data-sourcing-agent:8700", **overrides: object
) -> Registration:
    base = {
        "agent_id": "0b5c6345-1038-4e6d-82e9-81c52e032fe9",
        "name": "data-sourcing-agent",
        "url": url,
        "protocol_version": "1.0",
    }
    base.update(overrides)
    return Registration(**base)  # type: ignore[arg-type]


def _card(
    name: str = "data-sourcing-agent",
    url: str = "http://data-sourcing-agent:8700",
    protocol_version: str = "1.0",
) -> dict:
    return {
        "name": name,
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": protocol_version,
            }
        ],
    }


# --- parse_registry -----------------------------------------------------------


def test_parse_registry_reads_both_shapes():
    row = {
        "agent_id": "abc",
        "agent_name": "data-sourcing-agent",
        "agent_card_params": {"url": "http://x:8700", "protocolVersion": "1.0"},
    }
    assert parse_registry({"agents": [row]})[0] == parse_registry([row])[0]


def test_parse_registry_keeps_a_url_less_row():
    # An agent with no url is a finding, not a row to drop: dropping it would
    # hide the very defect the probe exists for.
    found = parse_registry([{"agent_id": "abc", "agent_name": "x"}])
    assert len(found) == 1
    assert found[0].url == ""


def test_parse_registry_ignores_noise():
    assert parse_registry(None) == []
    assert parse_registry({"agents": "nope"}) == []
    assert parse_registry([1, "two", None]) == []


def test_parse_registry_falls_back_to_card_name():
    found = parse_registry(
        [{"agent_id": "a", "agent_card_params": {"name": "from-card", "url": "u"}}]
    )
    assert found[0].name == "from-card"


# --- undialable_reason --------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:8700",
        "http://[::]:8700",
        "http://127.0.0.1:8700",
        "http://localhost:8700",
    ],
)
def test_undialable_flags_self_addressed_urls(url):
    assert undialable_reason(url) is not None


@pytest.mark.parametrize(
    "url",
    ["", "   ", "ftp://host:8700", "http://"],
)
def test_undialable_flags_malformed_urls(url):
    assert undialable_reason(url) is not None


@pytest.mark.parametrize(
    "url",
    [
        "http://data-sourcing-agent:8700",
        "http://192.0.2.10:8700",
        "https://agent.example.com",
    ],
)
def test_undialable_accepts_a_dialable_address(url):
    assert undialable_reason(url) is None


# --- compare ------------------------------------------------------------------


def test_compare_ok_when_card_agrees():
    finding = compare(_reg(), served=_card(), latency_ms=3.0)
    assert finding.verdict == OK
    assert finding.ok


def test_compare_undialable_never_dials():
    finding = compare(_reg("http://0.0.0.0:8700"), served=_card())
    assert finding.verdict == UNDIALABLE
    # The reason must name the wildcard, not merely say "bad".
    assert "0.0.0.0" in finding.detail


def test_compare_unreachable_carries_the_error():
    finding = compare(_reg(), error="ConnectError: refused")
    assert finding.verdict == UNREACHABLE
    assert "refused" in finding.detail


def test_compare_mismatch_names_every_disagreement():
    finding = compare(
        _reg(),
        served=_card(
            name="someone-else", url="http://other:9000", protocol_version="0.3"
        ),
    )
    assert finding.verdict == MISMATCH
    assert "someone-else" in finding.detail
    assert "0.3" in finding.detail
    assert "http://other:9000" in finding.detail


def test_compare_tolerates_a_trailing_slash():
    finding = compare(
        _reg("http://data-sourcing-agent:8700"),
        served=_card(url="http://data-sourcing-agent:8700/"),
    )
    assert finding.verdict == OK


def test_compare_reads_proto_field_names_too():
    served = {
        "name": "data-sourcing-agent",
        "supported_interfaces": [
            {"url": "http://data-sourcing-agent:8700", "protocol_version": "1.0"}
        ],
    }
    assert compare(_reg(), served=served).verdict == OK


# --- probe (end to end, no network) -------------------------------------------


def _transport(
    *, registry: dict, cards: dict[str, httpx.Response | Exception]
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/agents":
            return httpx.Response(200, json=registry)
        result = cards.get(str(request.url))
        if result is None:
            return httpx.Response(404)
        if isinstance(result, Exception):
            raise result
        return result

    return httpx.MockTransport(handler)


def test_probe_classifies_a_whole_registry():
    registry = {
        "agents": [
            {
                "agent_id": "a",
                "agent_name": "good",
                "agent_card_params": {
                    "url": "http://good:8700",
                    "protocolVersion": "1.0",
                },
            },
            {
                "agent_id": "b",
                "agent_name": "wildcard",
                "agent_card_params": {
                    "url": "http://0.0.0.0:8700",
                    "protocolVersion": "1.0",
                },
            },
            {
                "agent_id": "c",
                "agent_name": "dead",
                "agent_card_params": {
                    "url": "http://dead:8700",
                    "protocolVersion": "1.0",
                },
            },
        ]
    }
    transport = _transport(
        registry=registry,
        cards={
            "http://good:8700/.well-known/agent-card.json": httpx.Response(
                200, json=_card(name="good", url="http://good:8700")
            ),
            "http://dead:8700/.well-known/agent-card.json": httpx.ConnectError(
                "refused"
            ),
        },
    )
    findings = asyncio.run(
        probe(base_url="http://litellm:4000", key="k", transport=transport)
    )
    by_name = {f.registration.name: f.verdict for f in findings}
    assert by_name == {"good": OK, "wildcard": UNDIALABLE, "dead": UNREACHABLE}


def test_probe_flags_a_mismatched_card():
    registry = {
        "agents": [
            {
                "agent_id": "a",
                "agent_name": "x",
                "agent_card_params": {"url": "http://x:8700", "protocolVersion": "1.0"},
            }
        ]
    }
    transport = _transport(
        registry=registry,
        cards={
            "http://x:8700/.well-known/agent-card.json": httpx.Response(
                200, json=_card(name="x", url="http://x:8700", protocol_version="0.3")
            )
        },
    )
    findings = asyncio.run(probe(key="k", transport=transport))
    assert findings[0].verdict == MISMATCH


def test_probe_flags_a_non_2xx_card():
    registry = {
        "agents": [
            {
                "agent_id": "a",
                "agent_name": "x",
                "agent_card_params": {"url": "http://x:8700"},
            }
        ]
    }
    transport = _transport(
        registry=registry,
        cards={
            "http://x:8700/.well-known/agent-card.json": httpx.Response(
                500, text="boom"
            )
        },
    )
    assert asyncio.run(probe(key="k", transport=transport))[0].verdict == UNREACHABLE


# --- main (exit code) ---------------------------------------------------------


def test_main_exits_zero_when_everything_is_ok(monkeypatch):
    registry = {
        "agents": [
            {
                "agent_id": "a",
                "agent_name": "x",
                "agent_card_params": {"url": "http://x:8700", "protocolVersion": "1.0"},
            }
        ]
    }
    transport = _transport(
        registry=registry,
        cards={
            "http://x:8700/.well-known/agent-card.json": httpx.Response(
                200, json=_card(name="x", url="http://x:8700")
            )
        },
    )

    async def fake_probe(**kwargs):
        return await probe(key="k", transport=transport)

    monkeypatch.setattr("agent.probe.probe", fake_probe)
    assert asyncio.run(main([])) == 0


def test_main_exits_nonzero_on_a_failing_agent(monkeypatch):
    registry = {
        "agents": [
            {
                "agent_id": "a",
                "agent_name": "x",
                "agent_card_params": {"url": "http://0.0.0.0:8700"},
            }
        ]
    }
    transport = _transport(registry=registry, cards={})

    async def fake_probe(**kwargs):
        return await probe(key="k", transport=transport)

    monkeypatch.setattr("agent.probe.probe", fake_probe)
    assert asyncio.run(main([])) == 1
