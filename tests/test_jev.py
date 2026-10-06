"""Tests for the Jev client.

The load-bearing behaviours are *parsing* and *fail-open*, and neither needs a
network — so these drive the client through an `httpx.MockTransport`, the same
seam the probe and validator tests use. The truth pinned here is ours: that a
good response becomes typed answers, and that a bad one becomes `failed=True`
rather than an exception.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent.jev import (
    SYSTEMONE_PATH,
    Choice,
    ChoiceAnswer,
    JevClient,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)


def _client(handler: httpx.MockTransport) -> JevClient:
    return JevClient(
        base_url="http://litellm:4000",
        api_key="test-key",
        client=httpx.AsyncClient(base_url="http://litellm:4000", transport=handler),
    )


def _ok_body() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "q_noul": {"type": "noul", "noul": 0.99},
            "q_choice": {
                "type": "choice",
                "choice": "b",
                "confidence": 0.78,
                "probabilities": {"a": 0.2, "b": 0.8},
            },
            "q_score": {
                "type": "score",
                "score": 2.0,
                "confidence": 1.0,
                "legend": {"0": "none", "2": "exact"},
                "probabilities": {"2": 1.0},
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_ok_body())


def _questions() -> dict:
    return {
        "q_noul": Noul(instructions="Is it urgent?"),
        "q_choice": Choice(instructions="Which team?", criteria={"a": "billing", "b": "tech"}),
        "q_score": Score(instructions="How frustrated?", criteria=["none", "some", "exact"]),
    }


def test_ask_parses_each_answer_kind():
    answers = asyncio.run(_client(httpx.MockTransport(_ok_handler)).ask("s", _questions()))
    assert isinstance(answers["q_noul"], NoulAnswer)
    assert answers["q_noul"].noul == pytest.approx(0.99)
    assert isinstance(answers["q_choice"], ChoiceAnswer)
    assert answers["q_choice"].choice == "b"
    assert answers["q_choice"].probabilities == {"a": 0.2, "b": 0.8}
    assert isinstance(answers["q_score"], ScoreAnswer)
    assert answers["q_score"].score == 2.0
    assert answers["q_score"].legend == {0: "none", 2: "exact"}
    assert not any(a.failed for a in answers.values())


def test_ask_posts_the_native_body_to_the_proxy_path():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok_body())

    asyncio.run(_client(httpx.MockTransport(handler)).ask("the state", _questions()))
    assert seen["path"] == SYSTEMONE_PATH == "/typesafe/v1/systemone"
    assert seen["body"]["state"] == "the state"
    assert seen["body"]["model"] == "jev-latest"
    assert set(seen["body"]["questions"]) == {"q_noul", "q_choice", "q_score"}
    # The native body shape: a `type` discriminator and `instructions` per question.
    assert seen["body"]["questions"]["q_noul"]["type"] == "noul"
    assert seen["body"]["questions"]["q_choice"]["criteria"] == {"a": "billing", "b": "tech"}


def test_ask_with_no_questions_makes_no_call():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not be called")

    assert asyncio.run(_client(httpx.MockTransport(handler)).ask("s", {})) == {}


def test_ask_fails_open_on_transport_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    answers = asyncio.run(_client(httpx.MockTransport(handler)).ask("s", _questions()))
    assert all(a.failed for a in answers.values())
    assert answers["q_noul"].noul is None


def test_ask_fails_open_on_http_error():
    handler = lambda request: httpx.Response(401, text="unauthorized")  # noqa: E731
    answers = asyncio.run(_client(httpx.MockTransport(handler)).ask("s", _questions()))
    assert all(a.failed for a in answers.values())


def test_ask_fails_open_on_non_json():
    handler = lambda request: httpx.Response(200, text="<html>nope</html>")  # noqa: E731
    answers = asyncio.run(_client(httpx.MockTransport(handler)).ask("s", _questions()))
    assert all(a.failed for a in answers.values())


def test_ask_fails_open_when_answers_object_is_absent():
    handler = lambda request: httpx.Response(200, json={"model": "jev"})  # noqa: E731
    answers = asyncio.run(_client(httpx.MockTransport(handler)).ask("s", _questions()))
    assert all(a.failed for a in answers.values())


def test_one_malformed_answer_fails_only_that_question():
    body = _ok_body()
    del body["answers"]["q_choice"]["choice"]  # the others are still fine

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    answers = asyncio.run(_client(httpx.MockTransport(handler)).ask("s", _questions()))
    assert answers["q_choice"].failed
    assert not answers["q_noul"].failed
    assert not answers["q_score"].failed
    assert answers["q_noul"].noul == pytest.approx(0.99)
