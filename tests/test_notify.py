"""The drift push: one POST to ntfy, and only when out of sync."""

from __future__ import annotations

from agent import notify as n


class _Resp:
    def read(self) -> bytes:
        return b"ok"

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _capture(monkeypatch) -> dict:
    calls: dict = {}

    def fake_urlopen(request, timeout=None):
        calls["url"] = request.full_url
        calls["data"] = request.data
        calls["title"] = request.headers.get("Title")
        return _Resp()

    monkeypatch.setattr(n.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_notify_posts_the_message_to_the_topic(monkeypatch):
    calls = _capture(monkeypatch)
    n.notify("drift!", url="https://ntfy.example/", topic="some-topic")
    assert calls["url"] == "https://ntfy.example/some-topic"
    assert calls["data"] == b"drift!"
    assert calls["title"] == "gate out of sync"


def test_the_env_overrides_the_topic(monkeypatch):
    monkeypatch.setenv(n.NTFY_URL_ENV, "https://ntfy.example")
    monkeypatch.setenv(n.NTFY_TOPIC_ENV, "env-topic")
    calls = _capture(monkeypatch)
    n.notify("hello")
    assert calls["url"] == "https://ntfy.example/env-topic"


def test_main_pushes_every_issue_when_out_of_sync(monkeypatch):
    from agent import summary as s

    monkeypatch.setattr(
        s, "build_summary", lambda **kw: {"status": "warn", "issue": "re-freeze due", "issues": ["re-freeze due", "x"]}
    )
    calls = _capture(monkeypatch)
    assert n.main([]) == 0
    assert calls["data"] == b"re-freeze due\nx"


def test_main_stays_silent_when_in_sync(monkeypatch):
    from agent import summary as s

    monkeypatch.setattr(s, "build_summary", lambda **kw: {"status": "ok", "issue": "in sync", "issues": []})
    calls = _capture(monkeypatch)
    assert n.main([]) == 0
    assert calls == {}
