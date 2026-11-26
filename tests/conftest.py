"""Test configuration.

`agent.main` builds its model at import time, and constructing the OpenAI client
requires *an* api key — a real one is not needed, but the client refuses an empty
string. So the tests supply a placeholder, which keeps them hermetic: they run in
CI (where no gateway secret exists) and on a laptop alike, and they never talk to
the gateway anyway.

The Databento key is deliberately NOT defaulted here: the Validator must be
explicit that it cannot run without the oracle's credential, so the tests inject
a fake validator rather than papering over its absence.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("LITELLM_API_KEY", "test-key-not-a-secret")
os.environ.setdefault("LITELLM_BASE_URL", "http://litellm:4000")


@pytest.fixture
def stub_filer():
    """Filing, without a runtime to file against.

    `agent/main.py` files the validated ticket through `agent/filing.py`, which
    speaks MCP to mcphub. A test that drives a whole turn has neither, so it
    injects this instead: the ticket comes back carrying the id a real filing
    would have returned, which is what the artifact then names.

    The order is the property these turns are testing — the ticket that was
    validated is the body that was filed — so a stub here is the honest seam, not
    a hole: `tests/test_filing.py` covers the real mapping and the real call.
    """

    async def _file(ticket, *, turn_id=None):
        return ticket.model_copy(update={"request_id": f"req_{turn_id or 'stub'}"})

    return _file
