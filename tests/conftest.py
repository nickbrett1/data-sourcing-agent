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

os.environ.setdefault("LITELLM_API_KEY", "test-key-not-a-secret")
os.environ.setdefault("LITELLM_BASE_URL", "http://litellm:4000")
