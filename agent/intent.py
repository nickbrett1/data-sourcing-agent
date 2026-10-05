"""Is this message a market-data request, or just conversation?

The agent's runtime contract is a single `TicketProposal` (see
`agent/contract.py`): `PromptedOutput(TicketProposal)` plus the Validator's
bounded repair loop. That contract is exactly right for a data request and
exactly wrong for anything else. A greeting ("Hello") carries no
dataset/schema/symbol intent, so there is no legal proposal to make and the
model has only two ways out:

* it returns something malformed and the output retries are exhausted — the run
  fails with `Exceeded maximum output retries (3)`, surfaced to the caller as
  `TASK_STATE_FAILED`; or
* it invents a plausible-looking ticket to satisfy the schema — a fabricated
  `XNAS.ITCH` / `ALL_SYMBOLS` / `ohlcv-1d` download with a placeholder `why`.

Both are wrong. A greeting is a normal conversational turn: it should return a
normal assistant message asking what market data the user wants, and it should
never reach the ticket machinery at all.

This module is that decision, made **before** the model runs and made
deterministically — same input, same answer, no extra model call, no extra cost.
`TicketAgentExecutor.execute` calls `is_data_request`; on `False` it short-circuits
to `NON_REQUEST_REPLY` and completes the task, and the ticket path stays exactly
as it was for real requests.

The bias is deliberate: **anything without a recognisable market-data signal is
treated as conversation**. A vaguely-phrased request ("I need some data") gets
the same friendly clarification a greeting does — which is the right answer for a
vague request too — instead of a fabricated ticket. Under-classifying costs one
clarifying turn; over-classifying fabricates a download request, which is the
failure we are fixing.
"""

from __future__ import annotations

import re

# The reply for a turn that is not a data request. It does the one job the old
# behaviour could not: it states what the agent needs to draft a ticket, so the
# next turn can carry the intent the ticket path requires.
NON_REQUEST_REPLY = (
    "Hi! I draft Databento historical download tickets. Tell me what market "
    "data you want — the dataset or symbols (for example `OPRA.PILLAR` or the "
    "`SPY` options chain), the schema (for example `trades`, `ohlcv-1d`), and "
    "the date window — and I'll draft the request for your sign-off."
)

# Dataset ids look like `XNAS.ITCH`, `OPRA.PILLAR`, `GLBX.MDP3`, `EQUS.MINI`.
# Uppercase, two-or-more letters, a dot, then an uppercase alphanumeric code.
_DATASET_RE = re.compile(r"\b[A-Z]{2,6}\.[A-Z][A-Z0-9]{1,}\b")

# A bare ticker or `$TICKER`: two to five uppercase letters standing alone.
# Requiring length >= 2 keeps the pronoun "I" out, and the smalltalk check below
# keeps an all-caps greeting ("HELLO") from reading as a symbol.
_TICKER_RE = re.compile(r"(?<![A-Za-z])\$?[A-Z]{2,5}(?![A-Za-z])")

# Whole words (lower-cased) that signal a data request. Matched as tokens, not
# substrings, so "bar" does not fire on "barista" or "chain" on "blockchain".
_REQUEST_WORDS = frozenset(
    {
        # the schema vocabulary, including its unambiguous long forms
        "mbo",
        "mbp",
        "mbp-1",
        "mbp-10",
        "tbbo",
        "bbo",
        "bbo-1s",
        "bbo-1m",
        "cmbp",
        "cmbp-1",
        "cmbo",
        "ohlcv",
        "ohlcv-1s",
        "ohlcv-1m",
        "ohlcv-1h",
        "ohlcv-1d",
        "ohlcv-eod",
        "imbalance",
        # the market-data nouns a request tends to use
        "dataset",
        "datasets",
        "schema",
        "schemas",
        "symbol",
        "symbols",
        "all_symbols",
        "ticker",
        "tickers",
        "instrument",
        "instruments",
        "tick",
        "ticks",
        "trade",
        "trades",
        "quote",
        "quotes",
        "depth",
        "bar",
        "bars",
        "candle",
        "candles",
        "futures",
        "future",
        "equities",
        "equity",
        "options",
        "option",
        "contract",
        "contracts",
        "expiry",
        "expiries",
        "expiration",
        "strike",
        "strikes",
        "backfill",
        "dataset_id",
        # dataset families, so "OPRA data" reads as a request
        "opra",
        "glbx",
        "equs",
        "xnas",
        "dbn",
        "ifeu",
        "xeur",
        "ndex",
        "xnys",
        "databento",
    }
)

# Multi-word phrases, matched against the lower-cased text.
_REQUEST_PHRASES = (
    "market data",
    "options chain",
    "option chain",
    "order book",
    "time series",
    "time-series",
    "get range",
    "get_range",
)

# The words a greeting / thank-you / pleasantry is made of. When *every* word of
# a short message is one of these, it is smalltalk regardless of its casing, so
# an all-caps "HELLO" is not mistaken for a ticker.
_SMALLTALK_WORDS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "heya",
        "yo",
        "howdy",
        "greetings",
        "good",
        "morning",
        "afternoon",
        "evening",
        "day",
        "there",
        "how",
        "are",
        "you",
        "doing",
        "whats",
        "what's",
        "up",
        "thanks",
        "thank",
        "thanx",
        "thx",
        "cheers",
        "bye",
        "goodbye",
        "see",
        "later",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
        "sure",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "please",
        "pls",
        "help",
        "welcome",
        "sorry",
    }
)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*")


def _tokens(text: str) -> list[str]:
    """Lower-cased word tokens, so matching is casing- and punctuation-tolerant."""
    return _WORD_RE.findall(text.lower())


def _is_smalltalk(text: str) -> bool:
    """True when a short message is made only of greeting/pleasantry words.

    Bounded to eight tokens so a long message that happens to start with "hey"
    ("hey, pull the SPY chain for last week") is never reduced to smalltalk.
    """
    words = _tokens(text)
    return bool(words) and len(words) <= 8 and all(w in _SMALLTALK_WORDS for w in words)


def is_data_request(text: str) -> bool:
    """Whether the message asks for market data (and so warrants a ticket).

    Returns `False` for a greeting or any message with no dataset/schema/symbol
    signal; the executor answers those conversationally and never drafts a
    ticket. See the module docstring for why the exception for all-caps
    greetings exists.
    """
    if not text or not text.strip():
        return False

    lowered = text.lower()
    if any(phrase in lowered for phrase in _REQUEST_PHRASES):
        return True
    if _DATASET_RE.search(text):
        return True

    tokens = set(_tokens(text))
    if tokens & _REQUEST_WORDS:
        return True

    # A bare uppercase ticker counts as symbol intent — unless the whole message
    # is smalltalk, so "HELLO" is not read as a symbol.
    if _TICKER_RE.search(text) and not _is_smalltalk(text):
        return True

    return False
