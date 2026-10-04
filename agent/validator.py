"""The Validator — Layer 2 of the design (memo `data-acquisition-agent-v1` §4).

Deterministic. **No model lives in here, and none may be added.** This is the
only component allowed to call a request *valid* or a cost *known*, and it does
so by asking the API, never by reasoning.

The oracle is the Databento historical API's own `metadata.get_cost` endpoint:

* it returns a **200 with a cost** for a legal `(dataset, schema, symbols,
  stype_in, start, end)` combination — so a priced request is also a valid one;
* it returns an authoritative **400/422** for an illegal one, and the error body
  names the *case* and, where it can, the **legal values**. That error text is
  the runtime vocabulary the repair loop feeds back to the model — the legal set
  is data the API owns, not something written into a prompt.

Wire detail worth recording: `metadata.get_cost` is **query/form-encoded**, not
JSON. A JSON body returns a confusing `422 ... Field required` that names fields
that *were* sent. The endpoint was found by probing; see the handover memo §5.1.

The two public methods are the design's, verbatim:

* `validate_request(request)` — raise `DatabentoRequestError` carrying the API's
  own case and message if the request is illegal; return `None` if it is legal.
* `price_request(request)` — return the API's cost, raising the same error if the
  request cannot be priced.

Prices are not cached on purpose: a stale price is a wrong number standing in
for a right one at a money gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from agent.ticket import RequestSpec

DEFAULT_BASE_URL = "https://hist.databento.com"
# The API key is presented by config, never hard-coded. Basic auth with the key
# as the username and an empty password (Databento's convention).
API_KEY_ENV = "DATABENTO_API_KEY"
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class DatabentoRequestError(Exception):
    """An authoritative refusal from the API, carrying its own vocabulary.

    `case` and `message` are the API's, not ours. The repair loop passes them
    through to the model unchanged: rewording the oracle is how a repair loop
    starts "fixing" a request against a whisper instead of against the truth.
    """

    status_code: int
    case: str
    message: str

    def __str__(self) -> str:  # noqa: D105 - dataclass Exception string
        return f"[{self.status_code} {self.case}] {self.message}"


def _parse_error(status_code: int, body: object) -> DatabentoRequestError:
    """Turn the API's error body into a `DatabentoRequestError`.

    Two shapes are handled: `{"detail": {"case": ..., "message": ...}}` (the
    documented one) and a bare `{"detail": [...]}` validation-error list (what a
    malformed body produces). Anything else is reported as-is rather than
    flattened into a shrug — an unrecognised refusal is still a refusal.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return DatabentoRequestError(
            status_code=status_code,
            case=str(detail.get("case", "unknown")),
            message=str(detail.get("message", body)),
        )
    return DatabentoRequestError(
        status_code=status_code, case="unrecognized_error", message=str(body)
    )


class DatabentoValidator:
    """The Validator layer, backed by the live Databento API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                f"{API_KEY_ENV} is not set. The Validator must ask the API; it "
                "cannot be built without the credential that reaches it."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        # An injectable client is the seam tests use to exercise the truth-
        # handling without a network; production passes none and gets a real one.
        self._client = client or httpx.Client(
            base_url=self._base_url,
            auth=httpx.BasicAuth(api_key, ""),
            timeout=timeout,
        )

    @classmethod
    def from_env(cls, **kwargs: object) -> DatabentoValidator:
        """Build from the environment, the only way the agent constructs one."""
        return cls(
            os.environ.get(API_KEY_ENV, ""),
            base_url=os.environ.get("DATABENTO_BASE_URL", DEFAULT_BASE_URL),
            **kwargs,  # type: ignore[arg-type]
        )

    def close(self) -> None:
        self._client.close()

    def _get_cost(self, request: RequestSpec) -> float:
        """The API's cost for this request, or raise its refusal verbatim."""
        params = {
            "dataset": request.dataset,
            "schema": request.schema_name,
            "symbols": ",".join(request.symbols),
            "stype_in": request.stype_in.value,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
        }
        response = self._client.get("/v0/metadata.get_cost", params=params)
        if response.status_code >= 400:
            try:
                body: object = response.json()
            except ValueError:
                body = response.text
            raise _parse_error(response.status_code, body)
        # A successful cost estimate is a bare float in the body (e.g.
        # `1.219747960567`), not a JSON object.
        try:
            return float(response.text.strip())
        except ValueError as exc:  # pragma: no cover - defensive
            raise DatabentoRequestError(
                status_code=response.status_code,
                case="unparseable_cost",
                message=f"Expected a numeric cost, got: {response.text[:200]!r}",
            ) from exc

    def price_request(self, request: RequestSpec) -> float:
        """The live cost of this request in USD, exactly as the API states it."""
        return self._get_cost(request)

    def validate_request(self, request: RequestSpec) -> None:
        """Raise the API's refusal if the request is illegal; return None if legal.

        Legality is decided by the same call that prices the request: the API
        rejects an illegal combination with a 4xx before it returns a cost. There
        is deliberately no separate local model of what is legal — that model
        would be the second opinion this layer exists to remove.
        """
        self._get_cost(request)
