"""API keys and rate limiting (NFR-SEC-01).

The forecasts are the whole product, so on a public address the API needs a door.
Both checks are deliberately simple and dependency-free:

  keys        a header must carry one of settings.api_keys. With no keys set the
              API stays open, which is what a laptop wants; a deployment sets
              them and is then closed by default.
  rate limit  a fixed window per caller, counted in memory. Enough to stop a
              script hammering the box; a multi-process deployment would need
              Redis for an exact count, and the limit is per process until then.
"""

import secrets
import time
from collections import defaultdict, deque

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from app.config import settings

API_KEY_HEADER = "X-API-Key"
_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# caller -> timestamps of recent requests
_recent: dict[str, deque[float]] = defaultdict(deque)


class Unauthorised(Exception):
    """Raised instead of HTTPException so the handler can return RFC 7807."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class RateLimited(Exception):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"retry in {retry_after}s")


async def require_api_key(key: str | None = Depends(_header)) -> str | None:
    """Check the header against the configured keys, in constant time."""
    if not settings.api_keys:
        return None  # open locally; a deployment sets keys
    if not key:
        raise Unauthorised(f"Provide an API key in the {API_KEY_HEADER} header.")
    if not any(secrets.compare_digest(key, valid) for valid in settings.api_keys):
        raise Unauthorised("That API key is not recognised.")
    return key


def _caller(request: Request) -> str:
    key = request.headers.get(API_KEY_HEADER)
    if key:
        return f"key:{key[:8]}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


async def rate_limit(request: Request) -> None:
    """Allow settings.rate_limit_per_minute requests per caller per minute."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return
    now = time.monotonic()
    window = _recent[_caller(request)]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise RateLimited(retry_after=max(1, int(61 - (now - window[0]))))
    window.append(now)


def reset_rate_limits() -> None:
    """For tests: forget what has been counted so far."""
    _recent.clear()
