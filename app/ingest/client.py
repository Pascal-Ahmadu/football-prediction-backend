"""HTTP client for apifootball.com"""

import asyncio
import logging 
from typing import Any
import httpx


from app.config import settings

logger = logging.getLogger(__name__)

PROVIDER = "apifootball"
MAX_ATTEMPTS =3

class ProviderError(RuntimeError):
    """The provider returned something unusable"""

class ApiFootballClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
       self._api_key = api_key or settings.api_football_key
       if not self._api_key:
          raise ProviderError("API_FOOTBALL_KEY is not set")
       self._timeout = timeout
    
    async def get(self, action: str, **params: Any) -> Any: 
        """Call one apifootball action and return the parsed JSON."""
        query = {"action": action, "APIkey": self._api_key, **params}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = await client.get(
                        settings.api_football_base_url, params=query
                    )
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    if attempt == MAX_ATTEMPTS:
                        raise ProviderError(
                            f"{action} failed after {MAX_ATTEMPTS} attempts: {exc}"
                        ) from exc
                    wait = 2**attempt
                    logger.warning(
                        "%s attempts %s failed (%s); retrying in %ss",
                        action, attempt, exc, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                payload = response.json()
                if isinstance(payload, dict) and "error" in payload:
                    raise ProviderError(f"{action}: {payload.get('message')}")
                return payload
        raise ProviderError("unreachable")
                