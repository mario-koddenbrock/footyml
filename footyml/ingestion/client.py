import asyncio
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from footyml.config import (
    TRANSFERMARKT_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_WAIT_SECONDS,
    RATE_LIMIT_RPS,
)


class TransfermarktClient:
    """Async HTTP client for transfermarkt-api with rate limiting and automatic retry."""

    def __init__(self, base_url: str = TRANSFERMARKT_BASE_URL):
        self._base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._last_call: float = 0.0
        self._min_interval = 1.0 / RATE_LIMIT_RPS

    async def __aenter__(self) -> "TransfermarktClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_WAIT_SECONDS, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )
    async def get(self, path: str, params: dict[str, str] | None = None) -> dict:  # type: ignore[type-arg]
        """Rate-limited GET with automatic retry on transient errors."""
        if self._client is None:
            raise RuntimeError("Client not started — use 'async with TransfermarktClient()'")

        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)

        response = await self._client.get(path, params=params)
        self._last_call = time.monotonic()
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
