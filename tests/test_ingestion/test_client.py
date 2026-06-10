from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from footyml.ingestion.client import TransfermarktClient


@pytest.mark.asyncio
async def test_successful_get() -> None:
    with respx.mock:
        respx.get("https://transfermarkt-api.fly.dev/test").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        async with TransfermarktClient() as client:
            result = await client.get("/test")
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_rate_limiting_enforces_interval() -> None:
    with respx.mock:
        respx.get("https://transfermarkt-api.fly.dev/test").mock(
            return_value=httpx.Response(200, json={})
        )
        async with TransfermarktClient() as client:
            client._min_interval = 0.2
            t0 = time.monotonic()
            await client.get("/test")
            await client.get("/test")
            elapsed = time.monotonic() - t0
    assert elapsed >= 0.15, f"Rate limiting not enforced: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_raises_on_404() -> None:
    with respx.mock:
        respx.get("https://transfermarkt-api.fly.dev/missing").mock(
            return_value=httpx.Response(404)
        )
        async with TransfermarktClient() as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.get("/missing")


@pytest.mark.asyncio
async def test_context_manager_required() -> None:
    client = TransfermarktClient()
    with pytest.raises(RuntimeError, match="async with"):
        await client.get("/test")
