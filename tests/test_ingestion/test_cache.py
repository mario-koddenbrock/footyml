from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from footyml.ingestion.cache import ParquetCache


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    assert cache.get("nonexistent") is None


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    cache.set("test_key", df)
    loaded = cache.get("test_key")
    assert loaded is not None
    assert loaded.shape == df.shape
    assert loaded.columns == df.columns


def test_cache_key_is_deterministic(tmp_path: Path) -> None:
    key1 = ParquetCache.make_key("competitions", competition_id="L1", season_id="2023")
    key2 = ParquetCache.make_key("competitions", season_id="2023", competition_id="L1")
    assert key1 == key2


def test_has_returns_false_then_true(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path)
    key = "mykey"
    assert not cache.has(key)
    cache.set(key, pl.DataFrame({"x": [1]}))
    assert cache.has(key)
