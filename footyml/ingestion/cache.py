import hashlib
import json
import polars as pl
from pathlib import Path

from footyml.config import RAW_DIR


class ParquetCache:
    """Parquet-backed cache for API responses, keyed by endpoint + params."""

    def __init__(self, cache_dir: Path = RAW_DIR):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.parquet"

    def get(self, key: str) -> pl.DataFrame | None:
        path = self._path(key)
        if path.exists():
            return pl.read_parquet(path)
        return None

    def set(self, key: str, df: pl.DataFrame) -> None:
        self._path(key).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self._path(key))

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    @staticmethod
    def make_key(endpoint: str, **kwargs: str | int) -> str:
        """Deterministic cache key from endpoint + sorted params."""
        params_str = json.dumps(kwargs, sort_keys=True)
        digest = hashlib.sha1(params_str.encode()).hexdigest()[:8]
        safe_endpoint = endpoint.strip("/").replace("/", "_")
        return f"{safe_endpoint}_{digest}"
