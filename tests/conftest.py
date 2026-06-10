from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from footyml.data.store import DataStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> DataStore:
    return DataStore(db_path=tmp_path / "test.duckdb")


@pytest.fixture
def sample_matches() -> pl.DataFrame:
    """100 synthetic matches with correct schema."""
    rng = random.Random(42)
    start = date(2022, 8, 6)
    clubs = [f"club_{i}" for i in range(10)]
    rows = []
    game_id = 0

    for week in range(34):
        d = start + timedelta(weeks=week)
        matchups = list(range(0, 10, 2))
        for i in matchups:
            home = clubs[i]
            away = clubs[i + 1]
            hg = rng.randint(0, 4)
            ag = rng.randint(0, 3)
            result = 2 if hg > ag else (1 if hg == ag else 0)
            rows.append({
                "game_id": str(game_id),
                "competition_id": "L1",
                "season": 2022,
                "matchday": week + 1,
                "date": d,
                "home_club_id": home,
                "away_club_id": away,
                "home_goals": hg,
                "away_goals": ag,
                "result": result,
            })
            game_id += 1

    return pl.DataFrame(rows)


@pytest.fixture
def populated_store(tmp_store: DataStore, sample_matches: pl.DataFrame) -> DataStore:
    tmp_store.upsert_matches(sample_matches)
    return tmp_store
