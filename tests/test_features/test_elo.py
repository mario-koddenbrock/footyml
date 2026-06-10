from __future__ import annotations

import polars as pl
import pytest

from footyml.features.elo import EloBuilder, _expected_score, _update


def test_expected_score_equal_ratings() -> None:
    assert abs(_expected_score(1500.0, 1500.0) - 0.5) < 1e-9


def test_expected_score_higher_rating_favored() -> None:
    assert _expected_score(1600.0, 1400.0) > 0.5


def test_elo_increases_after_win() -> None:
    new = _update(1500.0, actual=1.0, expected=0.5)
    assert new > 1500.0


def test_elo_decreases_after_loss() -> None:
    new = _update(1500.0, actual=0.0, expected=0.5)
    assert new < 1500.0


def test_elo_builder_output_columns() -> None:
    matches = pl.DataFrame({
        "game_id": ["g1", "g2"],
        "home_club_id": ["A", "B"],
        "away_club_id": ["B", "A"],
        "date": ["2023-01-01", "2023-01-08"],
        "result": [2, 1],
    })
    builder = EloBuilder()
    result = builder.build(matches)
    assert "home_elo" in result.columns
    assert "away_elo" in result.columns
    assert "elo_diff" in result.columns
    assert len(result) == 2


def test_elo_builder_no_leakage() -> None:
    """First match Elo values should be initial ratings (no prior history)."""
    matches = pl.DataFrame({
        "game_id": ["g1"],
        "home_club_id": ["A"],
        "away_club_id": ["B"],
        "date": ["2023-01-01"],
        "result": [2],
    })
    builder = EloBuilder()
    result = builder.build(matches)
    from footyml.config import ELO_INITIAL
    assert result["home_elo"][0] == ELO_INITIAL
    assert result["away_elo"][0] == ELO_INITIAL


def test_elo_builder_persists_to_store(tmp_path: "Path") -> None:
    from pathlib import Path
    from footyml.data.store import DataStore

    store = DataStore(db_path=tmp_path / "test.duckdb")
    matches = pl.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_club_id": ["A", "A", "A"],
        "away_club_id": ["B", "B", "B"],
        "date": ["2023-01-01", "2023-01-08", "2023-01-15"],
        "result": [2, 2, 2],
    })
    builder = EloBuilder(store=store)
    builder.build(matches)
    elo_df = store.get_elo("A")
    assert len(elo_df) > 0
