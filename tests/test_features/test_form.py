from __future__ import annotations

import polars as pl
import pytest

from footyml.features.form import FormFeatureBuilder, _flatten_to_team_rows


def _make_matches(n: int = 10) -> pl.DataFrame:
    """Minimal match sequence: team A always wins at home."""
    rows = []
    for i in range(n):
        rows.append({
            "game_id": str(i),
            "competition_id": "L1",
            "season": 2022,
            "matchday": i + 1,
            "date": f"2022-{8 + i // 4:02d}-{1 + (i % 4) * 7:02d}",
            "home_club_id": "A",
            "away_club_id": "B",
            "home_goals": 2,
            "away_goals": 0,
            "result": 2,
        })
    return pl.DataFrame(rows)


def test_form_output_has_game_ids() -> None:
    matches = _make_matches(5)
    result = FormFeatureBuilder().build(matches)
    assert "game_id" in result.columns
    assert len(result) == len(matches)


def test_no_leakage_first_match() -> None:
    """Team A's form stats for the first match should be None/0 (no prior history)."""
    matches = _make_matches(5)
    result = FormFeatureBuilder().build(matches)
    first = result.filter(pl.col("game_id") == "0")
    assert first["home_wins_3"][0] is None or first["home_wins_3"][0] == 0


def test_wins_accumulate_correctly() -> None:
    """After 5 consecutive home wins, home_wins_5 should be 5."""
    matches = _make_matches(10)
    result = FormFeatureBuilder().build(matches)
    last = result.filter(pl.col("game_id") == "9")
    wins_5 = last["home_wins_5"][0]
    assert wins_5 == 5, f"Expected 5 wins in last 5, got {wins_5}"


def test_away_team_losses_accumulate() -> None:
    matches = _make_matches(10)
    result = FormFeatureBuilder().build(matches)
    last = result.filter(pl.col("game_id") == "9")
    losses = last["away_losses_5"][0]
    assert losses == 5


def test_flatten_produces_two_rows_per_match() -> None:
    matches = _make_matches(3)
    team_rows = _flatten_to_team_rows(matches)
    assert len(team_rows) == 6
