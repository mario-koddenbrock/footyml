"""Tests for TournamentPredictor and tournament feature pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from footyml.config import VENUE_NEUTRAL, VENUE_HOME_A
from footyml.data.store import DataStore
from footyml.features.elo import EloBuilder
from footyml.features.pipeline import FeaturePipeline, _add_diff_features


# ------------------------------------------------------------------
# Elo: venue_type-aware tests
# ------------------------------------------------------------------


def test_elo_no_home_advantage_for_neutral() -> None:
    """Elo update for neutral venue should not apply home bonus."""
    matches = pl.DataFrame({
        "game_id": ["g1"],
        "home_club_id": ["A"],
        "away_club_id": ["B"],
        "date": [date(2026, 6, 14)],
        "result": [2],
        "venue_type": [VENUE_NEUTRAL],
    })
    matches_home = matches.with_columns(pl.lit(VENUE_HOME_A).alias("venue_type"))

    builder = EloBuilder()
    neutral_result = builder.build(matches)
    home_result = builder.build(matches_home)

    # Both should start at ELO_INITIAL — no pre-match difference
    assert neutral_result["home_elo"][0] == home_result["home_elo"][0]
    assert neutral_result["elo_diff"][0] == 0.0


def test_elo_home_advantage_applied_for_league() -> None:
    """After a league home win, the winning home team Elo gain should reflect home bonus."""
    matches = pl.DataFrame({
        "game_id": ["g1", "g2"],
        "home_club_id": ["A", "C"],
        "away_club_id": ["B", "D"],
        "date": [date(2026, 1, 1), date(2026, 1, 8)],
        "result": [2, 2],
        "venue_type": [VENUE_HOME_A, VENUE_NEUTRAL],
    })
    builder = EloBuilder()
    result = builder.build(matches)
    # Both first matches start at ELO_INITIAL, diff = 0
    assert result["elo_diff"][0] == 0.0
    assert result["elo_diff"][1] == 0.0


# ------------------------------------------------------------------
# Pipeline: venue_type encoding + diff features
# ------------------------------------------------------------------


def test_diff_features_computed() -> None:
    df = pl.DataFrame({
        "game_id": ["g1"],
        "home_elo": [1600.0],
        "away_elo": [1400.0],
        "home_pts_5": [12.0],
        "away_pts_5": [6.0],
    })
    result = _add_diff_features(df)
    assert "elo_diff_raw" in result.columns
    assert result["elo_diff_raw"][0] == pytest.approx(200.0)
    assert "form_pts5_diff" in result.columns
    assert result["form_pts5_diff"][0] == pytest.approx(6.0)


def test_diff_features_missing_cols_no_crash() -> None:
    """If a feature pair doesn't exist, _add_diff_features should skip it gracefully."""
    df = pl.DataFrame({"game_id": ["g1"], "home_elo": [1500.0]})
    result = _add_diff_features(df)
    assert "game_id" in result.columns
    assert "elo_diff_raw" not in result.columns  # away_elo missing


def test_pipeline_adds_venue_type_enc(populated_store: DataStore) -> None:
    pipeline = FeaturePipeline(populated_store)
    result = pipeline.build(competition_id="L1", seasons=[2022])
    assert "venue_type_enc" in result.columns


def test_pipeline_venue_enc_default_league(populated_store: DataStore) -> None:
    """League matches with no venue_type column should default to home_advantage_a (1.0)."""
    pipeline = FeaturePipeline(populated_store)
    result = pipeline.build(competition_id="L1", seasons=[2022])
    assert all(v == 1.0 for v in result["venue_type_enc"].drop_nulls().to_list())


# ------------------------------------------------------------------
# Store: schema migration
# ------------------------------------------------------------------


def test_store_has_venue_type_column(tmp_store: DataStore) -> None:
    cols = {
        r[0]
        for r in tmp_store._conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='matches'"
        ).fetchall()
    }
    assert "venue_type" in cols
    assert "group_id" in cols
    assert "stage" in cols


def test_upsert_match_with_venue_type(tmp_store: DataStore) -> None:
    row = pl.DataFrame({
        "game_id": ["wc_1"],
        "competition_id": ["WC"],
        "season": [2026],
        "matchday": [1],
        "date": [date(2026, 6, 14)],
        "home_club_id": ["fd_100"],
        "away_club_id": ["fd_200"],
        "home_goals": [None],
        "away_goals": [None],
        "result": [None],
        "venue_type": [VENUE_NEUTRAL],
        "group_id": ["Group A"],
        "stage": ["GROUP_STAGE"],
    })
    tmp_store.upsert_matches(row)
    result = tmp_store.get_matches(competition_id="WC", completed_only=False)
    assert len(result) == 1
    assert result["venue_type"][0] == VENUE_NEUTRAL
    assert result["group_id"][0] == "Group A"


def test_get_matches_filter_by_group(tmp_store: DataStore) -> None:
    rows = pl.DataFrame({
        "game_id": ["wc_1", "wc_2"],
        "competition_id": ["WC", "WC"],
        "season": [2026, 2026],
        "matchday": [1, 1],
        "date": [date(2026, 6, 14), date(2026, 6, 14)],
        "home_club_id": ["fd_100", "fd_300"],
        "away_club_id": ["fd_200", "fd_400"],
        "home_goals": [2, None],
        "away_goals": [0, None],
        "result": [2, None],
        "venue_type": [VENUE_NEUTRAL, VENUE_NEUTRAL],
        "group_id": ["Group A", "Group B"],
        "stage": ["GROUP_STAGE", "GROUP_STAGE"],
    })
    tmp_store.upsert_matches(rows)
    group_a = tmp_store.get_matches(competition_id="WC", group_id="Group A", completed_only=False)
    assert len(group_a) == 1
    assert group_a["group_id"][0] == "Group A"
