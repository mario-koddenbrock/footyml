from __future__ import annotations

import polars as pl

from footyml.data.store import DataStore

WINDOWS = [3, 5, 10]
STATS = ["wins", "draws", "losses", "pts", "gf", "ga", "gd"]


class FormFeatureBuilder:
    """Rolling team form over last N matches (3, 5, 10) for both teams."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        """Return game_id + rolling form features for home and away teams."""
        team_rows = _flatten_to_team_rows(matches)
        form_df = _compute_rolling_form(team_rows)
        result = _pivot_to_match_features(matches, form_df)
        return result


class HomeAwayFormBuilder:
    """Home-specific form for the home team; away-specific form for the away team."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        team_rows = _flatten_to_team_rows(matches)

        home_rows = team_rows.filter(pl.col("venue") == "home")
        away_rows = team_rows.filter(pl.col("venue") == "away")

        home_form = _compute_rolling_form(home_rows, prefix_suffix="_venue")
        away_form = _compute_rolling_form(away_rows, prefix_suffix="_venue")

        home_merged = (
            matches.select(["game_id", "home_club_id", "date"])
            .join(
                home_form.rename({"club_id": "home_club_id"}),
                on=["home_club_id", "date"],
                how="left",
            )
            .drop(["home_club_id", "date"])
        )
        home_merged = _prefix_cols(home_merged, "home_venue_", skip={"game_id"})

        away_merged = (
            matches.select(["game_id", "away_club_id", "date"])
            .join(
                away_form.rename({"club_id": "away_club_id"}),
                on=["away_club_id", "date"],
                how="left",
            )
            .drop(["away_club_id", "date"])
        )
        away_merged = _prefix_cols(away_merged, "away_venue_", skip={"game_id"})

        result = home_merged.join(away_merged, on="game_id", how="left")
        return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _flatten_to_team_rows(matches: pl.DataFrame) -> pl.DataFrame:
    """Explode match rows into per-team perspective rows."""
    home_df = matches.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("home_club_id").alias("club_id"),
        pl.col("home_goals").alias("gf"),
        pl.col("away_goals").alias("ga"),
        pl.lit("home").alias("venue"),
        pl.col("result").alias("match_result"),
    ])
    away_df = matches.select([
        pl.col("game_id"),
        pl.col("date"),
        pl.col("away_club_id").alias("club_id"),
        pl.col("away_goals").alias("gf"),
        pl.col("home_goals").alias("ga"),
        pl.lit("away").alias("venue"),
        pl.col("result").alias("match_result"),
    ])

    combined = pl.concat([home_df, away_df]).sort(["club_id", "date"])

    combined = combined.with_columns([
        pl.when(
            (pl.col("venue") == "home") & (pl.col("match_result") == 2) |
            (pl.col("venue") == "away") & (pl.col("match_result") == 0)
        ).then(1).otherwise(0).alias("win"),
        pl.when(pl.col("match_result") == 1).then(1).otherwise(0).alias("draw"),
        pl.when(
            (pl.col("venue") == "home") & (pl.col("match_result") == 0) |
            (pl.col("venue") == "away") & (pl.col("match_result") == 2)
        ).then(1).otherwise(0).alias("loss"),
    ])
    combined = combined.with_columns(
        (pl.col("win") * 3 + pl.col("draw")).alias("pts"),
        (pl.col("gf") - pl.col("ga")).alias("gd"),
    )
    return combined


def _compute_rolling_form(team_rows: pl.DataFrame, prefix_suffix: str = "") -> pl.DataFrame:
    """Compute rolling stats for each club, shifted by 1 to avoid leakage."""
    result_frames = []

    for n in WINDOWS:
        agg_cols = {
            f"wins_{n}{prefix_suffix}": pl.col("win"),
            f"draws_{n}{prefix_suffix}": pl.col("draw"),
            f"losses_{n}{prefix_suffix}": pl.col("loss"),
            f"pts_{n}{prefix_suffix}": pl.col("pts"),
            f"gf_{n}{prefix_suffix}": pl.col("gf"),
            f"ga_{n}{prefix_suffix}": pl.col("ga"),
            f"gd_{n}{prefix_suffix}": pl.col("gd"),
        }
        exprs = [
            v.shift(1).rolling_sum(window_size=n, min_samples=1).alias(k)
            for k, v in agg_cols.items()
        ]
        rolled = team_rows.with_columns(exprs).select(
            ["game_id", "club_id", "date"] + list(agg_cols.keys())
        )
        result_frames.append(rolled)

    merged = result_frames[0]
    for frame in result_frames[1:]:
        merged = merged.join(frame, on=["game_id", "club_id", "date"], how="left")

    return merged


def _pivot_to_match_features(matches: pl.DataFrame, form_df: pl.DataFrame) -> pl.DataFrame:
    home_form = (
        matches.select(["game_id", "home_club_id", "date"])
        .join(form_df.rename({"club_id": "home_club_id"}), on=["home_club_id", "date"], how="left")
        .drop(["home_club_id", "date"])
    )
    home_form = _prefix_cols(home_form, "home_", skip={"game_id"})

    away_form = (
        matches.select(["game_id", "away_club_id", "date"])
        .join(form_df.rename({"club_id": "away_club_id"}), on=["away_club_id", "date"], how="left")
        .drop(["away_club_id", "date"])
    )
    away_form = _prefix_cols(away_form, "away_", skip={"game_id"})

    return home_form.join(away_form, on="game_id", how="left")


def _prefix_cols(df: pl.DataFrame, prefix: str, skip: set[str]) -> pl.DataFrame:
    return df.rename({c: f"{prefix}{c}" for c in df.columns if c not in skip})
