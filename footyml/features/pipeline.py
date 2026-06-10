from __future__ import annotations

import polars as pl

from footyml.config import LEAGUE_IDS, PROCESSED_DIR, VENUE_TYPE_ENCODING
from footyml.data.store import DataStore
from footyml.features.elo import EloBuilder
from footyml.features.form import FormFeatureBuilder, HomeAwayFormBuilder
from footyml.features.h2h import HeadToHeadBuilder
from footyml.features.market_value import MarketValueBuilder
from footyml.features.squad import SquadBuilder
from footyml.features.standings import StandingsBuilder
from footyml.features.transfers import TransferBuilder


class FeaturePipeline:
    """Orchestrate all feature builders and produce a flat feature DataFrame."""

    def __init__(self, store: DataStore | None = None):
        self._store = store or DataStore()

    def build(
        self,
        competition_id: str | None = None,
        seasons: list[int] | None = None,
        tournament_mode: bool = False,
    ) -> pl.DataFrame:
        """Build the full feature matrix.

        In tournament_mode, also computes difference features and encodes
        venue_type as a numeric column so the model sees neutral-venue context.

        Returns a Polars DataFrame with game_id + all feature columns.
        Also persists to DuckDB features table and data/processed/features.parquet.
        """
        matches = self._store.get_matches(
            competition_id=competition_id,
            seasons=seasons,
            completed_only=True,
        )

        if len(matches) == 0:
            return pl.DataFrame({"game_id": []})

        matches = _ensure_competition_col(matches)
        matches = _ensure_season_col(matches)
        matches = _ensure_venue_type_col(matches)

        builders = [
            FormFeatureBuilder(self._store),
            HomeAwayFormBuilder(self._store),
            HeadToHeadBuilder(self._store),
            EloBuilder(self._store),
            MarketValueBuilder(self._store),
            SquadBuilder(self._store),
            TransferBuilder(self._store),
            StandingsBuilder(self._store),
        ]

        result = matches.select("game_id")
        for builder in builders:
            frame = builder.build(matches)
            if "game_id" in frame.columns:
                duplicate_cols = [c for c in frame.columns if c in result.columns and c != "game_id"]
                if duplicate_cols:
                    frame = frame.drop(duplicate_cols)
                result = result.join(frame, on="game_id", how="left")

        # Add venue_type as numeric feature
        venue_encoded = matches.select([
            pl.col("game_id"),
            pl.col("venue_type")
            .replace(VENUE_TYPE_ENCODING)
            .cast(pl.Float64)
            .alias("venue_type_enc"),
        ])
        result = result.join(venue_encoded, on="game_id", how="left")

        # Add diff features (important for tournament mode where home/away labels are arbitrary)
        result = _add_diff_features(result)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        result.write_parquet(PROCESSED_DIR / "features.parquet")
        self._store.upsert_features(result)

        return result


def _add_diff_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add home−away difference columns for key feature pairs."""
    diff_pairs = [
        # Elo
        ("home_elo", "away_elo", "elo_diff_raw"),
        # Form rolling 5
        ("home_pts_5", "away_pts_5", "form_pts5_diff"),
        ("home_wins_5", "away_wins_5", "form_wins5_diff"),
        ("home_gf_5", "away_gf_5", "form_gf5_diff"),
        ("home_gd_5", "away_gd_5", "form_gd5_diff"),
        # Form rolling 3
        ("home_pts_3", "away_pts_3", "form_pts3_diff"),
        ("home_gd_3", "away_gd_3", "form_gd3_diff"),
        # Market value
        ("home_mv_total_eur", "away_mv_total_eur", "mv_total_diff"),
        ("home_mv_avg_eur", "away_mv_avg_eur", "mv_avg_diff"),
        # Squad
        ("home_avg_age", "away_avg_age", "avg_age_diff"),
        ("home_squad_size", "away_squad_size", "squad_size_diff"),
        # Table
        ("home_table_pts", "away_table_pts", "table_pts_diff"),
        ("home_table_pos", "away_table_pos", "table_pos_diff"),
        ("home_table_gd", "away_table_gd", "table_gd_diff"),
    ]

    new_cols = []
    for col_a, col_b, alias in diff_pairs:
        if col_a in df.columns and col_b in df.columns:
            new_cols.append((pl.col(col_a) - pl.col(col_b)).alias(alias))

    if new_cols:
        df = df.with_columns(new_cols)

    return df


def _ensure_competition_col(matches: pl.DataFrame) -> pl.DataFrame:
    if "competition_id" not in matches.columns and "competition_id_league" in matches.columns:
        return matches.rename({"competition_id_league": "competition_id"})
    return matches


def _ensure_season_col(matches: pl.DataFrame) -> pl.DataFrame:
    if "season" not in matches.columns:
        return matches.with_columns(pl.lit(0).alias("season"))
    return matches


def _ensure_venue_type_col(matches: pl.DataFrame) -> pl.DataFrame:
    from footyml.config import VENUE_HOME_A
    if "venue_type" not in matches.columns:
        return matches.with_columns(pl.lit(VENUE_HOME_A).alias("venue_type"))
    return matches.with_columns(
        pl.col("venue_type").fill_null(pl.lit(VENUE_HOME_A))
    )
