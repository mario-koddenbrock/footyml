"""Tournament prediction pipeline.

Handles World Cup and other neutral-venue international tournaments.
Fetches fixtures from football-data.org, builds features, and runs predictions.

Usage:
    import asyncio
    from footyml.prediction.tournament import TournamentPredictor

    async def main():
        predictor = TournamentPredictor(model_path="data/models/tabpfn_WC_2026.pkl")
        async with predictor:
            predictions = await predictor.predict_all_upcoming("WC")
            for p in predictions:
                print(p)

    asyncio.run(main())
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from footyml.config import LEAGUE_IDS, MODELS_DIR, TOURNAMENT_IDS
from footyml.data.store import DataStore
from footyml.features.elo import EloBuilder, _expected_score
from footyml.features.pipeline import FeaturePipeline
from footyml.models.predictor import TabPFNMatchPredictor
from footyml.providers.football_data import FootballDataProvider

console = Console()

RESULT_LABELS = {0: "B Win", 1: "Draw", 2: "A Win"}


class TournamentPredictor:
    """Fetch fixtures, build features, and predict tournament match outcomes.

    For World Cup matches, no home advantage is assumed (venue_type=neutral)
    unless the host nation is explicitly involved.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        api_key: str | None = None,
        store: DataStore | None = None,
    ):
        self._model_path = model_path
        self._api_key = api_key
        self._store = store or DataStore()
        self._pipeline = FeaturePipeline(self._store)
        self._predictor: TabPFNMatchPredictor | None = None
        self._provider: FootballDataProvider | None = None

    async def __aenter__(self) -> "TournamentPredictor":
        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._provider = FootballDataProvider(**kwargs)
        await self._provider.__aenter__()

        if self._model_path and self._model_path.exists():
            self._predictor = TabPFNMatchPredictor.load(self._model_path)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._provider:
            await self._provider.__aexit__(*args)

    # ------------------------------------------------------------------
    # Ingest tournament data
    # ------------------------------------------------------------------

    async def sync_fixtures(
        self, competition: str, season: int | None = None
    ) -> pl.DataFrame:
        """Fetch all matches from the API and store in DuckDB.

        Also updates results for previously scheduled matches.
        Returns the full match DataFrame.
        """
        assert self._provider is not None, "Use 'async with TournamentPredictor()'"
        all_matches = await self._provider.get_all_matches(competition, season=season)

        if len(all_matches) == 0:
            console.print(f"[yellow]No matches found for competition {competition}[/yellow]")
            return all_matches

        # Map to the internal matches schema (drop extra columns)
        store_cols = [
            "game_id", "competition_id", "season", "matchday", "date",
            "home_club_id", "away_club_id", "home_goals", "away_goals",
            "result", "venue_type", "group_id", "stage",
            "home_club_name", "away_club_name",
        ]
        available = [c for c in store_cols if c in all_matches.columns]
        to_store = all_matches.select(available)

        if "date" in to_store.columns and to_store["date"].dtype != pl.Date:
            to_store = to_store.with_columns(
                pl.col("date").cast(pl.Date)
            )

        self._store.upsert_matches(to_store)
        console.print(f"[green]Synced {len(to_store)} matches for {competition}[/green]")
        return all_matches

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    async def predict_all_upcoming(
        self,
        competition: str,
        season: int | None = None,
        sync: bool = True,
    ) -> list[dict[str, object]]:
        """Fetch upcoming fixtures and return predictions for all of them.

        Args:
            competition: competition code, e.g. "WC"
            season: season year, e.g. 2026
            sync: if True, also fetch and store completed matches first
        """
        assert self._provider is not None, "Use 'async with TournamentPredictor()'"

        if sync:
            await self.sync_fixtures(competition, season=season)

        upcoming = await self._provider.get_upcoming(competition, season=season)

        if len(upcoming) == 0:
            console.print(f"[yellow]No upcoming matches for {competition}[/yellow]")
            return []

        if self._predictor is None:
            console.print("[yellow]No model loaded — returning fixtures only (no predictions)[/yellow]")
            return _format_fixtures_only(upcoming)

        # Build features for all matches in the competition (so we have context for form etc.)
        features_df = self._pipeline.build(
            competition_id=competition,
            tournament_mode=True,
        )

        predictions = []
        for row in upcoming.iter_rows(named=True):
            gid = str(row["game_id"])
            feat_row = features_df.filter(pl.col("game_id") == gid)

            if len(feat_row) == 0:
                # Upcoming match not in features yet — use Elo-only fallback
                pred = self._elo_fallback_prediction(
                    home_id=str(row.get("home_club_id", "")),
                    away_id=str(row.get("away_club_id", "")),
                    venue_type=str(row.get("venue_type", "neutral")),
                )
            else:
                skip = {"game_id"}
                feat_cols = [c for c in feat_row.columns if c not in skip]
                X = feat_row.select(feat_cols).to_numpy().astype(np.float32)
                proba = self._predictor.predict_proba(X)[0]
                pred = {
                    "home_win_prob": round(float(proba[2]) * 100, 1),
                    "draw_prob": round(float(proba[1]) * 100, 1),
                    "away_win_prob": round(float(proba[0]) * 100, 1),
                    "predicted_result": RESULT_LABELS[int(np.argmax(proba))],
                    "method": "tabpfn",
                }

            predictions.append(
                {
                    "game_id": gid,
                    "date": str(row.get("date", ""))[:10],
                    "stage": str(row.get("stage", "") or ""),
                    "group_id": str(row.get("group_id", "") or ""),
                    "home_team": str(row.get("home_club_name", row.get("home_club_id", ""))),
                    "away_team": str(row.get("away_club_name", row.get("away_club_id", ""))),
                    "venue_type": str(row.get("venue_type", "neutral")),
                    **pred,
                }
            )

        return sorted(predictions, key=lambda x: str(x.get("date", "")))

    def _elo_fallback_prediction(
        self, home_id: str, away_id: str, venue_type: str
    ) -> dict[str, object]:
        """Simple Elo-based probability estimate when full features unavailable."""
        from footyml.config import ELO_INITIAL, HOME_ADVANTAGE_ELO, VENUE_HOME_A

        elo_df = self._store.get_elo()
        home_elo = ELO_INITIAL
        away_elo = ELO_INITIAL

        if len(elo_df) > 0 and "club_id" in elo_df.columns:
            home_rows = elo_df.filter(pl.col("club_id") == home_id)
            away_rows = elo_df.filter(pl.col("club_id") == away_id)
            if len(home_rows) > 0:
                home_elo = float(home_rows.sort("date")["elo"][-1])
            if len(away_rows) > 0:
                away_elo = float(away_rows.sort("date")["elo"][-1])

        bonus = HOME_ADVANTAGE_ELO if venue_type == VENUE_HOME_A else 0.0
        p_home = _expected_score(home_elo + bonus, away_elo)
        p_draw = 0.25  # heuristic
        p_home_adj = p_home * (1 - p_draw)
        p_away_adj = (1 - p_home) * (1 - p_draw)

        predicted = "A Win" if p_home_adj > p_away_adj else ("Draw" if p_draw > max(p_home_adj, p_away_adj) else "B Win")

        return {
            "home_win_prob": round(p_home_adj * 100, 1),
            "draw_prob": round(p_draw * 100, 1),
            "away_win_prob": round(p_away_adj * 100, 1),
            "predicted_result": predicted,
            "method": "elo_fallback",
        }

    # ------------------------------------------------------------------
    # Training data collection
    # ------------------------------------------------------------------

    async def collect_training_data(
        self, competition: str, seasons: list[int]
    ) -> pl.DataFrame:
        """Ingest completed matches for multiple seasons into the store.

        Returns combined DataFrame of all completed matches.
        """
        assert self._provider is not None
        frames = []
        for s in seasons:
            completed = await self._provider.get_completed(competition, season=s)
            if len(completed) > 0:
                await self.sync_fixtures(competition, season=s)
                frames.append(completed)
                console.print(f"[green]Season {s}: {len(completed)} completed matches[/green]")

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames)


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def _format_fixtures_only(upcoming: pl.DataFrame) -> list[dict[str, object]]:
    rows = []
    for row in upcoming.iter_rows(named=True):
        rows.append(
            {
                "game_id": str(row.get("game_id", "")),
                "date": str(row.get("date", ""))[:10],
                "stage": str(row.get("stage", "") or ""),
                "group_id": str(row.get("group_id", "") or ""),
                "home_team": str(row.get("home_club_name", row.get("home_club_id", ""))),
                "away_team": str(row.get("away_club_name", row.get("away_club_id", ""))),
                "venue_type": str(row.get("venue_type", "neutral")),
                "home_win_prob": None,
                "draw_prob": None,
                "away_win_prob": None,
                "predicted_result": "—",
                "method": "no_model",
            }
        )
    return sorted(rows, key=lambda x: str(x.get("date", "")))
