from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import polars as pl

from footyml.config import LEAGUE_IDS, MODELS_DIR
from footyml.data.store import DataStore
from footyml.features.pipeline import FeaturePipeline
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.fetchers import TransfermarktFetcher
from footyml.models.predictor import TabPFNMatchPredictor

RESULT_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}


class PredictionService:
    """High-level prediction logic used by CLI scripts."""

    def __init__(
        self,
        predictor: TabPFNMatchPredictor,
        store: DataStore | None = None,
    ):
        self._predictor = predictor
        self._store = store or DataStore()
        self._pipeline = FeaturePipeline(self._store)

    def predict_from_features(self, feature_vector: np.ndarray) -> dict[str, float | str]:
        """Run prediction on a single pre-built feature vector."""
        X = feature_vector.reshape(1, -1)
        proba = self._predictor.predict_proba(X)[0]
        predicted = int(np.argmax(proba))
        return {
            "home_win_prob": round(float(proba[2]) * 100, 1),
            "draw_prob": round(float(proba[1]) * 100, 1),
            "away_win_prob": round(float(proba[0]) * 100, 1),
            "predicted_result": RESULT_LABELS[predicted],
        }

    def latest_model_path(self, league: str) -> Path | None:
        """Find the most recently saved model for a league."""
        pattern = f"tabpfn_{league}_*.pkl"
        candidates = sorted(MODELS_DIR.glob(pattern))
        return candidates[-1] if candidates else None

    @classmethod
    def from_saved_model(
        cls, league: str, store: DataStore | None = None
    ) -> "PredictionService":
        """Load the latest model for a league and return a ready-to-use service."""
        dummy = cls(TabPFNMatchPredictor(), store)
        model_path = dummy.latest_model_path(league)
        if model_path is None:
            raise FileNotFoundError(
                f"No trained model found for league '{league}'. "
                f"Run: python scripts/train.py --league {league} --season <year>"
            )
        predictor = TabPFNMatchPredictor.load(model_path)
        return cls(predictor, store)

    async def _resolve_club_id(self, name: str) -> str | None:
        """Search Transfermarkt for a club ID by name."""
        async with TransfermarktClient() as client:
            data = await client.get(f"/clubs/search/{name}")
            clubs = data.get("clubs", [])
            if clubs:
                return str(clubs[0].get("id", ""))
        return None

    def predict_upcoming(
        self, league: str, season: int | None = None
    ) -> list[dict[str, object]]:
        """Fetch upcoming fixtures and predict each match."""
        competition_id = LEAGUE_IDS[league]
        upcoming = self._store.query(f"""
            SELECT * FROM matches
            WHERE competition_id = '{competition_id}'
              AND result IS NULL
            ORDER BY date
            LIMIT 50
        """)

        if len(upcoming) == 0:
            return []

        results = []
        features = self._pipeline.build(competition_id=competition_id)

        for row in upcoming.iter_rows(named=True):
            gid = str(row["game_id"])
            feat_row = features.filter(pl.col("game_id") == gid)
            if len(feat_row) == 0:
                continue
            skip = {"game_id"}
            feat_cols = [c for c in feat_row.columns if c not in skip]
            X = feat_row.select(feat_cols).to_numpy().astype(float)
            pred = self.predict_from_features(X[0])
            results.append(
                {
                    "game_id": gid,
                    "date": str(row.get("date", "")),
                    "home_team": str(row.get("home_club_id", "")),
                    "away_team": str(row.get("away_club_id", "")),
                    **pred,
                }
            )

        return results
