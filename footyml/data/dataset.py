from __future__ import annotations

import numpy as np
import polars as pl

from footyml.config import LEAGUE_IDS
from footyml.data.store import DataStore


class MatchDataset:
    """Load feature-engineered match data ready for ML training."""

    def __init__(
        self,
        league: str,
        seasons: range | list[int],
        store: DataStore | None = None,
    ):
        if league != "__all__" and league not in LEAGUE_IDS:
            raise ValueError(f"Unknown league '{league}'. Valid: {list(LEAGUE_IDS)} or '__all__'")
        self.league = league
        self.seasons = list(seasons)
        self._store = store or DataStore()

    @classmethod
    def all_leagues(
        cls,
        seasons: range | list[int],
        store: DataStore | None = None,
    ) -> "MatchDataset":
        """Load from all domestic leagues combined (excludes tournament matches)."""
        obj = cls.__new__(cls)
        obj.league = "__all__"
        obj.seasons = list(seasons)
        obj._store = store or DataStore()
        return obj

    @property
    def competition_id(self) -> str:
        if self.league == "__all__":
            raise ValueError("competition_id is not defined for all-leagues mode")
        return LEAGUE_IDS[self.league]

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) numpy arrays ordered chronologically.

        X: float32, shape (n_matches, n_features)
        y: int32,   shape (n_matches,) — 0=away win, 1=draw, 2=home win
        """
        df = self.load_polars()
        y_col = "result"
        skip_cols = {"game_id", "date", y_col}
        feature_cols = [c for c in df.columns if c not in skip_cols]
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df[y_col].to_numpy().astype(np.int32)
        return X, y

    def load_polars(self) -> pl.DataFrame:
        """Return feature DataFrame including game_id, date, result, ordered by date."""
        seasons_str = ", ".join(str(s) for s in self.seasons)
        if self.league == "__all__":
            league_ids = ", ".join(f"'{cid}'" for cid in LEAGUE_IDS.values())
            cid_filter = f"AND m.competition_id IN ({league_ids})"
        else:
            cid_filter = f"AND m.competition_id = '{self.competition_id}'"
        return self._store.query(f"""
            SELECT f.*, m.date, m.result
            FROM features f
            JOIN matches m USING (game_id)
            WHERE m.season IN ({seasons_str})
              AND m.result IS NOT NULL
              {cid_filter}
            ORDER BY m.date
        """)

    def feature_names(self) -> list[str]:
        df = self.load_polars()
        skip = {"game_id", "date", "result"}
        return [c for c in df.columns if c not in skip]

    @staticmethod
    def time_split(
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        test_cutoff: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split arrays at test_cutoff date string (YYYY-MM-DD). Train < cutoff, test >= cutoff."""
        mask = np.array([str(d) < test_cutoff for d in dates])
        return X[mask], X[~mask], y[mask], y[~mask]
