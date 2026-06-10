from __future__ import annotations

import polars as pl

from footyml.config import ELO_INITIAL, ELO_K_FACTOR, HOME_ADVANTAGE_ELO, VENUE_NEUTRAL, VENUE_HOME_A
from footyml.data.store import DataStore


class EloBuilder:
    """Compute historical Elo ratings and produce pre-match Elo features."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        """Return DataFrame with game_id, home_elo, away_elo, elo_diff.

        Processes matches chronologically. Stores ratings in elo_ratings table.
        """
        matches_sorted = matches.sort("date")
        elos: dict[str, float] = {}
        rows = []

        for row in matches_sorted.iter_rows(named=True):
            home_id = str(row["home_club_id"])
            away_id = str(row["away_club_id"])
            home_elo = elos.get(home_id, ELO_INITIAL)
            away_elo = elos.get(away_id, ELO_INITIAL)

            rows.append(
                {
                    "game_id": str(row["game_id"]),
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_diff": home_elo - away_elo,
                }
            )

            result = row.get("result")
            if result is not None:
                actual_home, actual_away = _result_to_scores(int(result))
                venue_type = str(row.get("venue_type") or VENUE_HOME_A)
                home_bonus = HOME_ADVANTAGE_ELO if venue_type == VENUE_HOME_A else 0.0
                expected_home = _expected_score(home_elo + home_bonus, away_elo)
                expected_away = 1.0 - expected_home
                elos[home_id] = _update(home_elo, actual_home, expected_home)
                elos[away_id] = _update(away_elo, actual_away, expected_away)

        result_df = pl.DataFrame(rows)

        if self._store is not None and len(result_df) > 0:
            _persist_elos(elos, matches_sorted, self._store)

        return result_df


def _result_to_scores(result: int) -> tuple[float, float]:
    if result == 2:
        return 1.0, 0.0
    elif result == 1:
        return 0.5, 0.5
    else:
        return 0.0, 1.0


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update(rating: float, actual: float, expected: float) -> float:
    return rating + ELO_K_FACTOR * (actual - expected)


def _persist_elos(
    elos: dict[str, float], matches: pl.DataFrame, store: DataStore
) -> None:
    last_date = matches["date"].max()
    rows = [{"club_id": cid, "date": last_date, "elo": elo} for cid, elo in elos.items()]
    if rows:
        store.upsert_elo(pl.DataFrame(rows))
