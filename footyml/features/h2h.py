from __future__ import annotations

import polars as pl

from footyml.data.store import DataStore

H2H_WINDOWS = [3, 5]


class HeadToHeadBuilder:
    """Head-to-head features between home and away team (last 3/5 meetings)."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        matches_sorted = matches.sort("date")
        rows = []

        for row in matches_sorted.iter_rows(named=True):
            game_id = str(row["game_id"])
            home_id = str(row["home_club_id"])
            away_id = str(row["away_club_id"])
            match_date = row["date"]

            prior = matches_sorted.filter(
                (pl.col("date") < match_date)
                & (
                    (pl.col("home_club_id") == home_id) & (pl.col("away_club_id") == away_id)
                    | (pl.col("home_club_id") == away_id) & (pl.col("away_club_id") == home_id)
                )
            )

            feat: dict[str, object] = {"game_id": game_id}
            for n in H2H_WINDOWS:
                last_n = prior.tail(n)
                feat.update(_compute_h2h_stats(last_n, home_id, n))

            rows.append(feat)

        return pl.DataFrame(rows)


def _compute_h2h_stats(games: pl.DataFrame, home_id: str, n: int) -> dict[str, float | int | None]:
    prefix = f"h2h_{n}_"
    if len(games) == 0:
        return {
            f"{prefix}home_wins": None,
            f"{prefix}away_wins": None,
            f"{prefix}draws": None,
            f"{prefix}home_gf": None,
            f"{prefix}home_ga": None,
        }

    home_wins = draws = away_wins = 0
    home_gf = home_ga = 0

    for row in games.iter_rows(named=True):
        hg = row.get("home_goals") or 0
        ag = row.get("away_goals") or 0
        if str(row["home_club_id"]) == home_id:
            home_gf += hg
            home_ga += ag
            r = row.get("result")
            if r == 2:
                home_wins += 1
            elif r == 1:
                draws += 1
            else:
                away_wins += 1
        else:
            home_gf += ag
            home_ga += hg
            r = row.get("result")
            if r == 0:
                home_wins += 1
            elif r == 1:
                draws += 1
            else:
                away_wins += 1

    return {
        f"{prefix}home_wins": home_wins,
        f"{prefix}away_wins": away_wins,
        f"{prefix}draws": draws,
        f"{prefix}home_gf": home_gf,
        f"{prefix}home_ga": home_ga,
    }
