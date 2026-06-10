from __future__ import annotations

import polars as pl

from footyml.data.store import DataStore


class StandingsBuilder:
    """Reconstruct live league table at each match's date (no leakage)."""

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

            group_id = row.get("group_id")
            if group_id:
                # Tournament group mode: filter to same group
                prior = matches_sorted.filter(
                    (pl.col("date") < match_date)
                    & (pl.col("group_id") == group_id)
                    & (pl.col("result").is_not_null())
                )
            else:
                prior = matches_sorted.filter(
                    (pl.col("date") < match_date)
                    & (pl.col("competition_id") == row.get("competition_id", ""))
                    & (pl.col("result").is_not_null())
                )

            table = _compute_table(prior)
            home_row = _get_team_row(table, home_id)
            away_row = _get_team_row(table, away_id)

            feat: dict[str, object] = {"game_id": game_id}
            for key, val in home_row.items():
                feat[f"home_table_{key}"] = val
            for key, val in away_row.items():
                feat[f"away_table_{key}"] = val
            rows.append(feat)

        return pl.DataFrame(rows)


def _compute_table(matches: pl.DataFrame) -> dict[str, dict[str, int | float]]:
    """Compute league table from completed matches."""
    table: dict[str, dict[str, int | float]] = {}

    for row in matches.iter_rows(named=True):
        home_id = str(row["home_club_id"])
        away_id = str(row["away_club_id"])
        result = row.get("result")
        if result is None:
            continue

        hg = row.get("home_goals") or 0
        ag = row.get("away_goals") or 0

        for cid in (home_id, away_id):
            if cid not in table:
                table[cid] = {"wins": 0, "draws": 0, "losses": 0, "pts": 0, "gf": 0, "ga": 0}

        if result == 2:
            table[home_id]["wins"] += 1
            table[home_id]["pts"] += 3
            table[away_id]["losses"] += 1
        elif result == 1:
            table[home_id]["draws"] += 1
            table[home_id]["pts"] += 1
            table[away_id]["draws"] += 1
            table[away_id]["pts"] += 1
        else:
            table[away_id]["wins"] += 1
            table[away_id]["pts"] += 3
            table[home_id]["losses"] += 1

        table[home_id]["gf"] += hg
        table[home_id]["ga"] += ag
        table[away_id]["gf"] += ag
        table[away_id]["ga"] += hg

    return table


def _get_team_row(
    table: dict[str, dict[str, int | float]], club_id: str
) -> dict[str, int | float | None]:
    if club_id not in table:
        return {"pos": None, "pts": None, "ppg": None, "gd": None, "wins": None, "draws": None, "losses": None}

    t = table[club_id]
    played = t["wins"] + t["draws"] + t["losses"]
    gd = t["gf"] - t["ga"]
    ppg = t["pts"] / played if played > 0 else 0.0

    # Compute position by ranking all teams by points (desc), then gd (desc)
    ranking = sorted(
        table.items(),
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"]),
        reverse=True,
    )
    position = next((i + 1 for i, (cid, _) in enumerate(ranking) if cid == club_id), None)

    return {
        "pos": position,
        "pts": t["pts"],
        "ppg": round(ppg, 3),
        "gd": gd,
        "wins": t["wins"],
        "draws": t["draws"],
        "losses": t["losses"],
    }
