from __future__ import annotations

from datetime import date

import polars as pl

from footyml.data.store import DataStore


class SquadBuilder:
    """Squad demographic features: average age, foreign player %, squad size."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        if self._store is None:
            return matches.select("game_id")

        rows = []
        for row in matches.iter_rows(named=True):
            season_id = str(row.get("season", row.get("season_id", "")))
            match_date = row["date"]
            home_feats = self._squad_feats(str(row["home_club_id"]), season_id, match_date, "home")
            away_feats = self._squad_feats(str(row["away_club_id"]), season_id, match_date, "away")
            feat: dict[str, object] = {"game_id": str(row["game_id"])}
            feat.update(home_feats)
            feat.update(away_feats)
            rows.append(feat)

        return pl.DataFrame(rows)

    def _squad_feats(
        self, club_id: str, season_id: str, match_date: object, side: str
    ) -> dict[str, object]:
        assert self._store is not None
        squad = self._store.get_squad(club_id, season_id)
        prefix = side

        if len(squad) == 0:
            return {
                f"{prefix}_avg_age": None,
                f"{prefix}_median_age": None,
                f"{prefix}_foreign_pct": None,
                f"{prefix}_squad_size": None,
            }

        squad_size = len(squad)
        ages: list[float] = []
        foreign_count = 0

        if "date_of_birth" in squad.columns and "nationality" in squad.columns:
            ref_date = _to_date(match_date)
            for prow in squad.iter_rows(named=True):
                dob = _parse_date(str(prow.get("date_of_birth", "") or ""))
                if dob and ref_date:
                    age = (ref_date - dob).days / 365.25
                    ages.append(age)
                nat = str(prow.get("nationality", "") or "")
                if nat and nat.lower() not in ("", "german", "english", "spanish", "italian", "french"):
                    foreign_count += 1

        return {
            f"{prefix}_avg_age": float(sum(ages) / len(ages)) if ages else None,
            f"{prefix}_median_age": float(sorted(ages)[len(ages) // 2]) if ages else None,
            f"{prefix}_foreign_pct": round(foreign_count / squad_size * 100, 1) if squad_size else None,
            f"{prefix}_squad_size": squad_size,
        }


def _to_date(d: object) -> date | None:
    if isinstance(d, date):
        return d
    try:
        from datetime import date as dt
        return dt.fromisoformat(str(d))
    except (ValueError, TypeError):
        return None


def _parse_date(s: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%b %d, %Y"):
        try:
            from datetime import datetime
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None
