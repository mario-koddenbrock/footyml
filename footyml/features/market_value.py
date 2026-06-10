from __future__ import annotations

import polars as pl

from footyml.data.store import DataStore


class MarketValueBuilder:
    """Squad market value features for both home and away teams."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        if self._store is None:
            return matches.select("game_id")

        rows = []
        for row in matches.iter_rows(named=True):
            season_id = str(row.get("season", row.get("season_id", "")))
            home_feats = self._squad_features(str(row["home_club_id"]), season_id, "home")
            away_feats = self._squad_features(str(row["away_club_id"]), season_id, "away")
            feat = {"game_id": str(row["game_id"])}
            feat.update(home_feats)
            feat.update(away_feats)
            home_total = feat.get("home_mv_total_eur")
            away_total = feat.get("away_mv_total_eur")
            if home_total and away_total and away_total != 0:
                feat["mv_ratio"] = home_total / away_total
                feat["mv_diff_eur"] = home_total - away_total
            else:
                feat["mv_ratio"] = None
                feat["mv_diff_eur"] = None
            rows.append(feat)

        return pl.DataFrame(rows)

    def _squad_features(self, club_id: str, season_id: str, side: str) -> dict[str, object]:
        assert self._store is not None
        squad = self._store.get_squad(club_id, season_id)
        prefix = f"{side}_mv"

        if len(squad) == 0 or "market_value_eur" not in squad.columns:
            return {f"{prefix}_{k}": None for k in ["total_eur", "avg_eur", "median_eur", "top1_eur", "top3_eur", "top5_eur"]}

        vals = squad["market_value_eur"].drop_nulls().sort(descending=True)
        if len(vals) == 0:
            return {f"{prefix}_{k}": None for k in ["total_eur", "avg_eur", "median_eur", "top1_eur", "top3_eur", "top5_eur"]}

        return {
            f"{prefix}_total_eur": int(vals.sum()),
            f"{prefix}_avg_eur": float(vals.mean()),  # type: ignore[arg-type]
            f"{prefix}_median_eur": float(vals.median()),  # type: ignore[arg-type]
            f"{prefix}_top1_eur": int(vals[0]) if len(vals) >= 1 else None,
            f"{prefix}_top3_eur": int(vals[:3].sum()) if len(vals) >= 3 else int(vals.sum()),
            f"{prefix}_top5_eur": int(vals[:5].sum()) if len(vals) >= 5 else int(vals.sum()),
        }
