from __future__ import annotations

import polars as pl

from footyml.data.store import DataStore


class TransferBuilder:
    """Transfer window features: incoming, outgoing, net spend per team."""

    def __init__(self, store: DataStore | None = None):
        self._store = store

    def build(self, matches: pl.DataFrame) -> pl.DataFrame:
        if self._store is None:
            return matches.select("game_id")

        rows = []
        for row in matches.iter_rows(named=True):
            season_id = str(row.get("season", row.get("season_id", "")))
            home_feats = self._transfer_feats(str(row["home_club_id"]), season_id, "home")
            away_feats = self._transfer_feats(str(row["away_club_id"]), season_id, "away")
            feat: dict[str, object] = {"game_id": str(row["game_id"])}
            feat.update(home_feats)
            feat.update(away_feats)
            rows.append(feat)

        return pl.DataFrame(rows)

    def _transfer_feats(self, club_id: str, season_id: str, side: str) -> dict[str, object]:
        assert self._store is not None
        transfers = self._store.get_transfers(club_id, season_id)
        prefix = side

        if len(transfers) == 0:
            return {
                f"{prefix}_transfer_in_eur": None,
                f"{prefix}_transfer_out_eur": None,
                f"{prefix}_transfer_net_eur": None,
            }

        incoming = transfers.filter(pl.col("direction") == "in")["fee_eur"].drop_nulls()
        outgoing = transfers.filter(pl.col("direction") == "out")["fee_eur"].drop_nulls()
        in_spend = int(incoming.sum()) if len(incoming) > 0 else 0
        out_spend = int(outgoing.sum()) if len(outgoing) > 0 else 0

        return {
            f"{prefix}_transfer_in_eur": in_spend,
            f"{prefix}_transfer_out_eur": out_spend,
            f"{prefix}_transfer_net_eur": in_spend - out_spend,
        }
