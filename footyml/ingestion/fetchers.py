import asyncio
from datetime import date

import polars as pl

from footyml.config import LEAGUE_IDS
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.schemas import CompetitionClub, GameResult, TransferEntry
from footyml.utils.money import parse_market_value


class TransfermarktFetcher:
    """High-level data fetcher — returns normalized Polars DataFrames, caches to Parquet."""

    def __init__(self, client: TransfermarktClient, cache: ParquetCache):
        self._client = client
        self._cache = cache

    # ------------------------------------------------------------------
    # Competition / Club lists
    # ------------------------------------------------------------------

    async def fetch_competition_clubs(self, competition_id: str, season_id: str) -> pl.DataFrame:
        key = self._cache.make_key("competition_clubs", competition_id=competition_id, season_id=season_id)
        if cached := self._cache.get(key):
            return cached

        data = await self._client.get(
            f"/competitions/{competition_id}/clubs",
            params={"season_id": season_id},
        )
        clubs = data.get("clubs", [])
        rows = [
            {
                "club_id": str(c.get("id", "")),
                "club_name": str(c.get("name", "")),
                "competition_id": competition_id,
                "season_id": season_id,
            }
            for c in clubs
        ]
        df = pl.DataFrame(rows) if rows else _empty_clubs_df()
        self._cache.set(key, df)
        return df

    # ------------------------------------------------------------------
    # Club games
    # ------------------------------------------------------------------

    async def fetch_club_games(self, club_id: str, season_id: str) -> pl.DataFrame:
        key = self._cache.make_key("club_games", club_id=club_id, season_id=season_id)
        if cached := self._cache.get(key):
            return cached

        data = await self._client.get(
            f"/clubs/{club_id}/games",
            params={"season_id": season_id},
        )
        games_raw = data.get("games", [])
        rows = []
        for g in games_raw:
            result_raw = g.get("result", "")
            home_goals, away_goals = _parse_result(result_raw)
            rows.append(
                {
                    "game_id": str(g.get("id", "")),
                    "home_club_id": str(g.get("homeClub", {}).get("id", "") or ""),
                    "away_club_id": str(g.get("awayClub", {}).get("id", "") or ""),
                    "home_club_name": str(g.get("homeClub", {}).get("name", "") or ""),
                    "away_club_name": str(g.get("awayClub", {}).get("name", "") or ""),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "date": str(g.get("date", "") or ""),
                    "matchday": _safe_int(g.get("matchday")),
                    "competition_id": str(g.get("competition", {}).get("id", "") or ""),
                }
            )
        df = pl.DataFrame(rows) if rows else _empty_games_df()
        self._cache.set(key, df)
        return df

    # ------------------------------------------------------------------
    # Club profile
    # ------------------------------------------------------------------

    async def fetch_club_profile(self, club_id: str) -> pl.DataFrame:
        key = self._cache.make_key("club_profile", club_id=club_id)
        if cached := self._cache.get(key):
            return cached

        data = await self._client.get(f"/clubs/{club_id}/profile")
        row = {
            "club_id": club_id,
            "club_name": str(data.get("name", "")),
            "squad_size": _safe_int(data.get("squadSize")),
            "average_age": _safe_float(data.get("averageAge")),
            "foreigners_number": _safe_int(data.get("foreignersNumber")),
            "foreigners_percentage": _safe_float(_pct_str(data.get("foreignersPercentage"))),
            "total_market_value_eur": parse_market_value(data.get("totalMarketValue")),
        }
        df = pl.DataFrame([row])
        self._cache.set(key, df)
        return df

    # ------------------------------------------------------------------
    # Squad market values
    # ------------------------------------------------------------------

    async def fetch_squad_market_values(self, club_id: str, season_id: str) -> pl.DataFrame:
        key = self._cache.make_key("squad_mv", club_id=club_id, season_id=season_id)
        if cached := self._cache.get(key):
            return cached

        data = await self._client.get(
            f"/clubs/{club_id}/players",
            params={"season_id": season_id},
        )
        players = data.get("players", [])
        rows = [
            {
                "player_id": str(p.get("id", "")),
                "club_id": club_id,
                "season_id": season_id,
                "player_name": str(p.get("name", "")),
                "market_value_eur": parse_market_value(p.get("marketValue")),
                "date_of_birth": str(p.get("dateOfBirth", "") or ""),
                "nationality": str((p.get("nationality") or [""])[0]),
            }
            for p in players
        ]
        df = pl.DataFrame(rows) if rows else _empty_squad_df()
        self._cache.set(key, df)
        return df

    # ------------------------------------------------------------------
    # Club transfers
    # ------------------------------------------------------------------

    async def fetch_club_transfers(self, club_id: str, season_id: str) -> pl.DataFrame:
        key = self._cache.make_key("club_transfers", club_id=club_id, season_id=season_id)
        if cached := self._cache.get(key):
            return cached

        data = await self._client.get(
            f"/clubs/{club_id}/transfers",
            params={"season_id": season_id},
        )
        rows: list[dict] = []
        for direction in ("in", "out"):
            for t in data.get(f"transfers_{direction}", []):
                rows.append(
                    {
                        "club_id": club_id,
                        "season_id": season_id,
                        "direction": direction,
                        "player_id": str(t.get("id", "") or ""),
                        "player_name": str(t.get("name", "") or ""),
                        "fee_eur": parse_market_value(t.get("fee")),
                    }
                )
        df = pl.DataFrame(rows) if rows else _empty_transfers_df()
        self._cache.set(key, df)
        return df

    # ------------------------------------------------------------------
    # Top-level season orchestrator
    # ------------------------------------------------------------------

    async def fetch_league_season(self, league: str, season: int) -> pl.DataFrame:
        """Fetch all match data for a league season. Returns raw matches DataFrame."""
        competition_id = LEAGUE_IDS[league]
        season_id = str(season)

        clubs_df = await self.fetch_competition_clubs(competition_id, season_id)
        club_ids = clubs_df["club_id"].to_list()

        game_frames = await asyncio.gather(
            *[self.fetch_club_games(cid, season_id) for cid in club_ids],
            return_exceptions=True,
        )

        valid_frames = [f for f in game_frames if isinstance(f, pl.DataFrame) and len(f) > 0]
        if not valid_frames:
            return _empty_games_df()

        all_games = pl.concat(valid_frames).unique(subset=["game_id"])

        all_games = all_games.with_columns([
            pl.lit(competition_id).alias("competition_id_league"),
            pl.lit(season).alias("season"),
        ])

        return all_games


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_result(result: str | None) -> tuple[int | None, int | None]:
    if not result:
        return None, None
    import re
    m = re.match(r"(\d+)\s*[:\-]\s*(\d+)", str(result))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _safe_int(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _pct_str(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).replace("%", "").strip()
    return s if s else None


def _empty_clubs_df() -> pl.DataFrame:
    return pl.DataFrame({"club_id": [], "club_name": [], "competition_id": [], "season_id": []})


def _empty_games_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": [],
            "home_club_id": [],
            "away_club_id": [],
            "home_club_name": [],
            "away_club_name": [],
            "home_goals": [],
            "away_goals": [],
            "date": [],
            "matchday": [],
            "competition_id": [],
        }
    )


def _empty_squad_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [],
            "club_id": [],
            "season_id": [],
            "player_name": [],
            "market_value_eur": [],
            "date_of_birth": [],
            "nationality": [],
        }
    )


def _empty_transfers_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "club_id": [],
            "season_id": [],
            "direction": [],
            "player_id": [],
            "player_name": [],
            "fee_eur": [],
        }
    )
