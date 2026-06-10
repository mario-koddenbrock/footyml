"""football-data.org API provider.

API key required: set FOOTBALL_DATA_API_KEY environment variable.
Free tier: 10 requests/minute, covers World Cup, Euro, and major leagues.

Competition codes (examples):
    WC  — FIFA World Cup
    EC  — UEFA European Championship
    CL  — UEFA Champions League
    PL  — Premier League
    BL1 — Bundesliga

Discover all available codes: FootballDataProvider.list_competitions()
"""
from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from footyml.config import (
    FOOTBALL_DATA_API_KEY,
    FOOTBALL_DATA_BASE_URL,
    FOOTBALL_DATA_RATE_LIMIT_RPS,
    RAW_DIR,
    VENUE_NEUTRAL,
    VENUE_HOME_A,
    VENUE_HOME_B,
)
from footyml.ingestion.cache import ParquetCache


class FootballDataProvider:
    """Async client for the football-data.org v4 API.

    Usage:
        async with FootballDataProvider() as p:
            fixtures = await p.get_upcoming("WC")
            results = await p.get_completed("WC", season=2026)
    """

    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY):
        if not api_key:
            raise ValueError(
                "Set FOOTBALL_DATA_API_KEY environment variable. "
                "Get a free key at https://www.football-data.org/"
            )
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._last_call: float = 0.0
        self._min_interval = 1.0 / FOOTBALL_DATA_RATE_LIMIT_RPS
        self._cache = ParquetCache(RAW_DIR / "football_data_api")

    async def __aenter__(self) -> "FootballDataProvider":
        self._client = httpx.AsyncClient(
            base_url=FOOTBALL_DATA_BASE_URL,
            headers={"X-Auth-Token": self._api_key},
            timeout=30.0,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict:  # type: ignore[type-arg]
        if self._client is None:
            raise RuntimeError("Use 'async with FootballDataProvider()'")
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        response = await self._client.get(path, params=params)
        self._last_call = time.monotonic()
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_competitions(self) -> pl.DataFrame:
        """Return all competitions available on the API."""
        data = await self._get("/competitions")
        comps = data.get("competitions", [])
        rows = [
            {
                "id": str(c.get("id", "")),
                "code": str(c.get("code", "")),
                "name": str(c.get("name", "")),
                "area": str(c.get("area", {}).get("name", "")),
            }
            for c in comps
        ]
        return pl.DataFrame(rows) if rows else pl.DataFrame(
            {"id": [], "code": [], "name": [], "area": []}
        )

    async def get_upcoming(
        self, competition: str, season: int | None = None
    ) -> pl.DataFrame:
        """Fetch scheduled (not yet played) matches for a competition."""
        params: dict[str, str] = {"status": "SCHEDULED"}
        if season:
            params["season"] = str(season)
        return await self._fetch_matches(competition, params, cache_ttl_minutes=30)

    async def get_completed(
        self, competition: str, season: int | None = None
    ) -> pl.DataFrame:
        """Fetch finished matches for a competition."""
        params: dict[str, str] = {"status": "FINISHED"}
        if season:
            params["season"] = str(season)
        # Cache completed results for longer — they don't change
        cache_key = self._cache.make_key(
            f"fd_matches_{competition}_finished",
            season=str(season or ""),
        )
        if self._cache.has(cache_key):
            return self._cache.get(cache_key)  # type: ignore[return-value]
        df = await self._fetch_matches(competition, params)
        if len(df) > 0:
            self._cache.set(cache_key, df)
        return df

    async def get_all_matches(
        self, competition: str, season: int | None = None
    ) -> pl.DataFrame:
        """Fetch all matches (scheduled + finished) for a competition."""
        params: dict[str, str] = {}
        if season:
            params["season"] = str(season)
        return await self._fetch_matches(competition, params)

    async def get_standings(self, competition: str, season: int | None = None) -> pl.DataFrame:
        """Fetch tournament group standings."""
        params: dict[str, str] = {}
        if season:
            params["season"] = str(season)
        data = await self._get(f"/competitions/{competition}/standings", params=params or None)
        return _parse_standings(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch_matches(
        self,
        competition: str,
        params: dict[str, str],
        cache_ttl_minutes: int = 0,
    ) -> pl.DataFrame:
        cache_key = self._cache.make_key(
            f"fd_matches_{competition}",
            **{k: str(v) for k, v in sorted(params.items())},
        )
        if cache_ttl_minutes == 0 and self._cache.has(cache_key):
            return self._cache.get(cache_key)  # type: ignore[return-value]

        data = await self._get(f"/competitions/{competition}/matches", params=params or None)
        matches = data.get("matches", [])
        df = _parse_matches(matches, competition)
        if len(df) > 0 and cache_ttl_minutes == 0:
            self._cache.set(cache_key, df)
        return df


# ------------------------------------------------------------------
# Parser helpers
# ------------------------------------------------------------------


def _parse_matches(matches: list[dict], competition: str) -> pl.DataFrame:  # type: ignore[type-arg]
    rows = []
    for m in matches:
        game_id = f"fd_{competition}_{m.get('id', '')}"
        utc_date = str(m.get("utcDate", ""))
        match_date = _parse_date(utc_date)
        if match_date is None:
            continue

        home_team = m.get("homeTeam", {}) or {}
        away_team = m.get("awayTeam", {}) or {}
        score = m.get("score", {}) or {}
        full_time = score.get("fullTime", {}) or {}
        status = str(m.get("status", ""))
        winner = str(score.get("winner") or "")

        home_goals: int | None = None
        away_goals: int | None = None
        result: int | None = None

        if status == "FINISHED":
            hg = full_time.get("home")
            ag = full_time.get("away")
            if hg is not None and ag is not None:
                home_goals = int(hg)
                away_goals = int(ag)
                result = _result_from_goals(home_goals, away_goals)

        stage = str(m.get("stage", "") or "")
        group = str(m.get("group", "") or "")
        matchday = m.get("matchday")

        # Determine venue_type: neutral for all WC matches except host nation games
        venue_type = _infer_venue_type(
            home_id=str(home_team.get("id", "")),
            away_id=str(away_team.get("id", "")),
            competition=competition,
            stage=stage,
        )

        rows.append(
            {
                "game_id": game_id,
                "competition_id": competition,
                "season": match_date.year if match_date.month >= 6 else match_date.year - 1,
                "matchday": int(matchday) if matchday is not None else None,
                "date": match_date,
                "home_club_id": f"fd_{home_team.get('id', '')}",
                "away_club_id": f"fd_{away_team.get('id', '')}",
                "home_club_name": str(home_team.get("name", "") or ""),
                "away_club_name": str(away_team.get("name", "") or ""),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "venue_type": venue_type,
                "group_id": group if group else None,
                "stage": stage if stage else None,
                "status": status,
            }
        )

    if not rows:
        return pl.DataFrame(
            {
                "game_id": [], "competition_id": [], "season": [], "matchday": [],
                "date": [], "home_club_id": [], "away_club_id": [],
                "home_club_name": [], "away_club_name": [],
                "home_goals": [], "away_goals": [], "result": [],
                "venue_type": [], "group_id": [], "stage": [], "status": [],
            }
        )

    return pl.DataFrame(rows)


def _parse_standings(data: dict) -> pl.DataFrame:  # type: ignore[type-arg]
    rows = []
    for standing in data.get("standings", []):
        group = str(standing.get("group", "") or "")
        stage = str(standing.get("stage", "") or "")
        for entry in standing.get("table", []):
            team = entry.get("team", {}) or {}
            rows.append(
                {
                    "group_id": group,
                    "stage": stage,
                    "position": int(entry.get("position", 0)),
                    "team_id": f"fd_{team.get('id', '')}",
                    "team_name": str(team.get("name", "") or ""),
                    "played": int(entry.get("playedGames", 0)),
                    "won": int(entry.get("won", 0)),
                    "draw": int(entry.get("draw", 0)),
                    "lost": int(entry.get("lost", 0)),
                    "goals_for": int(entry.get("goalsFor", 0)),
                    "goals_against": int(entry.get("goalsAgainst", 0)),
                    "goal_difference": int(entry.get("goalDifference", 0)),
                    "points": int(entry.get("points", 0)),
                }
            )
    if not rows:
        return pl.DataFrame(
            {
                "group_id": [], "stage": [], "position": [], "team_id": [],
                "team_name": [], "played": [], "won": [], "draw": [], "lost": [],
                "goals_for": [], "goals_against": [], "goal_difference": [], "points": [],
            }
        )
    return pl.DataFrame(rows)


def _parse_date(utc_date: str) -> date | None:
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return datetime.strptime(utc_date[:19], fmt[:len(fmt.rstrip("%z"))]).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _result_from_goals(home: int, away: int) -> int:
    if home > away:
        return 2
    elif home == away:
        return 1
    return 0


# Host nation IDs for WC 2026 (USA, Canada, Mexico) in football-data.org format
_WC2026_HOST_IDS = {"fd_771", "fd_768", "fd_769"}  # approximate — update after checking API


def _infer_venue_type(home_id: str, away_id: str, competition: str, stage: str) -> str:
    """Default to neutral for all World Cup matches.

    Override to home_advantage_a/b only when a known host nation is playing at home.
    """
    if competition in ("WC", "EC", "CLI"):
        if home_id in _WC2026_HOST_IDS:
            return VENUE_HOME_A
        if away_id in _WC2026_HOST_IDS:
            return VENUE_HOME_B
        return VENUE_NEUTRAL
    return VENUE_HOME_A
