"""Fetch historical league match results from football-data.co.uk.

Free CSV downloads — no API key required.
URL pattern: https://www.football-data.co.uk/mmz4281/{YY}{YY+1}/{code}.csv
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

import httpx
import polars as pl

from footyml.config import LEAGUE_IDS, RAW_DIR

# football-data.co.uk league codes
_FD_CODES: dict[str, str] = {
    "bundesliga": "D1",
    "premier_league": "E0",
    "la_liga": "SP1",
    "serie_a": "I1",
    "ligue_1": "F1",
    "bundesliga2": "D2",
}

_BASE_URL = "https://www.football-data.co.uk/mmz4281"


class FootballDataCsvFetcher:
    """Download and parse football-data.co.uk CSV files into our match schema."""

    def __init__(self, cache_dir: Path = RAW_DIR / "fd_csv"):
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_season(
        self, league: str, season: int, name_to_id: dict[str, str] | None = None
    ) -> pl.DataFrame:
        """Return a DataFrame in the matches schema for the given league/season.

        Args:
            league: league key (e.g. "bundesliga")
            season: start year of the season (e.g. 2023 for 2023/24)
            name_to_id: optional mapping of team display name → club_id.
                        If provided, club IDs in the output will use those values.
                        Otherwise club IDs are "{competition_id}_{normalised_name}".
        """
        code = _FD_CODES.get(league)
        if code is None:
            raise ValueError(f"No football-data.co.uk code for league '{league}'")

        competition_id = LEAGUE_IDS[league]
        season_str = f"{str(season)[2:]}{str(season + 1)[2:]}"
        url = f"{_BASE_URL}/{season_str}/{code}.csv"

        cache_path = self._cache_dir / f"{league}_{season}.parquet"
        if cache_path.exists():
            return pl.read_parquet(cache_path)

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            csv_text = response.text

        df = _parse_csv(csv_text, competition_id, season, name_to_id)
        if len(df) > 0:
            df.write_parquet(cache_path)
        return df

    async def fetch_club_name_mapping(
        self, tm_client: object, competition_id: str, season_id: str
    ) -> dict[str, str]:
        """Fetch Transfermarkt club list and return {normalised_name: tm_club_id}."""
        from footyml.ingestion.fetchers import TransfermarktFetcher
        from footyml.ingestion.cache import ParquetCache

        fetcher = TransfermarktFetcher(tm_client, ParquetCache())  # type: ignore[arg-type]
        clubs_df = await fetcher.fetch_competition_clubs(competition_id, season_id)
        mapping: dict[str, str] = {}
        for row in clubs_df.iter_rows(named=True):
            normalised = _normalise(str(row["club_name"]))
            mapping[normalised] = str(row["club_id"])
        return mapping


def _parse_csv(
    csv_text: str,
    competition_id: str,
    season: int,
    name_to_id: dict[str, str] | None,
) -> pl.DataFrame:
    try:
        raw = pl.read_csv(io.StringIO(csv_text), infer_schema_length=0, ignore_errors=True)
    except Exception:
        return _empty_df()

    required = {"HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG", "FTR"}
    if not required.issubset(set(raw.columns)):
        return _empty_df()

    rows = []
    for row in raw.iter_rows(named=True):
        home_name = str(row.get("HomeTeam") or "").strip()
        away_name = str(row.get("AwayTeam") or "").strip()
        date_str = str(row.get("Date") or "").strip()
        fthg = row.get("FTHG")
        ftag = row.get("FTAG")
        ftr = str(row.get("FTR") or "").strip()

        if not home_name or not away_name or not date_str or ftr not in ("H", "D", "A"):
            continue

        match_date = _parse_date(date_str)
        if match_date is None:
            continue

        try:
            home_goals = int(fthg) if fthg is not None else None
            away_goals = int(ftag) if ftag is not None else None
        except (ValueError, TypeError):
            home_goals = away_goals = None

        result = {"H": 2, "D": 1, "A": 0}.get(ftr)

        home_id = _resolve_id(home_name, competition_id, name_to_id)
        away_id = _resolve_id(away_name, competition_id, name_to_id)

        game_id = _make_game_id(competition_id, season, home_id, away_id, match_date.isoformat())
        matchday = row.get("Wk") or row.get("Round") or None
        try:
            matchday_int = int(str(matchday).replace("Round ", "").strip()) if matchday else None
        except (ValueError, TypeError):
            matchday_int = None

        rows.append({
            "game_id": game_id,
            "competition_id": competition_id,
            "season": season,
            "matchday": matchday_int,
            "date": match_date.isoformat(),
            "home_club_id": home_id,
            "away_club_id": away_id,
            "home_club_name": home_name,
            "away_club_name": away_name,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "venue_type": "home_advantage_a",
            "group_id": None,
            "stage": None,
        })

    if not rows:
        return _empty_df()

    df = pl.DataFrame(rows)
    # Cast date and numeric columns
    df = df.with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    for col in ("home_goals", "away_goals", "result", "season", "matchday"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int32, strict=False))
    return df


def _resolve_id(name: str, competition_id: str, name_to_id: dict[str, str] | None) -> str:
    if name_to_id:
        # Try exact normalised match first
        key = _normalise(name)
        if key in name_to_id:
            return name_to_id[key]
        # Fuzzy fallback: find closest match
        import difflib
        close = difflib.get_close_matches(key, name_to_id.keys(), n=1, cutoff=0.6)
        if close:
            return name_to_id[close[0]]
    # Fallback: stable string ID from competition + normalised name
    return f"{competition_id}_{_normalise(name)}"


def _normalise(name: str) -> str:
    """Lowercase, strip common club prefixes/suffixes for fuzzy matching."""
    import re
    s = name.lower().strip()
    for word in ("fc", "cf", "sc", "ac", "as", "ss", "sv", "bv", "vfl", "vfb",
                 "tsv", "fsv", "rb", "rsca", "rcd", "rc", "us", "ud", "sd",
                 "deportivo", "atletico", "athletic", "real", "sporting"):
        s = re.sub(rf"\b{word}\b", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_date(s: str) -> datetime.date | None:  # type: ignore[name-defined]
    from datetime import date
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _make_game_id(competition_id: str, season: int, home: str, away: str, date: str) -> str:
    raw = f"{competition_id}_{season}_{home}_{away}_{date}"
    return f"fd_csv_{hashlib.sha1(raw.encode()).hexdigest()[:12]}"


def _empty_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": pl.Series([], dtype=pl.String),
        "competition_id": pl.Series([], dtype=pl.String),
        "season": pl.Series([], dtype=pl.Int32),
        "matchday": pl.Series([], dtype=pl.Int32),
        "date": pl.Series([], dtype=pl.Date),
        "home_club_id": pl.Series([], dtype=pl.String),
        "away_club_id": pl.Series([], dtype=pl.String),
        "home_club_name": pl.Series([], dtype=pl.String),
        "away_club_name": pl.Series([], dtype=pl.String),
        "home_goals": pl.Series([], dtype=pl.Int32),
        "away_goals": pl.Series([], dtype=pl.Int32),
        "result": pl.Series([], dtype=pl.Int32),
        "venue_type": pl.Series([], dtype=pl.String),
        "group_id": pl.Series([], dtype=pl.String),
        "stage": pl.Series([], dtype=pl.String),
    })
