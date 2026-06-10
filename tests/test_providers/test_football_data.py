"""Tests for the football-data.org provider.

Uses respx to mock HTTP without needing a real API key.
"""
from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from footyml.providers.football_data import (
    FootballDataProvider,
    _infer_venue_type,
    _parse_date,
    _result_from_goals,
    _parse_matches,
)
from footyml.config import VENUE_NEUTRAL, VENUE_HOME_A


# ------------------------------------------------------------------
# Unit tests (no HTTP)
# ------------------------------------------------------------------


def test_parse_date_utc() -> None:
    d = _parse_date("2026-06-14T17:00:00Z")
    assert d == date(2026, 6, 14)


def test_parse_date_plain() -> None:
    d = _parse_date("2026-06-14")
    assert d == date(2026, 6, 14)


def test_parse_date_invalid() -> None:
    assert _parse_date("not-a-date") is None


def test_result_from_goals_home_win() -> None:
    assert _result_from_goals(3, 1) == 2


def test_result_from_goals_draw() -> None:
    assert _result_from_goals(1, 1) == 1


def test_result_from_goals_away_win() -> None:
    assert _result_from_goals(0, 2) == 0


def test_infer_venue_type_wc_neutral() -> None:
    assert _infer_venue_type("fd_100", "fd_200", "WC", "GROUP_STAGE") == VENUE_NEUTRAL


def test_infer_venue_type_league_home() -> None:
    assert _infer_venue_type("fd_100", "fd_200", "BL1", "REGULAR_SEASON") == VENUE_HOME_A


def test_parse_matches_scheduled() -> None:
    raw = [
        {
            "id": 1,
            "utcDate": "2026-06-14T17:00:00Z",
            "status": "SCHEDULED",
            "matchday": 1,
            "stage": "GROUP_STAGE",
            "group": "Group A",
            "homeTeam": {"id": 100, "name": "Germany"},
            "awayTeam": {"id": 200, "name": "Scotland"},
            "score": {"winner": None, "fullTime": {"home": None, "away": None}},
        }
    ]
    df = _parse_matches(raw, "WC")
    assert len(df) == 1
    assert df["result"][0] is None
    assert df["venue_type"][0] == VENUE_NEUTRAL
    assert df["group_id"][0] == "Group A"
    assert df["home_club_name"][0] == "Germany"


def test_parse_matches_finished() -> None:
    raw = [
        {
            "id": 2,
            "utcDate": "2026-06-14T17:00:00Z",
            "status": "FINISHED",
            "matchday": 1,
            "stage": "GROUP_STAGE",
            "group": "Group A",
            "homeTeam": {"id": 100, "name": "Germany"},
            "awayTeam": {"id": 200, "name": "Scotland"},
            "score": {"winner": "HOME_TEAM", "fullTime": {"home": 2, "away": 0}},
        }
    ]
    df = _parse_matches(raw, "WC")
    assert df["result"][0] == 2
    assert df["home_goals"][0] == 2
    assert df["away_goals"][0] == 0


def test_parse_matches_empty() -> None:
    df = _parse_matches([], "WC")
    assert len(df) == 0
    assert "game_id" in df.columns


# ------------------------------------------------------------------
# HTTP mock tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_upcoming_returns_dataframe() -> None:
    mock_response = {
        "matches": [
            {
                "id": 42,
                "utcDate": "2026-06-15T18:00:00Z",
                "status": "SCHEDULED",
                "matchday": 2,
                "stage": "GROUP_STAGE",
                "group": "Group B",
                "homeTeam": {"id": 300, "name": "Brazil"},
                "awayTeam": {"id": 400, "name": "Argentina"},
                "score": {"winner": None, "fullTime": {"home": None, "away": None}},
            }
        ]
    }
    with respx.mock:
        respx.get("https://api.football-data.org/v4/competitions/WC/matches").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        async with FootballDataProvider(api_key="test_key") as p:
            df = await p.get_upcoming("WC")

    assert len(df) == 1
    assert df["home_club_name"][0] == "Brazil"
    assert df["result"][0] is None


@pytest.mark.asyncio
async def test_list_competitions() -> None:
    mock_response = {
        "competitions": [
            {"id": 2000, "code": "WC", "name": "FIFA World Cup", "area": {"name": "World"}},
            {"id": 2001, "code": "CL", "name": "UEFA Champions League", "area": {"name": "Europe"}},
        ]
    }
    with respx.mock:
        respx.get("https://api.football-data.org/v4/competitions").mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        async with FootballDataProvider(api_key="test_key") as p:
            df = await p.list_competitions()

    assert len(df) == 2
    assert "WC" in df["code"].to_list()


@pytest.mark.asyncio
async def test_missing_api_key_raises() -> None:
    import os
    original = os.environ.pop("FOOTBALL_DATA_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="FOOTBALL_DATA_API_KEY"):
            FootballDataProvider(api_key="")
    finally:
        if original:
            os.environ["FOOTBALL_DATA_API_KEY"] = original
