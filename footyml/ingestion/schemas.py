from datetime import date
from pydantic import BaseModel, field_validator


class ClubProfile(BaseModel):
    id: str
    name: str
    squad_size: int | None = None
    average_age: float | None = None
    foreigners_number: int | None = None
    foreigners_percentage: str | None = None
    total_market_value: str | None = None


class PlayerBasic(BaseModel):
    id: str
    name: str
    position: str | None = None
    date_of_birth: str | None = None
    nationality: list[str] | None = None
    market_value: str | None = None
    club_id: str | None = None


class MarketValueEntry(BaseModel):
    date: str
    market_value: str | None = None


class PlayerMarketValue(BaseModel):
    player_id: str = ""
    current_value: str | None = None
    history: list[MarketValueEntry] = []


class TransferEntry(BaseModel):
    player_id: str = ""
    player_name: str | None = None
    from_club_id: str | None = None
    from_club_name: str | None = None
    to_club_id: str | None = None
    to_club_name: str | None = None
    fee: str | None = None
    season: str | None = None
    transfer_date: str | None = None

    @field_validator("fee", mode="before")
    @classmethod
    def coerce_fee(cls, v: object) -> str | None:
        if v is None:
            return None
        return str(v)


class GameResult(BaseModel):
    id: str
    home_club_id: str | None = None
    away_club_id: str | None = None
    home_club_name: str | None = None
    away_club_name: str | None = None
    home_club_goals: int | None = None
    away_club_goals: int | None = None
    date: str | None = None
    matchday: int | None = None
    competition_id: str | None = None


class CompetitionClub(BaseModel):
    id: str
    name: str
    competition_id: str | None = None
    season: str | None = None
