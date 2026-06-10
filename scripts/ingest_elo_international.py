"""Build national-team Elo ratings from the martj42 international results dataset.

Downloads ~49,000 match results (1872-present) from GitHub and computes
chronological Elo ratings for every national team.  Competitive matches are
weighted twice as heavily as friendlies.  Results are stored in the
elo_ratings table with club_id = team name (e.g. "Germany", "France").

The tournament predictor uses these ratings as a fallback when no
features are available for a match.

Usage:
    python scripts/ingest_elo_international.py
    python scripts/ingest_elo_international.py --from-year 1990  # filter older matches
    python scripts/ingest_elo_international.py --no-friendlies    # competitive only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
from datetime import date, datetime

import polars as pl
import typer
from rich.console import Console

from footyml.config import ELO_INITIAL, ELO_K_FACTOR, HOME_ADVANTAGE_ELO
from footyml.data.store import DataStore

console = Console()
app = typer.Typer(add_completion=False)

DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# Tournaments to include (all others are treated as low-weight friendlies)
COMPETITIVE = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Copa América",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "CONCACAF Gold Cup",
    "CONCACAF Nations League",
    "Confederations Cup",
    "Olympic Games",
}

# Name aliases: martj42 name → football-data.org / our canonical name
NAME_MAP: dict[str, str] = {
    "USA":                          "United States",
    "Korea Republic":               "South Korea",
    "IR Iran":                      "Iran",
    "Côte d'Ivoire":                "Ivory Coast",
    "Cape Verde":                   "Cape Verde Islands",
    "Bosnia and Herzegovina":       "Bosnia-Herzegovina",
    "DR Congo":                     "Congo DR",
    "Trinidad and Tobago":          "Trinidad & Tobago",
    "China PR":                     "China",
    "Kyrgyz Republic":              "Kyrgyzstan",
    "São Tomé and Príncipe":        "São Tomé & Príncipe",
    "St. Kitts and Nevis":         "St Kitts & Nevis",
    "Antigua and Barbuda":          "Antigua & Barbuda",
    "Trinidad & Tobago":            "Trinidad & Tobago",
    "Northern Ireland":             "Northern Ireland",
}


def _normalise(name: str) -> str:
    return NAME_MAP.get(name, name)


def _expected(elo_a: float, elo_b: float, home: bool) -> float:
    bonus = HOME_ADVANTAGE_ELO if home else 0.0
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a - bonus) / 400.0))


def _k(tournament: str) -> float:
    """K-factor: doubled for competitive matches."""
    return ELO_K_FACTOR * 2 if tournament in COMPETITIVE else ELO_K_FACTOR


@app.command()
def main(
    from_year: int = typer.Option(1980, "--from-year", help="Only use matches from this year onward for Elo building (older results decay anyway)"),
    no_friendlies: bool = typer.Option(False, "--no-friendlies", help="Exclude friendlies entirely"),
    save_history: bool = typer.Option(False, "--save-history", help="Store one Elo row per match (large); default stores only the latest per team"),
) -> None:
    console.rule("[bold]Ingesting international Elo ratings from martj42 dataset")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    console.print(f"Downloading: {DATASET_URL}")
    import httpx
    resp = httpx.get(DATASET_URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    console.print(f"Downloaded {len(resp.content):,} bytes")

    df = pl.read_csv(resp.content, null_values=["NA", ""], infer_schema_length=0)
    console.print(f"Total rows: {len(df):,}  columns: {df.columns}")

    # ------------------------------------------------------------------
    # Filter & normalise
    # ------------------------------------------------------------------
    df = df.with_columns([
        pl.col("date").cast(pl.Date),
        pl.col("home_team").map_elements(_normalise, return_dtype=pl.String),
        pl.col("away_team").map_elements(_normalise, return_dtype=pl.String),
    ])

    df = df.filter(pl.col("date").dt.year() >= from_year)

    if no_friendlies:
        df = df.filter(pl.col("tournament").is_in(list(COMPETITIVE)))

    df = df.sort("date")
    console.print(f"After filter (from {from_year}, {'competitive only' if no_friendlies else 'all'}): {len(df):,} matches")

    # ------------------------------------------------------------------
    # Build Elo chronologically
    # ------------------------------------------------------------------
    elo: dict[str, float] = {}
    rows: list[dict] = []

    for row in df.iter_rows(named=True):
        home = str(row["home_team"])
        away = str(row["away_team"])
        hs   = int(row["home_score"] or 0) if row["home_score"] is not None else None
        as_  = int(row["away_score"] or 0) if row["away_score"] is not None else None
        if hs is None or as_ is None:
            continue
        dt   = row["date"]
        neutral = str(row["neutral"]).upper() == "TRUE"
        tournament = str(row["tournament"])

        eh = elo.get(home, ELO_INITIAL)
        ea = elo.get(away, ELO_INITIAL)

        if save_history:
            rows.append({"club_id": home, "date": dt, "elo": eh})
            rows.append({"club_id": away, "date": dt, "elo": ea})

        # Actual result from home perspective
        if hs > as_:
            actual = 1.0
        elif hs < as_:
            actual = 0.0
        else:
            actual = 0.5

        home_is_home = not neutral
        exp = _expected(eh, ea, home_is_home)
        k = _k(tournament)

        elo[home] = eh + k * (actual - exp)
        elo[away] = ea + k * (1.0 - actual - (1.0 - exp))

    console.print(f"Elo built for {len(elo):,} teams")

    # ------------------------------------------------------------------
    # Store latest Elo per team
    # ------------------------------------------------------------------
    today = date.today()
    if not save_history:
        rows = [{"club_id": name, "date": today, "elo": float(e)} for name, e in elo.items()]

    elo_df = pl.DataFrame(rows, schema={"club_id": pl.String, "date": pl.Date, "elo": pl.Float64})

    store = DataStore()
    store.upsert_elo(elo_df)

    # Print top 20
    top = sorted(elo.items(), key=lambda x: -x[1])
    console.rule("Top 20 national teams by Elo")
    for rank, (team, e) in enumerate(top[:20], 1):
        console.print(f"  {rank:2d}. {team:30s} {e:.0f}")

    console.print(f"\n[green]Stored {len(elo_df):,} Elo records in elo_ratings table[/green]")


if __name__ == "__main__":
    app()
