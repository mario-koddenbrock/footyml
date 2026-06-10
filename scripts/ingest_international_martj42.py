"""Ingest historical international competitive matches from martj42 dataset.

Downloads ~8,000 competitive matches (WC, WC Qual, Euro, Copa, AFCON, AFC Asian Cup,
CONCACAF Gold Cup, UEFA Nations League) since 2000 and stores them in the DuckDB
matches table so that FeaturePipeline.build() can compute features for training
a national-team TabPFN model.

Team names are used directly as club IDs (e.g., "Germany", "Spain").
MarketValueBuilder will look them up via the entries stored by store_national_team_mv.py
under team-name club IDs.

Usage:
    python scripts/ingest_international_martj42.py
    python scripts/ingest_international_martj42.py --from-year 2010
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
from datetime import date

import polars as pl
import typer
from rich.console import Console

from footyml.data.store import DataStore

console = Console()
app = typer.Typer(add_completion=False)

DATASET_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

TOURNAMENT_TO_COMP: dict[str, str] = {
    "FIFA World Cup":                  "WC",
    "FIFA World Cup qualification":    "WCQ",
    "UEFA Euro":                       "EC",
    "UEFA Euro qualification":         "ECQ",
    "UEFA Nations League":             "UNL",
    "Copa América":                    "CLI",
    "African Cup of Nations":          "CAN",
    "African Cup of Nations qualification": "CANQ",
    "AFC Asian Cup":                   "AAC",
    "AFC Asian Cup qualification":     "AACQ",
    "CONCACAF Gold Cup":               "CGC",
    "CONCACAF Nations League":         "CNL",
    "Confederations Cup":              "CC",
    "Olympic Games":                   "OG",
}

MAJOR_COMPS = set(TOURNAMENT_TO_COMP.keys())

# Name normalisation (same as ingest_elo_international.py)
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
    "Northern Ireland":             "Northern Ireland",
}


def _norm(name: str) -> str:
    return NAME_MAP.get(name, name)


def _game_id(date_str: str, home: str, away: str) -> str:
    key = f"nt_{date_str}_{home}_{away}"
    return "nt_" + hashlib.md5(key.encode()).hexdigest()[:12]


@app.command()
def main(
    from_year: int = typer.Option(2000, "--from-year", help="Only include matches from this year onward"),
    competition: str = typer.Option(None, "--competition", help="Filter to single competition code (e.g. WC)"),
) -> None:
    import httpx
    console.print(f"Downloading: {DATASET_URL}")
    resp = httpx.get(DATASET_URL, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    console.print(f"Downloaded {len(resp.content):,} bytes")

    raw = pl.read_csv(resp.content, null_values=["NA", ""], infer_schema_length=0)
    console.print(f"Total rows: {len(raw):,}")

    # Filter to major competitions and year range
    df = raw.filter(pl.col("tournament").is_in(list(MAJOR_COMPS)))
    df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.filter(pl.col("date").dt.year() >= from_year)

    if competition:
        # Map competition code back to tournament names
        rev = {v: k for k, v in TOURNAMENT_TO_COMP.items()}
        name = rev.get(competition.upper())
        if name:
            df = df.filter(pl.col("tournament") == name)
        else:
            console.print(f"[red]Unknown competition: {competition}[/red]")
            raise typer.Exit(1)

    df = df.sort("date")
    console.print(f"Matches after filter: {len(df):,}")

    store = DataStore()
    rows = []
    skipped = 0

    for row in df.iter_rows(named=True):
        hs = row.get("home_score")
        as_ = row.get("away_score")
        if hs is None or as_ is None:
            skipped += 1
            continue

        try:
            home_goals = int(hs)
            away_goals = int(as_)
        except (ValueError, TypeError):
            skipped += 1
            continue

        home = _norm(str(row["home_team"]))
        away = _norm(str(row["away_team"]))
        tournament = str(row["tournament"])
        comp_id = TOURNAMENT_TO_COMP.get(tournament, "INTL")
        neutral = str(row.get("neutral", "FALSE")).upper() == "TRUE"
        venue_type = "neutral" if neutral else "home_advantage_a"

        if home_goals > away_goals:
            result = 2
        elif home_goals < away_goals:
            result = 0
        else:
            result = 1

        dt = row["date"]
        rows.append({
            "game_id":        _game_id(str(dt), home, away),
            "competition_id": comp_id,
            "season":         dt.year,
            "matchday":       None,
            "date":           dt,
            "home_club_id":   home,
            "away_club_id":   away,
            "home_goals":     home_goals,
            "away_goals":     away_goals,
            "result":         result,
            "venue_type":     venue_type,
            "group_id":       None,
            "stage":          tournament,
            "home_club_name": home,
            "away_club_name": away,
        })

    if skipped:
        console.print(f"[yellow]Skipped {skipped} rows with missing scores[/yellow]")

    if not rows:
        console.print("[red]No rows to store[/red]")
        raise typer.Exit(1)

    schema = {
        "game_id":        pl.String,
        "competition_id": pl.String,
        "season":         pl.Int32,
        "matchday":       pl.Int32,
        "date":           pl.Date,
        "home_club_id":   pl.String,
        "away_club_id":   pl.String,
        "home_goals":     pl.Int32,
        "away_goals":     pl.Int32,
        "result":         pl.Int32,
        "venue_type":     pl.String,
        "group_id":       pl.String,
        "stage":          pl.String,
        "home_club_name": pl.String,
        "away_club_name": pl.String,
    }
    matches_df = pl.DataFrame(rows, schema=schema)
    store.upsert_matches(matches_df)

    # Summary
    summary = matches_df.group_by("competition_id").agg(pl.len().alias("n")).sort("n", descending=True)
    console.print(summary)
    console.print(f"[bold green]Stored {len(matches_df):,} international matches[/bold green]")


if __name__ == "__main__":
    app()
