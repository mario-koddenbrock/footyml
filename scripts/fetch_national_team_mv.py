"""Fetch national team total squad market values from Transfermarkt.

For each WC 2026 team that has a TM ID mapping, fetches the club profile
to get total squad market value. Teams without a mapping fall back to
hardcoded estimates based on publicly available data.

Saves results to data/processed/national_team_mv.parquet with columns:
    team (str), total_market_value_eur (float)

Usage:
    python scripts/fetch_national_team_mv.py
    python scripts/fetch_national_team_mv.py --season 2024
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from footyml.config import NATIONAL_TEAM_TM_IDS, PROCESSED_DIR
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.fetchers import TransfermarktFetcher

console = Console()
app = typer.Typer(add_completion=False)

# Hardcoded squad value estimates (million EUR, approximate 2024-2025)
# for teams that either lack a TM mapping or where the API call fails.
# Source: Transfermarkt public data, rounded to nearest 5M.
MV_FALLBACK_EUR: dict[str, float] = {
    # Top tier
    "England":            1_450_000_000,
    "France":             1_150_000_000,
    "Spain":              1_200_000_000,
    "Brazil":               950_000_000,
    "Germany":              900_000_000,
    "Portugal":             900_000_000,
    "Argentina":            850_000_000,
    "Netherlands":          750_000_000,
    "Belgium":              550_000_000,
    "United States":        650_000_000,
    "Turkey":               500_000_000,
    "Norway":               400_000_000,
    "Switzerland":          380_000_000,
    "Austria":              380_000_000,  # WC 2026 Group J
    "Colombia":             350_000_000,
    "Canada":               300_000_000,
    "Mexico":               280_000_000,
    "Croatia":              280_000_000,
    "Senegal":              280_000_000,
    "Scotland":             250_000_000,
    "Sweden":               220_000_000,
    "Czechia":              220_000_000,
    "Czech Republic":       220_000_000,
    "Japan":                200_000_000,
    "Morocco":              180_000_000,
    "South Korea":          180_000_000,
    "Australia":            160_000_000,
    "Ivory Coast":          130_000_000,
    "Bosnia-Herzegovina":   110_000_000,
    "Ecuador":              100_000_000,
    "Algeria":               90_000_000,
    "Egypt":                 90_000_000,
    "Ghana":                 80_000_000,
    "Saudi Arabia":          70_000_000,
    "Uruguay":              250_000_000,
    "Paraguay":              60_000_000,
    "Tunisia":               60_000_000,
    "Iran":                  50_000_000,
    "Congo DR":              45_000_000,
    "Qatar":                 50_000_000,
    "South Africa":          40_000_000,
    "Iraq":                  40_000_000,
    "Uzbekistan":            25_000_000,
    "Cape Verde Islands":    25_000_000,
    "New Zealand":           20_000_000,
    "Jordan":                20_000_000,
    "Panama":                25_000_000,
    "Curaçao":               15_000_000,
    "Haiti":                 15_000_000,
}

# All 48 WC 2026 teams (canonical football-data.org names)
WC2026_TEAMS: list[str] = [
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Bosnia-Herzegovina",
    "Brazil", "Canada", "Cape Verde Islands", "Colombia", "Congo DR",
    "Croatia", "Curaçao", "Czechia", "Ecuador", "Egypt", "England",
    "France", "Germany", "Ghana", "Haiti", "Iran", "Iraq", "Ivory Coast",
    "Japan", "Jordan", "Mexico", "Morocco", "Netherlands", "New Zealand",
    "Norway", "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
    "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
    "Uruguay", "Uzbekistan",
]


@app.command()
def main(
    season: int = typer.Option(2024, help="Season year for squad values"),
    skip_api: bool = typer.Option(False, "--skip-api", help="Use only hardcoded fallback values"),
) -> None:
    asyncio.run(_run(season, skip_api))


async def _run(season: int, skip_api: bool) -> None:
    results: dict[str, float] = {}

    if not skip_api:
        cache = ParquetCache()
        season_id = str(season)

        async with TransfermarktClient() as client:
            fetcher = TransfermarktFetcher(client, cache)

            for team in WC2026_TEAMS:
                tm_id = NATIONAL_TEAM_TM_IDS.get(team)
                if tm_id is None:
                    continue

                try:
                    profile_df = await fetcher.fetch_club_profile(tm_id)
                    if len(profile_df) == 0:
                        continue
                    mv = profile_df["total_market_value_eur"][0]
                    if mv is not None and mv > 0:
                        results[team] = float(mv)
                        console.print(f"  [green]{team:30s}  €{mv/1e6:,.0f}M[/green]")
                    else:
                        console.print(f"  [yellow]{team:30s}  no value from API[/yellow]")
                except Exception as e:
                    console.print(f"  [red]{team:30s}  error: {e}[/red]")

    # Fill in fallbacks for any team not fetched from API
    for team in WC2026_TEAMS:
        if team not in results:
            fb = MV_FALLBACK_EUR.get(team)
            if fb:
                results[team] = float(fb)
                if skip_api:
                    console.print(f"  [blue]{team:30s}  €{fb/1e6:,.0f}M  (estimate)[/blue]")
                else:
                    console.print(f"  [yellow]{team:30s}  €{fb/1e6:,.0f}M  (fallback estimate)[/yellow]")
            else:
                # Unknown team — use €30M as floor
                results[team] = 30_000_000.0
                console.print(f"  [dim]{team:30s}  €30M  (floor)[/dim]")

    # Build DataFrame
    rows = [{"team": t, "total_market_value_eur": v} for t, v in results.items()]
    df = pl.DataFrame(rows, schema={"team": pl.String, "total_market_value_eur": pl.Float64})
    df = df.sort("total_market_value_eur", descending=True)

    out_path = PROCESSED_DIR / "national_team_mv.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    console.print(f"\n[bold green]Saved {len(df)} team market values → {out_path}[/bold green]")

    # Print table
    t = Table(title=f"National Team Squad Values (season {season})")
    t.add_column("Rank", style="dim", width=5)
    t.add_column("Team", width=28)
    t.add_column("Squad Value (€M)", justify="right")
    for i, row in enumerate(df.iter_rows(named=True), 1):
        mv_m = row["total_market_value_eur"] / 1e6
        t.add_row(str(i), row["team"], f"{mv_m:,.0f}")
    console.print(t)


if __name__ == "__main__":
    app()
