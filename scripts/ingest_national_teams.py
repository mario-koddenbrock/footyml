"""Fetch national team squad market values from Transfermarkt.

Looks up each WC/international team's name in the matches table, maps it to
a Transfermarkt club ID via config.NATIONAL_TEAM_TM_IDS, then fetches and
stores squad values under the team's fd_ ID so MarketValueBuilder can find them.

Usage:
    python scripts/ingest_national_teams.py
    python scripts/ingest_national_teams.py --season 2025 --competition WC
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

from footyml.config import NATIONAL_TEAM_TM_IDS
from footyml.data.store import DataStore
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.fetchers import TransfermarktFetcher

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    season: int = typer.Option(2025, help="Most recent completed league season (for current squad values)"),
    competition: str = typer.Option("WC", help="Competition code to look up teams from"),
) -> None:
    asyncio.run(_run(season, competition))


async def _run(season: int, competition: str) -> None:
    store = DataStore()
    season_id = str(season)

    # Discover all national teams in the DB for this competition
    try:
        teams_df = store.query(f"""
            SELECT DISTINCT home_club_id as fd_id, home_club_name as name FROM matches
            WHERE competition_id = '{competition}' AND home_club_name IS NOT NULL AND home_club_name != ''
            UNION
            SELECT DISTINCT away_club_id, away_club_name FROM matches
            WHERE competition_id = '{competition}' AND away_club_name IS NOT NULL AND away_club_name != ''
        """)
    except Exception as e:
        console.print(f"[red]Failed to query teams: {e}[/red]")
        return

    if len(teams_df) == 0:
        console.print(f"[yellow]No teams found for {competition}. Run ingest_international.py first.[/yellow]")
        return

    console.print(f"Found {len(teams_df)} teams in {competition}")

    results_table = Table(title=f"National Team Squads (season {season})")
    results_table.add_column("Team")
    results_table.add_column("fd_id")
    results_table.add_column("TM ID")
    results_table.add_column("Players")

    cache = ParquetCache()
    async with TransfermarktClient() as client:
        fetcher = TransfermarktFetcher(client, cache)
        total_fetched = 0

        for row in teams_df.iter_rows(named=True):
            fd_id = str(row["fd_id"])
            name = str(row["name"])

            tm_id = NATIONAL_TEAM_TM_IDS.get(name)
            if tm_id is None:
                results_table.add_row(name, fd_id, "[yellow]no mapping[/yellow]", "—")
                continue

            try:
                squad_df = await fetcher.fetch_squad_market_values(tm_id, season_id)
            except Exception as e:
                results_table.add_row(name, fd_id, tm_id, f"[red]error: {e}[/red]")
                continue

            if len(squad_df) == 0:
                results_table.add_row(name, fd_id, tm_id, "[yellow]0[/yellow]")
                continue

            # Store under the fd_ club ID so MarketValueBuilder finds it by fd_id
            squad_df = squad_df.with_columns(pl.lit(fd_id).alias("club_id"))
            store.upsert_squad(squad_df)
            total_fetched += len(squad_df)
            results_table.add_row(name, fd_id, tm_id, str(len(squad_df)))

    console.print(results_table)
    console.print(f"[bold green]Done. {total_fetched} player market values stored.[/bold green]")


if __name__ == "__main__":
    app()
