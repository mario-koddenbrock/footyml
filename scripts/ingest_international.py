"""Ingest historical international tournament matches (WC, Euro) to build national team Elo/form.

Usage:
    python scripts/ingest_international.py
    python scripts/ingest_international.py --competition WC --seasons 2018 2022
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console

from footyml.config import INTERNATIONAL_HISTORY
from footyml.data.store import DataStore
from footyml.providers.football_data import FootballDataProvider

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    competition: str = typer.Option(
        None,
        help="Single competition code (e.g. WC, EC). Omit to run all from config.",
    ),
    seasons: list[int] = typer.Option(
        None,
        help="Seasons to ingest. Omit to use defaults from config.",
    ),
) -> None:
    asyncio.run(_run(competition, list(seasons) if seasons else None))


async def _run(competition: str | None, seasons: list[int] | None) -> None:
    store = DataStore()
    total = 0

    to_ingest: dict[str, list[int]]
    if competition:
        to_ingest = {competition: seasons or INTERNATIONAL_HISTORY.get(competition, [])}
    else:
        to_ingest = INTERNATIONAL_HISTORY

    async with FootballDataProvider() as provider:
        for comp, season_list in to_ingest.items():
            for season in season_list:
                console.print(f"Fetching {comp} {season}...")
                try:
                    matches = await provider.get_completed(comp, season=season)
                except Exception as e:
                    console.print(f"[red]Error fetching {comp} {season}: {e}[/red]")
                    continue

                if len(matches) == 0:
                    console.print(f"[yellow]{comp} {season}: no completed matches[/yellow]")
                    continue

                store_cols = [
                    "game_id", "competition_id", "season", "matchday", "date",
                    "home_club_id", "away_club_id", "home_goals", "away_goals",
                    "result", "venue_type", "group_id", "stage",
                    "home_club_name", "away_club_name",
                ]
                available = [c for c in store_cols if c in matches.columns]
                import polars as pl
                to_store = matches.select(available)
                if "date" in to_store.columns and to_store["date"].dtype != pl.Date:
                    to_store = to_store.with_columns(pl.col("date").cast(pl.Date))

                store.upsert_matches(to_store)
                total += len(to_store)
                console.print(f"[green]{comp} {season}: stored {len(to_store)} matches[/green]")

    console.print(f"[bold green]Done. {total} total matches ingested.[/bold green]")
    console.print("Next — rebuild features for international matches:")
    console.print('  python -c "from footyml.features import FeaturePipeline; FeaturePipeline().build(competition_id=\'WC\')"')


if __name__ == "__main__":
    app()
