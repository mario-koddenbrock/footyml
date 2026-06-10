"""Ingest league match data and squad market values.

Match results come from football-data.co.uk (free historical CSVs, no API key).
Squad market values come from the Transfermarkt API (local or fly.dev instance).

Usage:
    python scripts/ingest.py --league bundesliga --seasons 2020 2021 2022 2023 2024
    python scripts/ingest.py --league bundesliga --season 2024
    python scripts/ingest.py --league bundesliga --seasons 2020 2021 --no-squad
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from footyml.config import LEAGUE_IDS
from footyml.data.store import DataStore
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.fetchers import TransfermarktFetcher
from footyml.ingestion.footballdata_csv import FootballDataCsvFetcher

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: str = typer.Option("bundesliga", help=f"League name. Options: {list(LEAGUE_IDS)}"),
    seasons: list[int] = typer.Option(None, help="Season start years (e.g. --seasons 2020 2021)"),
    season: int | None = typer.Option(None, help="Single season (shorthand)"),
    also_squad: bool = typer.Option(True, "--squad/--no-squad", help="Also fetch squad market values from Transfermarkt"),
) -> None:
    if season is not None:
        season_list = [season]
    elif seasons:
        season_list = list(seasons)
    else:
        console.print("[red]Provide --season or --seasons[/red]")
        raise typer.Exit(1)

    if league not in LEAGUE_IDS:
        console.print(f"[red]Unknown league '{league}'. Valid: {list(LEAGUE_IDS)}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]FootyML Ingestion[/bold] | League: {league} | Seasons: {season_list}")
    asyncio.run(_run(league, season_list, also_squad))


async def _run(league: str, seasons: list[int], fetch_squad: bool) -> None:
    store = DataStore()
    competition_id = LEAGUE_IDS[league]
    csv_fetcher = FootballDataCsvFetcher()

    # Build Transfermarkt name→ID mapping for each season (for market value lookups)
    # This also lets us use proper TM club IDs in the match table
    async with TransfermarktClient() as client:
        fetcher = TransfermarktFetcher(client, ParquetCache())

        for season in seasons:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
                task = progress.add_task(f"Season {season}: fetching club list from Transfermarkt...")

                # Fetch TM club list to build name→id mapping
                name_to_id: dict[str, str] = {}
                try:
                    clubs_df = await fetcher.fetch_competition_clubs(competition_id, str(season))
                    if len(clubs_df) > 0:
                        from footyml.ingestion.footballdata_csv import _normalise
                        for row in clubs_df.iter_rows(named=True):
                            name_to_id[_normalise(str(row["club_name"]))] = str(row["club_id"])
                except Exception as e:
                    console.print(f"[yellow]TM club list unavailable for {season}: {e}[/yellow]")

                # Fetch match results from football-data.co.uk
                progress.update(task, description=f"Season {season}: downloading matches from football-data.co.uk...")
                try:
                    matches_df = await csv_fetcher.fetch_season(league, season, name_to_id if name_to_id else None)
                except Exception as e:
                    console.print(f"[red]Error fetching matches for season {season}: {e}[/red]")
                    continue

                if len(matches_df) == 0:
                    console.print(f"[yellow]No matches found for {league} {season}[/yellow]")
                    continue

                try:
                    store.upsert_matches(matches_df)
                except Exception as e:
                    console.print(f"[red]Error storing matches for season {season}: {e}[/red]")
                    continue

                console.print(f"[green]Season {season}: {len(matches_df)} matches stored[/green]")

                # Fetch squad market values from Transfermarkt
                if fetch_squad and len(clubs_df) > 0:
                    progress.update(task, description=f"Season {season}: fetching squad values...")
                    club_ids = clubs_df["club_id"].to_list()
                    squad_frames = await asyncio.gather(
                        *[fetcher.fetch_squad_market_values(cid, str(season)) for cid in club_ids],
                        return_exceptions=True,
                    )
                    valid = [f for f in squad_frames if isinstance(f, pl.DataFrame) and len(f) > 0]
                    if valid:
                        store.upsert_squad(pl.concat(valid))
                        console.print(f"[green]Season {season}: squad values stored for {len(valid)} clubs[/green]")

    console.print("[bold green]Ingestion complete.[/bold green]")
    console.print("Next step: build features with:")
    console.print(f'  python -c "from footyml.features import FeaturePipeline; FeaturePipeline().build(competition_id=\'{competition_id}\')"')


if __name__ == "__main__":
    app()
