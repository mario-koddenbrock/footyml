"""Ingest match data from the Transfermarkt API for a given league and seasons.

Usage:
    python scripts/ingest.py --league bundesliga --seasons 2020 2021 2022 2023 2024
    python scripts/ingest.py --league bundesliga --season 2024  # single season
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from footyml.config import LEAGUE_IDS
from footyml.data.store import DataStore
from footyml.ingestion.cache import ParquetCache
from footyml.ingestion.client import TransfermarktClient
from footyml.ingestion.fetchers import TransfermarktFetcher

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: str = typer.Option("bundesliga", help=f"League name. Options: {list(LEAGUE_IDS)}"),
    seasons: list[int] = typer.Option(None, help="Season start years (e.g. --seasons 2020 2021)"),
    season: int | None = typer.Option(None, help="Single season (shorthand)"),
    also_squad: bool = typer.Option(True, "--squad/--no-squad", help="Also fetch squad market values"),
    also_transfers: bool = typer.Option(True, "--transfers/--no-transfers", help="Also fetch transfers"),
) -> None:
    if season is not None:
        season_list = [season]
    elif seasons:
        season_list = list(seasons)
    else:
        console.print("[red]Provide --season or --seasons[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]FootyML Ingestion[/bold] | League: {league} | Seasons: {season_list}")
    asyncio.run(_run(league, season_list, also_squad, also_transfers))


async def _run(
    league: str, seasons: list[int], fetch_squad: bool, fetch_transfers: bool
) -> None:
    store = DataStore()
    cache = ParquetCache()
    competition_id = LEAGUE_IDS[league]

    async with TransfermarktClient() as client:
        fetcher = TransfermarktFetcher(client, cache)

        for season in seasons:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
                task = progress.add_task(f"Season {season}: fetching matches...")

                try:
                    matches_df = await fetcher.fetch_league_season(league, season)
                except Exception as e:
                    console.print(f"[red]Error fetching season {season}: {e}[/red]")
                    continue

                if len(matches_df) == 0:
                    console.print(f"[yellow]No matches found for {league} {season}[/yellow]")
                    continue

                # Normalise result column
                import polars as pl

                if "home_goals" in matches_df.columns and "away_goals" in matches_df.columns:
                    matches_df = matches_df.with_columns([
                        pl.when(pl.col("home_goals") > pl.col("away_goals")).then(2)
                        .when(pl.col("home_goals") == pl.col("away_goals")).then(1)
                        .otherwise(0)
                        .alias("result"),
                        pl.lit(competition_id).alias("competition_id"),
                        pl.lit(season).alias("season"),
                    ])

                required_cols = ["game_id", "competition_id", "season", "matchday", "date",
                                 "home_club_id", "away_club_id", "home_goals", "away_goals", "result"]
                available = [c for c in required_cols if c in matches_df.columns]
                if "date" not in available:
                    console.print(f"[red]No date column in matches for season {season}[/red]")
                    continue

                matches_store = matches_df.select(available)
                if "date" in matches_store.columns:
                    try:
                        matches_store = matches_store.with_columns(
                            pl.col("date").str.to_date("%Y-%m-%d").alias("date")
                        )
                    except Exception:
                        pass

                try:
                    store.upsert_matches(matches_store)
                except Exception as e:
                    console.print(f"[red]Error storing matches for season {season}: {e}[/red]")
                    continue

                progress.update(task, description=f"Season {season}: stored {len(matches_store)} matches")

                if fetch_squad or fetch_transfers:
                    clubs_df = await fetcher.fetch_competition_clubs(competition_id, str(season))
                    club_ids = clubs_df["club_id"].to_list()

                    if fetch_squad:
                        squad_frames = await asyncio.gather(
                            *[fetcher.fetch_squad_market_values(cid, str(season)) for cid in club_ids],
                            return_exceptions=True,
                        )
                        valid = [f for f in squad_frames if isinstance(f, pl.DataFrame) and len(f) > 0]
                        if valid:
                            store.upsert_squad(pl.concat(valid))

                    if fetch_transfers:
                        transfer_frames = await asyncio.gather(
                            *[fetcher.fetch_club_transfers(cid, str(season)) for cid in club_ids],
                            return_exceptions=True,
                        )
                        valid = [f for f in transfer_frames if isinstance(f, pl.DataFrame) and len(f) > 0]
                        if valid:
                            store.upsert_transfers(pl.concat(valid))

                console.print(f"[green]Season {season}: {len(matches_store)} matches ingested[/green]")

    console.print("[bold green]Ingestion complete.[/bold green]")
    console.print("Next step: build features with:")
    console.print(f"  python -c \"from footyml.features import FeaturePipeline; FeaturePipeline().build(competition_id='{competition_id}')\"")


if __name__ == "__main__":
    app()
