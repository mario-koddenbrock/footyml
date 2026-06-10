"""Predict outcomes for all upcoming tournament matches.

Usage:
    python scripts/predict_tournament.py --competition WC
    python scripts/predict_tournament.py --competition WC --season 2026
    python scripts/predict_tournament.py --competition WC --export predictions.csv

Requires: FOOTBALL_DATA_API_KEY environment variable.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from footyml.config import MODELS_DIR, TOURNAMENT_IDS

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    competition: str = typer.Option("WC", help=f"Competition code. Known: {list(TOURNAMENT_IDS.values())}"),
    season: int | None = typer.Option(None, help="Season year (e.g. 2026). Defaults to current."),
    model_path: Path | None = typer.Option(None, help="Path to trained .pkl model"),
    export: Path | None = typer.Option(None, help="Export predictions to Parquet file"),
    no_sync: bool = typer.Option(False, "--no-sync", help="Skip fixture sync (use cached data)"),
    list_competitions: bool = typer.Option(False, "--list", help="List all available competitions and exit"),
) -> None:
    asyncio.run(
        _run(
            competition=competition,
            season=season,
            model_path=model_path,
            export=export,
            sync=not no_sync,
            list_competitions=list_competitions,
        )
    )


async def _run(
    competition: str,
    season: int | None,
    model_path: Path | None,
    export: Path | None,
    sync: bool,
    list_competitions: bool,
) -> None:
    from footyml.prediction.tournament import TournamentPredictor

    if model_path is None:
        # Prefer all-leagues model, then competition-specific, then any tabpfn model
        candidates = (
            sorted(MODELS_DIR.glob("tabpfn_all_*.pkl"))
            or sorted(MODELS_DIR.glob(f"tabpfn_{competition}_*.pkl"))
            or sorted(MODELS_DIR.glob("tabpfn_*.pkl"))
        )
        if candidates:
            model_path = candidates[-1]
            console.print(f"Using model: {model_path.name}")
        else:
            console.print(
                f"[yellow]No model found. Running without predictions (fixtures only).[/yellow]"
            )
            console.print("  Train one: python scripts/train.py --league all --season 2025")

    async with TournamentPredictor(model_path=model_path) as predictor:
        if list_competitions:
            comps = await predictor._provider.list_competitions()  # type: ignore[union-attr]
            t = Table(title="Available Competitions")
            t.add_column("Code", style="cyan")
            t.add_column("Name")
            t.add_column("Area")
            for row in comps.iter_rows(named=True):
                t.add_row(str(row.get("code", "")), str(row.get("name", "")), str(row.get("area", "")))
            console.print(t)
            return

        predictions = await predictor.predict_all_upcoming(
            competition=competition,
            season=season,
            sync=sync,
        )

    if not predictions:
        console.print(Panel(
            f"No upcoming matches found for [bold]{competition}[/bold].",
            border_style="yellow",
        ))
        return

    _print_table(predictions, competition)

    import json
    import polars as pl

    out_path = export or Path(f"data/predictions/{competition}_upcoming.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Always write Parquet
    parquet_path = out_path.with_suffix(".parquet")
    pl.DataFrame([{k: v for k, v in p.items()} for p in predictions]).write_parquet(parquet_path)

    # Also write JSON for easy transfer / inspection
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    console.print(f"[dim]Predictions saved → {parquet_path}[/dim]")
    console.print(f"[dim]                  → {json_path}[/dim]")


def _print_table(predictions: list[dict], competition: str) -> None:
    t = Table(title=f"[bold]{competition}[/bold] — Upcoming Match Predictions")
    t.add_column("Date", style="cyan", no_wrap=True)
    t.add_column("Stage", style="dim")
    t.add_column("Group", style="dim")
    t.add_column("Team A (home)", style="bold")
    t.add_column("Team B (away)", style="bold")
    t.add_column("Venue", style="dim")
    t.add_column("A Win %", style="green", justify="right")
    t.add_column("Draw %", style="yellow", justify="right")
    t.add_column("B Win %", style="red", justify="right")
    t.add_column("Prediction", style="magenta")

    for p in predictions:
        hw = p.get("home_win_prob")
        dw = p.get("draw_prob")
        aw = p.get("away_win_prob")
        t.add_row(
            str(p.get("date", ""))[:10],
            str(p.get("stage", ""))[:15],
            str(p.get("group_id", "")),
            str(p.get("home_team", "")),
            str(p.get("away_team", "")),
            str(p.get("venue_type", ""))[:7],
            f"{hw:.1f}%" if hw is not None else "—",
            f"{dw:.1f}%" if dw is not None else "—",
            f"{aw:.1f}%" if aw is not None else "—",
            str(p.get("predicted_result", "")),
        )

    console.print(t)


if __name__ == "__main__":
    app()
