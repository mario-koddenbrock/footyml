"""Predict all upcoming matches for a league.

Usage:
    python scripts/predict_upcoming.py --league bundesliga
    python scripts/predict_upcoming.py --league bundesliga --matchday 30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from footyml.config import LEAGUE_IDS, MODELS_DIR
from footyml.data.store import DataStore
from footyml.models.predictor import TabPFNMatchPredictor
from footyml.prediction.service import PredictionService

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: str = typer.Option("bundesliga", help=f"League. Options: {list(LEAGUE_IDS)}"),
    matchday: int | None = typer.Option(None, help="Specific matchday to predict"),
) -> None:
    candidates = sorted(MODELS_DIR.glob(f"tabpfn_{league}_*.pkl"))
    if not candidates:
        console.print(f"[red]No model found for '{league}'. Train first:[/red]")
        console.print(f"  python scripts/train.py --league {league} --season <year>")
        raise typer.Exit(1)

    model_path = candidates[-1]
    console.print(f"Model: {model_path.name}")

    predictor = TabPFNMatchPredictor.load(model_path)
    store = DataStore()
    service = PredictionService(predictor, store)

    predictions = service.predict_upcoming(league=league)

    if not predictions:
        console.print("[yellow]No upcoming fixtures found in the database.[/yellow]")
        console.print("Make sure upcoming matches have been ingested (result IS NULL).")
        raise typer.Exit(0)

    table = Table(title=f"Upcoming {league.title()} Predictions")
    table.add_column("Date", style="cyan", no_wrap=True)
    table.add_column("Home Team", style="bold")
    table.add_column("Away Team", style="bold")
    table.add_column("Home Win %", style="green", justify="right")
    table.add_column("Draw %", style="yellow", justify="right")
    table.add_column("Away Win %", style="red", justify="right")
    table.add_column("Prediction", style="magenta")

    for pred in predictions:
        table.add_row(
            str(pred.get("date", ""))[:10],
            str(pred.get("home_team", "")),
            str(pred.get("away_team", "")),
            f"{pred.get('home_win_prob', 0):.1f}%",
            f"{pred.get('draw_prob', 0):.1f}%",
            f"{pred.get('away_win_prob', 0):.1f}%",
            str(pred.get("predicted_result", "")),
        )

    console.print(table)

    import polars as pl

    rows = [
        {
            "date": str(p.get("date", ""))[:10],
            "home_team": str(p.get("home_team", "")),
            "away_team": str(p.get("away_team", "")),
            "home_win_prob": float(p.get("home_win_prob", 0)),
            "draw_prob": float(p.get("draw_prob", 0)),
            "away_win_prob": float(p.get("away_win_prob", 0)),
            "predicted_result": str(p.get("predicted_result", "")),
        }
        for p in predictions
    ]
    output_path = Path("data/predictions") / f"{league}_upcoming.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)
    console.print(f"[dim]Predictions saved to {output_path}[/dim]")


if __name__ == "__main__":
    app()
