"""Train a TabPFN match predictor on a given league and season range.

Usage:
    python scripts/train.py --league bundesliga --season 2024
    python scripts/train.py --league bundesliga --season 2024 --train-from 2018 --eval
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from footyml.config import LEAGUE_IDS, MODELS_DIR
from footyml.data.dataset import MatchDataset
from footyml.models.evaluation import baseline_majority_metrics, evaluate_time_split
from footyml.models.predictor import TabPFNMatchPredictor

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    league: str = typer.Option("bundesliga", help=f"League name. Options: {list(LEAGUE_IDS)}. Use 'all' to train on all leagues."),
    season: int = typer.Option(..., help="Test season year (e.g. 2024)"),
    train_from: int = typer.Option(2015, help="First training season year"),
    evaluate: bool = typer.Option(False, "--eval", help="Print evaluation metrics after training"),
    device: str = typer.Option("cpu", help="Device for TabPFN: 'cpu' or 'cuda'"),
) -> None:
    all_leagues = league == "all"
    label = "all leagues" if all_leagues else league
    console.print(f"[bold]FootyML — Training[/bold] | {label} | Season: {season}")

    all_seasons = list(range(train_from, season + 1))
    train_seasons = list(range(train_from, season))

    if not train_seasons:
        console.print("[red]Need at least one training season before the test season.[/red]")
        raise typer.Exit(1)

    console.print(f"Loading data: train {train_seasons[0]}–{train_seasons[-1]}, test {season}")
    if all_leagues:
        dataset = MatchDataset.all_leagues(seasons=all_seasons)
    else:
        dataset = MatchDataset(league=league, seasons=all_seasons)

    try:
        full_df = dataset.load_polars()
    except Exception as e:
        console.print(f"[red]Failed to load data: {e}[/red]")
        console.print("Run the ingestion pipeline first to populate the database.")
        raise typer.Exit(1)

    if len(full_df) == 0:
        console.print("[red]No data found. Run ingestion first.[/red]")
        raise typer.Exit(1)

    skip_cols = {"game_id", "date", "result"}
    feature_cols = [c for c in full_df.columns if c not in skip_cols]

    import numpy as np

    full_df = full_df.sort("date")
    test_cutoff = f"{season}-01-01"
    train_df = full_df.filter(full_df["date"].cast(str) < test_cutoff)
    test_df = full_df.filter(full_df["date"].cast(str) >= test_cutoff)

    if len(train_df) == 0 or len(test_df) == 0:
        console.print("[red]Not enough data for the requested split.[/red]")
        raise typer.Exit(1)

    X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
    y_train = train_df["result"].to_numpy().astype(np.int32)
    X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
    y_test = test_df["result"].to_numpy().astype(np.int32)

    console.print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)} | Features: {X_train.shape[1]}")

    predictor = TabPFNMatchPredictor(device=device)

    if evaluate:
        console.print("Training and evaluating...")
        metrics = evaluate_time_split(predictor, X_train, y_train, X_test, y_test)
        baseline = baseline_majority_metrics(y_test)

        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("TabPFN", style="green")
        table.add_column("Majority Baseline", style="yellow")

        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}", f"{baseline['accuracy']:.4f}")
        table.add_row("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}", f"{baseline['balanced_accuracy']:.4f}")
        table.add_row("F1 (macro)", f"{metrics['f1_macro']:.4f}", "—")
        table.add_row("Log Loss", f"{metrics['log_loss']:.4f}", "—")

        console.print(table)
    else:
        console.print("Training model...")
        predictor.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = "all" if all_leagues else league
    model_path = MODELS_DIR / f"tabpfn_{model_name}_{season}.pkl"
    predictor.save(model_path)
    console.print(f"[green]Model saved to {model_path}[/green]")


if __name__ == "__main__":
    app()
