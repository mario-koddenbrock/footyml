"""Predict the outcome of a single match.

Usage:
    python scripts/predict.py --home "Bayern Munich" --away "Borussia Dortmund"
    python scripts/predict.py --home "Bayern Munich" --away "Borussia Dortmund" --league bundesliga
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel

from footyml.config import LEAGUE_IDS, MODELS_DIR
from footyml.data.store import DataStore
from footyml.models.predictor import TabPFNMatchPredictor

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    home: str = typer.Option(..., help="Home team name"),
    away: str = typer.Option(..., help="Away team name"),
    league: str = typer.Option("bundesliga", help=f"League. Options: {list(LEAGUE_IDS)}"),
    model_path: Path | None = typer.Option(None, help="Path to trained model .pkl file"),
) -> None:
    if model_path is None:
        candidates = sorted(MODELS_DIR.glob(f"tabpfn_{league}_*.pkl"))
        if not candidates:
            console.print(f"[red]No model found for league '{league}'. Train one first:[/red]")
            console.print(f"  python scripts/train.py --league {league} --season <year>")
            raise typer.Exit(1)
        model_path = candidates[-1]

    console.print(f"Loading model: {model_path.name}")
    predictor = TabPFNMatchPredictor.load(model_path)

    store = DataStore()
    competition_id = LEAGUE_IDS[league]

    import polars as pl

    home_match = store.query(f"""
        SELECT home_club_id AS club_id, home_club_name AS club_name
        FROM matches
        WHERE competition_id = '{competition_id}'
          AND LOWER(home_club_name) LIKE LOWER('%{home}%')
        LIMIT 1
        UNION ALL
        SELECT away_club_id AS club_id, away_club_name AS club_name
        FROM matches
        WHERE competition_id = '{competition_id}'
          AND LOWER(away_club_name) LIKE LOWER('%{home}%')
        LIMIT 1
    """)
    away_match = store.query(f"""
        SELECT home_club_id AS club_id, home_club_name AS club_name
        FROM matches
        WHERE competition_id = '{competition_id}'
          AND LOWER(home_club_name) LIKE LOWER('%{away}%')
        LIMIT 1
        UNION ALL
        SELECT away_club_id AS club_id, away_club_name AS club_name
        FROM matches
        WHERE competition_id = '{competition_id}'
          AND LOWER(away_club_name) LIKE LOWER('%{away}%')
        LIMIT 1
    """)

    if len(home_match) == 0 or len(away_match) == 0:
        console.print("[red]Could not find one or both teams in the database.[/red]")
        console.print("Ensure you have ingested data for this league.")
        raise typer.Exit(1)

    home_id = str(home_match["club_id"][0])
    away_id = str(away_match["club_id"][0])
    home_name = str(home_match["club_name"][0])
    away_name = str(away_match["club_name"][0])

    features_df = store.query(f"""
        SELECT f.*
        FROM features f
        JOIN matches m USING (game_id)
        WHERE m.home_club_id = '{home_id}' AND m.away_club_id = '{away_id}'
          AND m.competition_id = '{competition_id}'
        ORDER BY m.date DESC
        LIMIT 1
    """)

    if len(features_df) == 0:
        console.print("[yellow]No historical head-to-head feature data found.[/yellow]")
        console.print("Using the most recent home feature vector for this team.")
        features_df = store.query(f"""
            SELECT f.*
            FROM features f
            JOIN matches m USING (game_id)
            WHERE m.home_club_id = '{home_id}'
              AND m.competition_id = '{competition_id}'
            ORDER BY m.date DESC
            LIMIT 1
        """)

    if len(features_df) == 0:
        console.print("[red]No feature data available for this match. Run the feature pipeline first.[/red]")
        raise typer.Exit(1)

    import numpy as np

    skip_cols = {"game_id"}
    feat_cols = [c for c in features_df.columns if c not in skip_cols]
    X = features_df.select(feat_cols).to_numpy().astype(np.float32)
    proba = predictor.predict_proba(X)[0]

    home_win = round(float(proba[2]) * 100, 1)
    draw = round(float(proba[1]) * 100, 1)
    away_win = round(float(proba[0]) * 100, 1)

    predicted_idx = int(proba.argmax())
    labels = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    predicted = labels[predicted_idx]

    panel_text = (
        f"[bold]{home_name}[/bold] vs [bold]{away_name}[/bold]\n\n"
        f"Home Win:  [green]{home_win:.1f}%[/green]\n"
        f"Draw:      [yellow]{draw:.1f}%[/yellow]\n"
        f"Away Win:  [red]{away_win:.1f}%[/red]\n\n"
        f"Prediction: [bold]{predicted}[/bold]"
    )
    console.print(Panel(panel_text, title="Match Prediction", border_style="blue"))


if __name__ == "__main__":
    app()
