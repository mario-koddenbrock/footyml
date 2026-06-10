"""Train TabPFN on historical international competitive matches.

Reads completed matches from all international competitions stored by
ingest_international_martj42.py, builds features (Elo + MarketValue + rolling
form from match history), and trains a TabPFNClassifier saved as
data/models/tabpfn_international.pkl.

Usage:
    python scripts/train_international.py
    python scripts/train_international.py --competitions WC WCQ EC ECQ
    python scripts/train_international.py --eval
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from footyml.config import MODELS_DIR
from footyml.data.store import DataStore
from footyml.features.pipeline import FeaturePipeline
from footyml.models.predictor import TabPFNMatchPredictor

console = Console()
app = typer.Typer(add_completion=False)

ALL_INTL_COMPS = ["WC", "WCQ", "EC", "ECQ", "UNL", "CLI", "CAN", "CANQ",
                  "AAC", "AACQ", "CGC", "CNL", "CC"]


@app.command()
def main(
    competitions: list[str] = typer.Option(
        None,
        help="Competition codes to include. Defaults to all international competitions.",
    ),
    eval_last_year: bool = typer.Option(False, "--eval", help="Hold out last season for evaluation"),
    output: Path = typer.Option(None, help="Output model path (default: data/models/tabpfn_international.pkl)"),
) -> None:
    comps = list(competitions) if competitions else ALL_INTL_COMPS
    out_path = output or (MODELS_DIR / "tabpfn_international.pkl")

    store = DataStore()
    pipeline = FeaturePipeline(store)

    console.rule("[bold]Building features for international matches")

    # Build features per competition to avoid cross-competition form contamination
    frames = []
    for comp in comps:
        matches = store.get_matches(competition_id=comp, completed_only=True)
        if len(matches) == 0:
            continue
        console.print(f"  {comp}: {len(matches)} matches")
        try:
            feat = pipeline.build(competition_id=comp)
            if len(feat) > 0:
                # Join result label back
                result_col = matches.select(["game_id", "result"])
                feat = feat.join(result_col, on="game_id", how="inner")
                frames.append(feat)
        except Exception as e:
            console.print(f"  [red]{comp}: feature build failed — {e}[/red]")

    if not frames:
        console.print("[red]No features built — run ingest_international_martj42.py first[/red]")
        raise typer.Exit(1)

    all_feats = pl.concat(frames, how="diagonal")
    console.print(f"\nTotal feature rows: {len(all_feats):,}  columns: {len(all_feats.columns)}")

    # Drop game_id, result, and any non-numeric columns (e.g. join artifacts)
    skip = {"game_id", "result"}
    feature_cols = [
        c for c in all_feats.columns
        if c not in skip and all_feats[c].dtype not in (pl.String, pl.Categorical, pl.Date, pl.Datetime)
    ]
    X = all_feats.select(feature_cols).to_numpy().astype(np.float32)
    y = all_feats["result"].to_numpy().astype(np.int32)

    console.print(f"X shape: {X.shape}  y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    if eval_last_year:
        # Approximate time split: hold out ~15% newest matches for evaluation
        split = int(len(X) * 0.85)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        console.print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    console.rule("[bold]Training TabPFN")
    predictor = TabPFNMatchPredictor()
    predictor.fit(X_train, y_train)
    predictor.feature_names_ = feature_cols

    if eval_last_year and X_test is not None:
        from footyml.models.evaluation import evaluate_time_split
        metrics = evaluate_time_split(predictor, X_train, y_train, X_test, y_test)
        t = Table(title="Evaluation (hold-out)")
        t.add_column("Metric")
        t.add_column("Value", justify="right")
        for k, v in metrics.items():
            t.add_row(k, f"{v:.4f}")
        console.print(t)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved = predictor.save(out_path)
    console.print(f"[bold green]Model saved → {saved}[/bold green]")

    # Store feature names alongside model
    names_path = out_path.with_suffix(".features.txt")
    names_path.write_text("\n".join(feature_cols))
    console.print(f"Feature names → {names_path}")


if __name__ == "__main__":
    app()
