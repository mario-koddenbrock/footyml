"""Multi-model WC 2026 prediction script.

Trains 7 models on historical international data and generates predictions
for all upcoming WC 2026 matches, saving one JSON file per model.

Usage:
    python scripts/predict_multi_model.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from footyml.config import MODELS_DIR
from footyml.data.store import DataStore
from footyml.features.pipeline import FeaturePipeline
from footyml.models.predictor import TabPFNMatchPredictor

PREDICTIONS_DIR = Path("data/predictions")
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# International competitions to train on
TRAIN_COMPS = ["WC", "EC", "CLI", "CAN", "AAC", "CGC", "CC"]

# Base rates from historical WC/international data (away/draw/home)
BASE_RATES = [0.307, 0.232, 0.461]


# ---------------------------------------------------------------------------
# Baseline classifiers
# ---------------------------------------------------------------------------

class RandomBaselineClassifier:
    """Always returns fixed base-rate probabilities."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.tile(BASE_RATES, (n, 1))


class HomeAlwaysWinsClassifier:
    """Always predicts heavy home win."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else X.shape[0]
        return np.tile([0.05, 0.10, 0.85], (n, 1))


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------

def build_training_features(store: DataStore, pipeline: FeaturePipeline, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build concatenated feature matrix from all training competitions."""
    print("Building training features from:", TRAIN_COMPS)
    frames = []
    for comp in TRAIN_COMPS:
        matches = store.get_matches(competition_id=comp, completed_only=True)
        if len(matches) == 0:
            print(f"  {comp}: no matches, skipping")
            continue
        print(f"  {comp}: {len(matches)} completed matches")
        try:
            feat = pipeline.build(competition_id=comp)
            if len(feat) > 0:
                result_col = matches.select(["game_id", "result"])
                feat = feat.join(result_col, on="game_id", how="inner")
                frames.append(feat)
                print(f"    → {len(feat)} feature rows")
        except Exception as e:
            print(f"  {comp}: FAILED — {e}")

    if not frames:
        raise RuntimeError("No training features built. Run ingest first.")

    all_feats = pl.concat(frames, how="diagonal")
    print(f"\nTotal training rows: {len(all_feats):,}  Total columns: {len(all_feats.columns)}")

    # Extract X using canonical feature columns (fill missing with NaN)
    available = [c for c in feature_cols if c in all_feats.columns]
    missing = [c for c in feature_cols if c not in all_feats.columns]
    if missing:
        print(f"  Missing {len(missing)} feature cols in training data — padding with NaN")
        for col in missing:
            all_feats = all_feats.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    X = all_feats.select(feature_cols).to_numpy().astype(np.float32)
    y = all_feats["result"].to_numpy().astype(np.int32)
    print(f"X shape: {X.shape}  y distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def build_upcoming_features(store: DataStore, pipeline: FeaturePipeline, feature_cols: list[str]) -> tuple[np.ndarray, pl.DataFrame]:
    """Build feature matrix for upcoming WC 2026 matches."""
    upcoming_df = store.query(
        "SELECT * FROM matches WHERE competition_id='WC' AND season=2026 "
        "AND result IS NULL AND home_club_name != '' AND home_club_name IS NOT NULL"
    )
    print(f"\nUpcoming WC 2026 matches: {len(upcoming_df)}")

    feat = pipeline.build_upcoming(upcoming_df)
    print(f"Upcoming feature rows: {len(feat)}  columns: {len(feat.columns)}")

    # Align to canonical feature columns
    for col in feature_cols:
        if col not in feat.columns:
            feat = feat.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    X_upcoming = feat.select(feature_cols).to_numpy().astype(np.float32)

    # Join back match metadata
    meta = upcoming_df.select(["game_id", "date", "stage", "group_id", "home_club_name", "away_club_name", "venue_type"])
    feat_with_meta = feat.join(meta, on="game_id", how="left")

    return X_upcoming, feat_with_meta


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def format_prediction(row: dict, proba: np.ndarray, model_name: str) -> dict:
    """Format a single prediction as output dict."""
    away_prob, draw_prob, home_prob = float(proba[0]), float(proba[1]), float(proba[2])

    max_idx = int(np.argmax(proba))
    if max_idx == 2:
        predicted_result = "A Win"   # home wins (A = home team)
    elif max_idx == 1:
        predicted_result = "Draw"
    else:
        predicted_result = "B Win"   # away wins (B = away team)

    return {
        "game_id": str(row.get("game_id", "")),
        "date": str(row.get("date", ""))[:10],
        "stage": str(row.get("stage", "")),
        "group_id": str(row.get("group_id", "") or ""),
        "home_team": str(row.get("home_club_name", "")),
        "away_team": str(row.get("away_club_name", "")),
        "venue_type": str(row.get("venue_type", "neutral")),
        "home_win_prob": round(home_prob * 100, 1),
        "draw_prob": round(draw_prob * 100, 1),
        "away_win_prob": round(away_prob * 100, 1),
        "predicted_result": predicted_result,
        "method": model_name,
    }


def save_predictions(predictions: list[dict], model_name: str) -> Path:
    out_path = PREDICTIONS_DIR / f"wc2026_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    store = DataStore()
    pipeline = FeaturePipeline(store)

    # ── Load canonical feature column order from TabPFN model ──────────────
    tabpfn_model_path = MODELS_DIR / "tabpfn_international.pkl"
    if not tabpfn_model_path.exists():
        raise FileNotFoundError(f"TabPFN model not found: {tabpfn_model_path}")

    print(f"Loading TabPFN model: {tabpfn_model_path}")
    tabpfn_predictor = TabPFNMatchPredictor.load(tabpfn_model_path)
    feature_cols = tabpfn_predictor.feature_names_
    print(f"Canonical feature columns: {len(feature_cols)}")

    # ── Build training data ─────────────────────────────────────────────────
    X_train, y_train = build_training_features(store, pipeline, feature_cols)

    # ── Build upcoming WC 2026 features ─────────────────────────────────────
    X_upcoming, feat_with_meta = build_upcoming_features(store, pipeline, feature_cols)
    meta_rows = feat_with_meta.to_dicts()

    n_upcoming = X_upcoming.shape[0]
    print(f"\nPredicting {n_upcoming} upcoming matches with 7 models...\n")

    # ── Imputed versions for sklearn models ─────────────────────────────────
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_upcoming_imp = imputer.transform(X_upcoming)

    # ── Define model registry ───────────────────────────────────────────────
    models = {
        "tabpfn": None,          # loaded, not retrained
        "tabicl": None,          # fitted below
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "catboost": None,        # imported below
        "random_baseline": RandomBaselineClassifier(),
        "home_always_wins": HomeAlwaysWinsClassifier(),
    }

    # CatBoost import (optional dep)
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, verbose=0, random_seed=42
        )
    except ImportError:
        print("WARNING: catboost not installed — skipping catboost model")
        models.pop("catboost")

    # TabICL import (optional dep)
    try:
        from tabicl import TabICLClassifier
        models["tabicl"] = TabICLClassifier()
    except ImportError:
        print("WARNING: tabicl not installed — skipping tabicl model")
        models.pop("tabicl")

    # ── Fit and predict ─────────────────────────────────────────────────────
    for model_name, clf in models.items():
        out_path = PREDICTIONS_DIR / f"wc2026_{model_name}.json"
        if out_path.exists():
            print(f"[{model_name}] already exists — skipping (delete to re-run)")
            continue

        t0 = time.time()
        print(f"[{model_name}]", end=" ", flush=True)

        if model_name == "tabpfn":
            # Already loaded — just predict
            print("loaded (no retraining)", end=" ", flush=True)
            probas = tabpfn_predictor.predict_proba(X_upcoming)
        elif model_name in ("tabicl",):
            # TabICL: use NaN-imputed data to avoid internal feature-mask mismatch
            # when train/test have different NaN patterns
            print("fitting (may be slow)...", end=" ", flush=True)
            clf.fit(X_train_imp, y_train)
            print("predicting...", end=" ", flush=True)
            probas = clf.predict_proba(X_upcoming_imp)
        elif model_name in ("knn",):
            # Needs imputed data
            print("fitting (imputed)...", end=" ", flush=True)
            clf.fit(X_train_imp, y_train)
            probas = clf.predict_proba(X_upcoming_imp)
        elif model_name in ("random_forest", "catboost"):
            print("fitting (imputed)...", end=" ", flush=True)
            clf.fit(X_train_imp, y_train)
            probas = clf.predict_proba(X_upcoming_imp)
        else:
            # Baselines — no fitting needed
            print("predicting...", end=" ", flush=True)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_upcoming)

        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")

        # Ensure probas shape is (n, 3)
        if probas.shape[1] != 3:
            print(f"  WARNING: unexpected proba shape {probas.shape} — padding")
            padded = np.zeros((probas.shape[0], 3))
            padded[:, :probas.shape[1]] = probas
            probas = padded

        # Build prediction dicts
        predictions = []
        for i, row in enumerate(meta_rows):
            predictions.append(format_prediction(row, probas[i], model_name))

        save_predictions(predictions, model_name)

    print("\nAll predictions saved to data/predictions/")


if __name__ == "__main__":
    main()
