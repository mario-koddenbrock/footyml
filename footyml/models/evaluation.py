from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)

from footyml.models.predictor import TabPFNMatchPredictor


def evaluate_time_split(
    predictor: TabPFNMatchPredictor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Train on X_train/y_train and evaluate on the held-out time split."""
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    y_proba = predictor.predict_proba(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "log_loss": round(float(log_loss(y_test, y_proba)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def baseline_majority_metrics(y_test: np.ndarray) -> dict[str, float]:
    """Metrics for a naive majority-class baseline (upper bound for trivial models)."""
    from collections import Counter

    majority = Counter(y_test).most_common(1)[0][0]
    y_pred = np.full_like(y_test, majority)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
    }
