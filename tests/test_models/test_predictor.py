from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from footyml.models.predictor import TabPFNMatchPredictor


class _DummyClf:
    """Stand-in for TabPFNClassifier that avoids the actual download."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._classes = np.unique(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        proba = np.full((n, 3), 1 / 3)
        return proba


def _make_predictor_with_dummy() -> TabPFNMatchPredictor:
    pred = TabPFNMatchPredictor()
    pred._clf = _DummyClf()
    return pred


def test_predict_proba_shape() -> None:
    pred = _make_predictor_with_dummy()
    X = np.random.rand(10, 5).astype(np.float32)
    proba = pred.predict_proba(X)
    assert proba.shape == (10, 3)


def test_predict_proba_sums_to_one() -> None:
    pred = _make_predictor_with_dummy()
    X = np.random.rand(20, 5).astype(np.float32)
    proba = pred.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_predict_shape() -> None:
    pred = _make_predictor_with_dummy()
    X = np.random.rand(15, 5).astype(np.float32)
    preds = pred.predict(X)
    assert preds.shape == (15,)


def test_not_fitted_raises() -> None:
    pred = TabPFNMatchPredictor()
    with pytest.raises(RuntimeError, match="not fitted"):
        pred.predict(np.zeros((1, 5)))


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    pred = _make_predictor_with_dummy()
    save_path = tmp_path / "model.pkl"
    pred.save(save_path)

    loaded = TabPFNMatchPredictor.load(save_path)
    X = np.random.rand(5, 5).astype(np.float32)
    proba = loaded.predict_proba(X)
    assert proba.shape == (5, 3)
