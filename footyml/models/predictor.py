from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.base import ClassifierMixin

from footyml.config import MODELS_DIR


class TabPFNMatchPredictor(ClassifierMixin):
    """Sklearn-compatible wrapper around TabPFNClassifier for match outcome prediction.

    Target: 0 = away win, 1 = draw, 2 = home win.
    """

    classes_ = np.array([0, 1, 2])

    def __init__(self, device: str = "cpu", n_estimators: int = 32):
        self.device = device
        self.n_estimators = n_estimators
        self._clf: object | None = None
        self.feature_names_: list[str] = []

    def _build_clf(self) -> object:
        from tabpfn import TabPFNClassifier  # type: ignore[import]

        return TabPFNClassifier(device=self.device, n_estimators=self.n_estimators)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNMatchPredictor":
        self._clf = self._build_clf()
        self._clf.fit(X, y)  # type: ignore[union-attr]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self._clf.predict(X)  # type: ignore[union-attr, no-any-return]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 3) array: [P(away win), P(draw), P(home win)]."""
        if self._clf is None:
            raise RuntimeError("Model not fitted — call fit() first.")
        return self._clf.predict_proba(X)  # type: ignore[union-attr, no-any-return]

    def save(self, path: Path | None = None) -> Path:
        if self._clf is None:
            raise RuntimeError("Model not fitted — nothing to save.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = path or (MODELS_DIR / "tabpfn_model.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self._clf, f)
        return save_path

    @classmethod
    def load(cls, path: Path) -> "TabPFNMatchPredictor":
        predictor = cls()
        with open(path, "rb") as f:
            predictor._clf = pickle.load(f)
        return predictor
