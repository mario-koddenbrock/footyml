from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from footyml.data.dataset import MatchDataset
from footyml.data.store import DataStore
from footyml.features.pipeline import FeaturePipeline


def _setup_store_with_features(store: DataStore, matches: pl.DataFrame) -> None:
    store.upsert_matches(matches)
    pipeline = FeaturePipeline(store)
    pipeline.build(competition_id="L1", seasons=[2022])


def test_dataset_load_returns_correct_shapes(
    tmp_store: DataStore, sample_matches: pl.DataFrame
) -> None:
    _setup_store_with_features(tmp_store, sample_matches)
    dataset = MatchDataset(league="bundesliga", seasons=[2022], store=tmp_store)
    X, y = dataset.load()
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.dtype == np.float32


def test_dataset_invalid_league_raises() -> None:
    with pytest.raises(ValueError, match="Unknown league"):
        MatchDataset(league="invalid_league", seasons=[2022])


def test_time_split_preserves_order() -> None:
    dates = np.array(["2022-08-01", "2022-09-01", "2023-01-15", "2023-03-01"])
    X = np.arange(len(dates)).reshape(-1, 1).astype(np.float32)
    y = np.array([0, 1, 2, 0])

    X_train, X_test, y_train, y_test = MatchDataset.time_split(X, y, dates, "2023-01-01")
    assert len(X_train) == 2
    assert len(X_test) == 2
    assert all(d < "2023-01-01" for d in dates[: len(X_train)])
    assert all(d >= "2023-01-01" for d in dates[len(X_train) :])
