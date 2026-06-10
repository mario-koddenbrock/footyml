from __future__ import annotations

import polars as pl
import pytest

from footyml.features.pipeline import FeaturePipeline


def test_pipeline_returns_game_id_column(populated_store: "DataStore") -> None:
    pipeline = FeaturePipeline(populated_store)
    result = pipeline.build(competition_id="L1", seasons=[2022])
    assert "game_id" in result.columns
    assert len(result) > 0


def test_pipeline_no_duplicate_columns(populated_store: "DataStore") -> None:
    pipeline = FeaturePipeline(populated_store)
    result = pipeline.build(competition_id="L1", seasons=[2022])
    assert len(result.columns) == len(set(result.columns))


def test_pipeline_empty_db_returns_empty(tmp_store: "DataStore") -> None:
    pipeline = FeaturePipeline(tmp_store)
    result = pipeline.build(competition_id="L1", seasons=[2022])
    assert "game_id" in result.columns
    assert len(result) == 0
