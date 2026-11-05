import numpy as np
import pandas as pd
import pytest

from california_housing_lab.evaluation import regression_metrics
from california_housing_lab.features import HousingFeatureBuilder
from california_housing_lab.models import build_models
from california_housing_lab.train import split_by_geography


def sample_X(n=80):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "MedInc": rng.uniform(0.5, 12, n),
        "HouseAge": rng.integers(1, 52, n),
        "AveRooms": rng.uniform(2, 10, n),
        "AveBedrms": rng.uniform(0.5, 3, n),
        "Population": rng.integers(100, 5000, n),
        "AveOccup": rng.uniform(1, 6, n),
        "Latitude": rng.uniform(32, 42, n),
        "Longitude": rng.uniform(-124, -114, n),
        "Households": rng.integers(50, 1200, n),
    })


def test_feature_builder_adds_ratios_without_mutating_input():
    X = sample_X(8)
    before = X.copy(deep=True)
    transformed = HousingFeatureBuilder().fit_transform(X)
    assert "RoomsPerOccupant" in transformed
    assert "BedroomsPerRoom" in transformed
    pd.testing.assert_frame_equal(X, before)
    assert np.isfinite(transformed.to_numpy()).all()


def test_feature_builder_rejects_missing_required_predictor():
    with pytest.raises(ValueError, match="Missing required predictors"):
        HousingFeatureBuilder().fit(sample_X().drop(columns="AveOccup"))


def test_geographic_split_is_reproducible_and_disjoint():
    X = sample_X(200)
    y = pd.Series(np.arange(len(X), dtype=float))
    train, holdout, groups = split_by_geography(X, y, seed=42)
    train2, holdout2, _ = split_by_geography(X, y, seed=42)
    assert np.array_equal(train, train2)
    assert np.array_equal(holdout, holdout2)
    assert not set(groups.iloc[train]).intersection(groups.iloc[holdout])
    assert len(holdout) > 0 and len(train) > 0


def test_regression_metrics_and_models_are_well_formed():
    result = regression_metrics([1, 2, 3], [1, 2, 4])
    assert result["rmse_100k_usd"] == pytest.approx(np.sqrt(1 / 3))
    assert result["mae_100k_usd"] == pytest.approx(1 / 3)
    assert set(build_models()) == {"median_baseline", "ridge", "hist_gradient_boosting", "small_mlp"}
