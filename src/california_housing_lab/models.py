"""Comparable, reproducible model pipelines."""

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import HousingFeatureBuilder


def make_pipeline(regressor, scale: bool):
    steps = [("features", HousingFeatureBuilder()), ("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("regressor", regressor))
    return Pipeline(steps)


def build_models(seed: int = 42):
    """Return baseline, regularized linear, tree ensemble, and compact neural models."""
    return {
        "median_baseline": make_pipeline(DummyRegressor(strategy="median"), scale=False),
        "ridge": make_pipeline(Ridge(alpha=10.0), scale=True),
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=220,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=seed,
            ),
            scale=False,
        ),
        "small_mlp": make_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.002,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            ),
            scale=True,
        ),
    }
