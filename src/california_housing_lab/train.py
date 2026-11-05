"""Fetch, train, evaluate, explain, and export the complete benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from .evaluation import regression_metrics
from .models import build_models

SEED = 42


def spatial_groups(X: pd.DataFrame, tile_degrees: float = 0.5):
    """Group nearby census tracts into coarse geographic tiles for the holdout."""
    lon_bin = np.floor((X["Longitude"].to_numpy() + 180) / tile_degrees).astype(int)
    lat_bin = np.floor((X["Latitude"].to_numpy() + 90) / tile_degrees).astype(int)
    return pd.Series(lon_bin.astype(str) + ":" + lat_bin.astype(str), index=X.index, name="spatial_tile")


def split_by_geography(X, y, seed=SEED):
    groups = spatial_groups(X)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(X, y, groups))
    return train_idx, holdout_idx, groups


def run(output: Path, seed: int = SEED):
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    dataset = fetch_california_housing(as_frame=True, download_if_missing=True)
    X = dataset.data.copy()
    y = dataset.target.astype(float).copy()  # target unit: $100,000s
    train_idx, holdout_idx, groups = split_by_geography(X, y, seed)
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]

    summaries = []
    fitted = {}
    for name, model in build_models(seed).items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        predictions = model.predict(X_holdout)
        metrics = regression_metrics(y_holdout, predictions)
        model_path = output / "models" / f"{name}.joblib"
        joblib.dump(model, model_path, compress=3)
        fitted[name] = (model, np.asarray(predictions))
        summaries.append({
            "model": name,
            **metrics,
            "fit_seconds_machine_dependent": float(fit_seconds),
            "artifact_bytes": int(model_path.stat().st_size),
        })
    metrics_df = pd.DataFrame(summaries).sort_values("rmse_100k_usd")
    metrics_df.to_csv(output / "metrics.csv", index=False, float_format="%.6f")

    # Holdout is used only after model fitting for descriptive diagnostics, never tuning.
    best_name = metrics_df.iloc[0]["model"]
    best_model, best_predictions = fitted[best_name]
    errors = X_holdout.copy()
    errors.insert(0, "actual_100k_usd", np.asarray(y_holdout))
    errors.insert(1, "predicted_100k_usd", best_predictions)
    errors.insert(2, "signed_error_100k_usd", best_predictions - np.asarray(y_holdout))
    errors.insert(3, "absolute_error_100k_usd", np.abs(best_predictions - np.asarray(y_holdout)))
    errors.nlargest(20, "absolute_error_100k_usd").to_csv(output / "largest_holdout_errors.csv", index=False, float_format="%.6f")

    target_bands = pd.qcut(np.asarray(y_holdout), q=4, duplicates="drop")
    by_band = []
    for band in target_bands.categories:
        mask = np.asarray(target_bands == band)
        if mask.any():
            by_band.append({"target_band_100k_usd": str(band), "n": int(mask.sum()), **regression_metrics(np.asarray(y_holdout)[mask], best_predictions[mask])})
    pd.DataFrame(by_band).to_csv(output / "error_by_target_band.csv", index=False, float_format="%.6f")

    # Permutation importance is post-hoc explanation on the locked holdout, not selection.
    sample_n = min(1000, len(X_holdout))
    sample_idx = np.linspace(0, len(X_holdout) - 1, sample_n, dtype=int)
    importance = permutation_importance(
        best_model,
        X_holdout.iloc[sample_idx],
        y_holdout.iloc[sample_idx],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=seed,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mae_increase_100k_usd_mean": importance.importances_mean,
        "mae_increase_100k_usd_std": importance.importances_std,
    }).sort_values("mae_increase_100k_usd_mean", ascending=False)
    importance_df.to_csv(output / "permutation_importance.csv", index=False, float_format="%.6f")

    report = {
        "dataset": "scikit-learn California Housing / StatLib",
        "dataset_rows": int(len(X)),
        "raw_predictor_count": int(X.shape[1]),
        "train_rows": int(len(train_idx)),
        "holdout_rows": int(len(holdout_idx)),
        "spatial_tile_degrees": 0.5,
        "train_tiles": int(pd.Series(groups.iloc[train_idx]).nunique()),
        "holdout_tiles": int(pd.Series(groups.iloc[holdout_idx]).nunique()),
        "seed": seed,
        "split": "GroupShuffleSplit on 0.5-degree longitude/latitude tiles; disjoint tiles",
        "target_unit": "100,000 USD; source values capped at 5.00001 (approximately $500,001)",
        "best_holdout_rmse_model_for_diagnostics_only": best_name,
        "feature_importance_method": "3-repeat permutation importance on up to 1,000 holdout rows; descriptive only, not tuning",
        "sklearn_version": sklearn.__version__,
        "metrics_ordered_by_rmse": metrics_df.to_dict(orient="records"),
        "note": "Benchmark results are for coarse spatial holdout tiles; fit times and artifact sizes are environment-specific.",
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = run(args.output, args.seed)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
