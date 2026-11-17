"""Evaluation helpers kept independent of the dataset download."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return {
        "rmse_100k_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_100k_usd": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def grouped_regression_metrics(y_true, y_pred, groups):
    """Summarize regression errors per group without hiding small-group sizes."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    labels = np.asarray(groups).reshape(-1)
    if not (len(actual) == len(predicted) == len(labels)):
        raise ValueError("y_true, y_pred, and groups must have the same length")
    if len(actual) == 0:
        raise ValueError("at least one row is required")
    if any(label is None or (isinstance(label, (float, np.floating)) and np.isnan(label)) for label in labels):
        raise ValueError("group labels must not be missing")

    results = []
    for label in sorted(set(labels.tolist()), key=str):
        mask = labels == label
        if int(mask.sum()) < 2:
            residuals = predicted[mask] - actual[mask]
            metrics = {
                "rmse_100k_usd": float(np.sqrt(np.mean(residuals ** 2))),
                "mae_100k_usd": float(np.mean(np.abs(residuals))),
                "r2": None,
            }
        else:
            metrics = regression_metrics(actual[mask], predicted[mask])
        results.append({"group": str(label), "n": int(mask.sum()), **metrics})
    return results
