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
