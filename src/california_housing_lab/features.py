"""Feature construction using only the supplied census predictors."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HousingFeatureBuilder(TransformerMixin, BaseEstimator):
    """Add interpretable household ratios without changing input columns."""

    def fit(self, X: pd.DataFrame, y=None):
        required = {"AveRooms", "AveBedrms", "AveOccup"}
        missing = required.difference(X.columns)
        if missing:
            raise ValueError(f"Missing required predictors: {sorted(missing)}")
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.loc[:, self.feature_names_in_].copy()
        result["RoomsPerOccupant"] = result["AveRooms"] / result["AveOccup"].clip(lower=1e-6)
        result["BedroomsPerRoom"] = result["AveBedrms"] / result["AveRooms"].clip(lower=1e-6)
        return result

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return names + ["RoomsPerOccupant", "BedroomsPerRoom"]
