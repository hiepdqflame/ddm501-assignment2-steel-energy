"""The ten predeclared configurations and one shared prediction interface."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from steel_energy.data import feature_columns


class LagRegressor(RegressorMixin, BaseEstimator):
    """A deterministic baseline compatible with sklearn and MLflow serialization."""

    def __init__(self, lag: int = 2):
        self.lag = lag

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = len(X.columns)
        return self

    def predict(self, X):
        return X[f"lag_{self.lag}"].to_numpy()


def make_model(spec: dict, seed: int = 42):
    """Build a CPU estimator; disable random internal validation for time series."""
    params = {k: v for k, v in spec.items() if k not in {"id", "kind", "features"}}
    if spec["kind"] == "baseline":
        return LagRegressor(**params)
    if spec["kind"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if spec["kind"] == "rf":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=1)
    if spec["kind"] == "hgb":
        return HistGradientBoostingRegressor(
            **params, random_state=seed, early_stopping=False
        )
    raise ValueError("Unknown model kind")


def predict(model, frame, spec: dict) -> np.ndarray:
    """Use the exact feature whitelist and non-negative energy rule everywhere."""
    return np.maximum(0, model.predict(frame[feature_columns(spec["features"])]))
