from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

class ConstantRULBaseline:
    """Baseline that always predicts a constant RUL value."""

    def __init__(self, strategy: str) -> None:
        if strategy not in {"mean", "median"}:
            raise ValueError("strategy must be either 'mean' or 'median'")

        self.strategy = strategy
        self.constant_: float | None = None

    def fit(self, y_train: np.ndarray) -> "ConstantRULBaseline":
        """Compute the constant prediction from training targets only."""
        if self.strategy == "mean":
            self.constant_ = float(np.mean(y_train))
        elif self.strategy == "median":
            self.constant_ = float(np.median(y_train))

        return self

    def predict(self, num_samples: int) -> np.ndarray:
        """Return the same RUL prediction for every sample."""
        if self.constant_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        return np.full(shape=num_samples, fill_value=self.constant_, dtype=np.float32)


class CycleOnlyBaseline:
    """Linear regression baseline using only the final cycle of each window."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, cycles: np.ndarray, y: np.ndarray) -> "CycleOnlyBaseline":
        cycles = cycles.reshape(-1, 1)
        self.model.fit(cycles, y)
        return self

    def predict(self, cycles: np.ndarray) -> np.ndarray:
        cycles = cycles.reshape(-1, 1)
        return self.model.predict(cycles)


class FlattenedWindowRidgeBaseline:
    """Ridge regression baseline on flattened time-series windows."""

    def __init__(self) -> None:
        self.model = Ridge()

    def fit(self, windows: np.ndarray, y: np.ndarray) -> "FlattenedWindowRidgeBaseline":
        X = windows.reshape(windows.shape[0], -1)
        self.model.fit(X, y)
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        X = windows.reshape(windows.shape[0], -1)
        return self.model.predict(X)