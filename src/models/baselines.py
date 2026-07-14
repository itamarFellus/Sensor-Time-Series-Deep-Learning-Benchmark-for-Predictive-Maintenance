import numpy as np


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