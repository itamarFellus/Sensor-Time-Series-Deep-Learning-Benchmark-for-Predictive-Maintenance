from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from sensor_rul.data.cmapss import OP_SETTING_COLUMNS, SENSOR_COLUMNS

STD_EPS = 1e-6

DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = tuple(OP_SETTING_COLUMNS + SENSOR_COLUMNS)


@dataclass(frozen=True)
class FeatureStandardizer:
    """Fitted mean, std, and safe_std for feature-column z-scoring."""

    feature_columns: tuple[str, ...]
    mean: pd.Series
    std: pd.Series
    safe_std: pd.Series


def fit_standardizer(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> FeatureStandardizer:
    """
    Fit a standardizer on train_df only.

    Uses population std (ddof=0). Near-zero std columns use safe_std=1.0
    so constant features become 0 after normalization.
    """
    cols = tuple(feature_columns or DEFAULT_FEATURE_COLUMNS)

    missing = [col for col in cols if col not in train_df.columns]
    if missing:
        raise ValueError(f"train_df is missing feature columns: {missing}")

    features = train_df[list(cols)]
    mean = features.mean()
    std = features.std(ddof=0)
    safe_std = std.where(std > STD_EPS, 1.0)

    return FeatureStandardizer(
        feature_columns=cols,
        mean=mean,
        std=std,
        safe_std=safe_std,
    )


def apply_standardizer(
    df: pd.DataFrame,
    standardizer: FeatureStandardizer,
) -> pd.DataFrame:
    """Apply a fitted standardizer; returns a copy without recomputing stats."""
    out = df.copy()
    cols = list(standardizer.feature_columns)
    out[cols] = (out[cols] - standardizer.mean) / standardizer.safe_std
    return out
