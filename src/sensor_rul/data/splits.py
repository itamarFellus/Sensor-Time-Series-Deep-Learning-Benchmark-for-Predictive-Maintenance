from __future__ import annotations

import numpy as np
import pandas as pd


def split_engine_ids(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split engine IDs into train and validation groups.

    The split is engine-based, not row-based.
    This prevents leakage between train and validation sets.
    """
    if "engine_id" not in df.columns:
        raise ValueError("DataFrame must contain an 'engine_id' column.")

    engine_ids = df["engine_id"].unique() # Make sure you understand this line

    rng = np.random.default_rng(seed)
    shuffled_engine_ids = rng.permutation(engine_ids)

    num_val = int(round(len(shuffled_engine_ids) * val_fraction)) # Make sure you understand this line

    val_engine_ids = shuffled_engine_ids[:num_val]
    train_engine_ids = shuffled_engine_ids[num_val:]

    return train_engine_ids, val_engine_ids


def split_by_engine_ids(
    df: pd.DataFrame,
    train_engine_ids: np.ndarray,
    val_engine_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train and validation DataFrames from engine ID groups.
    """
    train_df = df[df["engine_id"].isin(train_engine_ids)].copy()
    val_df = df[df["engine_id"].isin(val_engine_ids)].copy()

    return train_df, val_df


def check_no_engine_overlap(
    train_engine_ids: np.ndarray,
    val_engine_ids: np.ndarray,
) -> None:
    """
    Raise an error if any engine appears in both train and validation.
    """
    overlap = set(train_engine_ids) & set(val_engine_ids)

    if overlap:
        raise ValueError(f"Engine leakage detected. Overlapping engines: {overlap}")