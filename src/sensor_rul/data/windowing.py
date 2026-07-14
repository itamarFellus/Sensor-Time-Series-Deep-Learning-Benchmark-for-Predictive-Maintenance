from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sensor_rul.data.cmapss import TARGET_COLUMN
from sensor_rul.data.normalization import DEFAULT_FEATURE_COLUMNS

WINDOW_SIZE = 30


@dataclass(frozen=True)
class WindowBatch:
    """Fixed-length supervised windows for RUL prediction."""

    windows: np.ndarray  # (n_windows, window_size, n_features)
    labels: np.ndarray  # (n_windows,)
    engine_ids: np.ndarray  # (n_windows,)
    end_cycles: np.ndarray  # (n_windows,)


def _validate_inputs(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    window_size: int,
    stride: int,
) -> None:
    if "engine_id" not in df.columns:
        raise ValueError("DataFrame must contain an 'engine_id' column.")
    if "cycle" not in df.columns:
        raise ValueError("DataFrame must contain a 'cycle' column.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain a '{TARGET_COLUMN}' column.")
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing feature columns: {missing}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")


def _windows_for_engine(
    engine_df: pd.DataFrame,
    engine_id: int,
    feature_columns: tuple[str, ...],
    window_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[float], list[int], list[int]]:
    engine_df = engine_df.sort_values("cycle")

    features = engine_df[list(feature_columns)].to_numpy(dtype=np.float32)
    labels = engine_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    cycles = engine_df["cycle"].to_numpy()

    n = len(engine_df)
    if n < window_size:
        return [], [], [], []

    windows: list[np.ndarray] = []
    window_labels: list[float] = []
    window_engine_ids: list[int] = []
    window_end_cycles: list[int] = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(features[start:end])
        window_labels.append(float(labels[end - 1]))
        window_engine_ids.append(engine_id)
        window_end_cycles.append(int(cycles[end - 1]))

    return windows, window_labels, window_engine_ids, window_end_cycles


def _empty_batch(window_size: int, n_features: int) -> WindowBatch:
    return WindowBatch(
        windows=np.empty((0, window_size, n_features), dtype=np.float32),
        labels=np.empty((0,), dtype=np.float32),
        engine_ids=np.empty((0,), dtype=np.int64),
        end_cycles=np.empty((0,), dtype=np.int64),
    )


def make_windows(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int = 1,
) -> WindowBatch:
    """
    Build fixed-length supervised windows from a C-MAPSS DataFrame.

    Each window uses cycles [t - window_size + 1, ..., t] and predicts
    RUL at cycle t. Windows are created per engine and never cross engines.
    Engines with fewer than window_size cycles are skipped.
    """
    cols = tuple(feature_columns or DEFAULT_FEATURE_COLUMNS)
    _validate_inputs(df, cols, window_size, stride)

    all_windows: list[np.ndarray] = []
    all_labels: list[float] = []
    all_engine_ids: list[int] = []
    all_end_cycles: list[int] = []

    for engine_id, engine_df in df.groupby("engine_id", sort=False):
        windows, labels, engine_ids, end_cycles = _windows_for_engine(
            engine_df,
            int(engine_id),
            cols,
            window_size,
            stride,
        )
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_engine_ids.extend(engine_ids)
        all_end_cycles.extend(end_cycles)

    if not all_windows:
        return _empty_batch(window_size, len(cols))

    return WindowBatch(
        windows=np.stack(all_windows, axis=0),
        labels=np.asarray(all_labels, dtype=np.float32),
        engine_ids=np.asarray(all_engine_ids, dtype=np.int64),
        end_cycles=np.asarray(all_end_cycles, dtype=np.int64),
    )


if __name__ == "__main__":
    from pathlib import Path

    from sensor_rul.data.cmapss import load_fd001
    from sensor_rul.data.splits import split_by_engine_ids, split_engine_ids

    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "cmapss-data" / "raw"
    window_size = WINDOW_SIZE
    feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    train_df, _, _ = load_fd001(data_dir)
    train_engine_ids, val_engine_ids = split_engine_ids(train_df, val_fraction=0.2, seed=42)
    train_split, _ = split_by_engine_ids(train_df, train_engine_ids, val_engine_ids)

    batch = make_windows(train_split, feature_columns=feature_columns, window_size=window_size)

    assert batch.windows.ndim == 3
    assert batch.windows.shape[1] == window_size
    assert batch.windows.shape[2] == len(feature_columns)
    assert batch.labels.shape == (batch.windows.shape[0],)
    assert batch.engine_ids.shape == (batch.windows.shape[0],)
    assert batch.end_cycles.shape == (batch.windows.shape[0],)

    sample_engine_id = int(train_split["engine_id"].iloc[0])
    engine_rows = train_split[train_split["engine_id"] == sample_engine_id]
    n_cycles = len(engine_rows)
    expected_windows = max(0, n_cycles - window_size + 1)
    actual_windows = int((batch.engine_ids == sample_engine_id).sum())
    assert actual_windows == expected_windows, (
        f"engine {sample_engine_id}: expected {expected_windows} windows, got {actual_windows}"
    )

    print("PASS: windowing self-check completed successfully.")
    print(f"  windows shape: {batch.windows.shape}")
    print(f"  labels shape:  {batch.labels.shape}")
