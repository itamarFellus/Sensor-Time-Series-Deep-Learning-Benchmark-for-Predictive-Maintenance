from __future__ import annotations

from pathlib import Path

import pandas as pd

from sensor_rul.data.cmapss import TARGET_COLUMN, load_fd001
from sensor_rul.data.normalization import (
    DEFAULT_FEATURE_COLUMNS,
    STD_EPS,
    apply_standardizer,
    fit_standardizer,
)
from sensor_rul.data.splits import (
    check_no_engine_overlap,
    split_by_engine_ids,
    split_engine_ids,
)

ID_TARGET_COLUMNS = ["engine_id", "cycle", TARGET_COLUMN]


def _assert_fitted_stats_match_train(
    train_split: pd.DataFrame,
    standardizer,
    feature_columns: list[str],
) -> None:
    for col in feature_columns:
        expected_mean = train_split[col].mean()
        expected_std = train_split[col].std(ddof=0)
        assert standardizer.mean[col] == expected_mean, (
            f"{col}: fitted mean {standardizer.mean[col]} != {expected_mean}"
        )
        fitted_std = standardizer.safe_std[col]
        if expected_std > STD_EPS:
            assert fitted_std == expected_std, (
                f"{col}: fitted safe_std {fitted_std} != {expected_std}"
            )
        else:
            assert fitted_std == 1.0, (
                f"{col}: expected safe_std=1.0 for near-constant feature, got {fitted_std}"
            )


def _assert_train_normalized_stats(
    train_norm: pd.DataFrame,
    train_split: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    for col in feature_columns:
        raw_std = train_split[col].std(ddof=0)
        if raw_std > STD_EPS:
            norm_mean = train_norm[col].mean()
            norm_std = train_norm[col].std(ddof=0)
            assert abs(norm_mean) < 1e-5, (
                f"{col}: normalized train mean {norm_mean} not near 0"
            )
            assert abs(norm_std - 1) < 1e-4, (
                f"{col}: normalized train std {norm_std} not near 1"
            )
        else:
            max_abs = train_norm[col].abs().max()
            assert max_abs < 1e-5, (
                f"{col}: near-constant feature should normalize to ~0, max abs={max_abs}"
            )


def _assert_id_target_unchanged(
    original: pd.DataFrame,
    normalized: pd.DataFrame,
) -> None:
    for col in ID_TARGET_COLUMNS:
        assert original[col].equals(normalized[col]), f"{col} changed after normalization"


def _print_val_normalized_stats(
    val_norm: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    print("\n=== Validation normalized feature stats (informational) ===")
    for col in feature_columns:
        mean = val_norm[col].mean()
        std = val_norm[col].std(ddof=0)
        print(f"  {col}: mean={mean:.6f}, std={std:.6f}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "cmapss-data" / "raw"

    feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    train_df, _, _ = load_fd001(data_dir)
    train_engine_ids, val_engine_ids = split_engine_ids(train_df, val_fraction=0.2, seed=42)
    check_no_engine_overlap(train_engine_ids, val_engine_ids)
    print("PASS: train/val engines are disjoint")

    train_split, val_split = split_by_engine_ids(train_df, train_engine_ids, val_engine_ids)
    train_split_orig = train_split.copy()
    val_split_orig = val_split.copy()

    standardizer = fit_standardizer(train_split, feature_columns=feature_columns)

    assert train_split.equals(train_split_orig), "train_split was mutated during fit"
    assert val_split.equals(val_split_orig), "val_split was mutated during fit"
    print("PASS: input DataFrames unchanged after fit")

    train_norm = apply_standardizer(train_split, standardizer)
    val_norm = apply_standardizer(val_split, standardizer)

    assert train_split.equals(train_split_orig), "train_split was mutated during apply"
    assert val_split.equals(val_split_orig), "val_split was mutated during apply"
    print("PASS: input DataFrames unchanged after apply")

    _assert_id_target_unchanged(train_split_orig, train_norm)
    _assert_id_target_unchanged(val_split_orig, val_norm)
    print("PASS: engine_id, cycle, rul unchanged")

    _assert_fitted_stats_match_train(train_split_orig, standardizer, feature_columns)
    print("PASS: fitted stats match train split")

    _assert_train_normalized_stats(train_norm, train_split_orig, feature_columns)
    print("PASS: train normalized feature means/stds are correct")

    near_constant = [
        col
        for col in feature_columns
        if train_split_orig[col].std(ddof=0) <= STD_EPS
    ]
    print(f"\nNormalized {len(feature_columns)} feature columns")
    print(f"Near-constant columns (std <= {STD_EPS}): {near_constant}")

    _print_val_normalized_stats(val_norm, feature_columns)
    print("\nAll normalization checks passed.")


if __name__ == "__main__":
    main()
