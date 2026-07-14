"""Run mean and median constant RUL baselines on FD001."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sensor_rul.evaluation.metrics import mae, rmse
from sensor_rul.models.baselines import ConstantRULBaseline
from sensor_rul.data.cmapss import load_fd001
from sensor_rul.data.normalization import (
    DEFAULT_FEATURE_COLUMNS,
    apply_standardizer,
    fit_standardizer,
)
from sensor_rul.data.splits import (
    check_no_engine_overlap,
    split_by_engine_ids,
    split_engine_ids,
)
from sensor_rul.data.windowing import make_windows

BASELINES = (
    ("mean_rul", "mean"),
    ("median_rul", "median"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run naive constant RUL baselines on FD001."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cmapss-data" / "raw",
        help="Directory containing C-MAPSS FD001 raw files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines",
        help="Directory for baseline result files.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Number of cycles per input window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride between consecutive windows.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of engines reserved for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for engine-based splitting.",
    )
    return parser.parse_args()


def evaluate_baseline(
    model_name: str,
    strategy: str,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float | str]:
    """Fit a constant baseline on y_train and score train/validation labels."""
    baseline = ConstantRULBaseline(strategy=strategy)
    baseline.fit(y_train)

    y_train_pred = baseline.predict(len(y_train))
    y_val_pred = baseline.predict(len(y_val))

    return {
        "model": model_name,
        "strategy": strategy,
        "constant_prediction": baseline.constant_,
        "train_rmse": rmse(y_train, y_train_pred),
        "train_mae": mae(y_train, y_train_pred),
        "val_rmse": rmse(y_val, y_val_pred),
        "val_mae": mae(y_val, y_val_pred),
    }


def print_summary(
    args: argparse.Namespace,
    train_engine_ids: np.ndarray,
    val_engine_ids: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    results: list[dict[str, float | str]],
    csv_path: Path,
    json_path: Path,
) -> None:
    print("Naive constant RUL baselines (FD001)")
    print(f"  data dir:       {args.data_dir}")
    print(f"  window size:    {args.window_size}")
    print(f"  stride:         {args.stride}")
    print(f"  val fraction:   {args.val_fraction}")
    print(f"  seed:           {args.seed}")
    print(f"  train engines:  {len(train_engine_ids)}")
    print(f"  val engines:    {len(val_engine_ids)}")
    print(f"  train windows:  {len(y_train)}")
    print(f"  val windows:    {len(y_val)}")
    print()
    print(f"{'model':<12} {'strategy':<8} {'constant':>10} "
          f"{'train_rmse':>12} {'train_mae':>11} {'val_rmse':>10} {'val_mae':>9}")
    print("-" * 78)
    for row in results:
        print(
            f"{row['model']:<12} {row['strategy']:<8} {row['constant_prediction']:>10.4f} "
            f"{row['train_rmse']:>12.4f} {row['train_mae']:>11.4f} "
            f"{row['val_rmse']:>10.4f} {row['val_mae']:>9.4f}"
        )
    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> None:
    args = parse_args()
    feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    train_df, _, _ = load_fd001(args.data_dir)
    train_engine_ids, val_engine_ids = split_engine_ids(
        train_df,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    check_no_engine_overlap(train_engine_ids, val_engine_ids)
    train_split, val_split = split_by_engine_ids(
        train_df, train_engine_ids, val_engine_ids
    )

    standardizer = fit_standardizer(train_split, feature_columns=feature_columns)
    train_norm = apply_standardizer(train_split, standardizer)
    val_norm = apply_standardizer(val_split, standardizer)

    train_windows = make_windows(
        train_norm,
        feature_columns=feature_columns,
        window_size=args.window_size,
        stride=args.stride,
    )
    val_windows = make_windows(
        val_norm,
        feature_columns=feature_columns,
        window_size=args.window_size,
        stride=args.stride,
    )

    y_train = train_windows.labels
    y_val = val_windows.labels

    results = [
        evaluate_baseline(model_name, strategy, y_train, y_val)
        for model_name, strategy in BASELINES
    ]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "naive_baselines_fd001.csv"
    json_path = args.results_dir / "naive_baselines_fd001.json"

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print_summary(
        args,
        train_engine_ids,
        val_engine_ids,
        y_train,
        y_val,
        results,
        csv_path,
        json_path,
    )


if __name__ == "__main__":
    main()
