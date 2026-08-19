"""Run flattened-window Ridge RUL baseline on FD001."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
from sensor_rul.evaluation.metrics import mae, rmse
from sensor_rul.models.baselines import FlattenedWindowRidgeBaseline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flattened-window Ridge RUL baseline on FD001."
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
    )
    return parser.parse_args()


def evaluate_ridge_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    window_size: int,
    stride: int,
    num_features: int,
    alpha: float,
) -> tuple[dict[str, float | str | int], FlattenedWindowRidgeBaseline]:
    """Fit flattened-window Ridge on train windows and score train/validation."""
    baseline = FlattenedWindowRidgeBaseline()
    baseline.model.set_params(alpha=alpha)
    baseline.fit(x_train, y_train)

    y_train_pred = baseline.predict(x_train)
    y_val_pred = baseline.predict(x_val)

    result = {
        "model": "ridge",
        "alpha": float(baseline.model.alpha),
        "window_size": window_size,
        "stride": stride,
        "num_features": num_features,
        "flattened_dim": int(x_train.shape[1] * x_train.shape[2]),
        "train_rmse": rmse(y_train, y_train_pred),
        "train_mae": mae(y_train, y_train_pred),
        "val_rmse": rmse(y_val, y_val_pred),
        "val_mae": mae(y_val, y_val_pred),
    }
    return result, baseline


def plot_binned_error_vs_true_rul(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str,
    split: str = "val",
) -> None:
    """Plot median prediction error by true-RUL bin with an interquartile range."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_df = predictions_df.loc[predictions_df["split"] == split].copy()

    bin_edges = [0, 25, 50, 75, 100, 125, 150, 175, 200, np.inf]
    bin_labels = [
        "0-25",
        "25-50",
        "50-75",
        "75-100",
        "100-125",
        "125-150",
        "150-175",
        "175-200",
        "200+",
    ]

    split_df["rul_bin"] = pd.cut(
        split_df["true_rul"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    medians: list[float] = []
    q25s: list[float] = []
    q75s: list[float] = []
    labels: list[str] = []
    for label in bin_labels:
        errors = split_df.loc[split_df["rul_bin"] == label, "error"]
        if errors.empty:
            continue
        labels.append(label)
        medians.append(float(errors.median()))
        q25s.append(float(errors.quantile(0.25)))
        q75s.append(float(errors.quantile(0.75)))

    if not labels:
        return

    x = np.arange(len(labels))
    medians_arr = np.array(medians)
    q25_arr = np.array(q25s)
    q75_arr = np.array(q75s)

    fig, ax = plt.subplots()
    ax.axhline(0.0, linestyle="--", color="black", label="error = 0")
    ax.fill_between(
        x,
        q25_arr,
        q75_arr,
        alpha=0.3,
        label="IQR (25th-75th)",
    )
    ax.plot(x, medians_arr, marker="o", label="median error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Binned prediction error vs true RUL ({split})")
    ax.set_xlabel("true RUL bin")
    ax.set_ylabel("error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{split}_binned_error_vs_true_rul.png")
    plt.close(fig)


def print_summary(
    args: argparse.Namespace,
    train_engine_ids: np.ndarray,
    val_engine_ids: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    result: dict[str, float | str | int],
    csv_path: Path,
    json_path: Path,
) -> None:
    print("Flattened-window Ridge RUL baseline (FD001)")
    print(f"  data dir:       {args.data_dir}")
    print(f"  window size:    {args.window_size}")
    print(f"  stride:         {args.stride}")
    print(f"  alpha:          {args.alpha}")
    print(f"  val fraction:   {args.val_fraction}")
    print(f"  seed:           {args.seed}")
    print(f"  train engines:  {len(train_engine_ids)}")
    print(f"  val engines:    {len(val_engine_ids)}")
    print(f"  train windows:  {len(y_train)}")
    print(f"  val windows:    {len(y_val)}")
    print()
    print(f"  num features:   {result['num_features']}")
    print(f"  flattened dim:  {result['flattened_dim']}")
    print()
    print(
        f"{'model':<12} {'train_rmse':>12} {'train_mae':>11} "
        f"{'val_rmse':>10} {'val_mae':>9}"
    )
    print("-" * 58)
    print(
        f"{result['model']:<12} {result['train_rmse']:>12.4f} {result['train_mae']:>11.4f} "
        f"{result['val_rmse']:>10.4f} {result['val_mae']:>9.4f}"
    )
    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


def main() -> None:
    args = parse_args()
    feature_columns = list(DEFAULT_FEATURE_COLUMNS)
    num_features = len(feature_columns)

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

    x_train = train_windows.windows
    y_train = train_windows.labels
    x_val = val_windows.windows
    y_val = val_windows.labels

    result, baseline = evaluate_ridge_baseline(
        x_train,
        y_train,
        x_val,
        y_val,
        window_size=args.window_size,
        stride=args.stride,
        num_features=num_features,
        alpha=args.alpha,
    )

    y_train_pred = baseline.predict(x_train)
    y_val_pred = baseline.predict(x_val)
    predictions_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "train",
                    "true_rul": y_train,
                    "pred_rul": y_train_pred,
                    "error": y_train_pred - y_train,
                }
            ),
            pd.DataFrame(
                {
                    "split": "val",
                    "true_rul": y_val,
                    "pred_rul": y_val_pred,
                    "error": y_val_pred - y_val,
                }
            ),
        ],
        ignore_index=True,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plot_prefix = "ridge_baseline_fd001"
    plots_dir = args.results_dir / "plots"
    plot_binned_error_vs_true_rul(predictions_df, plots_dir, plot_prefix, split="val")

    csv_path = args.results_dir / "ridge_baseline_fd001.csv"
    json_path = args.results_dir / "ridge_baseline_fd001.json"

    results_df = pd.DataFrame([result])
    results_df.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)
        f.write("\n")

    print_summary(
        args,
        train_engine_ids,
        val_engine_ids,
        y_train,
        y_val,
        result,
        csv_path,
        json_path,
    )


if __name__ == "__main__":
    main()
