"""Run sequence-window CNN RUL baseline on FD001."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

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
from sensor_rul.models.cnn import CNNRULRegressor
from sensor_rul.training.train_regressor import (
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    fit_regressor,
)
from sensor_rul.visualization.rul_diagnostics import (
    plot_engine_trajectories,
    plot_error_vs_true_rul,
    plot_predicted_vs_true,
    plot_training_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequence-window CNN RUL baseline on FD001."
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
        default=PROJECT_ROOT / "results" / "cnn",
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
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5e-3,
        help="Adam learning rate.",
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
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping on validation RMSE.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Number of epochs without meaningful validation RMSE improvement before stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Minimum validation RMSE improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--restore-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the best validation checkpoint before final prediction/evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_loader(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect model predictions for every batch in a DataLoader."""
    model.eval()
    predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            batch_pred = model(batch_x).cpu().numpy()
            predictions.append(batch_pred)

    return np.concatenate(predictions)


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
    result: dict[str, float | str | int | bool],
    csv_path: Path,
    json_path: Path,
    history_path: Path,
    history_csv_path: Path,
    predictions_path: Path,
    plots_dir: Path,
    plot_prefix: str,
) -> None:
    print("Sequence-window CNN RUL baseline (FD001)")
    print(f"  data dir:       {args.data_dir}")
    print(f"  window size:    {args.window_size}")
    print(f"  stride:         {args.stride}")
    print(f"  batch size:     {args.batch_size}")
    print(f"  epochs:         {args.epochs}")
    print(f"  lr:             {args.lr}")
    print(f"  early stopping: {args.early_stopping}")
    print(f"  early patience: {args.early_stop_patience}")
    print(f"  early min delta: {args.early_stop_min_delta}")
    print(f"  restore best:   {args.restore_best}")
    print(f"  trained epochs: {result['trained_epochs']}")
    print(f"  best epoch:     {result['best_epoch']}")
    print(f"  val fraction:   {args.val_fraction}")
    print(f"  seed:           {args.seed}")
    print(f"  train engines:  {len(train_engine_ids)}")
    print(f"  val engines:    {len(val_engine_ids)}")
    print(f"  train windows:  {len(y_train)}")
    print(f"  val windows:    {len(y_val)}")
    print()
    print(f"  num features:   {result['num_features']}")
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
    print(f"Saved CSV:          {csv_path}")
    print(f"Saved JSON:         {json_path}")
    print(f"Saved history JSON: {history_path}")
    print(f"Saved history CSV:  {history_csv_path}")
    print(f"Saved predictions:  {predictions_path}")
    print(f"Saved checkpoint:   {result['checkpoint_path']}")
    print(f"Saved plots:        {plots_dir}")
    for plot_path in sorted(plots_dir.glob(f"{plot_prefix}_*.png")):
        print(f"  {plot_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
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

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_eval_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CNNRULRegressor(
        window_size=args.window_size,
        num_features=num_features,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")
    model, history = fit_regressor(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        early_stopping_patience=(
            args.early_stop_patience if args.early_stopping else None
        ),
        early_stopping_min_delta=args.early_stop_min_delta,
        restore_best=args.restore_best,
    )

    best_row = min(history, key=lambda row: row["val_rmse"])
    trained_epochs = len(history)

    y_train_pred = predict_loader(model, train_eval_loader, device)
    y_val_pred = predict_loader(model, val_loader, device)

    train_predictions = pd.DataFrame(
        {
            "split": "train",
            "engine_id": train_windows.engine_ids,
            "end_cycle": train_windows.end_cycles,
            "true_rul": y_train,
            "pred_rul": y_train_pred,
            "error": y_train_pred - y_train,
        }
    )
    val_predictions = pd.DataFrame(
        {
            "split": "val",
            "engine_id": val_windows.engine_ids,
            "end_cycle": val_windows.end_cycles,
            "true_rul": y_val,
            "pred_rul": y_val_pred,
            "error": y_val_pred - y_val,
        }
    )
    predictions_df = pd.concat([train_predictions, val_predictions], ignore_index=True)

    result: dict[str, float | str | int | bool] = {
        "model": "cnn",
        "window_size": args.window_size,
        "stride": args.stride,
        "num_features": num_features,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stop_patience,
        "early_stopping_min_delta": args.early_stop_min_delta,
        "restore_best": args.restore_best,
        "trained_epochs": trained_epochs,
        "best_epoch": int(best_row["epoch"]),
        "best_val_rmse": float(best_row["val_rmse"]),
        "best_val_mae": float(best_row["val_mae"]),
        "seed": args.seed,
        "device": str(device),
        "train_rmse": float(rmse(y_train, y_train_pred)),
        "train_mae": float(mae(y_train, y_train_pred)),
        "val_rmse": float(rmse(y_val, y_val_pred)),
        "val_mae": float(mae(y_val, y_val_pred)),
    }

    output_stem = f"cnn_fd001"
    plot_prefix = output_stem
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    csv_path = args.results_dir / f"{output_stem}.csv"
    json_path = args.results_dir / f"{output_stem}.json"
    history_path = args.results_dir / f"{output_stem}_history.json"
    history_csv_path = args.results_dir / f"{output_stem}_history.csv"
    predictions_path = args.results_dir / f"{output_stem}_predictions.csv"
    checkpoint_path = args.results_dir / f"{output_stem}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    result["checkpoint_path"] = str(checkpoint_path)

    pd.DataFrame([result]).to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([result], f, indent=2)
        f.write("\n")

    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        f.write("\n")

    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    plot_training_history(history, plots_dir, plot_prefix)
    plot_predicted_vs_true(predictions_df, plots_dir, plot_prefix, split="val")
    plot_error_vs_true_rul(predictions_df, plots_dir, plot_prefix, split="val")
    plot_binned_error_vs_true_rul(predictions_df, plots_dir, plot_prefix, split="val")
    plot_engine_trajectories(predictions_df, plots_dir, plot_prefix, split="val")

    print_summary(
        args,
        train_engine_ids,
        val_engine_ids,
        y_train,
        y_val,
        result,
        csv_path,
        json_path,
        history_path,
        history_csv_path,
        predictions_path,
        plots_dir,
        plot_prefix,
    )


if __name__ == "__main__":
    main()
