"""Reusable matplotlib plots for RUL regression diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(output_dir: Path | str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_dataframe(history: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        return history
    return pd.DataFrame(history)


def _plot_train_val_curves(
    df: pd.DataFrame,
    train_col: str,
    val_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["epoch"], df[train_col], label="train")
    ax.plot(df["epoch"], df[val_col], label="val")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_history(
    history: list[dict[str, Any]] | pd.DataFrame,
    output_dir: Path | str,
    prefix: str,
) -> None:
    """Save loss/RMSE/MAE training curves when the corresponding columns exist."""
    out_dir = _ensure_output_dir(output_dir)
    df = _as_dataframe(history)

    if {"train_loss", "val_loss"}.issubset(df.columns):
        _plot_train_val_curves(
            df,
            train_col="train_loss",
            val_col="val_loss",
            output_path=out_dir / f"{prefix}_loss_curve.png",
            title="Training and validation loss",
            ylabel="loss",
        )

    if {"train_rmse", "val_rmse"}.issubset(df.columns):
        _plot_train_val_curves(
            df,
            train_col="train_rmse",
            val_col="val_rmse",
            output_path=out_dir / f"{prefix}_rmse_curve.png",
            title="Training and validation RMSE",
            ylabel="RMSE",
        )

    if {"train_mae", "val_mae"}.issubset(df.columns):
        _plot_train_val_curves(
            df,
            train_col="train_mae",
            val_col="val_mae",
            output_path=out_dir / f"{prefix}_mae_curve.png",
            title="Training and validation MAE",
            ylabel="MAE",
        )


def _filter_split(predictions_df: pd.DataFrame, split: str) -> pd.DataFrame:
    return predictions_df.loc[predictions_df["split"] == split].copy()


def plot_predicted_vs_true(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str,
    split: str = "val",
) -> None:
    """Scatter predicted RUL against true RUL with an ideal y=x reference line."""
    out_dir = _ensure_output_dir(output_dir)
    split_df = _filter_split(predictions_df, split)

    fig, ax = plt.subplots()
    ax.scatter(split_df["true_rul"], split_df["pred_rul"], alpha=0.5, s=12)

    lo = min(split_df["true_rul"].min(), split_df["pred_rul"].min())
    hi = max(split_df["true_rul"].max(), split_df["pred_rul"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="y = x")

    ax.set_title(f"Predicted vs true RUL ({split})")
    ax.set_xlabel("true_rul")
    ax.set_ylabel("pred_rul")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{split}_predicted_vs_true.png")
    plt.close(fig)


def plot_error_vs_true_rul(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str,
    split: str = "val",
) -> None:
    """Scatter prediction error against true RUL with a zero-error reference line."""
    out_dir = _ensure_output_dir(output_dir)
    split_df = _filter_split(predictions_df, split)

    fig, ax = plt.subplots()
    ax.scatter(split_df["true_rul"], split_df["error"], alpha=0.5, s=12)
    ax.axhline(0.0, linestyle="--", color="black", label="error = 0")

    ax.set_title(f"Prediction error vs true RUL ({split})")
    ax.set_xlabel("true_rul")
    ax.set_ylabel("error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_{split}_error_vs_true_rul.png")
    plt.close(fig)


def plot_engine_trajectories(
    predictions_df: pd.DataFrame,
    output_dir: Path | str,
    prefix: str,
    split: str = "val",
    max_engines: int = 5,
) -> None:
    """Plot true and predicted RUL trajectories for up to max_engines engines."""
    out_dir = _ensure_output_dir(output_dir)
    split_df = _filter_split(predictions_df, split)
    engine_ids = sorted(split_df["engine_id"].unique())[:max_engines]

    for engine_id in engine_ids:
        engine_df = split_df.loc[split_df["engine_id"] == engine_id].sort_values(
            "end_cycle"
        )

        fig, ax = plt.subplots()
        ax.plot(
            engine_df["end_cycle"],
            engine_df["true_rul"],
            label="true_rul",
            marker="o",
            markersize=3,
        )
        ax.plot(
            engine_df["end_cycle"],
            engine_df["pred_rul"],
            label="pred_rul",
            marker="o",
            markersize=3,
        )

        ax.set_title(f"Engine {engine_id} RUL trajectory ({split})")
        ax.set_xlabel("end_cycle")
        ax.set_ylabel("RUL")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{split}_engine_{engine_id}_trajectory.png")
        plt.close(fig)
