"""Run all uncapped (raw RUL target) FD001 experiments."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from sensor_rul.training.train_regressor import (
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ExperimentSpec = tuple[
    str,
    str,
    Callable[[argparse.Namespace], list[str]],
    str | None,
]

UNCAPPED_EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    (
        "naive baselines",
        "sensor_rul.experiments.run_naive_baselines",
        lambda args: [
            "--results-dir",
            str(args.baselines_dir),
        ],
        None,
    ),
    (
        "cycle baseline",
        "sensor_rul.experiments.run_cycle_baseline",
        lambda args: [
            "--results-dir",
            str(args.baselines_dir),
        ],
        None,
    ),
    (
        "ridge baseline",
        "sensor_rul.experiments.run_ridge_baseline",
        lambda args: [
            "--results-dir",
            str(args.baselines_dir),
            "--alpha",
            str(args.ridge_alpha),
        ],
        None,
    ),
    (
        "MLP",
        "sensor_rul.experiments.run_mlp_baseline",
        lambda args: [
            "--results-dir",
            str(args.baselines_dir),
            *deep_learning_args(args),
            *early_stopping_args(args),
        ],
        "mlp_baseline_fd001_raw",
    ),
    (
        "GRU",
        "sensor_rul.experiments.run_gru",
        lambda args: [
            "--results-dir",
            str(args.gru_dir),
            *deep_learning_args(args),
            "--num-layers",
            str(args.num_layers),
            *early_stopping_args(args),
        ],
        "gru_fd001",
    ),
    (
        "LSTM",
        "sensor_rul.experiments.run_lstm",
        lambda args: [
            "--results-dir",
            str(args.lstm_dir),
            *deep_learning_args(args),
            "--num-layers",
            str(args.num_layers),
            "--grad-clip-max-norm",
            str(args.grad_clip_max_norm),
            *early_stopping_args(args),
        ],
        "lstm_fd001_raw",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all uncapped (raw RUL target) FD001 experiments."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cmapss-data" / "raw",
        help="Directory containing C-MAPSS FD001 raw files.",
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines",
        help="Directory for baseline and MLP result files.",
    )
    parser.add_argument(
        "--gru-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gru",
        help="Directory for GRU result files.",
    )
    parser.add_argument(
        "--lstm-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lstm",
        help="Directory for LSTM result files.",
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
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden size for MLP, GRU, and LSTM models.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of stacked recurrent layers for GRU and LSTM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for deep models.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs for deep models.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for deep models.",
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for LSTM training.",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable early stopping on validation RMSE for deep models.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="Epochs without meaningful validation RMSE improvement before stopping.",
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
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength.",
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
    return parser.parse_args()


def common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-dir",
        str(args.data_dir),
        "--window-size",
        str(args.window_size),
        "--stride",
        str(args.stride),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
    ]


def deep_learning_args(args: argparse.Namespace) -> list[str]:
    return [
        "--hidden-dim",
        str(args.hidden_dim),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
    ]


def early_stopping_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
    ]
    cli_args.append("--early-stopping" if args.early_stopping else "--no-early-stopping")
    cli_args.append("--restore-best" if args.restore_best else "--no-restore-best")
    return cli_args


def build_command(
    module: str,
    extra_args: list[str],
    args: argparse.Namespace,
) -> list[str]:
    return [sys.executable, "-m", module, *common_args(args), *extra_args]


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_dir = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir
    return env


def epoch_output_suffix(epochs: int) -> str:
    return f"_epochs_{epochs}"


def renamed_output_name(stem: str, filename: str, epochs: int) -> str | None:
    suffix = epoch_output_suffix(epochs)
    if filename.startswith(f"{stem}{suffix}"):
        return None
    if filename == f"{stem}.csv" or filename == f"{stem}.json" or filename == f"{stem}.pt":
        return f"{stem}{suffix}{filename[len(stem):]}"
    if filename.startswith(f"{stem}_"):
        return f"{stem}{suffix}{filename[len(stem):]}"
    return None


def rename_epoch_outputs(results_dir: Path, stem: str, epochs: int) -> None:
    """Append _epochs_{epochs} to saved artifact names for a deep model run.

    Overwrites existing suffixed targets from prior runs.
    """
    plots_dir = results_dir / "plots"
    renamed_paths: list[tuple[Path, Path]] = []

    for directory in (results_dir, plots_dir):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            new_name = renamed_output_name(stem, path.name, epochs)
            if new_name is None or new_name == path.name:
                continue
            renamed_paths.append((path, path.with_name(new_name)))

    for source_path, target_path in renamed_paths:
        if target_path.exists():
            target_path.unlink()
        source_path.rename(target_path)


def run_experiment(
    label: str,
    module: str,
    extra_args: list[str],
    output_stem: str | None,
    args: argparse.Namespace,
) -> None:
    command = build_command(module, extra_args, args)
    print("=" * 72)
    print(f"Running uncapped experiment: {label}")
    print(f"Command: {' '.join(command)}")
    print("=" * 72)
    subprocess.run(command, check=True, env=subprocess_env())

    if output_stem is not None:
        results_dir = Path(extra_args[extra_args.index("--results-dir") + 1])
        rename_epoch_outputs(results_dir, output_stem, args.epochs)
        print(
            f"Renamed outputs with suffix {epoch_output_suffix(args.epochs)} "
            f"in {results_dir}"
        )


def main() -> None:
    args = parse_args()

    print("Uncapped FD001 experiments (raw RUL targets)")
    print(f"  data dir:       {args.data_dir}")
    print(f"  baselines dir:  {args.baselines_dir}")
    print(f"  gru dir:        {args.gru_dir}")
    print(f"  lstm dir:       {args.lstm_dir}")
    print(f"  window size:    {args.window_size}")
    print(f"  stride:         {args.stride}")
    print(f"  val fraction:   {args.val_fraction}")
    print(f"  max epochs:     {args.epochs}")
    print(f"  grad clip:      {args.grad_clip_max_norm}")
    print(f"  early stopping: {args.early_stopping}")
    print(f"  early patience: {args.early_stop_patience}")
    print(f"  early min delta: {args.early_stop_min_delta}")
    print(f"  restore best:   {args.restore_best}")
    print(f"  seed:           {args.seed}")
    print()

    for label, module, extra_args_builder, output_stem in UNCAPPED_EXPERIMENTS:
        run_experiment(label, module, extra_args_builder(args), output_stem, args)

    print()
    print("Finished all uncapped experiments.")


if __name__ == "__main__":
    main()
