"""Build a comparison table from saved LSTM and GRU hidden-dim sweep results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HIDDEN_DIM_RESULT_PATTERN = re.compile(r"_hidden_dim_(\d+)\.json$")
METRIC_COLUMNS = ("train_rmse", "train_mae", "val_rmse", "val_mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison table from saved LSTM and GRU hidden-dim results."
    )
    parser.add_argument(
        "--lstm-results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lstm",
        help="Directory containing LSTM result JSON files.",
    )
    parser.add_argument(
        "--gru-results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gru",
        help="Directory containing GRU result JSON files.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables",
        help="Directory for comparison table output files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fd001",
        help="Dataset suffix used in result filenames.",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="raw",
        help="Target type suffix used in LSTM result filenames, e.g. raw or capped_125.",
    )
    return parser.parse_args()


def load_results_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected a single-result JSON list in {path}")

    result = payload[0]
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return result


def result_glob_pattern(model: str, dataset: str, target_type: str) -> str:
    if model == "lstm":
        return f"lstm_{dataset}_{target_type}_hidden_dim_*.json"
    if model == "gru":
        return f"gru_{dataset}_hidden_dim_*.json"
    raise ValueError(f"Unsupported model: {model}")


def discover_hidden_dim_results(
    results_dir: Path,
    model: str,
    dataset: str,
    target_type: str,
) -> list[tuple[int, Path]]:
    pattern = result_glob_pattern(model, dataset, target_type)
    discovered: list[tuple[int, Path]] = []

    for path in sorted(results_dir.glob(pattern)):
        match = HIDDEN_DIM_RESULT_PATTERN.search(path.name)
        if match is None:
            continue
        discovered.append((int(match.group(1)), path))

    return sorted(discovered, key=lambda item: item[0])


def extract_comparison_metrics(result: dict[str, object]) -> dict[str, float]:
    """Prefer best-epoch validation metrics when saved in result JSON."""
    metrics = {metric: float(result[metric]) for metric in METRIC_COLUMNS}
    if "best_val_rmse" in result:
        metrics["val_rmse"] = float(result["best_val_rmse"])
    if "best_val_mae" in result:
        metrics["val_mae"] = float(result["best_val_mae"])
    return metrics


def append_model_rows(
    rows: list[dict[str, object]],
    model: str,
    results_dir: Path,
    dataset: str,
    target_type: str,
) -> None:
    discovered = discover_hidden_dim_results(results_dir, model, dataset, target_type)
    for hidden_dim, path in discovered:
        result = load_results_json(path)
        metrics = extract_comparison_metrics(result)

        row: dict[str, object] = {
            "model": model,
            "hidden_dim": hidden_dim,
            "target_type": result.get("target_type", target_type),
            "best_epoch": result.get("best_epoch"),
            "trained_epochs": result.get("trained_epochs"),
        }
        row.update(metrics)
        rows.append(row)


def build_comparison_table(
    lstm_results_dir: Path,
    gru_results_dir: Path,
    dataset: str,
    target_type: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    append_model_rows(rows, "lstm", lstm_results_dir, dataset, target_type)
    append_model_rows(rows, "gru", gru_results_dir, dataset, target_type)

    if not rows:
        lstm_pattern = result_glob_pattern("lstm", dataset, target_type)
        gru_pattern = result_glob_pattern("gru", dataset, target_type)
        raise FileNotFoundError(
            "No hidden-dim result files found. "
            f"Expected LSTM files matching {lstm_pattern} in {lstm_results_dir} "
            f"or GRU files matching {gru_pattern} in {gru_results_dir}."
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["model", "hidden_dim"])
        .reset_index(drop=True)
    )


def comparison_table_to_markdown(df: pd.DataFrame, dataset: str) -> str:
    lines = [
        f"# {dataset.upper()} LSTM/GRU Hidden-Dim Comparison",
        "",
        "Validation metrics from saved LSTM and GRU result files. "
        "Uses best-epoch validation metrics when available.",
        "",
        "| Model | Hidden Dim | Target | Best Epoch | Trained Epochs | "
        "Train RMSE | Train MAE | Val RMSE | Val MAE |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {int(row['hidden_dim'])} | {row['target_type']} | "
            f"{int(row['best_epoch'])} | {int(row['trained_epochs'])} | "
            f"{row['train_rmse']:.4f} | {row['train_mae']:.4f} | "
            f"{row['val_rmse']:.4f} | {row['val_mae']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_comparison_table(
    df: pd.DataFrame,
    tables_dir: Path,
    dataset: str,
    target_type: str,
) -> tuple[Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    output_stem = f"hidden_dim_comparison_{dataset}_{target_type}"
    csv_path = tables_dir / f"{output_stem}.csv"
    md_path = tables_dir / f"{output_stem}.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(
        comparison_table_to_markdown(df, dataset),
        encoding="utf-8",
    )

    return csv_path, md_path


def print_summary(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    print("LSTM/GRU hidden-dim comparison")
    print()
    print(
        f"{'model':<6} {'hidden_dim':>10} {'target':<12} {'best_epoch':>10} "
        f"{'trained_epochs':>14} {'train_rmse':>12} {'train_mae':>11} "
        f"{'val_rmse':>10} {'val_mae':>9}"
    )
    print("-" * 102)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<6} {int(row['hidden_dim']):>10} {row['target_type']:<12} "
            f"{int(row['best_epoch']):>10} {int(row['trained_epochs']):>14} "
            f"{row['train_rmse']:>12.4f} {row['train_mae']:>11.4f} "
            f"{row['val_rmse']:>10.4f} {row['val_mae']:>9.4f}"
        )
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


def main() -> None:
    args = parse_args()

    comparison_df = build_comparison_table(
        lstm_results_dir=args.lstm_results_dir,
        gru_results_dir=args.gru_results_dir,
        dataset=args.dataset,
        target_type=args.target_type,
    )
    csv_path, md_path = save_comparison_table(
        comparison_df,
        tables_dir=args.tables_dir,
        dataset=args.dataset,
        target_type=args.target_type,
    )
    print_summary(comparison_df, csv_path, md_path)


if __name__ == "__main__":
    main()
