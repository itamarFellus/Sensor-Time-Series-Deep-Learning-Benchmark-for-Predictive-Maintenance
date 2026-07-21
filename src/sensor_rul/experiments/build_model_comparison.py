"""Build a single FD001 model comparison table from saved experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COMPARISON_ROWS: tuple[tuple[str, str, str], ...] = (
    ("mean baseline", "naive_baselines", "mean_rul"),
    ("median baseline", "naive_baselines", "median_rul"),
    ("cycle-only", "cycle_baseline", "cycle_only"),
    ("ridge", "ridge_baseline", "ridge"),
    ("MLP", "mlp_baseline", "mlp"),
    ("GRU", "gru_baseline", "gru"),
)

METRIC_COLUMNS = ("train_rmse", "train_mae", "val_rmse", "val_mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison table from saved FD001 model results."
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "baselines",
        help="Directory containing baseline result JSON files.",
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
    return parser.parse_args()


def load_results_json(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")

    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")

    return payload


def result_path(
    results_dir: Path,
    stem: str,
    dataset: str,
) -> Path:
    return results_dir / f"{stem}_{dataset}.json"


def find_result_by_model(
    results: list[dict[str, object]],
    model_key: str,
    source_path: Path,
) -> dict[str, object]:
    for row in results:
        if row.get("model") == model_key:
            return row

    available = [row.get("model") for row in results]
    raise KeyError(
        f"Model '{model_key}' not found in {source_path}. Available: {available}"
    )


def build_comparison_table(
    baselines_dir: Path,
    gru_results_dir: Path,
    dataset: str,
) -> pd.DataFrame:
    loaded_files: dict[str, list[dict[str, object]]] = {}

    def get_results(stem: str, results_dir: Path) -> list[dict[str, object]]:
        if stem not in loaded_files:
            path = result_path(results_dir, stem, dataset)
            loaded_files[stem] = load_results_json(path)
        return loaded_files[stem]

    rows: list[dict[str, object]] = []
    for display_name, stem, model_key in COMPARISON_ROWS:
        results_dir = gru_results_dir if stem == "gru_baseline" else baselines_dir
        result = find_result_by_model(
            get_results(stem, results_dir),
            model_key,
            result_path(results_dir, stem, dataset),
        )

        row = {"model": display_name}
        for metric in METRIC_COLUMNS:
            if metric not in result:
                raise KeyError(
                    f"Metric '{metric}' missing for model '{model_key}' in {stem}_{dataset}.json"
                )
            row[metric] = float(result[metric])
        rows.append(row)

    return pd.DataFrame(rows)


def comparison_table_to_markdown(df: pd.DataFrame, dataset: str) -> str:
    lines = [
        f"# {dataset.upper()} Model Comparison",
        "",
        "Validation metrics from saved experiment result files.",
        "",
        "| Model | Train RMSE | Train MAE | Val RMSE | Val MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row['train_rmse']:.4f} | {row['train_mae']:.4f} | "
            f"{row['val_rmse']:.4f} | {row['val_mae']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_comparison_table(
    df: pd.DataFrame,
    tables_dir: Path,
    dataset: str,
) -> tuple[Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / f"model_comparison_{dataset}.csv"
    md_path = tables_dir / f"model_comparison_{dataset}.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(comparison_table_to_markdown(df, dataset), encoding="utf-8")

    return csv_path, md_path


def print_summary(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    print("FD001 model comparison")
    print()
    print(
        f"{'model':<16} {'train_rmse':>12} {'train_mae':>11} "
        f"{'val_rmse':>10} {'val_mae':>9}"
    )
    print("-" * 62)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<16} {row['train_rmse']:>12.4f} {row['train_mae']:>11.4f} "
            f"{row['val_rmse']:>10.4f} {row['val_mae']:>9.4f}"
        )
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


def main() -> None:
    args = parse_args()

    comparison_df = build_comparison_table(
        baselines_dir=args.baselines_dir,
        gru_results_dir=args.gru_results_dir,
        dataset=args.dataset,
    )
    csv_path, md_path = save_comparison_table(
        comparison_df,
        tables_dir=args.tables_dir,
        dataset=args.dataset,
    )
    print_summary(comparison_df, csv_path, md_path)


if __name__ == "__main__":
    main()
