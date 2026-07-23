"""Build a single FD001 model comparison table from saved experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASELINE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("mean baseline", "naive_baselines", "mean_rul"),
    ("median baseline", "naive_baselines", "median_rul"),
    ("cycle-only", "cycle_baseline", "cycle_only"),
    ("ridge", "ridge_baseline", "ridge"),
)

DEEP_MODEL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("MLP", "mlp_baseline", "mlp", "raw"),
    ("MLP", "mlp_baseline", "mlp", "capped_125"),
    ("GRU", "gru", "gru", "raw"),
    ("GRU", "gru_capped", "gru", "capped_125"),
    ("LSTM", "lstm", "lstm", "raw"),
    ("LSTM", "lstm", "lstm", "capped_125"),
)

GRU_STEM = "gru_capped"
LSTM_STEM = "lstm"
GRU_RESULTS_STEMS = frozenset({"gru", GRU_STEM})
TARGET_TYPE_STEMS = frozenset({"mlp_baseline", "gru", GRU_STEM, LSTM_STEM})

DEFAULT_RESULT_SUFFIX = "_epochs_100"

METRIC_COLUMNS = ("train_rmse", "train_mae", "val_rmse", "val_mae")


def extract_comparison_metrics(result: dict[str, object]) -> dict[str, float]:
    """Prefer best-epoch validation metrics when saved in result JSON."""
    metrics = {metric: float(result[metric]) for metric in METRIC_COLUMNS}
    if "best_val_rmse" in result:
        metrics["val_rmse"] = float(result["best_val_rmse"])
    if "best_val_mae" in result:
        metrics["val_mae"] = float(result["best_val_mae"])
    return metrics


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
        "--lstm-results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "lstm",
        help="Directory containing LSTM result JSON files.",
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
        "--result-suffix",
        type=str,
        default=DEFAULT_RESULT_SUFFIX,
        help="Result filename suffix, e.g. _epochs_100.",
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
    target_type: str = "raw",
    result_suffix: str = "",
) -> Path:
    candidates: list[Path] = []

    if stem in TARGET_TYPE_STEMS:
        candidates.append(
            results_dir / f"{stem}_{dataset}_{target_type}{result_suffix}.json"
        )

    if target_type == "raw":
        candidates.extend(
            (
                results_dir / f"{stem}_{dataset}{result_suffix}.json",
                results_dir / f"{stem}_{dataset}_raw{result_suffix}.json",
            )
        )

    if stem == GRU_STEM and target_type == "raw":
        candidates.extend(
            (
                results_dir / f"gru_{dataset}{result_suffix}.json",
                results_dir / f"gru_{dataset}_raw{result_suffix}.json",
            )
        )

    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


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


def results_dir_for_stem(
    stem: str,
    baselines_dir: Path,
    gru_results_dir: Path,
    lstm_results_dir: Path,
) -> Path:
    if stem in GRU_RESULTS_STEMS:
        return gru_results_dir
    if stem == LSTM_STEM:
        return lstm_results_dir
    return baselines_dir


def build_comparison_table(
    baselines_dir: Path,
    gru_results_dir: Path,
    lstm_results_dir: Path,
    dataset: str,
    result_suffix: str = "",
) -> pd.DataFrame:
    loaded_files: dict[str, list[dict[str, object]]] = {}

    def get_results(
        stem: str,
        results_dir: Path,
        target_type: str,
        suffix: str,
    ) -> list[dict[str, object]]:
        cache_key = f"{results_dir}:{stem}:{target_type}:{suffix}"
        if cache_key not in loaded_files:
            path = result_path(
                results_dir, stem, dataset, target_type, suffix
            )
            loaded_files[cache_key] = load_results_json(path)
        return loaded_files[cache_key]

    def append_row(
        rows: list[dict[str, object]],
        display_name: str,
        stem: str,
        model_key: str,
        target_type: str,
        *,
        use_result_suffix: bool = True,
    ) -> None:
        suffix = result_suffix if use_result_suffix else ""
        results_dir = results_dir_for_stem(
            stem, baselines_dir, gru_results_dir, lstm_results_dir
        )
        source_path = result_path(
            results_dir, stem, dataset, target_type, suffix
        )
        if not source_path.is_file():
            return
        result = find_result_by_model(
            get_results(stem, results_dir, target_type, suffix),
            model_key,
            source_path,
        )

        row: dict[str, object] = {"model": display_name, "target": target_type}
        metrics = extract_comparison_metrics(result)
        for metric in METRIC_COLUMNS:
            row[metric] = metrics[metric]
        rows.append(row)

    rows: list[dict[str, object]] = []
    for display_name, stem, model_key in BASELINE_ROWS:
        append_row(
            rows, display_name, stem, model_key, "raw", use_result_suffix=False
        )

    for display_name, stem, model_key, target_type in DEEP_MODEL_ROWS:
        append_row(rows, display_name, stem, model_key, target_type)

    return pd.DataFrame(rows)


def comparison_table_to_markdown(df: pd.DataFrame, dataset: str) -> str:
    lines = [
        f"# {dataset.upper()} Model Comparison",
        "",
        "Validation metrics from saved experiment result files. "
        "Deep models use best-epoch validation metrics when available.",
        "",
        "| Model | Target | Train RMSE | Train MAE | Val RMSE | Val MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['model']} | {row['target']} | {row['train_rmse']:.4f} | "
            f"{row['train_mae']:.4f} | {row['val_rmse']:.4f} | {row['val_mae']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_comparison_table(
    df: pd.DataFrame,
    tables_dir: Path,
    dataset: str,
    result_suffix: str = "",
) -> tuple[Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = result_suffix.lstrip("_")
    output_stem = f"model_comparison_{dataset}"
    if output_suffix:
        output_stem = f"{output_stem}_{output_suffix}"

    csv_path = tables_dir / f"{output_stem}.csv"
    md_path = tables_dir / f"{output_stem}.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(
        comparison_table_to_markdown(df, dataset),
        encoding="utf-8",
    )

    return csv_path, md_path


def print_summary(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    print("FD001 model comparison")
    print()
    print(
        f"{'model':<16} {'target':<12} {'train_rmse':>12} {'train_mae':>11} "
        f"{'val_rmse':>10} {'val_mae':>9}"
    )
    print("-" * 75)
    for _, row in df.iterrows():
        print(
            f"{row['model']:<16} {row['target']:<12} {row['train_rmse']:>12.4f} "
            f"{row['train_mae']:>11.4f} {row['val_rmse']:>10.4f} {row['val_mae']:>9.4f}"
        )
    print()
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")


def main() -> None:
    args = parse_args()

    comparison_df = build_comparison_table(
        baselines_dir=args.baselines_dir,
        gru_results_dir=args.gru_results_dir,
        lstm_results_dir=args.lstm_results_dir,
        dataset=args.dataset,
        result_suffix=args.result_suffix,
    )
    csv_path, md_path = save_comparison_table(
        comparison_df,
        tables_dir=args.tables_dir,
        dataset=args.dataset,
        result_suffix=args.result_suffix,
    )
    print_summary(comparison_df, csv_path, md_path)


if __name__ == "__main__":
    main()
