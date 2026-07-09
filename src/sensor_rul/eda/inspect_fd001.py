#%%
from pathlib import Path
from sensor_rul.data.cmapss import load_fd001

import matplotlib.pyplot as plt

def print_basic_summary(train_df, test_df, rul_df) -> None:
    print("\n=== FD001 Basic Summary ===")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"RUL shape:   {rul_df.shape}")

    print(f"\nTrain engines: {train_df['engine_id'].nunique()}")
    print(f"Test engines:  {test_df['engine_id'].nunique()}")

    print("\nTrain columns:")
    print(train_df.columns.tolist())

    print("\nFirst 5 train rows:")
    print(train_df.head())

    print("\nMissing values in train:")
    print(train_df.isna().sum())

    cycle_lengths = train_df.groupby("engine_id")["cycle"].max()

    print("\nCycle length summary:")
    print(cycle_lengths.describe())

    print("\nFirst 10 engine final cycles:")
    print(cycle_lengths.head(10))

def plot_cycle_lengths(train_df, figures_dir: Path) -> None:
    cycle_lengths = train_df.groupby("engine_id")["cycle"].max()

    plt.figure(figsize=(8, 5))
    plt.hist(cycle_lengths, bins=20)

    plt.title("FD001 Train Engine Cycle Lengths")
    plt.xlabel("Final cycle")
    plt.ylabel("Number of engines")

    plt.tight_layout()

    output_path = figures_dir / "fd001_cycle_lengths_hist.png"
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved figure to: {output_path}")

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "cmapss-data" /  "raw" 
    figures_dir = project_root / "results" / "figures"

    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, rul_df = load_fd001(data_dir)

    print_basic_summary(train_df, test_df, rul_df)
    plot_cycle_lengths(train_df, figures_dir)

if __name__ == "__main__":
    main()
#%
# %%
