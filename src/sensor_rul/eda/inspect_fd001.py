#%%
from pathlib import Path

from sensor_rul.data.cmapss import load_fd001


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

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data" / "cmapss-data" /  "raw" 

    train_df, test_df, rul_df = load_fd001(data_dir)

    print_basic_summary(train_df, test_df, rul_df)


if __name__ == "__main__":
    main()
#%
# %%
