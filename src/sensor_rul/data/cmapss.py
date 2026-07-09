#%%
from pathlib import Path

import pandas as pd

columns = (
    ['engine_id', 'cycle']
    + [f'op_setting_{i}' for i in range(1, 4)]
    + [f'sensor_{i}' for i in range(1, 22)]
)


def read_cmapss_file(path) -> pd.DataFrame:
    """
    Read a C-MAPSS train/test text file into a clean DataFrame.

    The raw files have no header row and are separated by whitespace.
    """
    df = pd.read_csv(path, header=None, sep=r'\s+', names=columns)
    return df

def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life labels to the training DataFrame.

    For each engine:
        RUL at cycle t = final_cycle_of_engine - current_cycle
    """
    df_train_rul = train_df.copy()

    final_cycle_of_engine = df_train_rul.groupby("engine_id")["cycle"].transform("max")

    df_train_rul["rul"] = final_cycle_of_engine - df_train_rul["cycle"]

    return df_train_rul

def read_rul_file(file_path: Path) -> pd.DataFrame:
    """
    Read the official C-MAPSS test RUL file.

    Each row gives the true RUL after the last observed cycle
    of the corresponding test engine.
    """
    rul_df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["final_rul"],
    )

    rul_df["engine_id"] = range(1, len(rul_df) + 1)

    return rul_df[["engine_id", "final_rul"]]

def load_fd001(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the FD001 subset of the C-MAPSS dataset.

    Returns:
        train_df: training data with a row-wise RUL column
        test_df: test data without row-wise RUL
        test_rul_df: final RUL for each test engine
    """
    data_dir = Path(data_dir)

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    train_df = read_cmapss_file(train_path)
    test_df = read_cmapss_file(test_path)
    test_rul_df = read_rul_file(rul_path)

    train_df = add_train_rul(train_df)

    return train_df, test_df, test_rul_df

#%
# %%
