# ============================================================
# data_loader.py — Load data and build targets
# ============================================================
import pandas as pd

import config


def load_data(path) -> pd.DataFrame:
    """Load the AI4I 2020 dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Place ai4i2020.csv inside the data/ directory."
        )
    return pd.read_csv(path)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the multi-class target while reconciling the known
    inconsistency in the AI4I dataset.

    Note: In the original AI4I data, 'Machine failure' does not
    exactly match the failure-type flags. We treat 'Machine failure'
    as the ground truth for overall failure and derive the failure
    type only from rows that actually failed.
    """
    df = df.copy()
    flag_sum = df[config.FAILURE_TYPE_COLUMNS].sum(axis=1)

    # Diagnose the mismatch
    type_without_failure = (flag_sum >= 1) & (df[config.BINARY_TARGET] == 0)
    failure_without_type = (flag_sum == 0) & (df[config.BINARY_TARGET] == 1)
    print(f"Rows with a failure type but no Machine failure: {type_without_failure.sum()}")
    print(f"Rows with Machine failure but no failure type : {failure_without_type.sum()}")

    # Derive failure type only from failed rows; join co-occurring types with '+'
    df[config.MULTI_CLASS_TARGET] = "No Failure"
    failed_mask = df[config.BINARY_TARGET] == 1
    df.loc[failed_mask, config.MULTI_CLASS_TARGET] = (
        df.loc[failed_mask, config.FAILURE_TYPE_COLUMNS]
        .apply(lambda row: "+".join(row.index[row == 1]) or "Unknown", axis=1)
    )
    return df


def load_and_prepare(path=None):
    """Load the dataset and return it with the multi-class target built."""
    path = path or config.DATA_FILE
    df = load_data(path)
    df = build_targets(df)
    return df
