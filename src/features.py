# ============================================================
# features.py — Physical feature engineering
# ============================================================
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create physical features derived from known failure rules.

    - Power [W]     : power = torque × speed (rad/s)
    - Temp Diff [K] : process temp - air temp
    - Overstrain    : tool wear × torque
    """
    df = df.copy()

    speed_rad_s = df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df["Power [W]"] = df["Torque [Nm]"] * speed_rad_s

    df["Temp Diff [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Overstrain [min·Nm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    return df
