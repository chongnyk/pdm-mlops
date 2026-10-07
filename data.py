"""Synthetic machine sensor data (modelled on the AI4I 2020 predictive-maintenance dataset)."""
import numpy as np
import pandas as pd

FEATURES = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
TARGET = "failure"


def make_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    air = rng.normal(300, 2, n_rows)                                   # ambient temperature (K)
    process = air + 10 + rng.normal(0, 1, n_rows)                      # process temperature (K)
    rpm = np.clip(rng.normal(1540, 180, n_rows), 1170, 2880)           # spindle speed
    torque = np.clip(40 - 0.03 * (rpm - 1540) + rng.normal(0, 6.5, n_rows), 4, 76)
    wear = rng.uniform(0, 240, n_rows)                                 # minutes the tool has been used
    power = torque * rpm * 2 * np.pi / 60                              # watts

    failure = (
        ((wear > 200) & (rng.random(n_rows) < 0.12))                   # tool wear failure
        | ((process - air < 8.6) & (rpm < 1380))                       # heat dissipation failure
        | (power < 3500) | (power > 9000)                              # power failure
        | (wear * torque > 11000)                                      # overstrain failure
        | (rng.random(n_rows) < 0.002)                                 # random failure
    )
    return pd.DataFrame({
        "air_temp_k": air.round(1),
        "process_temp_k": process.round(1),
        "rotational_speed_rpm": rpm.round(0),
        "torque_nm": torque.round(1),
        "tool_wear_min": wear.round(0),
        TARGET: failure.astype(int),
    })
