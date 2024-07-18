import pandas as pd
import numpy as np
import toml

# import matplotlib.pyplot as plt
from IPython.display import display
import polars as pl
from datetime import timedelta

import sys
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

features = toml.load(
    r"C:\Users\N000193384\Documents\sncf_project\sncf_playground\attendance\data\features.toml"
)
times_cols = features["times_cols"]
macro_horizon = features["MACRO_HORIZON"]

from nostradamus.preprocessing.quality import find_ts_outlier
from nostradamus.models.linear_baseline import FLinear
from attendance.experiment.project_utils import load_data

ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]


def freeze_validation_set(
    df: pl.DataFrame,
    date: str,
    val_size: int,
    return_train: bool = True,
) -> pl.DataFrame:
    max_dt = df[date].max()
    cut = max_dt - timedelta(days=val_size)
    valid = df.filter(pl.col(date) > cut)  # .select([ts_uid, date, target])
    if return_train:
        train = df.filter(pl.col(date) <= cut)
        return train, valid
    else:
        return valid


if __name__ == "__main__":

    train_data, test_data, submission = load_data(project_root)
    train_data = find_ts_outlier(
        train_data, ts_uid=ts_uid, y=y, date_col=date_col, window_size=90
    ).filter(pl.col("not_outlier") == 1)

    ll = FLinear(
        features=["job", "ferie", "vacances"],
        target=y,
        ts_uid=ts_uid,
        use_gam=True,
        ndays=364 * 3,
        horizon=181,
        forecast_end_dt=test_data["date"].max(),
    )

    ll.fit(train_data)
    out = ll.predict()
    out_ = (
        test_data.with_columns(pl.col(date_col).cast(pl.String).str.to_datetime().cast(pl.Date))
        .join(
            out.with_columns(
                pl.col(date_col).cast(pl.String).str.to_datetime().cast(pl.Date)
            ).select(["station", date_col, "y_hat"]),
            how="left",
            on=["station", "date"],
        )
        .drop("y")
        .rename({"y_hat": "y"})
        .with_columns(pl.col(y).clip_min(0).cast(pl.Float32).fill_null(0.0))
        .write_csv(project_root / "out/submit/gam.csv")
    )

    # ========== validation ===========
    tr, val = freeze_validation_set(train_data, date=date_col, val_size=181, return_train=True)

    ll = FLinear(
        features=["job", "ferie", "vacances"],
        target=y,
        ts_uid=ts_uid,
        use_gam=True,
        ndays=364 * 2,
        horizon=181,
        forecast_end_dt=val["date"].max(),
    )

    ll.fit(tr)
    out = ll.predict()
    out = out.write_csv(project_root / "out/gam.csv")
