import pandas as pd
import numpy as np
import toml
from src.models.linear_baseline import LogLinear

# import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
import polars as pl
from datetime import timedelta
import sys

features = toml.load(
    r"C:\Users\N000193384\Documents\sncf_project\sncf_playground\data\features.toml"
)
times_cols = features["times_cols"]
macro_horizon = features["MACRO_HORIZON"]
p = Path(features["ABS_DATA_PATH"])
sys.path.insert(1, p)
from src.project_utils import load_data

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

    train_data, test_data, submission = load_data(p)

    ll = LogLinear(
        features=["job", "ferie", "vacances"],
        target=y,
        ts_uid=ts_uid,
        use_gam=True,
        ndays=364 * 2,
        horizon=181,
        forecast_end_dt=test_data["date"].max(),
    )

    ll.fit(train_data)
    out = ll.predict()
    out_ = (
        test_data.with_columns(
            pl.col(date_col).cast(pl.String).str.to_datetime().cast(pl.Date)
        )
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
        .write_csv("out/submit/gam.csv")
    )

    # ========== validation ===========
    tr, val = freeze_validation_set(
        train_data, date=date_col, val_size=181, return_train=True
    )

    ll = LogLinear(
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
    out = out.write_csv("out/gam.csv")
