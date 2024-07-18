import pandas as pd
import polars as pl
import numpy as np

from datetime import timedelta

import sys
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nostradamus.preprocessing.quality import find_ts_outlier


def forecast_to_submit(
    test, unique_id_col_name, date_col_name, submission_name, y_col, submission_file
):
    test["index"] = test[date_col_name].astype(str) + "_" + test[unique_id_col_name].astype(str)
    test["y"] = test[y_col].copy()
    out = test[["index", "y"]]
    submission_file.merge(out, how="left", on=["index"]).to_csv(
        f"out/submit/{submission_name}.csv", index=False
    )
    return out


def submit_fcst(test_data, lgb_out, ts_uid):
    submit = (
        test_data.select(["index", "date", "station"])
        .with_columns(pl.col("date").cast(pl.Date))
        .join(
            lgb_out.select(["y_hat", "date", "station"]).with_columns(pl.col("date").cast(pl.Date)),
            how="left",
            on=["date", "station"],
        )
        .with_columns(pl.col("y_hat").clip(lower_bound=0))
    )

    if submit["y_hat"].is_null().sum() > 0:
        submit = submit.with_columns(pl.col("y_hat").fill_null(pl.col("y_hat").mean().over(ts_uid)))
    assert submit["y_hat"].is_null().sum() == 0, "forecast col has null"
    submit = submit.rename({"y_hat": "y"})
    return submit[["index", "y"]]


def load_data(p):
    train_data = pl.read_csv(p / "attendance/data/train_f_x.csv")
    test_data = pl.read_csv(p / "attendance/data/test_f_x_THurtzP.csv")
    submission = pl.read_csv(p / "attendance/data/y_exemple_sncf_d9so9pm.csv").drop(columns="y")

    train_data = (
        train_data.with_columns(
            pl.concat_str(["date", "station"], separator="_").alias("index"),
            pl.lit(True).alias("is_train"),
            pl.col("date").str.to_datetime("%Y-%m-%d"),
            # pl.when(pl.col("date") >= validation_dt).then(1).otherwise(0).alias("is_valid")
        )
        .join(pl.read_csv(p / "attendance/data/y_train_sncf.csv"), on="index", how="inner")
        .with_columns(pl.col("y").alias("y_copy"))
        .pipe(find_ts_outlier, ts_uid="station", y="y", date_col="date")
    )

    test_data = test_data.with_columns(
        pl.lit(np.nan).alias("y"),
        pl.lit(np.nan).alias("y_copy"),
        pl.lit(False).alias("is_train"),
        pl.col("date").str.to_datetime("%Y-%m-%d"),
    )
    return train_data, test_data, submission
