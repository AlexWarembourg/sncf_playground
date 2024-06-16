import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import datetime
import toml

import sys

sys.path.insert(0, r"C:\Users\N000193384\Documents\sncf_project\sncf_playground")
from src.preprocessing.quality import trim_timeseries
from src.project_utils import load_data
from src.preprocessing.times import from_day_to_time_fe

from itertools import chain
from pathlib import Path
from typing import List, Optional

import neuralforecast as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pli
from datasetsforecast.utils import download_file
from hyperopt import hp
from neuralforecast.auto import NHITS as autoNHITS
from neuralforecast.data.tsdataset import WindowsDataset
from neuralforecast.data.tsloader import TimeSeriesLoader
from neuralforecast.models.mqnhits.mqnhits import MQNHITS
from neuralforecast.models.nhits.nhits import NHITS

# GLOBAL PARAMETERS
DEFAULT_HORIZON = 181
HYPEROPT_STEPS = 10
MAX_STEPS = 1000
N_TS_VAL = 2 * 30

MODELS = {
    "Pretrained N-HiTS M4 Hourly": {
        "card": "nhitsh",
        "max_steps": 0,
        "model": "nhits_m4_hourly",
    },
    "Pretrained N-HiTS M4 Hourly (Tiny)": {
        "card": "nhitsh",
        "max_steps": 0,
        "model": "nhits_m4_hourly_tiny",
    },
    "Pretrained N-HiTS M4 Daily": {
        "card": "nhitsd",
        "max_steps": 0,
        "model": "nhits_m4_daily",
    },
    "Pretrained N-HiTS M4 Monthly": {
        "card": "nhitsm",
        "max_steps": 0,
        "model": "nhits_m4_monthly",
    },
    "Pretrained N-HiTS M4 Yearly": {
        "card": "nhitsy",
        "max_steps": 0,
        "model": "nhits_m4_yearly",
    },
    "Pretrained N-BEATS M4 Hourly": {
        "card": "nbeatsh",
        "max_steps": 0,
        "model": "nbeats_m4_hourly",
    },
    "Pretrained N-BEATS M4 Daily": {
        "card": "nbeatsd",
        "max_steps": 0,
        "model": "nbeats_m4_daily",
    },
    "Pretrained N-BEATS M4 Weekly": {
        "card": "nbeatsw",
        "max_steps": 0,
        "model": "nbeats_m4_weekly",
    },
    "Pretrained N-BEATS M4 Monthly": {
        "card": "nbeatsm",
        "max_steps": 0,
        "model": "nbeats_m4_monthly",
    },
    "Pretrained N-BEATS M4 Yearly": {
        "card": "nbeatsy",
        "max_steps": 0,
        "model": "nbeats_m4_yearly",
    },
}


def download_models():
    for _, meta in MODELS.items():
        if not Path(f'./models/{meta["model"]}.ckpt').is_file():
            download_file(
                "./models/",
                f'https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/{meta["model"]}.ckpt',
            )


download_models()


class StandardScaler:
    """This class helps to standardize a dataframe with multiple time series."""

    def __init__(self):
        self.norm: pd.DataFrame

    def fit(self, X: pd.DataFrame) -> "StandardScaler":
        self.norm = X.groupby("unique_id").agg({"y": [np.mean, np.std]})
        self.norm = self.norm.droplevel(0, 1).reset_index()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.merge(self.norm, how="left", on=["unique_id"])
        transformed["y"] = (transformed["y"] - transformed["mean"]) / transformed["std"]
        return transformed[["unique_id", "ds", "y"]]

    def inverse_transform(self, X: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        transformed = X.merge(self.norm, how="left", on=["unique_id"])
        for col in cols:
            transformed[col] = (
                transformed[col] * transformed["std"] + transformed["mean"]
            )
        return transformed[["unique_id", "ds"] + cols]


def forecast_pretrained_model(
    Y_df: pd.DataFrame, model: str, fh: int, max_steps: int = 0
):
    if "unique_id" not in Y_df:
        Y_df.insert(0, "unique_id", "ts_1")

    scaler = StandardScaler()
    scaler.fit(Y_df)
    Y_df = scaler.transform(Y_df)

    # Model
    file_ = f"./models/{model}.ckpt"
    mqnhits = MQNHITS.load_from_checkpoint(file_)

    # Fit
    if max_steps > 0:
        train_dataset = WindowsDataset(
            Y_df=Y_df,
            X_df=None,
            S_df=None,
            mask_df=None,
            f_cols=[],
            input_size=mqnhits.n_time_in,
            output_size=mqnhits.n_time_out,
            sample_freq=1,
            complete_windows=True,
            verbose=False,
        )

        train_loader = TimeSeriesLoader(
            dataset=train_dataset, batch_size=1, n_windows=32, shuffle=True
        )

        trainer = pli.Trainer(
            max_epochs=None,
            checkpoint_callback=False,
            logger=False,
            max_steps=max_steps,
            gradient_clip_val=1.0,
            progress_bar_refresh_rate=1,
            log_every_n_steps=1,
        )

        trainer.fit(mqnhits, train_loader)

    # Forecast
    forecast_df = mqnhits.forecast(Y_df=Y_df)
    forecast_df = scaler.inverse_transform(forecast_df, cols=["y_5", "y_50", "y_95"])

    # Foreoast
    n_ts = forecast_df["unique_id"].nunique()
    if fh * n_ts > len(forecast_df):
        forecast_df = (
            forecast_df.groupby("unique_id")
            .apply(lambda df: pd.concat([df] * fh).head(fh))
            .reset_index(drop=True)
        )
    else:
        forecast_df = forecast_df.groupby("unique_id").head(fh)
    # forecast_df["ds"] = compute_ds_future(Y_df, fh)
    return forecast_df


if __name__ == "__main__":
    train_data, test_data, submission = load_data(p)
    train_data = (
        train_data.pipe(trim_timeseries, target="y", uid="station", time="date")
        .pipe(from_day_to_time_fe, time="date", frequency="day")
        .filter(
            (pl.col("date") >= datetime(2017, 1, 1))
            & (pl.col("date") != pl.datetime(2019, 12, 1))
            & (pl.col("date") != pl.datetime(2021, 11, 1))
        )
        # .pipe(forecast_stats_base.nixtla_reformat, date_col="date")
    )
    for _, meta in MODELS.items():
        # test multiple time series
        multi_forecast = forecast_pretrained_model(
            train_data, model=meta["model"], fh=80
        )
        assert multi_forecast.shape == (80 * 2, 5)
