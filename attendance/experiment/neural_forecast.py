from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models import (
    PatchTST,
    TSMixerx,
    iTransformer,
    NHITS,
    NBEATSx,
    TimesNet,
)

import numpy as np
import datetime
import json
import toml
import sys
import argparse
from copy import deepcopy

from nostradamus.models.neural_wrapper import NeuralWrapper
from attendance.experiment.project_utils import load_data
from nostradamus.preprocessing.quality import trim_timeseries
from nostradamus.preprocessing.times import from_day_to_time_fe
from nostradamus.preprocessing.quality import minimum_length_uid

from datetime import datetime, timedelta
import polars as pl
import toml
from pathlib import Path

import sys
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

features = toml.load(
    r"C:\Users\N000193384\Documents\sncf_project\sncf_playground\attendance\data\features.toml"
)
macro_horizon = features["MACRO_HORIZON"]
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]
times_cols = features["times_cols"]
n_step = 1200
batch_no_serie = 548
win_batch_size = batch_no_serie * 5
SEED = 777
available_exog = ["job", "ferie", "vacances"]
exog = available_exog + times_cols

np.random.seed(SEED)

if __name__ == "__main__":

    train_data, test_data, submission = load_data(project_root)

    # some series may be to short to "validate" so we'll cut throw them for evaluation part
    validate_series = minimum_length_uid(
        train_data,
        uid=ts_uid,
        time=date_col,
        min_length=round(macro_horizon * 1.25),
    )
    train_data = train_data.filter(pl.col(ts_uid).is_in(validate_series))
    n_series = train_data[ts_uid].n_unique()

    time_consumer = [
        TimesNet(
            h=macro_horizon,  # Horizon
            input_size=round(1.5 * macro_horizon),  # Length of input window
            max_steps=n_step,  # Training iterations
            top_k=12,  # Number of periods (for FFT).
            num_kernels=4,  # Number of kernels for Inception module
            batch_size=batch_no_serie,  # Number of time series per batch
            windows_batch_size=win_batch_size,  # Number of windows per batch
            learning_rate=0.003,  # Learning rate
            scaler_type="robust",
            loss=MAE(),
            futr_exog_list=exog,  # Future exogenous variables
            random_seed=SEED,
        ),
        iTransformer(
            h=macro_horizon,
            batch_size=batch_no_serie,
            scaler_type="robust",
            input_size=2 * macro_horizon,
            n_series=n_series,
            loss=MAE(),
            max_steps=n_step,
            early_stop_patience_steps=5,
            random_seed=SEED,
        ),
    ]

    sf = NeuralForecast(
        [
            NHITS(
                h=macro_horizon,
                max_steps=n_step,
                batch_size=batch_no_serie,
                input_size=round(1.5 * macro_horizon),
                dropout_prob_theta=0.25,
                n_freq_downsample=[28, 7, 1],
                scaler_type="robust",
                loss=MAE(),
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
                stat_exog_list=[],  # static is rattached at unique id such as one line = one id with static feature set
                random_seed=SEED,
            ),
            NBEATSx(
                h=macro_horizon,
                max_steps=n_step,
                batch_size=batch_no_serie,
                n_harmonics=7,
                scaler_type="robust",
                loss=MAE(),
                windows_batch_size=win_batch_size,
                input_size=round(1.5 * macro_horizon),
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
                random_seed=SEED,
            ),
            PatchTST(
                h=macro_horizon,
                batch_size=batch_no_serie,
                scaler_type="robust",
                input_size=2 * macro_horizon,
                loss=MAE(),
                max_steps=n_step,
                early_stop_patience_steps=5,
                random_seed=SEED,
            ),
            TSMixerx(
                h=macro_horizon,
                batch_size=batch_no_serie,
                scaler_type="robust",
                input_size=2 * macro_horizon,
                n_series=n_series,
                max_steps=n_step,
                loss=MAE(),
                early_stop_patience_steps=5,
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
                random_seed=SEED,
            ),
        ],
        freq="1d",
    )

    nf_base = NeuralWrapper(
        models=sf,
        ts_uid=ts_uid,
        date_col=date_col,
        target=y,
        forecast_horizon=macro_horizon,
        fill_strategy="forward",
        frequency="1d",
        levels=[95],
        conformalised=False,
        fitted=False,
        exog=exog,
    )

    train_data = (
        train_data.with_columns(
            pl.col(y).log1p().alias(y), pl.col(date_col).cast(pl.Date).alias(date_col)
        )
        .pipe(nf_base.nixtla_reformat, date_col="date", exog=available_exog)
        .pipe(from_day_to_time_fe, time="ds", frequency="day")
        .filter(
            (pl.col("ds") >= datetime(2016, 1, 1))
            & (pl.col("ds") != pl.datetime(2019, 12, 1))
            & (pl.col("ds") != pl.datetime(2021, 11, 1))
        )
        .pipe(trim_timeseries, target=y, uid="unique_id", time="ds")
        .sort(by=["unique_id", "ds"])
        .with_columns(pl.col(y).forward_fill().over("unique_id").alias(y))
        # for exog that has been gap reconstructed.
        .fill_null(0)
    )

    valid_forecast, output_metrics = nf_base.evaluate_on_valid(
        train_data,
        val_size=macro_horizon,
    )
    valid_forecast.write_csv("out/neural_validation.csv")
    print(output_metrics)

    test_forecast = nf_base.forecast(
        train_data,
        val_size=macro_horizon,
        future_df=test_data.rename({ts_uid: "unique_id", date_col: "ds"}).pipe(
            from_day_to_time_fe, time="ds", frequency="day"
        ),
    )
    test_forecast.write_csv("out/submit/neural_test_set.csv")
