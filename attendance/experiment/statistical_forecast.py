import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import datetime
import toml
import sys

from nostradamus.preprocessing.quality import trim_timeseries
from attendance.experiment.project_utils import load_data
from nostradamus.preprocessing.times import from_day_to_time_fe
from nostradamus.models.stats_wrapper import StatsBaseline
from nostradamus.preprocessing.quality import minimum_length_uid

from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    AutoTheta,
    AutoARIMA,
    SeasonalNaive,
    AutoTBATS,
    AutoETS,
    MSTL,
)

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

features = toml.load("data/features.toml")
macro_horizon = features["MACRO_HORIZON"]
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]

in_dt = datetime.date(2019, 1, 1)
season_length = 28

my_way_to_go_baseline = [
    HoltWinters(),
    AutoTheta(),
    MSTL(season_length=[7, season_length], trend_forecaster=HoltWinters()),
    AutoARIMA(),
    AutoETS(),
    AutoTBATS(season_length=[season_length]),
]

# Instantiate StatsForecast class as sf
sf = StatsForecast(
    models=my_way_to_go_baseline,
    fallback_model=SeasonalNaive(season_length=season_length),
    n_jobs=-1,
    freq="1d",
)

forecast_stats_base = StatsBaseline(
    models=sf,
    ts_uid=ts_uid,
    date_col=date_col,
    target=y,
    forecast_horizon=macro_horizon,
    fill_strategy="forward",
    frequency="1d",
    levels=[99],
    conformalised=True,
    fitted=False,
)

if __name__ == "__main__":

    train_data, test_data, submission = load_data(project_root)
    train_data = (
        train_data.pipe(from_day_to_time_fe, time="date", frequency="day")
        .pipe(forecast_stats_base.nixtla_reformat, date_col="date")
        .fill_null(0)
        .pipe(trim_timeseries, target="y", uid="unique_id", time="ds")
        .filter(
            (pl.col("ds") >= in_dt)
            & (pl.col("ds") != pl.datetime(2019, 12, 1))
            & (pl.col("ds") != pl.datetime(2021, 11, 1))
        )
        .select(["unique_id", "ds", "y"])
    )
    test_forecast = forecast_stats_base.forecast(train_data)
    test_forecast.write_csv(project_root / "out/submit/nixtla_forecast.csv")
    # some series may be to short to "validate" so we'll cut throw them for evaluation part
    validate_series = minimum_length_uid(
        train_data,
        uid="unique_id",
        time="ds",
        min_length=round(macro_horizon * 2),
    )
    valid_forecast, output_metrics = forecast_stats_base.evaluate_on_valid(
        train_data.filter(pl.col("unique_id").is_in(validate_series))
    )
    valid_forecast.write_csv(project_root / "out/nixtla_validation.csv")
    print(output_metrics)
