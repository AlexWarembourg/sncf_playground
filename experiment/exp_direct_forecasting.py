import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
import polars as pl
from datetime import timedelta
import datetime
import json
import toml
import holidays

from src.analysis.describe import describe_timeseries
from src.preprocessing.polars.times import from_day_to_time_fe, get_covid_table
from src.preprocessing.polars.quality import trim_timeseries, minimum_length_uid
from src.preprocessing.polars.lags import (
    reference_shift_from_day,
    pl_compute_lagged_features,
    pl_compute_moving_features,
)
from src.analysis.metrics import display_metrics
from src.models.forecast.direct import DirectForecaster

# metrics of the competition.
from sklearn.metrics import mean_absolute_percentage_error
from src.project_utils import forecast_to_submit, load_data
from src.preprocessing.polars.validation import freeze_validation_set
from src.models.lgb_wrapper import GBTModel
from src.preprocessing.polars.scale import PanelStandardScaler
from src.project_utils import submit_fcst

features = toml.load("data/features.toml")

times_cols = features["times_cols"]
macro_horizon = features["MACRO_HORIZON"]
p = Path(features["ABS_DATA_PATH"])
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]
submit = False
flist = features["flist"]
long_horizon = np.arange(macro_horizon)
chains = np.array_split(long_horizon, 6)
exog = ["job", "ferie", "vacances"] + times_cols

with open("data/params.json", "rb") as stream:
    params_q = json.load(stream)

in_dt = datetime.date(2017, 12, 31)
