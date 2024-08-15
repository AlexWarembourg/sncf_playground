import pandas as pd
import numpy as np

from pathlib import Path
from IPython.display import display
import polars as pl
import datetime
import json
import toml
import sys
import argparse
from copy import deepcopy
from datetime import timedelta

from nostradamus.preprocessing.times import (
    from_day_to_time_fe,
    get_covid_table,
)
from nostradamus.preprocessing.quality import trim_timeseries, minimum_length_uid
from nostradamus.forecaster.direct import DirectForecaster
from nostradamus.preprocessing.lags import get_significant_lags
from nostradamus.preprocessing.times import get_basic_holidays
from nostradamus.preprocessing.lags import compute_autoreg_features

# metrics of the competition.
from attendance.experiment.project_utils import load_data
from nostradamus.models.cqr_tree_regressor import CqrGBT

import sys
from pathlib import Path


# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

features = toml.load("data/features.toml")

times_cols = features["times_cols"]
# macro_horizon = features["MACRO_HORIZON"]
macro_horizon = 60
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]
flist = features["flist"]
exog = ["job", "ferie", "vacances"] + times_cols
in_dt = datetime.date(2017, 1, 1)
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", default="global", required=True)

if __name__ == "__main__":

    flag = parser.parse_args()
    set_strategy = flag.strategy
    params_file_name = "params" if set_strategy == "global" else "individual_params"
    with open(f"data/{params_file_name}.json", "rb") as stream:
        params_q = json.load(stream)

    covid_df = get_covid_table(2015, 2024)
    df_dates = df_dates = get_basic_holidays()
    holidays_fe = list(filter(lambda x: date_col not in x, df_dates.columns))
    covid_fe = list(filter(lambda x: date_col not in x, covid_df.columns))
    exog = exog + holidays_fe + covid_fe

    train_data, _, _ = load_data(project_root)

    train_data = (
        train_data.pipe(trim_timeseries, target="y", uid=ts_uid, time=date_col)
        .pipe(from_day_to_time_fe, time=date_col, frequency="day")
        .join(df_dates, how="left", on=[date_col])
        .join(
            covid_df.with_columns(
                pl.lit(np.where(np.any(covid_df != 0, axis=1), 0, 1)).alias("covid_weight")
            ),
            how="left",
            on=[date_col],
        )
        # sncf strike | 2019-12-01 to 2021-11-01
        .filter(
            (pl.col(date_col) >= in_dt)
            & (pl.col(date_col) != pl.datetime(2019, 12, 1))
            & (pl.col(date_col) != pl.datetime(2021, 11, 1))
        )
    )

    # add exponentiel time weight over covid weight
    train_data = train_data.with_columns(
        (
            pl.col("covid_weight")
            + pl.col(date_col).cast(pl.String).cast(pl.Categorical).to_physical()
        ).alias("covid_weight")
    )

    good_ts = minimum_length_uid(train_data, uid=ts_uid, time=date_col, min_length=round(364 * 1.2))
    train_data = train_data.filter(pl.col(ts_uid).is_in(good_ts))

    # test data is a subset of train for dashboarding purpose
    test_data = train_data.filter(
        pl.col(date_col) > pl.col(date_col).max() - timedelta(days=macro_horizon)
    )
    train_data = train_data.filter(
        pl.col(date_col) <= pl.col(date_col).max() - timedelta(days=macro_horizon)
    )
    # test.
    left_term = train_data.select([date_col, ts_uid, "y"] + exog).with_columns(
        pl.lit(1).alias("train")
    )
    right_term = test_data.select([date_col, ts_uid, "y"] + exog).with_columns(
        pl.lit(0).alias("train")
    )
    full_data = pl.concat((left_term, right_term), how="vertical_relaxed")
    del left_term, right_term

    # define params
    significant_lags = get_significant_lags(train_data, date_col=date_col, target=y)
    significant_lags = [x for x in significant_lags if x <= macro_horizon and x % 7 == 0]

    model_reg = CqrGBT(
        params=params_q,
        early_stopping_value=200,
        features=None,
        weight=None,
        categorical_features=[],
        alpha=0.01,
    )

    lags = deepcopy(significant_lags)
    win_list = [7, 14, 28, 56]

    autoreg_dict = {
        ts_uid: {
            "groups": ts_uid,
            "horizon": lambda horizon: np.int32(horizon),
            "wins": np.array(win_list),
            "shifts": lambda horizon: np.int32([horizon, horizon + 28]),
            "lags": lambda horizon: np.array(significant_lags) + horizon,
            "funcs": np.array(flist),
            "delta": [(7, 7), (7, 14), (7, 28), (14, 14), (14, 28)],
        }
    }

    transform_strategy = "None"

    dir_forecaster = DirectForecaster(
        model=model_reg,
        ts_uid=ts_uid,
        forecast_range=np.arange(macro_horizon),
        target_str=y,
        date_str=date_col,
        exogs=exog,
        features_params=autoreg_dict,
        n_jobs=-1,
        transform_strategy=transform_strategy,
        transform_win_size=28,
    )

    # fit model through the wrapper
    tr, val = dir_forecaster.prepare(deepcopy(train_data))

    # fit model directly wo object proxy
    model_reg.fit(
        train_x=tr.select(dir_forecaster.features),
        train_y=tr.select(y),
        valid_x=val.select(dir_forecaster.features),
        valid_y=val.select(y),
        tune=True,
        n_trials=35,
        return_object=True,
    )

    x_test = dir_forecaster.make_future_dataframe().join(
        test_data.rename({"y": "outcome"}), how="left", on=[ts_uid, date_col]
    )
    x_test = pl.concat(
        (
            tr.with_columns(pl.lit(np.nan).alias("outcome")).select(x_test.columns),
            val.with_columns(pl.lit(np.nan).alias("outcome")).select(x_test.columns),
            x_test,
        ),
        how="vertical_relaxed",
    )

    shape = x_test[dir_forecaster.date_str].n_unique()
    assert (
        shape >= dir_forecaster.initial_trim
    ), f"the test length must contains at least {dir_forecaster.initial_trim} date observation to compute feature engineering"
    max_dt = x_test[dir_forecaster.date_str].max()
    minimal_cut = max_dt - timedelta(
        days=int(dir_forecaster.initial_trim + dir_forecaster.forecast_horizon)
    )
    x_test = (
        x_test.filter(pl.col(dir_forecaster.date_str) >= minimal_cut)
        .pipe(
            compute_autoreg_features,
            target_col=dir_forecaster.target_str,
            date_str=dir_forecaster.date_str,
            auto_reg_params=dir_forecaster.features_params,
        )
        .filter(
            pl.col(dir_forecaster.date_str).is_between(
                dir_forecaster.start_forecast, dir_forecaster.end_forecast
            )
        )
    )
    median_forecast, bands = model_reg.predict(x_test)
    display(bands)
    x_test = x_test.with_columns(
        pl.lit(median_forecast).alias("y_hat"),
        pl.lit(bands[:, 0].flatten().ravel()).alias("lower_band"),
        pl.lit(bands[:, 1].flatten().ravel()).alias("upper_band"),
    )
    x_test.write_csv(r"C:\Users\N000193384\Documents\sncf_project\sncf_playground\out\cqr_lgb.csv")
