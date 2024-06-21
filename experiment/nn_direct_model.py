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


sys.path.insert(0, r"C:\Users\N000193384\Documents\sncf_project\sncf_playground")

from src.preprocessing.times import (
    from_day_to_time_fe,
    get_covid_table,
)
from src.preprocessing.quality import trim_timeseries, minimum_length_uid
from src.models.forecast.direct import DirectForecaster
from src.preprocessing.lags import get_significant_lags
from src.preprocessing.times import get_basic_holidays

# metrics of the competition.
from src.project_utils import load_data
from src.models.mlp_wrapper import TorchWrapper

features = toml.load("data/features.toml")

times_cols = features["times_cols"]
macro_horizon = features["MACRO_HORIZON"]
p = Path(features["ABS_DATA_PATH"])
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

    train_data, test_data, submission = load_data(p)

    test_data = (
        test_data.pipe(from_day_to_time_fe, time=date_col, frequency="day")
        .join(df_dates, how="left", on=[date_col])
        .join(covid_df, how="left", on=[date_col])
    )

    train_data = (
        train_data.pipe(trim_timeseries, target="y", uid=ts_uid, time=date_col)
        .pipe(from_day_to_time_fe, time=date_col, frequency="day")
        .join(df_dates, how="left", on=[date_col])
        .join(
            covid_df.with_columns(
                pl.lit(np.where(np.any(covid_df != 0, axis=1), 0, 1)).alias(
                    "covid_weight"
                )
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

    good_ts = minimum_length_uid(
        train_data, target=y, uid=ts_uid, time=date_col, min_length=round(364 * 1.2)
    )
    train_data = train_data.filter(pl.col(ts_uid).is_in(good_ts))
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
    significant_lags = [
        x for x in significant_lags if x <= macro_horizon and x % 7 == 0
    ]

    if set_strategy == "global":
        mlp = TorchWrapper(
            batch_size=512,
            num_cols=scaled_num_cols,
            cat_cols=cat_cols,
            target=y,
            hidden_dim=254,
        )

    mlp.fit(train_set, val_set, num_epochs=100, patience=10)

    val_set = val_set.with_columns(
        pl.lit(np.expm1(mlp.predict(x_test=val_set))).alias("forecast_mlp")
    )

    lags = deepcopy(significant_lags)
    win_list = [7, 28, 56]

    autoreg_dict = {
        ts_uid: {
            "groups": ts_uid,
            "horizon": lambda horizon: np.int32(horizon),
            "wins": np.array(win_list),
            "shifts": lambda horizon: np.int32([horizon, horizon + 28]),
            "lags": lambda horizon: np.array(significant_lags) + horizon,
            "funcs": np.array(flist),
        },
        "ts_uid_dow": {
            "groups": [ts_uid, "day_of_week"],
            "horizon": lambda horizon: np.int32(np.ceil(horizon / 7) + 1),
            "wins": np.array([4, 12]),
            "shifts": lambda horizon: np.int32(
                [
                    np.ceil(horizon / 7) + 1,
                    np.ceil(horizon / 7) + round((macro_horizon / 4)),
                ]
            ),
            "lags": lambda horizon: np.arange(1, 7) + np.ceil(horizon / 7) + 1,
            "funcs": np.array(flist),
        },
        "ts_uid_week": {
            "groups": [ts_uid, "week"],
            "horizon": lambda horizon: np.int32(np.ceil(horizon / 7) + 1),
            "wins": np.array([4, 12]),
            "shifts": lambda horizon: np.int32(
                [
                    np.ceil(horizon / 7) + 1,
                    np.ceil(horizon / 7) + round((macro_horizon / 4)),
                ]
            ),
            "lags": lambda horizon: np.arange(1, 7) + np.ceil(horizon / 7) + 1,
            "funcs": np.array(flist),
        },
    }

    dir_forecaster = DirectForecaster(
        model=model_reg,
        ts_uid=ts_uid,
        forecast_range=np.arange(macro_horizon),
        target_str=y,
        date_str=date_col,
        exogs=exog,
        features_params=autoreg_dict,
        n_jobs=-1,
    )
    # fit model through the wrapper
    dir_forecaster.fit(train_data=train_data, strategy=set_strategy)
    display(dir_forecaster.evaluate())
    # and forecast test.
    (
        test_data.select(["index", date_col, ts_uid])
        .join(
            (
                dir_forecaster.predict(full_data)
                .filter(pl.col("train") == 0)
                .select([date_col, ts_uid, "y_hat"])
            ),
            how="left",
            on=[date_col, ts_uid],
        )
        .fill_null(0)
        .select(["index", "y_hat"])
        .rename({"y_hat": "y"})
    ).write_csv(f"out/submit/{set_strategy}_direct_lgb.csv")

    # write valid for evaluation purpose.
    if set_strategy == "global":
        valid_out = dir_forecaster.valid.with_columns(
            pl.lit(dir_forecaster.model.predict(dir_forecaster.valid)).alias(
                f"{set_strategy}_direct_y_hat"
            )
        )
    else:
        valid_out = (
            dir_forecaster.predict_local(dir_forecaster.valid)
            .select([ts_uid, date_col, "y_hat"])
            .rename({"y_hat": f"{set_strategy}_direct_y_hat"})
        )

    valid_out.write_csv(f"out/{set_strategy}_direct_lgb.csv")
