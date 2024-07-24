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

from nostradamus.preprocessing.times import (
    from_day_to_time_fe,
    get_covid_table,
)
from nostradamus.preprocessing.quality import trim_timeseries, minimum_length_uid
from nostradamus.forecaster.chain import ChainForecaster
from nostradamus.preprocessing.lags import get_significant_lags
from nostradamus.preprocessing.times import get_basic_holidays

# metrics of the competition.
from attendance.experiment.project_utils import load_data
from nostradamus.models.lgb_wrapper import GBTModel

import sys
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

features = toml.load("data/features.toml")

times_cols = features["times_cols"]
macro_horizon = features["MACRO_HORIZON"]
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

    train_data, test_data, submission = load_data(project_root)

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
    from copy import deepcopy

    significant_lags = get_significant_lags(train_data, date_col=date_col, target=y)
    significant_lags = [x for x in significant_lags if x <= macro_horizon and x % 7 == 0]
    lags = deepcopy(significant_lags)
    win_list = [7, 14, 28, 56]

    if set_strategy == "global":
        model_reg = GBTModel(
            params=params_q,
            early_stopping_value=150,
            features=None,
            custom_loss=params_q["objective"],
            weight="covid_weight",
            categorical_features=[],
        )

    else:
        from nostradamus.models.scikit_wrapper import ScikitWrapper
        from sklearn.ensemble import RandomForestRegressor

        model_reg = ScikitWrapper(
            model=RandomForestRegressor(
                max_depth=8,
                n_estimators=200,
                n_jobs=-1,
                min_samples_split=20,
                min_samples_leaf=15,
                max_features=0.85,
                random_state=42,
                ccp_alpha=0.3,
            ),
            params=None,
            features=None,
            categorical_features=[],
            weight="covid_weight",
        )

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

    for transform_strategy in [
        "rolling_zscore",
        "shiftn",
        "None",
        "rolling_mean",
        "rolling_median",
        "log",
        "mean",
        "median",
    ]:
        forecaster = ChainForecaster(
            model=model_reg,
            ts_uid=ts_uid,
            target_str=y,
            date_str=date_col,
            exogs=exog,
            params_dict=autoreg_dict,
            total_forecast_horizon=181,
            n_jobs=-1,
            n_splits=4,
            transform_strategy=transform_strategy,
            transform_win_size=28,
        )
        # display(test_data)
        forecaster.fit(data=train_data, strategy=set_strategy, optimize=True, n_trials=10)
        test_data = (
            deepcopy(test_data)
            .select(["index", date_col, ts_uid])
            .join(
                (forecaster.predict(full_data).select([date_col, ts_uid, "y_hat"])),
                how="left",
                on=[date_col, ts_uid],
            )
        )
        (test_data.fill_null(0).select(["index", "y_hat"]).rename({"y_hat": "y"})).write_csv(
            project_root / f"out/submit/{set_strategy}_{transform_strategy}_chain_lgb.csv"
        )

        # save valid for evaluation purpose.
        valid_out, metrics_out = forecaster.evaluate(return_valid=True)
        valid_out = valid_out.select([ts_uid, date_col, "y_hat"]).rename(
            {"y_hat": f"{set_strategy}_{transform_strategy}_chain_y_hat"}
        )
        display(metrics_out)
        valid_out.write_csv(project_root / f"out/{set_strategy}_{transform_strategy}_chain_lgb.csv")
