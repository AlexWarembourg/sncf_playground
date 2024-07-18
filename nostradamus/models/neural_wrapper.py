from typing import List, Union, Dict, Tuple
from nostradamus.analysis.metrics import display_metrics

import polars as pl
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class NeuralWrapper:

    def __init__(
        self,
        models: List,
        ts_uid: str,
        date_col: str,
        target: str,
        exog: List[str],
        forecast_horizon: int,
        fill_strategy: str = "forward",
        frequency: str = "1d",
        levels: List[int] = [95],
        conformalised: bool = False,
        fitted: bool = False,
    ):
        self.models = models
        self.fill_strategy = fill_strategy
        self.ts_uid = ts_uid
        self.forecast_horizon = forecast_horizon
        self.date_col = date_col
        self.target = target
        self.frequency = frequency
        self.conformalised = conformalised
        self.levels = levels
        self.exog = exog
        self.fitted = fitted
        self.forecast_range = np.arange(self.forecast_horizon)

    def fill_gap(self, x: pl.DataFrame, date_range, exog):
        base_join = [self.ts_uid, self.target, self.date_col]
        base_join += exog if exog is not None else []
        return (
            pl.DataFrame({self.ts_uid: x[self.ts_uid][0], self.date_col: date_range})
            .join(x, on=self.date_col, how="left")[base_join]
            .with_columns(pl.col(self.target).fill_null(strategy=self.fill_strategy))
        )

    def nixtla_reformat(
        self, Y_df: pl.DataFrame, date_col: str, exog: List[str] = None
    ) -> pl.DataFrame:
        # get min,  max date
        mn, mx = (
            Y_df[date_col].cast(pl.Date).min(),
            Y_df[date_col].cast(pl.Date).max(),
        )
        # construct the range
        r = pl.date_range(mn, mx, self.frequency, eager=True)
        # group by "id" and fill the missing dates
        Y_df = (
            Y_df.group_by(self.ts_uid, maintain_order=True)
            .map_groups(lambda x: self.fill_gap(x, r, exog))
            .rename({date_col: "ds", self.ts_uid: "unique_id", self.target: "y"})
            .with_columns(pl.col("ds").cast(pl.Date).alias("ds"))
        )
        return Y_df

    def ensemble(self, forecasts_df):
        all_col = forecasts_df.columns
        lb_cols = list(filter(lambda x: "-lo-" in x, all_col))
        ub_cols = list(filter(lambda x: "-hi-" in x, all_col))
        point_fcst_cols = list(
            filter(lambda x: x not in ["ds", "unique_id"] + lb_cols + ub_cols, all_col)
        )
        ensamble_expr = []
        for col_group in [lb_cols, ub_cols, point_fcst_cols]:
            if len(col_group) > 0:
                if "lo" in col_group[0]:
                    fname = "lower_bound"
                elif "hi" in col_group[0]:
                    fname = "upper_bound"
                else:
                    fname = "forecast"
                ensamble_expr.append(
                    pl.concat_list(col_group).list.mean().alias(f"ensamble_{fname}")
                )
        forecasts_df = forecasts_df.with_columns(ensamble_expr)
        self.point_fcst_cols = point_fcst_cols
        return forecasts_df

    def temporal_train_test_split(
        self, data: pl.DataFrame, date_col: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        date_series = data[date_col].cast(pl.Date)
        max_available_date = date_series.max()
        self.min_available_date = date_series.min()
        self.start_valid = max_available_date - timedelta(
            days=int(max(self.forecast_range))
        )
        self.end_valid = max_available_date
        # retrieve max date from data
        train = data.filter(pl.col(date_col) < self.start_valid)
        valid = data.filter(
            pl.col(date_col).is_between(self.start_valid, self.end_valid, closed="both")
        )
        return train, valid

    def forecast(
        self, Y_df: pl.DataFrame, val_size: int, future_df: pl.DataFrame = None
    ):
        self.models.fit(Y_df, val_size=val_size)
        futr_df = self.models.get_missing_future(future_df)

        if futr_df.shape[0] > 0:
            print(f'there"s missing : {futr_df.shape[0]}')
            future_df = pl.concat((future_df, futr_df), how="diagonal")

        forecasts_df = (
            self.models.predict()
            if len(self.exog) == 0 and future_df is not None
            else self.models.predict(futr_df=future_df)
        )
        forecasts_df = self.ensemble(forecasts_df=forecasts_df)
        # print(forecasts_df)
        return forecasts_df

    def evaluate_on_valid(self, Y_df, val_size):
        train, valid = self.temporal_train_test_split(Y_df, date_col="ds")
        forecast_valid = self.forecast(
            train.filter(pl.col("unique_id").is_in(valid["unique_id"].unique())),
            val_size=val_size,
            future_df=valid,
        )
        metrics = []
        for col in self.point_fcst_cols:
            metrics.append(
                display_metrics(
                    valid[self.target].to_numpy(),
                    forecast_valid[col].to_numpy(),
                    name=str(col),
                )
            )
        metrics = pd.concat(metrics)
        return forecast_valid, metrics
