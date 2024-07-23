import polars as pl
from typing import List
import numpy as np
import pandas as pd
from statsforecast.utils import ConformalIntervals
from datetime import timedelta
from typing import Tuple, List
from nostradamus.analysis.metrics import display_metrics
from nostradamus.preprocessing.quality import minimum_length_uid


class StatsBaseline:
    """_summary_"""

    def __init__(
        self,
        models: List,
        ts_uid: str,
        date_col: str,
        target: str,
        forecast_horizon: int,
        fill_strategy: str = "forward",
        frequency: str = "1d",
        levels: List[int] = [99],
        conformalised: bool = False,
        n_conformalised_win: int = 2,
        fitted: bool = False,
    ):
        self.models = models
        self.fill_strategy = fill_strategy
        self.ts_uid = ts_uid
        self.forecast_horizon = forecast_horizon
        self.date_col = date_col
        self.target = target
        self.n_conformalised_win = n_conformalised_win
        self.frequency = frequency
        self.conformalised = conformalised
        self.levels = levels
        self.fitted = fitted
        self.forecast_range = np.arange(self.forecast_horizon)

    def fill_gap(self, x: pl.DataFrame, date_range):
        """_summary_

        Args:
            x (pl.DataFrame): _description_
            date_range (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            pl.DataFrame({self.ts_uid: x[self.ts_uid][0], self.date_col: date_range})
            .join(x, on=self.date_col, how="left")[[self.ts_uid, self.target, self.date_col]]
            .with_columns(pl.col(self.target).fill_null(strategy=self.fill_strategy))
        )

    def nixtla_reformat(self, Y_df: pl.DataFrame, date_col: str) -> pl.DataFrame:
        """_summary_

        Args:
            Y_df (pl.DataFrame): _description_
            date_col (str): _description_

        Returns:
            pl.DataFrame: _description_
        """
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
            .map_groups(lambda x: self.fill_gap(x, r))
            .rename({date_col: "ds", self.ts_uid: "unique_id", self.target: "y"})
            .with_columns(pl.col("ds").cast(pl.Date).alias("ds"))
        )
        return Y_df

    def ensemble(self, forecasts_df):
        """_summary_

        Args:
            forecasts_df (_type_): _description_

        Returns:
            _type_: _description_
        """
        all_col = forecasts_df.columns
        lb_cols = list(filter(lambda x: "-lo-" in x, all_col))
        ub_cols = list(filter(lambda x: "-hi-" in x, all_col))
        point_fcst_cols = list(
            filter(lambda x: x not in ["ds", "unique_id"] + lb_cols + ub_cols, all_col)
        )
        if len(point_fcst_cols) > 1:
            forecasts_df = forecasts_df.with_columns(
                [
                    pl.concat_list(point_fcst_cols)
                    .list.mean()
                    .alias("arithmetic_forecast_ensamble"),
                    pl.concat_list(lb_cols).list.mean().alias("arithmetic_lower_bound_ensamble"),
                    pl.concat_list(ub_cols).list.mean().alias("arithmetic_upper_bound_ensamble"),
                ]
            )
        self.point_fcst_cols = point_fcst_cols
        return forecasts_df

    def temporal_train_test_split(
        self, data: pl.DataFrame, date_col: str
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """_summary_

        Args:
            data (pl.DataFrame): _description_
            date_col (str): _description_

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: _description_
        """
        date_series = data[date_col].cast(pl.Date)
        max_available_date = date_series.max()
        self.min_available_date = date_series.min()
        self.start_valid = max_available_date - timedelta(days=int(max(self.forecast_range)))
        self.end_valid = max_available_date
        # retrieve max date from data
        train = data.filter(pl.col(date_col) < self.start_valid)
        valid = data.filter(
            pl.col(date_col).is_between(self.start_valid, self.end_valid, closed="both")
        )
        return train, valid

    def forecast(self, Y_df: pl.DataFrame) -> pl.DataFrame:
        """_summary_

        Args:
            Y_df (pl.DataFrame): _description_

        Returns:
            pl.DataFrame: _description_
        """
        if self.conformalised:
            min_length = self.forecast_horizon * self.n_conformalised_win
            allowed_uid = minimum_length_uid(
                Y_df,
                uid="unique_id",
                time="ds",
                min_length=min_length,
            )
            self.fall_back_uid = (
                Y_df.filter(~pl.col("unique_id").is_in(allowed_uid))
                .select(pl.col("unique_id").unique())
                .to_numpy()
                .tolist()
            )

        forecasts_df = (
            self.models.fit(Y_df.filter(pl.col("unique_id").is_in(allowed_uid)))
            .forecast(
                fitted=self.fitted,
                level=self.levels,
                prediction_intervals=(
                    ConformalIntervals(h=self.forecast_horizon, n_windows=self.n_conformalised_win)
                    if self.conformalised
                    else None
                ),
                h=self.forecast_horizon,
            )
            .pipe(self.ensemble)
        )
        return forecasts_df

    def evaluate_on_valid(self, Y_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """_summary_

        Args:
            Y_df (pl.DataFrame): _description_

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: _description_
        """
        train, valid = self.temporal_train_test_split(Y_df, date_col="ds")
        forecast_valid = self.forecast(train)
        valid = valid.join(forecast_valid, how="left", on=["ds", "unique_id"])
        metrics = []
        for col in self.point_fcst_cols:
            metrics.append(
                display_metrics(
                    valid[self.target].fill_null(0).fill_nan(0).to_numpy().ravel(),
                    valid[col].fill_null(0).fill_nan(0).to_numpy().ravel(),
                    name=str(col),
                )
            )
        metrics = pl.from_pandas(pd.concat(metrics))
        return valid, metrics
