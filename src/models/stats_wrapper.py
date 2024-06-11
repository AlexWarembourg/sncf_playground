import polars as pl
from typing import List
import numpy as np
from statsforecast.utils import ConformalIntervals
from datetime import timedelta
from typing import Tuple, List
from src.analysis.metrics import display_metrics


class StatsBaseline:

    def __init__(
        self,
        models: List,
        ts_uid: str,
        date_col: str,
        target: str,
        forecast_horion: int,
        fill_strategy: str = "forward",
        frequency: str = "1d",
        levels: List[int] = [95],
        conformalised: bool = False,
        fitted: bool = False,
    ):
        self.models = models
        self.fill_strategy = fill_strategy
        self.ts_uid = ts_uid
        self.forecast_horion = forecast_horion
        self.date_col = date_col
        self.target = target
        self.frequency = frequency
        self.conformalised = conformalised
        self.levels = levels
        self.fitted = fitted

    def fill_gap(self, x: pl.DataFrame, date_range):
        return (
            pl.DataFrame({self.ts_uid: x[self.ts_uid][0], self.date_col: date_range})
            .join(x, on=self.date_col, how="left")[
                [self.ts_uid, self.target, self.date_col]
            ]
            .with_columns(pl.col(self.target).fill_null(strategy=self.fill_strategy))
        )

    def nixtla_reformat(self, Y_df: pl.DataFrame) -> pl.DataFrame:
        # get min,  max date
        mn, mx = (
            Y_df[self.date_col].cast(pl.Date).min(),
            Y_df[self.date_col].cast(pl.Date).max(),
        )
        # construct the range
        r = pl.date_range(mn, mx, self.frequency, eager=True)
        # group by "id" and fill the missing dates
        Y_df = (
            Y_df.groupby(self.ts_uid, maintain_order=True)
            .apply(lambda x: fill_gap(x, r))
            .rename({self.date_col: "ds", self.ts_uid: "unique_id", self.target: "y"})
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
        if len(point_fcst_cols) > 1:
            forecasts_df["arithmetic_forecast_ensamble"] = forecasts_df[
                point_fcst_cols
            ].mean(axis=1)
            forecasts_df["arithmetic_lower_bound_ensamble"] = forecasts_df[
                lb_cols
            ].mean(axis=1)
            forecasts_df["arithmetic_upper_bound_ensamble"] = forecasts_df[
                ub_cols
            ].mean(axis=1)
        self.point_fcst_cols = point_fcst_cols
        return forecasts_df

    def temporal_train_test_split(
        self, data: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        date_series = data[self.date_str].cast(pl.Date)
        max_available_date = date_series.max()
        self.min_available_date = date_series.min()
        # suppose we need to predict timestep 30 to 60
        # that means we have a range from 30 to 60 and we must shift by 60 and our validation is also 60
        self.start_valid = max_available_date - timedelta(
            days=int(max(self.forecast_range))
        )
        self.end_valid = max_available_date
        # retrieve max date from data
        train = data.filter(pl.col(self.date_col) < self.start_valid)
        valid = data.filter(
            pl.col(self.date_col).is_between(
                self.start_valid, self.end_valid, closed="both"
            )
        )
        return train, valid

    def forecast(self, Y_df):
        forecasts_df = (
            self.models.fit(Y_df)
            .forecast(
                fitted=self.fitted,
                level=self.levels,
                prediction_intervals=(
                    ConformalIntervals(h=self.forecast_horion, n_windows=2)
                    if self.conformalised
                    else None
                ),
                h=self.forecast_horion,
            )
            .reset_index()
            .pipe(self.ensemble)
        )
        return forecasts_df

    def evaluate_on_valid(self, Y_df):
        train, valid = self.temporal_train_test_split(Y_df)
        forecast_valid = self.forecast(train)
        metrics = pl.DataFrame()
        for col in self.point_fcst_cols:
            metrics.append(display_metrics(valid[self.target], forecast_valid[col]))
        return metrics


def fill_gap(
    x: pl.DataFrame,
    date_range,
    exog: List[str],
    ts_uid: str,
    target: str,
    date_col: str,
    fill_strategy: str = "forward",
):
    return (
        pl.DataFrame({ts_uid: x[ts_uid][0], date_col: date_range})
        .join(x, on=date_col, how="left")[[ts_uid, target, date_col] + exog]
        .with_columns(pl.col(target).fill_null(strategy=fill_strategy))
    )


def nixtla_reformat(
    Y_df: pl.DataFrame,
    exog: List[str],
    date_col: str = "date",
    ts_uid: str = "ts_uid",
    y_col: str = "y",
    frequency: str = "1d",
) -> pl.DataFrame:
    # get min,  max date
    mn, mx = Y_df[date_col].cast(pl.Date).min(), Y_df[date_col].cast(pl.Date).max()
    # construct the range
    r = pl.date_range(mn, mx, frequency, eager=True)
    # group by "id" and fill the missing dates
    Y_df = (
        Y_df.groupby(ts_uid, maintain_order=True)
        .apply(
            lambda x: fill_gap(
                x,
                r,
                ts_uid=ts_uid,
                target=y_col,
                date_col=date_col,
                exog=exog,
            )
        )
        .rename({date_col: "ds", ts_uid: "unique_id", y_col: "y"})
        .with_columns(pl.col("ds").cast(pl.Date).alias("ds"))
    )
    return Y_df


def nixtla_fast_baseline(
    df,
    sf_models,
    date_col,
    uid,
    y_col,
    horizon,
    exog: List[str],
    frequency="1d",
    fitted=False,
    levels=[95],
    conformal=False,
    to_pd: bool = False,
):

    nixtla_df = nixtla_reformat(
        Y_df=df,
        date_col=date_col,
        ts_uid=uid,
        y_col=y_col,
        frequency=frequency,
        exog=exog,
    )
    nixtla_df = nixtla_df.to_pandas() if to_pd else nixtla_df
    forecasts_df = (
        sf_models.fit(nixtla_df)
        .forecast(
            fitted=fitted,
            level=levels,
            prediction_intervals=(
                ConformalIntervals(h=horizon, n_windows=2) if conformal else None
            ),
            h=horizon,
        )
        .reset_index()
    )

    all_col = forecasts_df.columns
    lb_cols = list(filter(lambda x: "-lo-" in x, all_col))
    ub_cols = list(filter(lambda x: "-hi-" in x, all_col))
    point_fcst_cols = list(
        filter(lambda x: x not in ["ds", "unique_id"] + lb_cols + ub_cols, all_col)
    )
    if len(point_fcst_cols) > 1:
        forecasts_df["arithmetic_forecast_ensamble"] = forecasts_df[
            point_fcst_cols
        ].mean(axis=1)
        forecasts_df["arithmetic_lower_bound_ensamble"] = (
            forecasts_df[lb_cols].mean(axis=1).clip(0, None)
        )
        forecasts_df["arithmetic_upper_bound_ensamble"] = (
            forecasts_df[ub_cols].clip(0, None).mean(axis=1)
        )

    return forecasts_df


"""
def timegpt_draft(timeseries_df):
    from nixtla import NixtlaClient

    # api_key = "AHAHAH"
    api_key = "nixtla-tok-6dK88U0aPDIJocOOkowvJStVboPWdeNdd4jmN5FlHoxndK1PQDU1TXdw21CLpY5LRM9rJHsghmpVVY4O"
    nixtla_client = NixtlaClient(api_key=api_key)

    fcst_df = nixtla_client.forecast(
        df=timeseries_df,
        # X_df=future_exog_df,
        h=181,
        finetune_steps=40,  # Specify the number of steps for fine-tuning
        finetune_loss="mse",  # Use the MAE as the loss function for fine-tuning
        model="timegpt-1-long-horizon",  # Use the model for long-horizon forecasting
        time_col="ds",
        target_col="y",
        id_col="unique_id",
        # date_features=["dayofweek"],
    )

    fcst_df["TimeGPT"] = np.expm1(fcst_df["TimeGPT"])
    fcst_df["TimeGPT"] = fcst_df["TimeGPT"].clip(0)
    return fcst_df
"""
