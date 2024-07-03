import polars as pl
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Union, List, Optional


class LogTransform:

    @staticmethod
    def transform(
        data: Union[pd.DataFrame, pl.DataFrame], target: str
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        y = data[target].to_numpy().ravel()
        logify = np.log(
            np.float64(y),
            out=np.zeros_like(y, dtype=np.float64),
            where=y != 0,
            dtype=np.float64,
        )
        if isinstance(data, pl.DataFrame):
            data = data.with_columns(pl.lit(logify).alias(target))
        else:
            data[target] = logify
        return data

    @staticmethod
    def inverse_transform(
        data: Union[pd.DataFrame, pl.DataFrame], target: str
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        y = data[target]
        reverse_log_y = np.exp(y)
        if isinstance(data, pl.DataFrame):
            data = data.with_columns(pl.lit(reverse_log_y).alias(target))
        else:
            data[target] = reverse_log_y
        return data


class TargetTransform:
    # [todo] : nettoyer la class c'est le bordel ..

    def __init__(
        self,
        ts_uid: Union[List[str], str],
        target: str,
        strategy: str,
        time_col: str,
        forecast_horizon: Optional[int] = 1,
        win_size: Optional[int] = 1,
    ):
        self.target = target
        self.ts_uid = ts_uid
        self.time_col = time_col
        self.forecast_horizon = forecast_horizon
        self.win_size = win_size
        self.allowed_strategy = [
            "log",
            "mean",
            "median",
            "rolling_mean",
            "rolling_median",
            "rolling_zscore",
            "shiftn",
            "None",
        ]
        if strategy not in self.allowed_strategy:
            raise ValueError(f"Strategy must be one of those : {self.allowed_strategy}")
        else:
            self.strategy = strategy
            self.agg_fname = f"agg_{target}_{self.strategy}"
            self.join_key = (
                [self.ts_uid, self.time_col]
                if self.strategy not in ["mean", "median"]
                else [self.ts_uid]
            )
        self.fitted = False

    def make_future_dataframe(self, end_current, uids: List[str] = None):
        current_start_dt = end_current + timedelta(days=int(1))
        df_dates = pl.date_range(
            start=current_start_dt,
            end=current_start_dt + timedelta(days=int(self.forecast_horizon)),
            eager=True,
        ).to_frame("date")
        # define future_df
        future_df = df_dates.join(
            uids,
            how="cross",
        ).with_columns(pl.lit(np.nan).alias(self.target))
        return future_df

    def incremental_rols_fill_null(
        self, data: pl.DataFrame, agg_fname: str
    ) -> pl.DataFrame:
        data = (
            data.with_columns(
                pl.coalesce(
                    pl.col(agg_fname),
                    pl.col(agg_fname).forward_fill().over(self.ts_uid).alias(agg_fname),
                )
            )
            # forward is okay after backward
            .with_columns(
                pl.coalesce(
                    pl.col(agg_fname),
                    pl.col(agg_fname)
                    .backward_fill()
                    .over(self.ts_uid)
                    .alias(agg_fname),
                )
            )
        )
        return data

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.strategy == "None":
            data = data.with_columns(pl.lit(0).alias(self.agg_fname))
            self.aggregates = pl.DataFrame()
        elif self.strategy == "log":
            loggifier = LogTransform()
            data = loggifier.transform(data=data, target=self.target)
        elif self.strategy in ["mean", "median"]:
            grouped_data = data.sort(by=[self.ts_uid, self.time_col]).groupby(
                self.join_key
            )
            self.aggregates = grouped_data.agg(
                getattr(pl.col(self.target), self.strategy)().alias(self.agg_fname)
            )
            data = data.join(self.aggregates, how="left", on=self.join_key)
        elif self.strategy in ["rolling_mean", "rolling_median"]:
            # extend date.
            futr_df = self.make_future_dataframe(
                uids=data.select(pl.col(self.ts_uid).unique()),
                end_current=data[self.time_col].max(),
            )

            select_exp = [
                pl.col(self.time_col).cast(pl.String).str.to_datetime().cast(pl.Date),
                pl.col(self.ts_uid).cast(pl.String),
                pl.col(self.target).cast(pl.Float32),
            ]

            self.aggregates = (
                pl.concat(
                    (data.select(select_exp), futr_df.select(select_exp)),
                    how="vertical",
                )
                .sort(by=[self.ts_uid, self.time_col])
                .with_columns(
                    getattr(
                        pl.col(self.target).shift(int(self.forecast_horizon)),
                        self.strategy,
                    )(self.win_size)
                    .over(self.ts_uid)
                    .alias(self.agg_fname)
                )
                .drop(self.target)
                .pipe(self.incremental_rols_fill_null, agg_fname=self.agg_fname)
                .unique(subset=[self.ts_uid, self.time_col])
            )

            # join aggregates.
            data = data.with_columns(
                pl.col(self.time_col).cast(pl.String).str.to_datetime().cast(pl.Date)
            ).join(self.aggregates, how="left", on=self.join_key)

        elif self.strategy == "rolling_zscore":
            # extend date.
            futr_df = self.make_future_dataframe(
                uids=data.select(pl.col(self.ts_uid).unique()),
                end_current=data[self.time_col].max(),
            )

            select_exp = [
                pl.col(self.time_col).cast(pl.String).str.to_datetime().cast(pl.Date),
                pl.col(self.ts_uid).cast(pl.String),
                pl.col(self.target).cast(pl.Float32),
            ]

            self.aggregates = (
                pl.concat(
                    (data.select(select_exp), futr_df.select(select_exp)),
                    how="vertical",
                )
                .sort(by=[self.ts_uid, self.time_col])
                .with_columns(
                    getattr(
                        pl.col(self.target).shift(int(self.forecast_horizon)),
                        "rolling_mean",
                    )(self.win_size)
                    .over(self.ts_uid)
                    .alias(f"mean_{self.agg_fname}"),
                    getattr(
                        pl.col(self.target).shift(int(self.forecast_horizon)),
                        "rolling_std",
                    )(self.win_size)
                    .over(self.ts_uid)
                    .alias(f"std_{self.agg_fname}"),
                )
                .pipe(
                    self.incremental_rols_fill_null, agg_fname=f"mean_{self.agg_fname}"
                )
                .pipe(
                    self.incremental_rols_fill_null, agg_fname=f"std_{self.agg_fname}"
                )
                .drop(self.target)
                .unique(subset=[self.ts_uid, self.time_col])
            )
            # join aggregates.
            data = data.with_columns(
                pl.col(self.time_col).cast(pl.String).str.to_datetime().cast(pl.Date)
            ).join(self.aggregates, how="left", on=self.join_key)

        elif self.strategy == "shiftn":
            futr_df = self.make_future_dataframe(
                uids=data.select(pl.col(self.ts_uid).unique()),
                end_current=data[self.time_col].max(),
            )

            select_exp = [
                pl.col(self.time_col).cast(pl.String).str.to_datetime().cast(pl.Date),
                pl.col(self.ts_uid).cast(pl.String),
                pl.col(self.target).cast(pl.Float32),
            ]

            self.aggregates = (
                pl.concat(
                    (data.select(select_exp), futr_df.select(select_exp)),
                    how="vertical",
                )
                .sort(by=[self.ts_uid, self.time_col])
                .select([self.time_col, self.ts_uid, self.target])
                .with_columns(
                    pl.col(self.time_col)
                    .dt.offset_by(by=f"{self.forecast_horizon}d")
                    .cast(pl.String)
                    .str.to_datetime()
                    .cast(pl.Date)
                    .alias(self.time_col)
                )
                .rename({self.target: self.agg_fname})
                .pipe(self.incremental_rols_fill_null, agg_fname=self.agg_fname)
                .unique(subset=[self.ts_uid, self.time_col])
            )

            # join aggregates.
            data = (
                data.with_columns(
                    pl.col(self.time_col)
                    .cast(pl.String)
                    .str.to_datetime()
                    .cast(pl.Date)
                    .alias(self.time_col)
                )
                .join(self.aggregates, how="left", on=self.join_key)
                .with_columns(
                    pl.coalesce(pl.col(self.agg_fname).cast(pl.Int32), pl.lit(0)).alias(
                        self.agg_fname
                    )
                )
            )

        if self.strategy not in ("log", "None", "rolling_zscore"):
            data = data.with_columns(
                (pl.col(self.target) - pl.col(self.agg_fname)).alias(self.target)
            ).drop(self.agg_fname)
        elif self.strategy == "rolling_zscore":
            data = data.with_columns(
                (
                    (
                        pl.col(self.target).cast(pl.Float32)
                        - pl.col(f"mean_{self.agg_fname}").cast(pl.Float32)
                    )
                    / pl.col(f"std_{self.agg_fname}").cast(pl.Float32)
                ).alias(self.target)
            ).drop([f"std_{self.agg_fname}", f"mean_{self.agg_fname}"])
        self.fitted = True
        return data

    def inverse_transform(self, data: pl.DataFrame, target: str = None) -> pl.DataFrame:
        fname_target = self.target if not target else target
        data = data.with_columns(
            pl.col(self.time_col)
            .cast(pl.String)
            .str.to_datetime()
            .cast(pl.Date)
            .alias(self.time_col)
        )
        if self.fitted:
            if self.strategy not in ("log", "None", "rolling_zscore"):
                output = data.join(
                    self.aggregates.select(self.join_key + [self.agg_fname]),
                    how="left",
                    on=self.join_key,
                )
                output = output.with_columns(
                    (pl.col(fname_target) + pl.col(self.agg_fname)).alias(fname_target)
                )  # .drop(self.agg_fname)
                return output
            elif self.strategy == "log":
                logifier = LogTransform()
                output = logifier.inverse_transform(data=data, target=fname_target)
                return output
            elif self.strategy == "rolling_zscore":
                return data.join(
                    self.aggregates.select(
                        self.join_key
                        + [f"std_{self.agg_fname}", f"mean_{self.agg_fname}"]
                    ),
                    how="left",
                    on=self.join_key,
                ).with_columns(
                    (
                        (pl.col(fname_target) * pl.col(f"std_{self.agg_fname}"))
                        + pl.col(f"mean_{self.agg_fname}")
                    ).alias(fname_target)
                )
            else:
                return data
