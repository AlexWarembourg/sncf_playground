from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from datetime import timedelta
from IPython.display import display
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy

import polars as pl
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe

from src.preprocessing.lags import (
    pl_compute_lagged_features,
    pl_compute_moving_features,
    compute_autoreg_features,
)
from src.analysis.describe import timeseries_length
from src.analysis.metrics import display_metrics


class DirectForecaster:

    def __init__(
        self,
        model,
        ts_uid: str,
        exogs: List[str],
        forecast_range: Union[List[int], np.ndarray, pl.Series],
        features_params: Dict[str, Dict[str, List[Union[str, int]]]],
        total_forecast_horizon: int = 1,
        target_str: str = "y",
        date_str: str = "date",
        scaler: str = None,
        n_jobs: int = -1,
    ):
        self.model = model
        self.ts_uid = ts_uid
        self.target_str = target_str
        self.date_str = date_str
        self.exogs = exogs
        self.n_jobs = n_jobs
        self.forecast_range = forecast_range
        self.total_forecast_horizon = total_forecast_horizon
        self.forecast_horizon = max(self.forecast_range) + 1
        self.fitted = False
        self.ar_features = None
        self.scaler = scaler
        self.output_name = "y_hat"
        self.features_params = self.update_params_dict(
            deepcopy(features_params), horizon=self.forecast_horizon
        )
        self.initial_trim = max(
            max(self.features_params[ts_uid]["wins"])
            + max(self.features_params[ts_uid]["shifts"]),
            max(self.features_params[ts_uid]["lags"]),
        )
        self.strategy = None
        self.except_list = []

        if scaler is not None:
            assert scaler.name in [
                "robust",
                "standard",
                "temporal",
            ], 'Le Scaler specifié n"est pas connus'

        mandatory_keys = ["groups", "horizon", "wins", "shifts", "lags", "funcs"]

        for key in self.features_params.keys():
            keys = list(self.features_params[key].keys())
            intersct = np.intersect1d(mandatory_keys, keys)
            assert len(keys) == len(
                intersct
            ), "Shape not match, that imply there's missing or foreign key in the dictionnary"
            # print(keys, intersct)

    def update_params_dict(
        self, params: Dict[str, Dict[str, List[Union[str, int]]]], horizon: int
    ):
        for key in params.keys():
            params[key]["horizon"] = params[key]["horizon"](horizon)
            params[key]["shifts"] = params[key]["shifts"](horizon)
            params[key]["lags"] = params[key]["lags"](horizon)
        return params

    def set_date_scope(self, date_series: pl.Series) -> None:
        # max_available_date - this is the maximal date available let's it's 1000
        max_available_date = date_series.max()
        self.min_available_date = date_series.min()
        # suppose we need to predict timestep 30 to 60
        # that means we have a range from 30 to 60 and we must shift by 60 and our validation is also 60
        self.start_valid = max_available_date - timedelta(
            days=int(max(self.forecast_range))
        )
        self.end_valid = max_available_date
        # forecast is de facto the max available date + min of the range and the max of the range
        self.start_forecast = max_available_date + timedelta(
            days=int(min(self.forecast_range) + 1)
        )
        self.end_forecast = max_available_date + timedelta(
            days=int(max(self.forecast_range) + 1)
        )

    def trim(self, data: polars_dataframe) -> polars_dataframe:
        trim_dt = self.min_available_date + timedelta(days=int(self.initial_trim))
        return data.filter(pl.col(self.date_str) >= pl.lit(trim_dt))

    def temporal_train_test_split(
        self, data: polars_dataframe
    ) -> Tuple[polars_dataframe, polars_dataframe]:
        # retrieve max date from data
        train = data.filter(pl.col(self.date_str) < self.start_valid)
        valid = data.filter(
            pl.col(self.date_str).is_between(
                self.start_valid, self.end_valid, closed="both"
            )
        )
        return train, valid

    def prepare(
        self, data: polars_dataframe
    ) -> Tuple[polars_dataframe, polars_dataframe]:
        self.uids = data[[self.ts_uid]].unique()
        # this is the maximal date available let's it's 100
        self.set_date_scope(data[self.date_str].cast(pl.Date))
        # if we need to predict
        if self.scaler is not None:
            self.scaler.fit(data)
            data = self.scaler.transform(data)
        # compute features engineering and split.
        train, valid = (
            data
            # .pipe(self.compute_autoregressive_features)
            .pipe(
                compute_autoreg_features,
                target_col=self.target_str,
                date_str=self.date_str,
                auto_reg_params=self.features_params,
            )
            .pipe(self.trim)
            .pipe(self.temporal_train_test_split)
        )
        self.ar_features = list(filter(lambda x: x.startswith("ar_"), train.columns))
        self.features = self.exogs + self.ar_features
        self.model.features = self.features
        return train, valid

    def fit_local(self, train_data: polars_dataframe) -> None:
        except_uid = (
            timeseries_length(train_data, ts_uid=self.ts_uid, date=self.date_str)
            .filter(pl.col("date") <= self.initial_trim)
            .select(pl.col(self.ts_uid).unique())
        )[self.ts_uid].to_list()
        train_data = train_data.filter(~pl.col(self.ts_uid).is_in(except_uid))
        self.except_list.append(except_uid)
        self.train, self.valid = self.prepare(train_data)
        # call model and fit with a validation set
        self.models_locals = {}
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(deepcopy(self.model).fit)(
                tr,
                tr.select(self.target_str),
                val,
                val.select(self.target_str),
                return_object=True,
            )
            for (_, tr), (_, val) in tqdm(
                zip(
                    self.train.group_by([self.ts_uid]),
                    self.valid.group_by([self.ts_uid]),
                )
            )
        )
        for key, model in zip(train_data[self.ts_uid].unique(), results):
            self.models_locals[key] = model

    def fit_global(self, train_data: polars_dataframe) -> None:
        self.train, self.valid = self.prepare(train_data)
        # call model and fit with a validation set
        # model must have a quadruple fit method with an early stopping setting on the validation set.
        self.model.fit(
            self.train,
            self.train.select(self.target_str),
            self.valid,
            self.valid.select(self.target_str),
        )

    def fit(self, train_data: polars_dataframe, strategy: str = "global"):
        if strategy == "global":
            self.fit_global(train_data)
        elif strategy == "local":
            self.fit_local(train_data)
        else:
            raise ValueError("Unknown Strategy")
        self.strategy = strategy
        self.fitted = True

    def predict_global(
        self, x_test: Union[polars_dataframe, pandas_dataframe] = None
    ) -> np.ndarray:
        y_hat = self.model.predict(x_test)
        x_test = x_test.with_columns(pl.lit(y_hat).alias(self.output_name))
        if self.scaler is not None:
            x_test = self.scaler.inverse_transform(x_test, target=self.output_name)
        return x_test

    def single_predict(self, models_list: list, key: str, X):
        try:
            model = models_list[str(key[0])]
            output = model.predict(X)
            X = X.with_columns(pl.lit(output).alias(self.output_name))
            return X
        except KeyError:
            self.except_list.append(key[0])
            msg = f"The following uids cannot be forecasted. : {key}"
            print(msg)
            """
            if len(self.except_list) > 0:
                missing_uids = self.make_future_dataframe(uids=[key[0]])
                msg = f"The following uids cannot be forecasted. : {key}"
                print(msg)
                missing_df = missing_uids.with_columns(
                    pl.lit(np.nan).alias(self.output_name)
                ).select([self.date_str, self.ts_uid, self.output_name])
                return missing_df
            """

    def predict_local(
        self, x_test: Union[polars_dataframe, pandas_dataframe] = None
    ) -> np.ndarray:
        grouped_dataframe = x_test.group_by([self.ts_uid])
        conditional_back = (
            "threading" if x_test.select(self.ts_uid).n_unique() < 1000 else "loky"
        )
        predictions = Parallel(n_jobs=self.n_jobs, backend=conditional_back)(
            delayed(self.single_predict)(self.models_locals, key, df)
            for key, df in grouped_dataframe
        )
        predictions = pl.concat(list(filter(lambda x: x is not None, predictions)))
        return predictions

    def predict(self, x_test: Union[polars_dataframe, pandas_dataframe] = None):
        if x_test is None:
            x_test = self.make_future_dataframe()
            # reconstruct dataset
            x_test = pl.concat(
                (
                    self.train.select(x_test.columns),
                    self.valid.select(x_test.columns),
                    x_test,
                ),
                how="vertical_relaxed",
            )
        shape = x_test[self.date_str].n_unique()
        assert (
            shape >= self.initial_trim
        ), f"the test length must contains at least {self.initial_trim} observation to compute feature engineering"
        max_dt = x_test[self.date_str].max()
        minimal_cut = max_dt - timedelta(
            days=int(self.initial_trim + self.forecast_horizon)
        )
        x_test = (
            x_test.filter(pl.col(self.date_str) >= minimal_cut)
            .pipe(
                compute_autoreg_features,
                target_col=self.target_str,
                date_str=self.date_str,
                auto_reg_params=self.features_params,
            )
            .filter(
                pl.col(self.date_str).is_between(self.start_forecast, self.end_forecast)
            )
        )
        if self.strategy == "global":
            return self.predict_global(x_test=x_test)
        elif self.strategy == "local":
            return self.predict_local(x_test=x_test)
        else:
            raise ValueError("Unknown Strategy")

    def make_future_dataframe(self, uids: List[str] = None):
        df_dates = pl.date_range(
            start=self.start_forecast,
            end=self.end_forecast,
            eager=True,
        ).to_frame("date")
        future_df = df_dates.join(
            (
                self.uids
                if uids is None
                else self.uids.filter(pl.col(self.ts_uid).is_in(uids))
            ),
            how="cross",
        )
        return future_df

    def evaluate(self):
        if self.fitted:
            y_real = self.valid[self.target_str].to_numpy()
            if self.strategy == "global":
                y_hat = (
                    self.predict_global(x_test=self.valid)
                    .select(self.output_name)
                    .to_numpy()
                    .flatten()
                )
            elif self.strategy == "local":
                y_hat = (
                    self.predict_local(x_test=self.valid)
                    .select(self.output_name)
                    .to_numpy()
                    .flatten()
                )
            else:
                raise ValueError("Unknown Strategy")
            metrics_valid = display_metrics(y_real, y_hat)
            return metrics_valid
        else:
            raise ValueError("Model is not fitted.")

    def backtest(self, data):
        pass
