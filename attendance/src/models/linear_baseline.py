import pandas as pd
import numpy as np

from typing import List, Union, Optional, Dict
from sklearn.linear_model import RidgeCV
from datetime import timedelta
from pygam import GAM

from joblib import Parallel, delayed
import polars as pl
from datetime import datetime
from statsmodels.tsa.deterministic import DeterministicProcess
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


class FLinear:
    def __init__(
        self,
        features: List[str],
        target: str,
        ts_uid: str,
        n_jobs: int = -1,
        frequency: str = "D",
        use_gam: bool = True,
        date_col: str = "date",
        ndays: int = 364,
        horizon: int = 181,
        forecast_end_dt: str = None,
        smoothing: bool = True,
    ) -> None:
        self.n_jobs = n_jobs
        self.models = {}
        self.ts_uid = ts_uid
        self.spline_transformer = None
        self.target = target
        self.features = features
        self.date_col = date_col
        self.frequency = frequency
        self.use_gam = use_gam
        self.ndays = ndays
        self.horizon = horizon
        self.forecast_end_dt = forecast_end_dt
        self.smoothing = smoothing

    def basic_features(self, end_date) -> pl.DataFrame:
        index = pd.date_range(
            end_date - timedelta(days=self.ndays), end_date, freq=self.frequency
        )
        dp = DeterministicProcess(
            index=index,
            constant=True,
            order=1,
            seasonal=True,
            fourier=0,
            additional_terms=(),
            drop=True,
        )
        return dp

    def backup_model(self, y: pl.Series, n: int = 30) -> np.ndarray:
        scalar = np.mean(y[-n:]) if len(y) > n else np.mean(y)
        return np.tile(
            scalar,
            self.horizon,
        )

    def fit_single(self, y, max_dt):
        size = y.shape[0]
        dp = self.basic_features(end_date=max_dt)
        features = dp.in_sample()
        training_obs = size if self.ndays >= size else self.ndays
        features = features[-training_obs:]
        y = np.log(
            sm.nonparametric.lowess(
                exog=np.arange(len(y[-training_obs:])),
                endog=y[-training_obs:],
                frac=1.0 / 3,
            )[:, 1]
            if self.smoothing
            else y[-training_obs:]
        )
        if size > 364:
            model = (
                GAM(distribution="normal", link="identity")
                if self.use_gam
                else RidgeCV(cv=4)
            )
            model.fit(features, y)
            return (model, dp, True)
        else:
            return (y, dp, False)

    def fit(self, data: pl.DataFrame) -> pl.DataFrame:
        self.all_keys = data[self.ts_uid].unique().to_list()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single)(y=df[self.target], max_dt=df[self.date_col].max())
            for key, df in data.group_by([self.ts_uid])
        )
        for key, (model, dp, statut) in zip(data[self.ts_uid].unique(), results):
            self.models[key] = (model, dp, statut)

    def predict_single(self, key: str) -> pl.DataFrame:
        model, dp, state = self.models[str(key)]
        end_times = dp.index.max() + timedelta(days=self.horizon)
        forecast_horizon = self.horizon
        resize = False
        if end_times != self.forecast_end_dt:
            add = abs(
                (
                    (dp.index.max() + timedelta(days=self.horizon))
                    - pd.to_datetime(end_times)
                ).days
            )
            forecast_horizon = self.horizon + add
            resize = True
        features = dp.out_of_sample(forecast_horizon)
        features = (
            features.loc[features.index.max() - timedelta(days=self.horizon) :]
            if resize
            else features
        )
        dt = features.index.to_numpy().ravel()
        forecast = (
            np.clip(np.exp(model.predict(features)), a_min=0, a_max=None)
            if state
            else np.exp(self.backup_model(y=model, n=90))
        )
        output = pl.DataFrame()
        output = output.with_columns(
            pl.lit(forecast).alias("y_hat"),
            pl.lit(key).alias(self.ts_uid),
            pl.lit(dt).alias(self.date_col),
        )
        return output

    def predict(self) -> pl.DataFrame:
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(key) for key in self.all_keys
        )
        results = pl.concat(predictions)
        return results
