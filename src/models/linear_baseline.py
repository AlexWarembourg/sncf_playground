import pandas as pd
import numpy as np

from typing import List, Union, Optional, Dict
from sklearn.linear_model import RidgeCV
from pygam import GAM
from datetime import datetime, timedelta

from joblib import Parallel, delayed
import polars as pl
from datetime import datetime
from statsmodels.tsa.deterministic import DeterministicProcess


class LogLinear:
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

    def fit_single(self, y, max_dt):
        dp = self.basic_features(end_date=max_dt)
        features = dp.in_sample()
        size = y.shape[0]
        training_obs = size if self.ndays >= size else self.ndays
        features = features[-training_obs:]
        y = y[-training_obs:]
        model = GAM(distribution="normal") if self.use_gam else RidgeCV(cv=4)
        model.fit(features, np.log1p(y))
        return (model, dp)

    def fit(self, data: pl.DataFrame) -> pl.DataFrame:
        self.all_keys = data[self.ts_uid].unique().to_list()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single)(y=df[self.target], max_dt=df[self.date_col].max())
            for key, df in data.group_by([self.ts_uid])
        )
        for key, (model, dp) in zip(data[self.ts_uid].unique(), results):
            self.models[key] = (model, dp)

    def predict_single(self, key: str) -> pl.DataFrame:
        model, dp = self.models[str(key)]
        features = dp.out_of_sample(self.horizon)
        forecast = np.expm1(model.predict(features))
        output = pl.DataFrame()
        output = output.with_columns(
            pl.lit(forecast).alias("y_hat"),
            pl.lit(key[0]).alias(self.ts_uid),
            # pl.lit(X["date"]),
        )
        return output

    def predict(self) -> pl.DataFrame:
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(key) for key in self.all_keys
        )
        results = pl.concat(predictions)
        return results
