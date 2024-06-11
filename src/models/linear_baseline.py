import pandas as pd
import numpy as np

from typing import List, Union, Optional, Dict
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer
from joblib import Parallel, delayed
import polars as pl
from datetime import datetime


class SplineLogLinearRegression:
    def __init__(
        self,
        features: List[str],
        target: str,
        ts_uid: str,
        n_splines: int = 12,
        degree: int = 3,
        n_jobs: int = -1,
        frequency: str = "D",
    ) -> None:
        self.n_splines = n_splines
        self.degree = degree
        self.n_jobs = n_jobs
        self.models = {}
        self.ts_uid = ts_uid
        self.spline_transformer = None
        self.target = target
        self.features = features
        if frequency == "D":
            self.base_d = np.arange(1, 366)[:, None]
        else:
            self.base_d = np.arange(1, 54)[:, None]
        self.set_spline(self.base_d)

    def set_spline(self, ordinal_dt):
        self.spline_transformer = SplineTransformer(
            n_knots=self.n_splines, degree=self.degree
        )
        self.spline_transformer.fit(ordinal_dt)

    def prepare_features(self, X):
        dates = X["date"].dt.ordinal_day().alias("day_of_year").to_numpy()[:, None]
        splines = self.spline_transformer.transform(dates)
        additional_features = X.select(self.features).to_numpy()
        return np.hstack((splines, additional_features))

    def fit_single(self, X, y):
        features = self.prepare_features(X)
        model = RidgeCV(cv=3)
        # WLS
        model.fit(features, np.log1p(y), sample_weight=y / np.sum(y))
        return model

    def fit(self, data):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_single)(df, df[self.target])
            for key, df in data.group_by([self.ts_uid])
        )
        for key, model in zip(data[self.ts_uid].unique(), results):
            self.models[key] = model

    def predict_single(self, key, X):
        features = self.prepare_features(X)
        model = self.models[str(key[0])]
        forecast = np.expm1(model.predict(features))
        output = pl.DataFrame()
        output = output.with_columns(
            pl.lit(forecast).alias("y_hat"),
            pl.lit(key[0]).alias(self.ts_uid),
            pl.lit(X["date"]),
        )
        return output

    def predict(self, data):
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self.predict_single)(key, df)
            for key, df in data.group_by([self.ts_uid])
        )
        results = pl.concat(predictions)
        return results
