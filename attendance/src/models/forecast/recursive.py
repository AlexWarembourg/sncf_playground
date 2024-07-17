from typing import List, Optional, Union, Tuple
import numpy as np
from datetime import timedelta

import polars as pl
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe

from src.models.forecast.direct import DirectForecaster


class RecursiveForecaster:

    def __init__(
        self,
        model,
        ts_uid: str,
        lags: List[int],
        windows: List[int],
        shifts: List[int],
        funcs: List[str],
        exogs: List[str],
        target_str: str = "y",
        date_str: str = "date",
    ):
        self.fitted = False
        self.direct_wrapper = self.DirectForecaster(
            model=model,
            ts_uid=ts_uid,
            lags=lags,
            windows=windows,
            shifts=shifts,
            funcs=funcs,
            target_str=target_str,
            date_str=date_str,
            forecast_horizon=1,
            exogs=exogs,
        )

    def fit(self, data: polars_dataframe) -> None:
        self.direct_wrapper.fit(data)
        self.fitted = True

    def predict_single(self, x_test):
        pass

    def backtest(self, data):
        pass

    def predict(self, x_test: Union[polars_dataframe, pandas_dataframe]) -> np.ndarray:
        return self.model.predict(x_test)
