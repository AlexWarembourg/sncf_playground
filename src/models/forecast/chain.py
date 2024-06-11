from typing import List, Optional, Union, Tuple
import numpy as np
from datetime import timedelta
from IPython.display import display
from joblib import Parallel, delayed

import polars as pl
from copy import deepcopy
from forecast.direct import DirectForecaster
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe

from src.preprocessing.polars.lags import (
    pl_compute_lagged_features,
    pl_compute_moving_features,
)
from src.analysis.metrics import display_metrics


class ChainForecaster:

    def __init__(
        self,
        model,
        n_splits: int,
        ts_uid: str,
        lags: List[int],
        windows: List[int],
        funcs: List[str],
        exogs: List[str],
        total_forecast_horizon: int = 1,
        target_str: str = "y",
        date_str: str = "date",
    ):
        self.chains = np.array_split(np.arange(total_forecast_horizon), n_splits)
        self.model = model
        self.ts_uid = ts_uid
        self.target_str = target_str
        self.date_str = date_str
        self.windows = windows
        self.funcs = funcs
        self.exogs = exogs
        self.total_forecast_horizon = total_forecast_horizon
        self.output_name = "y_hat"
        self.lags = lags

    def fit_single(self, chain_pieces, data, strategy: str = "global"):
        dir_forecaster = DirectForecaster(
            model=deepcopy(self.model),
            ts_uid=self.ts_uid,
            lags=self.lags,
            windows=self.windows,
            # no shift equal to one horizon shift only
            # shifts=shift_list,
            funcs=self.funcs,
            target_str=self.target_str,
            date_str=self.date_str,
            forecast_range=chain_pieces,
            exogs=self.exogs,
            total_forecast_horizon=self.total_forecast_horizon,
            # scaler=PanelStandardScaler(ts_uid=ts_uid, target=y)
        )
        dir_forecaster.fit(data, strategy=strategy)

    def fit(self, data):
        with Parallel(n_jobs=-1) as parralel:
            delayed_func = delayed(self.fit_single)
            self.models_out = parralel(
                delayed_func()(chain_pieces=subset, data=data) for subset in self.chains
            )

    def evaluate(self):
        if self.fitted:
            y_real = self.valid[self.target_str].to_numpy()
            y_hat = self.model.predict(self.valid)
            metrics_valid = display_metrics(y_real, y_hat)
            return metrics_valid
        else:
            raise ValueError("Model is not fitted.")

    def backtest(self, data):
        pass

    def predict(
        self, x_test: Union[polars_dataframe, pandas_dataframe] = None
    ) -> np.ndarray:
        pass
