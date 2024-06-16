from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from datetime import timedelta
from IPython.display import display
from tqdm import tqdm
from joblib import Parallel, delayed

import polars as pl
from copy import deepcopy
from .direct import DirectForecaster
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe
from src.analysis.metrics import display_metrics


class ChainForecaster:

    def __init__(
        self,
        model,
        n_splits: int,
        ts_uid: str,
        params_dict: Dict[str, Dict[str, List[int]]],
        exogs: List[str],
        total_forecast_horizon: int = 1,
        target_str: str = "y",
        date_str: str = "date",
        n_jobs: int = -1,
    ):
        self.chains = np.array_split(np.arange(total_forecast_horizon), n_splits)
        self.model = model
        self.ts_uid = ts_uid
        self.target_str = target_str
        self.params_dict = params_dict
        self.date_str = date_str
        self.exogs = exogs
        self.total_forecast_horizon = total_forecast_horizon
        self.output_name = "y_hat"
        self.n_jobs = n_jobs

    def fit_single(self, chain_pieces: List[int], data: pl.DataFrame, strategy: str):
        forecaster = DirectForecaster(
            model=deepcopy(self.model),
            ts_uid=self.ts_uid,
            forecast_range=chain_pieces,
            target_str=self.target_str,
            date_str=self.date_str,
            exogs=self.exogs,
            features_params=self.params_dict,
        )
        forecaster.fit(data, strategy=strategy)
        return forecaster

    def fit(self, data, strategy: str = "global"):
        with Parallel(n_jobs=self.n_jobs) as parralel:
            delayed_func = delayed(self.fit_single)
            self.models_out = parralel(
                delayed_func(chain_pieces=pieces, data=data, strategy=strategy)
                for pieces in tqdm(self.chains)
            )
        self.fitted = True

    def evaluate(self, return_valid: bool = False):
        if self.fitted:
            reconstruct_valid = []
            for chain_model in self.models_out:
                reconstruct_valid.append(
                    (
                        chain_model.valid.with_columns(
                            pl.lit(chain_model.model.predict(chain_model.valid)).alias(
                                self.output_name
                            )
                        ).select(
                            [
                                self.ts_uid,
                                self.date_str,
                                self.output_name,
                                self.target_str,
                            ]
                        )
                    )
                )
            reconstruct_valid = pl.concat(reconstruct_valid, how="vertical")
            y_real = reconstruct_valid[self.target_str].to_numpy().ravel()
            y_hat = reconstruct_valid[self.output_name].to_numpy().ravel()
            metrics_valid = display_metrics(y_real, y_hat)
            if return_valid:
                return reconstruct_valid, metrics_valid
            else:
                return metrics_valid
        else:
            raise ValueError("Model is not fitted.")

    def backtest(self, data):
        pass

    def predict(
        self, x_test: Union[polars_dataframe, pandas_dataframe] = None
    ) -> np.ndarray:
        reconstruct_test = []
        for chain_model in self.models_out:
            reconstruct_test.append(
                (
                    chain_model.predict(x_test)
                    .filter(
                        pl.col(self.date_str).is_between(
                            chain_model.start_forecast, chain_model.end_forecast
                        )
                    )
                    .select([self.ts_uid, self.date_str, self.output_name])
                )
            )
        reconstruct_test = pl.concat(reconstruct_test, how="vertical_relaxed")
        return reconstruct_test
