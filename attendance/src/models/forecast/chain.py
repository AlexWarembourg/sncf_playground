from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from datetime import timedelta
from IPython.display import display
from tqdm import tqdm
from joblib import Parallel, delayed

import polars as pl
from copy import deepcopy
from attendance.src.models.forecast.direct import DirectForecaster
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe
from attendance.src.analysis.metrics import display_metrics


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
        transform_win_size: Optional[int] = 30,
        transform_strategy: Optional[str] = "None",
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
        self.transform_strategy = transform_strategy
        self.transform_win_size = transform_win_size

    def fit_single(self, chain_pieces: List[int], data: pl.DataFrame, strategy: str):
        forecaster = DirectForecaster(
            model=deepcopy(self.model),
            ts_uid=self.ts_uid,
            forecast_range=chain_pieces,
            target_str=self.target_str,
            date_str=self.date_str,
            exogs=self.exogs,
            features_params=self.params_dict,
            transform_win_size=self.transform_win_size,
            transform_strategy=self.transform_strategy,
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
            for model in self.models_out:
                _, valid_output = model.evaluate(return_output=True)
                valid_output = valid_output.select(
                    [self.ts_uid, self.date_str, self.target_str, self.output_name]
                )
                reconstruct_valid.append(valid_output)
            valid_output = pl.concat(reconstruct_valid, how="vertical_relaxed")
            y_real = valid_output[self.target_str].fill_null(0).to_numpy().ravel()
            y_hat = valid_output[self.output_name].fill_null(0).to_numpy().ravel()
            metrics_valid = display_metrics(y_real, y_hat)
            if return_valid:
                return valid_output, metrics_valid
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
