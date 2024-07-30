from typing import List, Optional, Union, Tuple, Dict, Callable
import numpy as np
from datetime import timedelta
from IPython.display import display
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy
import os

import polars as pl
import optuna
from pandas import DataFrame as pandas_dataframe
from polars import DataFrame as polars_dataframe

from nostradamus.preprocessing.lags import compute_autoreg_features
from nostradamus.preprocessing.target_transform import TargetTransform
from nostradamus.analysis.describe import timeseries_length
from nostradamus.analysis.metrics import display_metrics


def parameters_tuning(
    initial_params: Dict,
    tuning_objective,
    n_trials: int = 25,
    njobs: int = -1,
):
    """parameter for tuning over sudy

    Args:
        tuning_objective (_type_): _description_
        n_trials (int, optional): _description_. Defaults to 25.

    Returns:
        _type_: _description_

    example :

    func = lambda trial: objective(trial=trial,
                                    train_x=train_x,
                                    test=residualised_test,
                                    covariates=covariates,
                                    target=y,
                                    seed=12345
                                    )
    study_df, best_params = parameters_tuning(tuning_objective=func, n_trials=25, initial_params={})
        print(best_params)
        print(study_df)
        study_df.to_csv('bparamslgb_new.csv', sep="|", index=False)
    """
    study = optuna.create_study(direction="minimize")
    # study.enqueue_trial(initial_params)
    study.optimize(tuning_objective, n_trials=n_trials, n_jobs=njobs, gc_after_trial=True)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    study_df = study.trials_dataframe()
    return study_df, study.best_params


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
        transform_win_size: Optional[int] = 30,
        transform_strategy: Optional[str] = "None",
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
            max(self.features_params[ts_uid]["wins"]) + max(self.features_params[ts_uid]["shifts"]),
            max(self.features_params[ts_uid]["lags"]),
        )
        self.strategy = None
        self.except_list = []

        self.target_transformer = TargetTransform(
            ts_uid=self.ts_uid,
            time_col=self.date_str,
            target=self.target_str,
            forecast_horizon=self.forecast_horizon,
            win_size=transform_win_size,
            strategy=transform_strategy,
        )

        if scaler is not None:
            assert scaler.name in [
                "robust",
                "standard",
                "temporal",
            ], 'Le Scaler specifié n"est pas connus'

        mandatory_keys = [
            "groups",
            "horizon",
            "wins",
            "shifts",
            "lags",
            "funcs",
            "delta",
        ]

        for key in self.features_params.keys():
            keys = list(self.features_params[key].keys())
            intersct = np.intersect1d(mandatory_keys, keys)
            assert len(keys) == len(
                intersct
            ), "Shape not match, that imply there's missing or foreign key in the dictionnary"
            # print(keys, intersct)

    def update_params_dict(self, params: Dict[str, Dict[str, List[Union[str, int]]]], horizon: int):
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
        self.start_valid = max_available_date - timedelta(days=int(max(self.forecast_range)))
        self.end_valid = max_available_date
        # forecast is de facto the max available date + min of the range and the max of the range
        self.start_forecast = max_available_date + timedelta(days=int(min(self.forecast_range) + 1))
        self.end_forecast = max_available_date + timedelta(days=int(max(self.forecast_range) + 1))

    def trim(self, data: polars_dataframe) -> polars_dataframe:
        trim_dt = self.min_available_date + timedelta(days=int(self.initial_trim))
        return data.filter(pl.col(self.date_str) >= pl.lit(trim_dt))

    def temporal_train_test_split(
        self, data: polars_dataframe
    ) -> Tuple[polars_dataframe, polars_dataframe]:
        # retrieve max date from data
        train = data.filter(pl.col(self.date_str) < self.start_valid)
        valid = data.filter(
            pl.col(self.date_str).is_between(self.start_valid, self.end_valid, closed="both")
        )
        return train, valid

    def prepare(self, data: polars_dataframe) -> Tuple[polars_dataframe, polars_dataframe]:
        self.uids = data[[self.ts_uid]].unique()
        # this is the maximal date available.
        self.set_date_scope(data[self.date_str].cast(pl.Date))
        # compute features engineering and split.
        train, valid = (
            data.pipe(self.target_transformer.transform)
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
        # if we need to predict
        if self.scaler is not None:
            self.scaler.fit(data)
            data = self.scaler.transform(data)
        self.features = self.exogs + self.ar_features
        # special case for scaling
        if hasattr(self.model, "num_cols"):
            self.model.num_cols = list(set(self.model.num_cols + self.ar_features))
        self.model.features = self.features
        return train, valid

    def fit_local(
        self, train_data: polars_dataframe, optimize: bool = True, n_trials: int = 15
    ) -> None:
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
                train_x=tr,
                train_y=tr.select(self.target_str),
                valid_x=val,
                valid_y=val.select(self.target_str),
                return_object=True,
                optimize=optimize,
                n_trials=n_trials,
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

    def objective(
        self,
        model,
        trial: optuna.trial,
        train_data: pl.DataFrame,
        forecaster_object,
        seed: int = 12345,
    ):
        """_summary_

        Args:
            trial (optuna.trial): _description_
            train_x (pd.DataFrame): _description_
            test (pd.DataFrame): _description_
            features (List): _description_
            target (str, optional): _description_. Defaults to "".
            seed (int, optional): _description_. Defaults to 12345.

        Returns:
            _type_: _description_
        """
        model = deepcopy(model)
        forecaster_object = deepcopy(forecaster_object)
        optuna_params = {
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "objective": trial.suggest_categorical(
                "objective", ["regression", "huber", "regression_l1", "quantile"]
            ),
            "metric": trial.suggest_categorical("metric", ["rmse"]),
            "alpha": trial.suggest_categorical(
                "alpha",
                [0.5],
            ),
            "force_row_wise": trial.suggest_categorical("force_row_wise", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=False),
            "max_depth": trial.suggest_int("max_depth", 4, 15),
            "sub_row": trial.suggest_categorical("sub_row", [0.6, 0.7, 0.8, 1.0]),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 4, log=True),
            "num_iterations": trial.suggest_int(
                "n_estimators",
                50,
                2500,
            ),
            "num_leaves": trial.suggest_int("num_leaves", 25, 800),
            "max_bins": trial.suggest_int("max_bins", 24, 1000),
            "min_data_in_bin": trial.suggest_int("min_data_in_bin", 25, 1000),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 1000),
            "feature_fraction_seed": trial.suggest_categorical("feature_fraction_seed", [seed]),
            "bagging_seed": trial.suggest_categorical("bagging_seed", [seed]),
            "seed": trial.suggest_categorical("seed", [seed]),
            "verbose": trial.suggest_categorical("verbose", [-1]),
        }
        model.params = optuna_params
        forecaster_object.fit(train_data=train_data)
        return forecaster_object.evaluate()["rmse"].values[0]

    def fit_global(
        self, train_data: polars_dataframe, optimize: bool = False, n_trials: int = 50
    ) -> None:
        self.train, self.valid = self.prepare(train_data)
        # call model and fit with a validation set
        # model must have a quadruple fit method with an early stopping setting on the validation set.
        self.model.fit(
            train_x=self.train,
            train_y=self.train.select(self.target_str),
            valid_x=self.valid,
            valid_y=self.valid.select(self.target_str),
            tune=optimize,
            n_trials=n_trials,
        )

    def fit(
        self,
        train_data: polars_dataframe,
        strategy: str = "global",
        optimize: bool = False,
        n_trials: int = 50,
    ):
        if strategy == "global":
            self.fit_global(train_data, optimize=optimize, n_trials=n_trials)
        elif strategy == "local":
            self.fit_local(train_data, optimize=optimize, n_trials=n_trials)
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

    def predict_local(self, x_test: Union[polars_dataframe, pandas_dataframe] = None) -> np.ndarray:
        grouped_dataframe = x_test.group_by([self.ts_uid])
        conditional_back = "threading" if x_test.select(self.ts_uid).n_unique() < 1000 else "loky"
        predictions = Parallel(n_jobs=self.n_jobs, backend=conditional_back)(
            delayed(self.single_predict)(self.models_locals, key, subset_data)
            for key, subset_data in grouped_dataframe
        )
        predictions = pl.concat(list(filter(lambda x: x is not None, predictions)))
        return predictions

    def predict(self, x_test: Union[polars_dataframe, pandas_dataframe] = None):
        if x_test is None:
            x_test = self.make_future_dataframe()
            x_test = pl.concat(
                (
                    self.train.select(x_test.columns),
                    self.valid.select(x_test.columns),
                    x_test,
                ),
                how="vertical",
            )
        shape = x_test[self.date_str].n_unique()
        assert (
            shape >= self.initial_trim
        ), f"the test length must contains at least {self.initial_trim} date observation to compute feature engineering"
        max_dt = x_test[self.date_str].max()
        minimal_cut = max_dt - timedelta(days=int(self.initial_trim + self.forecast_horizon))
        x_test = (
            x_test.filter(pl.col(self.date_str) >= minimal_cut)
            .pipe(
                compute_autoreg_features,
                target_col=self.target_str,
                date_str=self.date_str,
                auto_reg_params=self.features_params,
            )
            .filter(pl.col(self.date_str).is_between(self.start_forecast, self.end_forecast))
        )
        if self.strategy == "global":
            output = self.predict_global(x_test=x_test)
        elif self.strategy == "local":
            output = self.predict_local(x_test=x_test)
        else:
            raise ValueError("Unknown Strategy")
        output = self.target_transformer.inverse_transform(output, target=self.output_name)
        return output

    def make_future_dataframe(self, uids: List[str] = None):
        df_dates = pl.date_range(
            start=self.start_forecast,
            end=self.end_forecast,
            eager=True,
        ).to_frame("date")
        # define future_df
        future_df = df_dates.join(
            (self.uids if uids is None else self.uids.filter(pl.col(self.ts_uid).is_in(uids))),
            how="cross",
        )
        future_df = future_df.with_columns(pl.lit(np.nan).cast(pl.Float32).alias(self.target_str))
        return future_df

    def evaluate(self, return_output: bool = False):
        if self.fitted:
            if self.strategy == "global":
                hat = self.predict_global(x_test=self.valid)
                y_hat = (
                    self.target_transformer.inverse_transform(hat, target=self.output_name)
                    .select(self.output_name)
                    .fill_null(0.0)
                    .to_numpy()
                    .flatten()
                )
                y_real = (
                    self.target_transformer.inverse_transform(hat, target=self.target_str)
                    .select(self.target_str)
                    .fill_null(0.0)
                    .to_numpy()
                    .flatten()
                )
            elif self.strategy == "local":
                hat = self.predict_local(x_test=self.valid)
                y_hat = (
                    self.target_transformer.inverse_transform(hat, target=self.output_name)
                    .select(self.output_name)
                    .fill_null(0.0)
                    .to_numpy()
                    .flatten()
                )
                y_real = (
                    self.target_transformer.inverse_transform(hat, target=self.target_str)
                    .select(self.target_str)
                    .fill_null(0.0)
                    .to_numpy()
                    .flatten()
                )
            else:
                raise ValueError("Unknown Strategy")
            # display(y_real, y_hat, hat.select([self.output_name, self.target_str]))
            metrics_valid = display_metrics(y_real, y_hat)
            if return_output:
                hat = hat.with_columns(
                    pl.lit(y_real).alias(self.target_str),
                    pl.lit(y_hat).alias(self.output_name),
                )
                return metrics_valid, hat
            else:
                return metrics_valid
        else:
            raise ValueError("Model is not fitted.")

    def backtest(self):
        pass

    def conformalised(self, train_data, strategy, optimize, n_trials, alpha=0.95):
        estimators: List[object] = []
        for alpha_ in [alpha / 2, (1 - (alpha / 2)), 0.5]:
            m = deepcopy(self.model.params)
            m["objective"] = "quantile"
            m["alpha"] = alpha_
            m = m.fit(train_data, strategy=strategy, optimize=optimize, n_trials=n_trials)
            estimators.append(m)
