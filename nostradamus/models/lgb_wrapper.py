import pickle
from typing import Any, Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import mean_squared_error
import optuna
from lightgbm import plot_importance


def parameters_tuning(
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
    study.optimize(tuning_objective, n_trials=n_trials, n_jobs=njobs)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    study_df = study.trials_dataframe()
    return study_df, study.best_params


class GBTModel:
    """_summary_"""

    def __init__(
        self,
        params: Dict,
        early_stopping_value: int,
        features: List,
        categorical_features: List,
        weight: Optional[Union[List, Any]] = None,
        custom_loss: str = None,
        num_ite: int = 1000,
        seed: int = 123456,
    ):
        """_summary_

        Args:
            params (Dict): _description_
            early_stopping_value (int): _description_
            features (List): _description_
            categorical_features (List): _description_
            weight (Optional[Union[List, Any]], optional): _description_. Defaults to None.
            custom_loss (str, optional): _description_. Defaults to "l2".
            num_ite (int, optional): _description_. Defaults to 1000.
        """
        self.early_stopping_value = early_stopping_value
        self.features = features
        self.categorical_features = categorical_features
        self.weight = weight
        self.params = params
        if custom_loss is not None:
            self.params["objective"] = custom_loss
        self.num_ite = num_ite
        self.lgb_model = None
        self.is_fitted = False
        self.seed = seed

    def weight_estimation(self, paneled_y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """_summary_

        Args:
            paneled_y (Union[pd.Series, np.ndarray]): _description_

        Returns:
            np.ndarray: _description_
        """
        weights = paneled_y / np.sum(paneled_y)
        weights.loc[:, weights == 0] = weights[~weights.eq(0)].min()
        return weights

    def pooling(
        self,
        x: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series, np.ndarray],
        weight: Union[pd.Series, np.ndarray] = None,
    ):
        """_summary_

        Args:
            x (pd.DataFrame): _description_
            y (Union[pd.DataFrame, pd.Series]): _description_
            weight (Union[pd.Series, np.ndarray]): _description_

        Returns:
            _type_: _description_
        """
        if weight is None:
            weight = np.ones(len(x))
            # weight = self.weight_estimation(y)
        return lgb.Dataset(
            data=x,
            label=y,
            weight=weight,
            feature_name=self.features,
            categorical_feature=self.categorical_features,
        )

    def is_an_array(self, cols_array):
        if not isinstance(cols_array, np.ndarray):
            return cols_array.to_numpy()
        else:
            return cols_array

    def objective(
        self,
        trial: optuna.trial,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: pd.DataFrame,
        valid_y: Union[pd.DataFrame, pd.Series, np.ndarray],
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
            "num_leaves": trial.suggest_int("num_leaves", 15, 800),
            "max_bins": trial.suggest_int("max_bins", 15, 1000),
            "min_data_in_bin": trial.suggest_int("min_data_in_bin", 15, 1000),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 15, 1000),
            "feature_fraction_seed": trial.suggest_categorical(
                "feature_fraction_seed", [self.seed]
            ),
            "bagging_seed": trial.suggest_categorical("bagging_seed", [self.seed]),
            "seed": trial.suggest_categorical("seed", [self.seed]),
            "verbose": trial.suggest_categorical("verbose", [-1]),
        }
        self.train(
            train_x,
            train_y,
            valid_x,
            valid_y,
            return_object=False,
            tune=True,
            bounds_params=optuna_params,
        )
        valid_yhat = self.predict(valid_x)
        metric = mean_squared_error(valid_yhat, valid_y)
        return metric

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: pd.DataFrame,
        valid_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        tune: bool = False,
        return_object: bool = False,
        n_trials: int = 1,
    ):
        if tune:
            func = lambda trial: self.objective(
                trial=trial,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
            )

            self.study_df, best_params = parameters_tuning(tuning_objective=func, n_trials=n_trials)
            # update params with best one.
            self.params = best_params

        # train model with best params.
        self.train(
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
            return_object=return_object,
            tune=False,
            bounds_params=None,
        )

    def online(
        self,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: pd.DataFrame,
        valid_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        return_object: bool = False,
    ) -> None:

        if not self.lgb_model:
            tmp_model = self.train(
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
                return_object=return_object,
                tune=False,
                bounds_params=None,
            )
            GBTModel.save(tmp_model, path="", name="tmp")

        for bound_in, bound_out in ExpandingWindow(dates):
            tmp_model = lgb.train(
                params=self.params,
                train_set=self.pooling(
                    x=self.is_an_array(train_x[self.features]),
                    y=self.is_an_array(train_y).ravel(),
                    weight=(
                        self.is_an_array(train_x[self.weight]) if self.weight is not None else None
                    ),
                ),
                valid_sets=self.pooling(
                    x=valid_x[self.features],
                    y=self.is_an_array(valid_y).ravel(),
                    weight=(
                        self.is_an_array(valid_x[self.weight]) if self.weight is not None else None
                    ),
                ),
                feature_name=self.features,
                num_boost_round=self.num_ite,
                callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_value)],
                init_model=self.lgb_model,
                keep_training_booster=True,
            )
            GBTModel.save(tmp_model, path="", name="tmp")

    def train(
        self,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: pd.DataFrame,
        valid_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        return_object: bool = False,
        tune: bool = False,
        bounds_params: dict = None,
    ) -> None:
        """_summary_

        Args:
            train_x (pd.DataFrame): _description_
            train_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
            valid_x (pd.DataFrame): _description_
            valid_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        """
        self.lgb_model = lgb.train(
            params=self.params if not tune else bounds_params,
            train_set=self.pooling(
                x=self.is_an_array(train_x[self.features]),
                y=self.is_an_array(train_y).ravel(),
                weight=(
                    self.is_an_array(train_x[self.weight]) if self.weight is not None else None
                ),
            ),
            valid_sets=self.pooling(
                x=valid_x[self.features],
                y=self.is_an_array(valid_y).ravel(),
                weight=(
                    self.is_an_array(valid_x[self.weight]) if self.weight is not None else None
                ),
            ),
            feature_name=self.features,
            num_boost_round=self.num_ite,
            callbacks=[lgb.early_stopping(stopping_rounds=self.early_stopping_value)],
        )
        self.is_fitted = True
        if return_object:
            return self

    @staticmethod
    def importance_plot(lgb_model, importance_type: str = "gain", num_feat: int = 15):
        """_summary_

        Args:
            lgb_model (_type_): _description_
            importance_type (str, optional): _description_. Defaults to "gain".
            num_feat (int, optional): _description_. Defaults to 15.
        """
        plot_importance(lgb_model, importance_type=importance_type, max_num_features=num_feat)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """_summary_

        Args:
            x_test (pd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.lgb_model.predict(self.is_an_array(x_test[self.features]))

    @staticmethod
    def save(lgb_model, path: str, name: str) -> None:
        """_summary_

        Args:
            path (str): _description_
            name (str): _description_
        """
        with open(f"{path}/{name}.pickle", "wb") as f:
            pickle.dump(lgb_model, f)
