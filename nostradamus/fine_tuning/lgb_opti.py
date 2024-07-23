import os
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def objective(
    model,
    trial: optuna.trial,
    train_data: pd.DataFrame,
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
    study.optimize(tuning_objective, n_trials=n_trials, n_jobs=njobs)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    study_df = study.trials_dataframe()
    return study_df, study.best_params
