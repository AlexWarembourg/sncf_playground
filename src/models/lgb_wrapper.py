import pickle
from typing import Any, Dict, List, Optional, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import plot_importance


class GBTModel:
    """_summary_"""

    def __init__(
        self,
        params: Dict,
        early_stopping_value: int,
        features: List,
        categorical_features: List,
        weight: Optional[Union[List, Any]] = None,
        custom_loss: str = "l2",
        num_ite: int = 1000,
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
        self.custom_loss = custom_loss
        self.num_ite = num_ite
        self.lgb_model = None
        self.is_fitted = False

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

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: pd.DataFrame,
        valid_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        return_object: bool = False,
    ) -> None:
        """_summary_

        Args:
            train_x (pd.DataFrame): _description_
            train_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
            valid_x (pd.DataFrame): _description_
            valid_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        """
        self.params["objective"] = self.custom_loss
        self.lgb_model = lgb.train(
            params=self.params,
            train_set=self.pooling(
                x=self.is_an_array(train_x[self.features]),
                y=self.is_an_array(train_y),
                weight=(
                    self.is_an_array(train_x[self.weight])
                    if self.weight is not None
                    else None
                ),
            ),
            valid_sets=self.pooling(
                x=valid_x[self.features],
                y=self.is_an_array(valid_y),
                weight=(
                    self.is_an_array(valid_x[self.weight])
                    if self.weight is not None
                    else None
                ),
            ),
            feature_name=self.features,
            num_boost_round=self.num_ite,
            # objective = self.custom_loss,
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
        plot_importance(
            lgb_model, importance_type=importance_type, max_num_features=num_feat
        )

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """_summary_

        Args:
            x_test (pd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.lgb_model.predict(self.is_an_array(x_test[self.features]))

    def save(self, path: str, name: str) -> None:
        """_summary_

        Args:
            path (str): _description_
            name (str): _description_
        """
        with open(f"{path}/{name}.pickle", "wb") as f:
            pickle.dump(self.lgb_model, f)
