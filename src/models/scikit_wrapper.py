import pandas as pd
import numpy as np
import pickle

from typing import List, Union, Optional, Dict, Any
from datetime import datetime


class ScikitWrapper:
    """_summary_"""

    def __init__(
        self,
        model,
        params: Dict,
        features: List,
        categorical_features: List,
        weight: Optional[Union[List, Any]] = None,
    ):
        self.features = features
        self.categorical_features = categorical_features
        self.weight = weight
        self.params = params
        self.model = model
        self.is_fitted = False
        if params is not None:
            self.model.set_params(**params)

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

    def is_an_array(self, cols_array):
        if not isinstance(cols_array, np.ndarray):
            return cols_array.to_numpy()
        else:
            return cols_array

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: Union[pd.DataFrame, pd.Series, np.ndarray],
        valid_x: Any = None,
        valid_y: Any = None,
        return_object: bool = False,
    ) -> None:
        """_summary_

        Args:
            train_x (pd.DataFrame): _description_
            train_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
            valid_x (pd.DataFrame): _description_
            valid_y (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        """
        self.model = self.model.fit(
            X=self.is_an_array(train_x[self.features]),
            y=self.is_an_array(train_y),
            sample_weight=(
                self.is_an_array(train_x[self.weight])
                if self.weight is not None
                else None
            ),
        )
        self.is_fitted = True
        if return_object:
            return self

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """_summary_

        Args:
            x_test (pd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.model.predict(self.is_an_array(x_test[self.features]))

    def save(self, path: str, name: str) -> None:
        """_summary_

        Args:
            path (str): _description_
            name (str): _description_
        """
        with open(f"{path}/{name}.pickle", "wb") as f:
            pickle.dump(self.model, f)
