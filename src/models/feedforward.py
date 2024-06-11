import pickle
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


class FFModel(nn.Module):
    def __init__(
        self,
        activation_fn,
        num_embeddings_list,
        embedding_dim,
        num_numerical_features,
        hidden_size,
    ):
        super(FFModel, self).__init__()

        # Define Embedding Layers
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, embedding_dim)
                for num_embeddings in num_embeddings_list
            ]
        )
        self.embeddingLayer = nn.Linear(
            embedding_dim * len(num_embeddings_list), hidden_size
        )

        # Define weight matrix and bias term
        self.weight0 = nn.Parameter(
            torch.randn(hidden_size, embedding_dim * len(num_embeddings_list)),
            requires_grad=True,
        )
        self.weight1 = nn.Parameter(
            torch.randn(hidden_size, num_numerical_features), requires_grad=True
        )
        self.weight2 = nn.Parameter(torch.randn(1, hidden_size), requires_grad=True)

        self.bias1 = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Custom monotonic activation function
        self.activation_fn = activation_fn

    def forward(self, x_num, x_cat):

        # Fully connected layer for features
        activation_output1 = self.activation_fn(self.weight1)

        # Categorical output
        x_embedded = [self.embeddings[i](x_cat[i]) for i in range(len(self.embeddings))]
        # Concatenate embeddings along the feature dimension
        x_embedded = torch.cat(x_embedded, dim=1).squeeze(1)
        # Perform linear transformation on Embedded Data
        x_embedded = torch.matmul(x_embedded, self.weight0.t())

        # Perform linear transformation on Numerical Data
        x_continiuous = torch.matmul(x_num, activation_output1.t())

        # Concatenate embeddings and continuous features
        linear_output1 = x_embedded + x_continiuous + self.bias1.unsqueeze(0)
        # Apply ReLU on first linear transforamtion
        linear_output1 = F.relu(linear_output1)

        # Perform second linear transformation
        activation_output2 = self.weight2
        linear_output2 = torch.matmul(
            linear_output1, activation_output2.t()
        ) + self.bias2.unsqueeze(0)
        return linear_output2


class feedforward:
    def __init__(
        self,
        params: Dict,
        features: List,
        tags: List,
        custom_loss=nn.MSELoss(),
        learning_rate=0.01,
        optimizer=optim.Adam,
        custom_monotone: Optional = None,
    ):
        """_summary_

        Args:
            params (Dict): _description_
            features (List): _description_
            custom_loss (str, optional): _description_. Defaults to l2.


        Example:
                # Change data types to int for boolean data and category to categorical data
                    X[mask_columns]=X[mask_columns].astype('int')
                    X['clusterCode']=X['clusterCode'].astype('category')

                # If custome_montone parameter is not None
                # Define monotonical tags for all but categorical columns.
                    tags=[0]*47+[-1]
                    custome_monotone= CReLU(tags)

                # Split train and validation data
                    x_train, x_valid, y_train, y_valid = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                # Define the followinf parameter dictionary
                    params_dict={'num_epochs' : 20,
                                 'hidden_size' : 40,
                                 'batch_size' : 64,
                                 'num_embeddings_list':[6],
                                 'num_numerical_features':48,
                                 'embedding_dim':5
                                  }

                # Creat Model Object with the parameters defined above
                    ff=feedforward(params=params_dict,features=X.columns,tags=tags,custom_monotone=custom_montone)

                # Fit Model on data
                    ff.fit(x_train,x_valid,y_train,y_valid)

                # Predict on test data
                  ff.predict(x_train)

        """
        self.features = features
        self.params = params
        self.tags = tags
        self.custom_loss = custom_loss
        self.custom_monotone = custom_monotone
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.is_fitted = False
        self.train_losses = []
        self.test_losses = []
        if custom_monotone is not None:
            self.custom_monotone = custom_monotone
        else:
            self.custom_monotone = nn.Identity()

        self.ff_model = FFModel(
            hidden_size=self.params["hidden_size"],
            num_embeddings_list=self.params["num_embeddings_list"],
            num_numerical_features=self.params["num_numerical_features"],
            embedding_dim=self.params["embedding_dim"],
            activation_fn=self.custom_monotone,
        )

    def split_numerical_categorical(self, x):
        # Divide categircal and numercial data
        x_num = x.select_dtypes(["float", "int"])
        x_cat_indices = [x.select_dtypes("category")]

        # Convert data to PyTorch tensors
        categ_tensors = [
            torch.tensor(col.values, dtype=int).long() for col in x_cat_indices
        ]
        num_tensors = torch.tensor(x_num.values, dtype=torch.float32)

        return categ_tensors, num_tensors

    def prepare(self, x_train, x_valid, y_train, y_valid):

        # Divide into categircal and numercial tensors

        train_targets = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        train_categ_tensors, train_num_tensors = self.split_numerical_categorical(
            x_train
        )

        valid_targets = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)
        valid_categ_tensors, valid_num_tensors = self.split_numerical_categorical(
            x_valid
        )

        # Create datasets
        train_dataset = TensorDataset(
            train_num_tensors, *train_categ_tensors, train_targets
        )
        test_dataset = TensorDataset(
            valid_num_tensors, *valid_categ_tensors, valid_targets
        )

        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset, batch_size=self.params["batch_size"], shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=self.params["batch_size"])
        return train_loader, test_loader

    def fit(
        self,
        x_train: pd.DataFrame,
        x_valid: pd.DataFrame,
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        y_valid: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> None:
        """_summary_

        Args:
            x_train (pd.DataFrame): _description_
            y_train (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
            x_valid (pd.DataFrame): _description_
            y_valid (Union[pd.DataFrame, pd.Series, np.ndarray]): _description_
        """

        train_loader, test_loader = self.prepare(x_train, x_valid, y_train, y_valid)

        # Define loss function and optimizer
        criterion = self.custom_loss
        optimizer = self.optimizer(self.ff_model.parameters(), lr=self.learning_rate)

        for epoch in range(self.params["num_epochs"]):
            # Train the model
            self.ff_model.train()

            for batch in train_loader:

                num_inputs = batch[0]
                # The last element is the target
                targets = batch[-1]

                # All but the first and last elements are categorical inputs
                cat_inputs = batch[1:-1]
                # Prediction Step
                outputs = self.ff_model(num_inputs, cat_inputs)

                # Backward process
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Evaluate on train set
            self.ff_model.eval()
            with torch.no_grad():
                train_loss = sum(
                    criterion(self.ff_model(batch[0], batch[1:-1]), batch[-1])
                    for batch in train_loader
                ) / len(train_loader)
                self.train_losses.append(train_loss.item())

            # Evaluate on test set
            with torch.no_grad():
                test_loss = sum(
                    criterion(self.ff_model(batch[0], batch[1:-1]), batch[-1])
                    for batch in test_loader
                ) / len(test_loader)
                self.test_losses.append(test_loss.item())

        self.is_fitted = True

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """_summary_

        Args:
            x_test (pd.DataFrame): _description_

        Returns:
            np.ndarray: _description_
        """
        # Divide categircal and numercial tensors
        test_categ_tensors, test_num_tensors = self.split_numerical_categorical(x_test)

        return (
            self.ff_model(test_num_tensors, *test_categ_tensors)
            .detach()
            .numpy()
            .reshape(-1)
        )

    def save(self, path: str, name: str) -> None:
        """_summary_

        Args:
            path (str): _description_
            name (str): _description_
        """
        with open(f"{path}/{name}.pickle", "wb") as f:
            pickle.dump(self.ff_model, f)
