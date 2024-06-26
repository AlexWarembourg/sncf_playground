import numpy as np
import torch
from typing import Any, List, Dict, Union, Optional
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.optim as optim
import copy

from sklearn.preprocessing import StandardScaler

import polars as pl


class MixedDataset(Dataset):
    def __init__(
        self,
        dataset: pl.DataFrame,
        continuous_data: List[str],
        categorical_data: List[str],
        target: str,
        is_test: bool = False,
    ):
        self.categorical_data = dataset[categorical_data].to_numpy()
        self.continuous_data = dataset[continuous_data].to_numpy()
        if not is_test:
            self.targets = dataset[target].to_numpy().ravel()
        else:
            # placeholder y doest not exist in future
            self.targets = np.ones(dataset.shape[0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.continuous_data[idx], dtype=torch.float32),
            "categorical_features": torch.tensor(
                self.categorical_data[idx], dtype=torch.long
            ),
            "y": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


class EmbedMLP(nn.Module):
    def __init__(
        self,
        num_features,
        embedding_sizes,
        hidden_dim=64,
        dropout_level: float = 0.25,
        output_dim=1,
    ):
        super(EmbedMLP, self).__init__()
        self.output_dim = output_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories + 1, size) for categories, size in embedding_sizes]
        )
        embedding_dim = sum(e.embedding_dim for e in self.embeddings)
        input_dim = num_features + embedding_dim
        self.dropout = nn.Dropout(dropout_level)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x, categorical_features):
        embedded = [
            embedding(categorical_features[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([x, embedded], dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.squeeze(x) if self.output_dim == 1 else x


class TorchWrapper:

    def __init__(
        self,
        batch_size: int,
        hidden_dim: int,
        num_cols: List[str],
        cat_cols: List[str],
        target: str,
        scaler: callable = None,
    ):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.features = self.cat_cols + self.num_cols
        self.unique_modalities = None
        self.num_lags = len(self.num_cols)
        self.num_categorical_features = len(cat_cols)
        self.criterion = nn.MSELoss()
        self.scaler = scaler
        self.scaler_is_fitted = False

    @property
    def num_cols(self):
        return self._num_cols

    @num_cols.setter
    def num_cols(self, value):
        self._num_cols = value
        self.num_lags = len(self._num_cols)

    def make_loader(
        self,
        data: pl.DataFrame,
        shuffle: bool = True,
        batch_size: int = None,
        is_test: bool = False,
    ):
        if batch_size is not None:
            loader_batch_size = batch_size
        else:
            loader_batch_size = self.batch_size
        # calling custom struct
        data = MixedDataset(
            data,
            continuous_data=self.num_cols,
            categorical_data=self.cat_cols,
            target=self.target,
            is_test=is_test,
        )
        data_loader = DataLoader(data, batch_size=loader_batch_size, shuffle=shuffle)
        return data_loader

    def scale_numerical(
        self, data: pl.DataFrame, is_train: bool = True, default_value: float = 0.0
    ) -> pl.DataFrame:
        if is_train:
            scaled_values = self.scaler.fit_transform(
                data.select(self.num_cols)
                .fill_null(default_value)
                .fill_nan(default_value)
            )
            self.scaler_is_fitted = True
        else:
            if self.scaler_is_fitted:
                scaled_values = self.scaler.transform(
                    data.select(self.num_cols)
                    .fill_null(default_value)
                    .fill_nan(default_value)
                )
            else:
                raise ValueError("Scaler is not fitted.")
        # apply the transformation
        for column in self.num_cols:
            data = data.with_columns(pl.Series(column, scaled_values[column]))
        return data

    def prepare(self, train_set: pl.DataFrame, val_set: pl.DataFrame):

        self.unique_modalities = {
            col: len(
                set(np.concatenate((train_set[col].unique(), val_set[col].unique())))
            )
            for col in self.cat_cols
        }
        print(f"Unique modalities: {self.unique_modalities}")  # Debug print
        if self.scaler:
            train_set = self.scale_numerical(train_set, is_train=True)
            val_set = self.scale_numerical(val_set, is_train=False)
        # data loader.
        train_loader = self.make_loader(data=train_set, shuffle=True)
        val_loader = self.make_loader(data=val_set, shuffle=False)

        # define model during preparation
        embedding_sizes = [
            (nmodality, min(max(nmodality // 4, 1), 7))
            for nmodality in self.unique_modalities.values()
        ]
        print(f"Embedding sizes: {embedding_sizes}")  # Debug print

        self.model = EmbedMLP(
            num_features=self.num_lags,
            embedding_sizes=embedding_sizes,
            hidden_dim=self.hidden_dim,
        )
        return train_loader, val_loader

    def fit(
        self,
        train_x: pl.DataFrame,
        valid_x: pl.DataFrame,
        num_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 0.0007,
        decay: float = 0.2,
        train_y: Optional[Any] = None,  # this is a placeholder don't meant to be used
        valid_y: Optional[Any] = None,  # this is a placeholder don't meant to be used
    ):
        train_loader, val_loader = self.prepare(train_set=train_x, val_set=valid_x)
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=decay
        )
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch["x"]
                y = batch["y"]
                categorical_features = batch["categorical_features"]
                optimizer.zero_grad()
                outputs = self.model(x, categorical_features)
                loss = self.criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)

            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"]
                    y = batch["y"]
                    categorical_features = batch["categorical_features"]
                    outputs = self.model(x, categorical_features)
                    loss = self.criterion(outputs.squeeze(), y)
                    val_loss += loss.item() * x.size(0)

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break
        self.model.load_state_dict(best_model_wts)
        return self.model

    def predict(self, x_test: pl.DataFrame):
        if self.scaler:
            x_test = self.scale_numerical(data=x_test, is_train=False)
        # wrap loader then precict.
        forecast_loader = self.make_loader(
            data=x_test, shuffle=False, batch_size=1, is_test=True
        )
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in forecast_loader:
                x = batch["x"]
                categorical_features = batch["categorical_features"]
                outputs = self.model(x, categorical_features)
                predictions.append(outputs.item())
        return np.array(predictions).ravel()
