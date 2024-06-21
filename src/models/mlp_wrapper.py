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

from pandas import DataFrame as pd_DataFrame
from polars import DataFrame as pl_Dataframe


class MixedDataset(Dataset):
    def __init__(
        self,
        dataset: Union[pl_Dataframe, pd_DataFrame],
        continuous_data: List[str],
        categorical_data: List[str],
        target: str,
    ):
        self.continuous_data = dataset[continuous_data].to_numpy()
        self.categorical_data = dataset[categorical_data].to_numpy()
        self.targets = dataset[target].to_numpy().ravel()

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
        num_features: List[str],
        embedding_sizes: List[str],
        hidden_dim=64,
        output_dim=1,
    ):
        super(EmbedMLP, self).__init__()
        self.output_dim = output_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories + 1, size) for categories, size in embedding_sizes]
        )
        embedding_dim = sum(e.embedding_dim for e in self.embeddings)
        input_dim = num_features + embedding_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x, categorical_features: List[str]):
        embedded = [
            embedding(categorical_features[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        embedded = torch.cat(embedded, dim=1)
        x = torch.cat([x, embedded], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x) if self.output_dim == 1 else x


class TorchWrapper:

    def __init__(
        self,
        batch_size: int,
        hidden_dim: int,
        num_cols: List[str],
        cat_cols: List[int],
        target: str,
    ):
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.unique_modalities = None
        self.num_lags = len(num_cols)
        self.num_categorical_features = len(cat_cols)
        self.criterion = nn.MSELoss()

    def make_loader(
        self, data: pl_Dataframe, shuffle: bool = True, batch_size: int = None
    ):
        if batch_size is not None:
            loader_batch_size = batch_size
        else:
            loader_batch_size = self.batch_size
        data = MixedDataset(
            data,
            continuous_data=self.num_cols,
            categorical_data=self.cat_cols,
            target=self.target,
        )
        data_loader = DataLoader(data, batch_size=loader_batch_size, shuffle=shuffle)
        return data_loader

    def prepare(self, train_set: pl_Dataframe, val_set: pl_Dataframe):

        self.unique_modalities = {
            col: len(set(list(train_set[col]) + list(val_set[col])))
            for col in self.cat_cols
        }
        print(f"Unique modalities: {self.unique_modalities}")  # Debug print

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
        train_x: Union[pl_Dataframe, pd_DataFrame],
        valid_x: Union[pd_DataFrame, pl_Dataframe],
        num_epochs: int = 100,
        patience: int = 10,
        train_y: Optional[Any] = None,  # this is a placeholder don't meant to be used
        valid_y: Optional[Any] = None,  # this is a placeholder don't meant to be used
    ):
        train_loader, val_loader = self.prepare(train_x, valid_x)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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

    def predict(self, x_test: Union[pl_Dataframe, pd_DataFrame]) -> List[float]:
        forecast_loader = self.make_loader(data=x_test, shuffle=False, batch_size=1)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in forecast_loader:
                x = batch["x"]
                categorical_features = batch["categorical_features"]
                outputs = self.model(x, categorical_features)
                predictions.append(outputs.item())
        return predictions
