import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import polars as pl


class MixedDataset(Dataset):
    def __init__(
        self,
        dataset: pl.DataFrame,
        continuous_data: List[str],
        categorical_data: List[str],
        target: str,
    ):
        self.continuous_data = dataset.select(continuous_data).to_numpy()
        self.categorical_data = dataset.select(categorical_data).to_numpy()
        self.targets = dataset.select(target).to_numpy().ravel()

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
    def __init__(self, num_features, embedding_sizes, hidden_dim=64, output_dim=1):
        super(EmbedMLP, self).__init__()
        self.output_dim = output_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for categories, size in embedding_sizes]
        )
        embedding_dim = sum(e.embedding_dim for e in self.embeddings)
        input_dim = num_features + embedding_dim
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x) if self.output_dim == 1 else x


class TorchWrapper:

    def __init__(
        self, batch_size: int, num_cols: List[str], cat_cols: List[str], target: str
    ):
        self.batch_size = batch_size
        self.target = target
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.unique_modalities = None
        self.num_lags = len(num_cols)
        self.num_categorical_features = len(cat_cols)

    def prepare(self, train_set, val_set):

        self.unique_modalities = {
            col: pl.concat(
                (train_set.select(self.cat_cols), val_set.select(self.cat_cols)),
                how="vertical_relaxed",
            )[col].n_unique()
            + 1
            for col in self.cat_cols
        }

        train_ds = MixedDataset(
            train_set,
            continuous_data=self.num_cols,
            categorical_data=self.cat_cols,
            target=self.target,
        )

        val_ds = MixedDataset(
            val_set,
            continuous_data=self.num_cols,
            categorical_data=self.cat_cols,
            target=self.target,
        )

        # Create dataloaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        return train_loader, val_loader

    def fit(
        self,
        model,
        train_set,
        val_set,
        criterion,
        optimizer,
        num_epochs=100,
        patience=10,
        hidden_dim=128,
    ):
        embedding_sizes = [
            (nmodality, max(nmodality // 4, 1))
            for nmodality in self.unique_modalities.values()
        ]
        model = EmbedMLP(
            num_features=self.num_lags,
            embedding_sizes=embedding_sizes,
            hidden_dim=hidden_dim,
        )
        train_loader, val_loader = self.prepare(train_set, val_set)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch["x"]
                y = batch["y"]
                categorical_features = batch["categorical_features"]
                optimizer.zero_grad()
                outputs = model(x, categorical_features)
                loss = criterion(outputs.squeeze(), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"]
                    y = batch["y"]
                    categorical_features = batch["categorical_features"]
                    outputs = model(x, categorical_features)
                    loss = criterion(outputs.squeeze(), y)
                    val_loss += loss.item() * x.size(0)

            val_loss /= len(val_loader.dataset)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

        model.load_state_dict(best_model_wts)
        return model

    def predict(self, model, x_test):
        x_test = MixedDataset(
            x_test,
            continuous_data=self.num_cols,
            categorical_data=self.cat_cols,
            target=self.target,
        )
        forecast_loader = DataLoader(x_test, batch_size=1, shuffle=False)
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in forecast_loader:
                x = batch["x"]
                categorical_features = batch["categorical_features"]
                outputs = model(x, categorical_features)
                predictions.append(outputs.item())
        return predictions
