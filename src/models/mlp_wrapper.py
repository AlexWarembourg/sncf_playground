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


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


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
        kan_layer: bool = False,
    ):
        super(EmbedMLP, self).__init__()
        self.output_dim = output_dim
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories + 1, size) for categories, size in embedding_sizes]
        )
        embedding_dim = sum(e.embedding_dim for e in self.embeddings)
        input_dim = num_features + embedding_dim
        self.dropout = nn.Dropout(dropout_level)
        if not kan_layer:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        else:
            # efficient kan layer : https://github.com/Blealtan/efficient-kan/blob/master/src%2Fefficient_kan%2Fkan.py
            self.fc1 = KANLinear(
                input_dim,
                hidden_dim,
                grid_size=6,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            )
            self.fc2 = KANLinear(
                hidden_dim,
                hidden_dim // 2,
                grid_size=6,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            )
            self.fc3 = KANLinear(
                hidden_dim // 2,
                output_dim,
                grid_size=6,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
            )

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
