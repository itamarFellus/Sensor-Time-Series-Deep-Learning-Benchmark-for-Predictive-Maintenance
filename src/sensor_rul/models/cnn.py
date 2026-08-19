from __future__ import annotations

import torch
import torch.nn as nn


class CNNRULRegressor(nn.Module):
    """1D CNN model for sequence-to-one RUL regression.

    Input shape:
        x: (batch_size, window_size, num_features)

    Output shape:
        y_pred: (batch_size,)
    """

    def __init__(self, window_size: int, num_features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            num_features, 32, kernel_size=9, stride=1, padding="same"
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding="same")
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.head = nn.Linear(64 * window_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        return self.head(x).squeeze(-1)
