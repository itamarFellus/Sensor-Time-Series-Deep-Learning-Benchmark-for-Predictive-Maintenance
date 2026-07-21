from __future__ import annotations

import torch
import torch.nn as nn


class GRURULRegressor(nn.Module):
    """GRU model for sequence-to-one RUL regression.

    Input shape:
        x: (batch_size, window_size, num_features)

    Output shape:
        y_pred: (batch_size,)
    """
    def __init__(self, num_features: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)
