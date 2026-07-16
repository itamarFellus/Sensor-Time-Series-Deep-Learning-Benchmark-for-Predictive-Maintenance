from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP on flattened time-series windows. Input shape: (batch, window, features)."""

    def __init__(self, window_size: int, num_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = window_size * num_features
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
