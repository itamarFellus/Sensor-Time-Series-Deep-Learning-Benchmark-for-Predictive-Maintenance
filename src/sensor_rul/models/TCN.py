from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1d with explicit left-only zero padding for causal temporal mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, window_size) -> same length after conv
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):
    """Residual TCN block: Conv1 -> LN -> ReLU -> Conv2 -> LN -> + shortcut -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        # Project shortcut when channel dimensions differ (e.g. num_features -> 32).
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)

        shortcut = self.residual(x) if self.residual is not None else x
        return self.relu(out + shortcut)


class TCNRULRegressor(nn.Module):
    """TCN model for sequence-to-one RUL regression.

    Input shape:
        x: (batch_size, window_size, num_features)

    Output shape:
        y_pred: (batch_size,)
    """

    def __init__(self, window_size: int, num_features: int) -> None:
        super().__init__()
        kernel_size = 3
        hidden_channels = 32
        dilations = (1, 2, 4)

        self.block1 = TCNResidualBlock(
            num_features, hidden_channels, kernel_size, dilations[0]
        )
        self.block2 = TCNResidualBlock(
            hidden_channels, hidden_channels, kernel_size, dilations[1]
        )
        self.block3 = TCNResidualBlock(
            hidden_channels, hidden_channels, kernel_size, dilations[2]
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, window_size, num_features) -> (batch_size, num_features, window_size)
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # (batch_size, 32, window_size) -> (batch_size, 32)
        x = x[:, :, -1]
        return self.head(x).squeeze(-1)
