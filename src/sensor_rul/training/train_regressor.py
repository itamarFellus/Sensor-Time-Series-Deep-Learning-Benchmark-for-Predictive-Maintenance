from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_abs_error = 0.0
    num_samples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(predictions - batch_y).sum().item()
        num_samples += batch_size

    if num_samples == 0:
        return {"loss": 0.0, "rmse": 0.0, "mae": 0.0}

    avg_loss = total_loss / num_samples
    return {
        "loss": avg_loss,
        "rmse": avg_loss**0.5,
        "mae": total_abs_error / num_samples,
    }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_abs_error += torch.abs(predictions - batch_y).sum().item()
            num_samples += batch_size

    if num_samples == 0:
        return {"loss": 0.0, "rmse": 0.0, "mae": 0.0}

    avg_loss = total_loss / num_samples
    return {
        "loss": avg_loss,
        "rmse": avg_loss**0.5,
        "mae": total_abs_error / num_samples,
    }


def fit_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device | str,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss: {train_metrics['loss']:.6f}, val_loss: {val_metrics['loss']:.6f}, "
            f"train_rmse: {train_metrics['rmse']:.4f}, val_rmse: {val_metrics['rmse']:.4f}, "
            f"train_mae: {train_metrics['mae']:.4f}, val_mae: {val_metrics['mae']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
            }
        )

    return model, history
