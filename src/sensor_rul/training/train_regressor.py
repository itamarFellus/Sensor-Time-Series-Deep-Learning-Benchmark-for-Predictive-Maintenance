from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.1

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device | str,
    grad_clip_max_norm: float | None = None,
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
        if grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_abs_error += torch.abs(predictions.detach() - batch_y).sum().item()
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
    grad_clip_max_norm: float | None = None,
    early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    restore_best: bool = True,
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Train a regression model with Adam and MSE loss.

    Tracks validation RMSE each epoch and always records the true best validation
    checkpoint (lowest val_rmse seen). Set early_stopping_patience=None to disable
    early stopping. When restore_best=True, the model is loaded with the true best
    validation checkpoint before returning. early_stopping_min_delta is used only for
    the patience/early-stopping decision, not for deciding whether a checkpoint is
    the true best.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float | int]] = []

    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    early_stop_reference = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            grad_clip_max_norm=grad_clip_max_norm,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_rmse = float(val_metrics["rmse"])
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss: {train_metrics['loss']:.6f}, val_loss: {val_metrics['loss']:.6f}, "
            f"train_rmse: {train_metrics['rmse']:.4f}, val_rmse: {val_rmse:.4f}, "
            f"train_mae: {train_metrics['mae']:.4f}, val_mae: {val_metrics['mae']:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_rmse,
                "train_mae": train_metrics["mae"],
                "val_mae": val_metrics["mae"],
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if val_rmse < early_stop_reference - early_stopping_min_delta:
            early_stop_reference = val_rmse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"no validation RMSE improvement >= {early_stopping_min_delta} "
                f"for {early_stopping_patience} epochs."
            )
            break

    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    trained_epochs = int(history[-1]["epoch"]) if history else 0
    print(
        f"Best validation epoch: {best_epoch} "
        f"(val_rmse={best_val_rmse:.4f}, val_mae={best_val_mae:.4f}, "
        f"trained {trained_epochs}/{epochs} epochs)"
    )

    return model, history
