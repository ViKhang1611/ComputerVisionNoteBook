"""
Các hàm huấn luyện (training loop) và đánh giá (validation loop) tường minh cho PyTorch:
- `train_one_epoch`: Thực hiện 1 epoch huấn luyện chuẩn (`model.train()` -> `zero_grad()` -> `forward` -> `loss` -> `backward()` -> `clip_grad_norm_` -> `optimizer.step()`).
- `evaluate_classifier`: Đánh giá mô hình ở chế độ `model.eval()` kết hợp `torch.no_grad()`.
- `fit_classifier`: Chu trình huấn luyện nhiều epoch kèm theo dõi lịch sử, lưu checkpoint tốt nhất theo validation metric và dừng sớm (Early Stopping).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Huấn luyện mô hình phân loại trong 1 epoch.
    Kiểm tra sự cố NaN/Inf của hàm mất mát (loss) ngay trong quá trình huấn luyện.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Phát hiện giá trị loss không hữu hạn (NaN/Inf): {loss.item()}")

        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += float(loss.item()) * batch_size
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += batch_size

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return {"loss": avg_loss, "accuracy": acc}


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình phân loại trên tập validation hoặc test.
    Lưu ý phân biệt:
    - `model.eval()` chuyển BatchNorm sang dùng thống kê tích lũy (running mean/var) và tắt Dropout.
    - `@torch.no_grad()` tắt bộ máy tính đạo hàm tự động (`autograd`) để tiết kiệm RAM và tăng tốc độ.
    Trả về: `(metrics_dict, y_true, y_pred, y_prob)`.
    """
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device, dtype=torch.long)

        logits = model(inputs)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        batch_size = inputs.size(0)
        running_loss += float(loss.item()) * batch_size
        total += batch_size

        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32)

    avg_loss = running_loss / max(1, total)
    acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else 0.0
    return {"loss": avg_loss, "accuracy": acc}, y_true, y_pred, y_prob


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 10,
    scheduler: Optional[Any] = None,
    patience: int = 5,
    checkpoint_path: Optional[Path | str] = None,
    class_names: Optional[List[str]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[float] | int | float]:
    """
    Huấn luyện mô hình nhiều epoch với cơ chế:
    - Ghi nhận lịch sử `train_loss`, `train_acc`, `val_loss`, `val_acc`.
    - Lưu checkpoint tốt nhất dựa trên `val_loss` (hoặc `val_acc` khi hòa).
    - Dừng sớm (`early stopping`) nếu `val_loss` không cải thiện sau `patience` epoch liên tiếp.
    """
    model.to(device)
    history: Dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_acc": 0.0,
    }

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_res, _, _, _ = evaluate_classifier(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_res["loss"])
            else:
                scheduler.step()

        history["train_loss"].append(train_res["loss"])
        history["train_acc"].append(train_res["accuracy"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["accuracy"])

        improved = val_res["loss"] < (history["best_val_loss"] - 1e-4)
        if improved:
            history["best_val_loss"] = val_res["loss"]
            history["best_val_acc"] = val_res["accuracy"]
            history["best_epoch"] = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0

            if checkpoint_path is not None:
                save_checkpoint(
                    filepath=checkpoint_path,
                    model=model,
                    class_names=class_names or [str(i) for i in range(4)],
                    preprocessing_config=preprocessing_config or {},
                    epoch=epoch,
                    val_metrics=val_res,
                    optimizer=optimizer,
                )
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.to(device)
    return history
