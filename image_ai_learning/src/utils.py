"""
Các hàm tiện ích cốt lõi (core utilities):
- Xác định thư mục gốc dự án (project root) bằng đường dẫn tương đối.
- Cố định hạt giống ngẫu nhiên (random seed) để hỗ trợ tái lập kết quả (reproducibility).
- Tự động chọn thiết bị tính toán (CPU/CUDA).
- Đếm số lượng tham số (parameters) và lập bảng tóm tắt kiến trúc mạng Deep Learning.
- Lưu và nạp checkpoint an toàn kèm metadata tiền xử lý và ánh xạ nhãn (class mapping).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Đảm bảo Matplotlib có thư mục cache ghi được trên mọi môi trường
_default_mpl_dir = Path(__file__).resolve().parent.parent / "outputs" / ".mplconfig"
_default_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_default_mpl_dir))

import numpy as np
import torch
import torch.nn as nn


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Tìm thư mục gốc của dự án `image_ai_learning` dựa trên các tệp đánh dấu (`README.md`, `src`).
    Giúp notebook chạy ổn định dù khởi động từ thư mục gốc hay từ `notebooks/`.
    """
    current = (start_path or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
        if (candidate / "image_ai_learning" / "src").is_dir():
            return candidate / "image_ai_learning"
    return current


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Thiết lập hạt giống ngẫu nhiên (seed) cho Python `random`, `NumPy` và `PyTorch`.
    Lưu ý: Kết quả vẫn có thể lệch nhỏ giữa CPU và GPU hoặc giữa các phiên bản phần cứng khác nhau.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Trả về `torch.device('cuda')` nếu có GPU NVIDIA khả dụng và `prefer_gpu=True`,
    ngược lại trả về `torch.device('cpu')`.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Đếm tổng số tham số (total parameters) và số tham số có học (trainable parameters, `requires_grad=True`).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "frozen": int(frozen_params),
    }


def summarize_model_layers(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Chạy một lượt lan truyền tiến (forward pass) với tensor giả lập có kích thước `input_shape` (ví dụ `(1, 3, 32, 32)`)
    để ghi nhận tên lớp (layer), kích thước đầu ra (output shape) và số lượng tham số của từng lớp lá (leaf module).
    """
    if device is None:
        device = next(model.parameters(), torch.tensor(0.0)).device
    summary_rows: List[Dict[str, Any]] = []
    hooks = []

    def register_hook(name: str, module: nn.Module):
        def hook(mod: nn.Module, inp: Any, out: Any):
            if isinstance(out, torch.Tensor):
                out_shape = tuple(out.shape)
            elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                out_shape = tuple(out[0].shape)
            else:
                out_shape = ()
            params = sum(p.numel() for p in mod.parameters())
            trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            summary_rows.append(
                {
                    "layer_name": name,
                    "layer_type": mod.__class__.__name__,
                    "output_shape": str(out_shape),
                    "total_params": int(params),
                    "trainable_params": int(trainable),
                }
            )

        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name != "":
            register_hook(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape, device=device)
        _ = model(dummy)
    if was_training:
        model.train()

    for h in hooks:
        h.remove()

    return summary_rows


def save_checkpoint(
    filepath: Path | str,
    model: nn.Module,
    class_names: List[str],
    preprocessing_config: Dict[str, Any],
    epoch: int,
    val_metrics: Dict[str, float],
    optimizer: Optional[torch.optim.Optimizer] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Lưu checkpoint chuẩn phục vụ cả việc tiếp tục huấn luyện (resume training) và suy luận (inference).
    Bao gồm:
    - `model_state_dict`: Trọng số của mô hình.
    - `class_names`, `class_to_idx`, `idx_to_class`: Ánh xạ nhãn để không bị nhầm thứ tự lớp khi dự đoán.
    - `preprocessing_config`: Kích thước ảnh, mean, std chuẩn hóa đã học trên tập train.
    - `epoch`, `val_metrics`: Thông tin thời điểm lưu và chất lượng trên tập validation.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    payload: Dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "class_names": list(class_names),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "preprocessing_config": dict(preprocessing_config),
        "epoch": int(epoch),
        "val_metrics": dict(val_metrics),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_metadata is not None:
        payload["extra_metadata"] = dict(extra_metadata)

    torch.save(payload, path)
    return path


def load_checkpoint(
    filepath: Path | str,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Nạp checkpoint từ đĩa và khôi phục `state_dict` vào `model` (nếu truyền vào).
    """
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(Path(filepath), map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
    return checkpoint
