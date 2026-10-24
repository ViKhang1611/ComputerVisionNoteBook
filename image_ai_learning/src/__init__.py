"""
Gói thư viện dùng chung (shared utilities package) cho dự án học tập image_ai_learning.
Tất cả các hàm trong gói này đều được giải thích chi tiết tại các Jupyter Notebook tương ứng.
"""

from .utils import (
    find_project_root,
    set_seed,
    get_device,
    count_parameters,
    summarize_model_layers,
    save_checkpoint,
    load_checkpoint,
)
from .synthetic_data import (
    generate_synthetic_shapes_dataset,
    generate_detection_dataset,
    generate_segmentation_dataset,
    generate_anomaly_dataset,
)
from .features import (
    extract_pixel_flatten,
    extract_color_histogram,
    extract_hog_manual,
    extract_lbp_manual,
    extract_combined_features,
)
from .models import (
    SimpleMLP,
    SimpleCNN,
    ResidualCNN,
    MiniUNet,
    ConvAutoencoder,
    SiameseEmbeddingNet,
    MiniViT,
)
from .training import (
    train_one_epoch,
    evaluate_classifier,
    fit_classifier,
)
from .evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix_np,
    compute_bbox_iou,
    compute_segmentation_iou_dice,
    measure_inference_latency,
)

__all__ = [
    "find_project_root",
    "set_seed",
    "get_device",
    "count_parameters",
    "summarize_model_layers",
    "save_checkpoint",
    "load_checkpoint",
    "generate_synthetic_shapes_dataset",
    "generate_detection_dataset",
    "generate_segmentation_dataset",
    "generate_anomaly_dataset",
    "extract_pixel_flatten",
    "extract_color_histogram",
    "extract_hog_manual",
    "extract_lbp_manual",
    "extract_combined_features",
    "SimpleMLP",
    "SimpleCNN",
    "ResidualCNN",
    "MiniUNet",
    "ConvAutoencoder",
    "SiameseEmbeddingNet",
    "MiniViT",
    "train_one_epoch",
    "evaluate_classifier",
    "fit_classifier",
    "compute_classification_metrics",
    "compute_confusion_matrix_np",
    "compute_bbox_iou",
    "compute_segmentation_iou_dice",
    "measure_inference_latency",
]
