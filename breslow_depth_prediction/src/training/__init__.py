"""Training utilities and metrics."""

from .trainer import Trainer
from .multitask_trainer import MultiTaskTrainer
from .metrics import (
    # Main metrics classes
    SegmentationMetrics,
    BreslowMetrics,
    # Utility functions
    compute_confusion_matrix,
    print_metrics_summary,
    # Legacy functions (backward compatibility)
    dice_coefficient,
    iou_score,
    pixel_accuracy,
)

__all__ = [
    # Trainer
    "Trainer",
    "MultiTaskTrainer",
    # Main metrics classes
    "SegmentationMetrics",
    "BreslowMetrics",
    # Utility functions
    "compute_confusion_matrix",
    "print_metrics_summary",
    # Legacy functions
    "dice_coefficient",
    "iou_score",
    "pixel_accuracy",
]
