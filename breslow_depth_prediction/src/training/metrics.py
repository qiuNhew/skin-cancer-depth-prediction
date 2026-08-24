"""Evaluation metrics for segmentation and Breslow-depth prediction.

Two streaming accumulator classes:
    - SegmentationMetrics — per-class TP/FP/FN -> Dice, IoU, precision, recall, pixel acc.
    - BreslowMetrics — per-sample (pred, true) -> MAE/RMSE/R²/T-cat acc/within-bounds.

Plus helpers: compute_confusion_matrix, print_metrics_summary, and one-shot
versions for the trainer's per-batch progress bar.

All Dice/IoU/accuracy in [0, 1]. All Breslow regression metrics in micrometres.
Inter-observer variability constant = 860 µm (pathologist agreement threshold).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


# AJCC T-stage thresholds in µm. Half-open intervals [min, max).
T_CATEGORY_THRESHOLDS = {
    "T1a": (0, 800),
    "T1b": (800, 1000),
    "T2": (1000, 2000),
    "T3": (2000, 4000),
    "T4": (4000, float("inf")),
}

INTER_OBSERVER_VARIABILITY = 860  # µm — pathologist 95% limits-of-agreement


def get_t_category(depth_um: float) -> str:
    """Map Breslow depth (µm) to AJCC T-stage. Duplicate of dataset.py to avoid circular imports."""
    for category, (min_val, max_val) in T_CATEGORY_THRESHOLDS.items():
        if min_val <= depth_um < max_val:
            return category
    return "T4"


class SegmentationMetrics:
    """Streaming accumulator for multi-class segmentation metrics.

    Keeps running per-class TP/FP/FN counters; `.compute()` returns the full
    metric dict. `.compute_per_sample()` is a one-shot variant that doesn't
    touch the accumulators.

    Usage:
        m = SegmentationMetrics(num_classes=4, class_names=[...])
        for batch in dataloader:
            m.update(model(images), targets)
        results = m.compute()

    Args:
        num_classes:  4 for this project.
        class_names:  Per-class names for output (default Class_0 .. Class_N).
        ignore_index: Excluded target value (default -100, matches PyTorch CE).
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        """Clear accumulators. Call between evaluation runs."""
        # float64 to avoid precision loss summing millions of pixels.
        self.true_positives = torch.zeros(self.num_classes, dtype=torch.float64)
        self.false_positives = torch.zeros(self.num_classes, dtype=torch.float64)
        self.false_negatives = torch.zeros(self.num_classes, dtype=torch.float64)
        self.total_correct = 0
        self.total_pixels = 0
        self.num_batches = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Ingest one batch. Updates internal TP/FP/FN counters.

        Args:
            predictions: (B, C, H, W) logits OR (B, H, W) class indices.
            targets:     (B, H, W) class indices.
        """
        if predictions.dim() == 4:
            predictions = predictions.argmax(dim=1)

        # Detach so metrics don't leak gradients.
        predictions = predictions.detach()
        targets = targets.detach()

        valid_mask = targets != self.ignore_index

        # Pixel accuracy contribution.
        correct = ((predictions == targets) & valid_mask).sum().item()
        total = valid_mask.sum().item()
        self.total_correct += correct
        self.total_pixels += total

        # Per-class TP/FP/FN.
        for class_idx in range(self.num_classes):
            pred_class = (predictions == class_idx) & valid_mask
            target_class = (targets == class_idx) & valid_mask
            tp = (pred_class & target_class).sum().item()
            fp = (pred_class & ~target_class).sum().item()
            fn = (~pred_class & target_class).sum().item()
            self.true_positives[class_idx] += tp
            self.false_positives[class_idx] += fp
            self.false_negatives[class_idx] += fn

        self.num_batches += 1

    def compute(self) -> Dict[str, Any]:
        """Compute Dice/IoU/precision/recall/F1 per class + means + pixel accuracy.

        Returns dict with `_per_class` arrays, `_mean` (all classes), `_mean_no_bg`
        (excluding class 0), `pixel_accuracy`, `class_names`, `num_batches`.
        """
        smooth = 1e-8

        # Dice = 2*TP / (2*TP + FP + FN); IoU = TP / (TP + FP + FN).
        dice_per_class = (
            2 * self.true_positives /
            (2 * self.true_positives + self.false_positives + self.false_negatives + smooth)
        ).tolist()
        iou_per_class = (
            self.true_positives /
            (self.true_positives + self.false_positives + self.false_negatives + smooth)
        ).tolist()
        precision_per_class = (
            self.true_positives / (self.true_positives + self.false_positives + smooth)
        ).tolist()
        recall_per_class = (
            self.true_positives / (self.true_positives + self.false_negatives + smooth)
        ).tolist()
        # F1 (= Dice for multi-class one-vs-rest; reported for clarity).
        f1_per_class = [
            2 * p * r / (p + r + smooth)
            for p, r in zip(precision_per_class, recall_per_class)
        ]

        pixel_accuracy = self.total_correct / (self.total_pixels + smooth)

        # Two mean variants: with and without background (V1/V2 headline excludes bg).
        dice_mean = np.mean(dice_per_class)
        iou_mean = np.mean(iou_per_class)
        dice_mean_no_bg = np.mean(dice_per_class[1:]) if len(dice_per_class) > 1 else dice_per_class[0]
        iou_mean_no_bg = np.mean(iou_per_class[1:]) if len(iou_per_class) > 1 else iou_per_class[0]

        return {
            "dice_per_class": dice_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "dice_mean": dice_mean,
            "iou_mean": iou_mean,
            "dice_mean_no_bg": dice_mean_no_bg,
            "iou_mean_no_bg": iou_mean_no_bg,
            "pixel_accuracy": pixel_accuracy,
            "class_names": self.class_names,
            "num_batches": self.num_batches,
        }

    def compute_per_sample(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """One-shot per-sample metrics (does NOT touch the running accumulators)."""
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=0)

        smooth = 1e-8
        dice_scores = []
        iou_scores = []

        for class_idx in range(self.num_classes):
            pred_mask = predictions == class_idx
            target_mask = targets == class_idx
            intersection = (pred_mask & target_mask).sum().float()
            pred_sum = pred_mask.sum().float()
            target_sum = target_mask.sum().float()
            dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
            iou = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

        correct = (predictions == targets).sum().float()
        pixel_acc = correct / targets.numel()

        return {
            "dice_per_class": dice_scores,
            "dice_mean": np.mean(dice_scores),
            "iou_per_class": iou_scores,
            "iou_mean": np.mean(iou_scores),
            "pixel_accuracy": pixel_acc.item(),
        }


class BreslowMetrics:
    """Streaming accumulator for Breslow regression + T-cat classification metrics.

    Stores (pred, true) pairs as lists; `.compute()` returns MAE/RMSE/R²/Pearson r,
    T-category accuracy (exact + adjacent), within-µm fractions, T-cat confusion matrix.

    Args:
        observer_variability: Inter-observer threshold in µm (default 860).
    """

    def __init__(self, observer_variability: float = INTER_OBSERVER_VARIABILITY):
        self.observer_variability = observer_variability
        self.reset()

    def reset(self) -> None:
        """Clear accumulators."""
        self.predicted_depths: List[float] = []
        self.true_depths: List[float] = []
        self.predicted_categories: List[str] = []
        self.true_categories: List[str] = []

    def update(
        self,
        predicted_depths: Union[float, List[float], np.ndarray, torch.Tensor],
        true_depths: Union[float, List[float], np.ndarray, torch.Tensor],
    ) -> None:
        """Ingest one or more (pred, true) depth pairs. Normalises types -> Python list."""
        if isinstance(predicted_depths, (torch.Tensor, np.ndarray)):
            predicted_depths = predicted_depths.flatten().tolist()
        elif isinstance(predicted_depths, (int, float)):
            predicted_depths = [float(predicted_depths)]

        if isinstance(true_depths, (torch.Tensor, np.ndarray)):
            true_depths = true_depths.flatten().tolist()
        elif isinstance(true_depths, (int, float)):
            true_depths = [float(true_depths)]

        self.predicted_depths.extend(predicted_depths)
        self.true_depths.extend(true_depths)
        # Eagerly assign T-categories so `.compute()` doesn't have to.
        for pred, true in zip(predicted_depths, true_depths):
            self.predicted_categories.append(get_t_category(pred))
            self.true_categories.append(get_t_category(true))

    def compute(self) -> Dict[str, Any]:
        """Compute every Breslow metric the Research Article reports."""
        if len(self.predicted_depths) == 0:
            return self._empty_results()

        pred = np.array(self.predicted_depths)
        true = np.array(self.true_depths)
        errors = pred - true               # signed -> tracks systematic bias
        abs_errors = np.abs(errors)        # for MAE and within-X-µm

        # Regression metrics.
        mae = np.mean(abs_errors)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        # R² in [-inf, 1]; negative when worse than predicting the mean.
        r_squared = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else 0.0

        if len(pred) > 1:
            pearson_r = np.corrcoef(pred, true)[0, 1]
            if np.isnan(pearson_r):
                pearson_r = 0.0
        else:
            pearson_r = 0.0

        # Classification metrics. Binary @ 0.8 mm = T1a/T1b decision boundary.
        pred_binary = pred >= 800
        true_binary = true >= 800
        accuracy_at_800um = np.mean(pred_binary == true_binary)

        t_cat_correct = sum(
            p == t for p, t in zip(self.predicted_categories, self.true_categories)
        )
        t_category_accuracy = t_cat_correct / len(self.predicted_categories)

        # Adjacent-T-cat accuracy: off-by-one stage tolerated (clinically reasonable).
        t_cat_order = ["T1a", "T1b", "T2", "T3", "T4"]
        adjacent_correct = sum(
            abs(t_cat_order.index(p) - t_cat_order.index(t)) <= 1
            for p, t in zip(self.predicted_categories, self.true_categories)
        )
        t_category_adjacent_accuracy = adjacent_correct / len(self.predicted_categories)

        # Clinical-relevance fractions.
        within_observer_var = np.mean(abs_errors <= self.observer_variability)
        within_500um = np.mean(abs_errors <= 500)
        within_200um = np.mean(abs_errors <= 200)

        # Error percentiles for "how bad is the worst case" reporting.
        error_percentiles = {
            "p25": np.percentile(abs_errors, 25),
            "p50": np.percentile(abs_errors, 50),  # median absolute error
            "p75": np.percentile(abs_errors, 75),
            "p90": np.percentile(abs_errors, 90),
            "p95": np.percentile(abs_errors, 95),
        }

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mse": float(mse),
            "r_squared": float(r_squared),
            "pearson_r": float(pearson_r),
            "accuracy_at_800um": float(accuracy_at_800um),
            "t_category_accuracy": float(t_category_accuracy),
            "t_category_adjacent_accuracy": float(t_category_adjacent_accuracy),
            "within_observer_var": float(within_observer_var),
            "within_500um": float(within_500um),
            "within_200um": float(within_200um),
            "observer_variability_threshold": self.observer_variability,
            "error_percentiles": error_percentiles,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "t_category_confusion": self._compute_t_category_confusion(),
            "n_samples": len(self.predicted_depths),
        }

    def _compute_t_category_confusion(self) -> Dict[str, Dict[str, int]]:
        """T-category confusion matrix as `{true: {pred: count}}` nested dict."""
        categories = ["T1a", "T1b", "T2", "T3", "T4"]
        confusion = {true_cat: {pred_cat: 0 for pred_cat in categories} for true_cat in categories}
        for pred, true in zip(self.predicted_categories, self.true_categories):
            confusion[true][pred] += 1
        return confusion

    def _empty_results(self) -> Dict[str, Any]:
        """Placeholder dict when no samples have been ingested. Keeps API type-stable."""
        return {
            "mae": 0.0, "rmse": 0.0, "mse": 0.0, "r_squared": 0.0, "pearson_r": 0.0,
            "accuracy_at_800um": 0.0, "t_category_accuracy": 0.0,
            "t_category_adjacent_accuracy": 0.0, "within_observer_var": 0.0,
            "within_500um": 0.0, "within_200um": 0.0,
            "observer_variability_threshold": self.observer_variability,
            "error_percentiles": {}, "mean_error": 0.0, "std_error": 0.0,
            "t_category_confusion": {}, "n_samples": 0,
        }

    def get_predictions_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw accumulated arrays (used by notebook 02 for custom plots)."""
        return np.array(self.predicted_depths), np.array(self.true_depths)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    normalize: bool = False,
) -> np.ndarray:
    """Pixel-level confusion matrix. Rows = true class, cols = predicted class.

    Args:
        predictions: (B, H, W) class indices or (B, C, H, W) logits.
        targets:     (B, H, W) class indices.
        num_classes: Expected count.
        normalize:   If True, normalise rows -> per-true-class recall in [0, 1].
    """
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)

    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()

    valid = (targets >= 0) & (targets < num_classes)
    predictions = predictions[valid]
    targets = targets[valid]

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        confusion[int(t), int(p)] += 1

    if normalize:
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion = confusion.astype(np.float64) / (row_sums + 1e-8)
    return confusion


def print_metrics_summary(
    seg_metrics: Optional[Dict[str, Any]] = None,
    breslow_metrics: Optional[Dict[str, Any]] = None,
    title: str = "EVALUATION METRICS",
) -> None:
    """Pretty-print seg + Breslow metrics to stdout. Called at the end of `evaluate.py`."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    if seg_metrics is not None:
        print("\n SEGMENTATION METRICS")
        print("-" * 70)
        class_names = seg_metrics.get("class_names", [])
        print(f"\n {'Class':<15} {'Dice':>10} {'IoU':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 57)
        dpc = seg_metrics.get("dice_per_class", [])
        ipc = seg_metrics.get("iou_per_class", [])
        ppc = seg_metrics.get("precision_per_class", [])
        rpc = seg_metrics.get("recall_per_class", [])
        for i, name in enumerate(class_names):
            dice = dpc[i] if i < len(dpc) else 0
            iou = ipc[i] if i < len(ipc) else 0
            prec = ppc[i] if i < len(ppc) else 0
            rec = rpc[i] if i < len(rpc) else 0
            print(f" {name:<15} {dice:>10.4f} {iou:>10.4f} {prec:>10.4f} {rec:>10.4f}")
        print("-" * 57)
        print(f"\n {'Metric':<30} {'Value':>15}")
        print("-" * 47)
        print(f" {'Mean Dice (all classes)':<30} {seg_metrics.get('dice_mean', 0):>15.4f}")
        print(f" {'Mean Dice (excl. background)':<30} {seg_metrics.get('dice_mean_no_bg', 0):>15.4f}")
        print(f" {'Mean IoU (mIoU)':<30} {seg_metrics.get('iou_mean', 0):>15.4f}")
        print(f" {'Mean IoU (excl. background)':<30} {seg_metrics.get('iou_mean_no_bg', 0):>15.4f}")
        print(f" {'Pixel Accuracy':<30} {seg_metrics.get('pixel_accuracy', 0):>15.4f}")
        print(f" {'Number of batches':<30} {seg_metrics.get('num_batches', 0):>15d}")

    if breslow_metrics is not None:
        print("\n BRESLOW DEPTH METRICS")
        print("-" * 70)
        print(f"\n {'Metric':<40} {'Value':>20}")
        print("-" * 62)
        print(f" {'Mean Absolute Error (MAE)':<40} {breslow_metrics.get('mae', 0):>17.2f} um")
        print(f" {'Root Mean Squared Error (RMSE)':<40} {breslow_metrics.get('rmse', 0):>17.2f} um")
        print(f" {'R-squared (R2)':<40} {breslow_metrics.get('r_squared', 0):>20.4f}")
        print(f" {'Pearson Correlation (r)':<40} {breslow_metrics.get('pearson_r', 0):>20.4f}")
        print("-" * 62)
        print(f" {'T-category Accuracy':<40} {breslow_metrics.get('t_category_accuracy', 0)*100:>17.1f} %")
        print(f" {'T-category Adjacent Accuracy':<40} {breslow_metrics.get('t_category_adjacent_accuracy', 0)*100:>17.1f} %")
        print(f" {'Binary Accuracy at 800um':<40} {breslow_metrics.get('accuracy_at_800um', 0)*100:>17.1f} %")
        print("-" * 62)
        obs_var = breslow_metrics.get('observer_variability_threshold', 860)
        print(f" {'Within {0}um (inter-observer var)':<40} {breslow_metrics.get('within_observer_var', 0)*100:>17.1f} %".format(int(obs_var)))
        print(f" {'Within 500um':<40} {breslow_metrics.get('within_500um', 0)*100:>17.1f} %")
        print(f" {'Within 200um':<40} {breslow_metrics.get('within_200um', 0)*100:>17.1f} %")
        print("-" * 62)
        error_pct = breslow_metrics.get("error_percentiles", {})
        if error_pct:
            print(f" {'Median Absolute Error (P50)':<40} {error_pct.get('p50', 0):>17.2f} um")
            print(f" {'90th Percentile Error (P90)':<40} {error_pct.get('p90', 0):>17.2f} um")
            print(f" {'95th Percentile Error (P95)':<40} {error_pct.get('p95', 0):>17.2f} um")
        print(f"\n {'Number of samples':<40} {breslow_metrics.get('n_samples', 0):>20d}")

    print("\n" + "=" * 70)


# === Legacy one-shot helpers (used by the trainer's per-batch progress) ===

def dice_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Dice for a single binary mask. Legacy."""
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    predictions = predictions.float()
    targets = targets.float()
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """IoU for a single binary mask. Legacy. (Always ≤ Dice.)"""
    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)
    if targets.dim() == 4:
        targets = targets.squeeze(1)
    predictions = predictions.float()
    targets = targets.float()
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pixel-wise accuracy. Legacy."""
    if predictions.dim() == 4:
        predictions = predictions.argmax(dim=1)
    correct = (predictions == targets).float().sum()
    return correct / targets.numel()


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, Any]:
    """One-shot multi-class segmentation metrics for a single batch. Used by Trainer."""
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics.update(predictions, targets)
    return metrics.compute()


# === Smoke test ==========================================================

if __name__ == "__main__":
    print("Metrics Module Examples")
    print("=" * 70)

    batch_size, num_classes = 4, 4
    height, width = 256, 256
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))

    print("\n1. Segmentation Metrics:")
    class_names = ["Background", "Tumour", "Epidermis", "Dermis"]
    seg_metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)
    for _ in range(5):
        preds = torch.randn(batch_size, num_classes, height, width)
        targs = torch.randint(0, num_classes, (batch_size, height, width))
        seg_metrics.update(preds, targs)
    seg_results = seg_metrics.compute()
    print(f"   Mean Dice: {seg_results['dice_mean']:.4f}")
    print(f"   Mean IoU: {seg_results['iou_mean']:.4f}")
    print(f"   Pixel Accuracy: {seg_results['pixel_accuracy']:.4f}")

    print("\n2. Breslow Depth Metrics:")
    breslow_metrics = BreslowMetrics()
    np.random.seed(42)
    true_depths = np.random.uniform(200, 5000, 50)
    predicted_depths = np.clip(true_depths + np.random.normal(0, 300, 50), 0, 10000)
    breslow_metrics.update(predicted_depths, true_depths)
    breslow_results = breslow_metrics.compute()
    print(f"   MAE: {breslow_results['mae']:.2f} um")
    print(f"   RMSE: {breslow_results['rmse']:.2f} um")
    print(f"   R-squared: {breslow_results['r_squared']:.4f}")
    print(f"   T-category Accuracy: {breslow_results['t_category_accuracy']*100:.1f}%")
    print(f"   Within 860um: {breslow_results['within_observer_var']*100:.1f}%")

    print("\n3. Confusion Matrix:")
    confusion = compute_confusion_matrix(predictions, targets, num_classes)
    print(f"   Shape: {confusion.shape}")
    print(f"   Total pixels: {confusion.sum()}")

    print("\n4. Metrics Summary:")
    print_metrics_summary(seg_results, breslow_results, title="EXAMPLE EVALUATION")

    print("\n" + "=" * 70)
    print("Examples completed!")
