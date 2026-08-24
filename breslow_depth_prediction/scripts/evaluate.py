#!/usr/bin/env python
"""Evaluation entry point — runs a trained model on the test set and writes a report.

Loads test set, runs forward pass, computes per-class segmentation metrics
(Dice/IoU/precision/recall) and Breslow depth metrics (MAE/RMSE/T-cat
accuracy/etc.), saves JSON + text report + plots under `results/evaluation/`.

Usage:
    python breslow_depth_prediction/scripts/evaluate.py
    python breslow_depth_prediction/scripts/evaluate.py --checkpoint checkpoints/best_model_v1.pth
    python breslow_depth_prediction/scripts/evaluate.py --no-visualize
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / CI use
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from tqdm import tqdm

# Add project root to sys.path so `breslow_depth_prediction.*` imports work.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from breslow_depth_prediction.src.config import load_config, resolve_paths
from breslow_depth_prediction.src.data import BreslowDataset, get_val_transforms
from breslow_depth_prediction.src.data.transforms import denormalize
from breslow_depth_prediction.src.models import get_model
from breslow_depth_prediction.src.models.unet import load_model_checkpoint
from breslow_depth_prediction.src.training.metrics import (
    BreslowMetrics,
    SegmentationMetrics,
    compute_confusion_matrix,
    get_t_category,
    print_metrics_summary,
)
from breslow_depth_prediction.src.visualization import save_figure, visualize_sample


# === Breslow depth calculation ===========================================
# Core algorithmic contribution of the project. See docs/METHODS_AND_RESULTS.md §5.

def get_original_image_size(image_path: Path) -> tuple:
    """Return (height, width) of the on-disk image. Needed for the per-image scale factor."""
    with Image.open(image_path) as img:
        w, h = img.size  # PIL: (width, height)
    return h, w


def _calculate_breslow_depth_um_vertical(
    predicted_mask: np.ndarray,
    original_height: int,
    resized_height: int = 512,
    resolution_um_per_pixel: float = 4.0,
    reference_percentile: float = 10.0,
) -> dict:
    """V1/V2 baseline: vertical (row-distance) Breslow depth.

    Algorithm: (1) per-image scale factor; (2) extract tumour & epidermis masks;
    (3) reference row = `reference_percentile`-th percentile of epidermis rows;
    (4) deepest tumour row = max row index containing any tumour pixel;
    (5) depth = (tumour_bottom - reference) × µm/pixel.

    `reference_percentile` controls how robust-vs-aggressive the surface estimate
    is: 10 (default) rejects single-stray epidermis pixels at the top; lower
    values (5, 3, 0) place the reference row higher in the image, yielding
    LARGER predicted depths. Useful when the model under-predicts (e.g. V5
    misclassifying T4 as T3 because the reference row drifted downward).

    Class mapping: 0=background, 1=tumour, 2=epidermis, 3=dermis.

    Example (sample Sample-15, V1 rerun):
        >>> # original 933x1865, mask 512x512, GT = 3883.5 µm
        >>> out = _calculate_breslow_depth_um_vertical(mask, original_height=933)
        >>> out["depth_um"]    # 3374.8 (error 509 µm, within inter-observer 860 µm)
        >>> out["um_per_pixel"]  # 7.29 = 4.0 * (933/512)
    """
    # Step 1: scale factor and µm-per-resized-pixel.
    scale_factor = original_height / resized_height
    um_per_resized_pixel = resolution_um_per_pixel * scale_factor

    # Step 2: per-class binary masks.
    tumour_mask = (predicted_mask == 1).astype(np.uint8)
    epidermis_mask = (predicted_mask == 2).astype(np.uint8)

    result = {
        "depth_um": 0.0,
        "depth_pixels": 0.0,
        "um_per_pixel": float(um_per_resized_pixel),
        "scale_factor": float(scale_factor),
        "tumour_pixels": int(tumour_mask.sum()),
        "epidermis_pixels": int(epidermis_mask.sum()),
        "confidence": 1.0,
        "method": "vertical",
    }

    if tumour_mask.sum() == 0:  # degenerate — no tumour predicted
        result["confidence"] = 0.0
        return result

    # Step 3: reference row (top of epidermis). Configurable percentile.
    # 10 (default) rejects stray top pixels; 0 = absolute min; lower => higher
    # surface => bigger predicted depth.
    epidermis_rows = np.where(epidermis_mask.sum(axis=1) > 0)[0]
    if len(epidermis_rows) > 0:
        reference_row = int(np.percentile(epidermis_rows, reference_percentile))
    else:
        # Fallback: no epidermis predicted — use top-of-tumour, halve confidence.
        tumour_rows = np.where(tumour_mask.sum(axis=1) > 0)[0]
        reference_row = tumour_rows.min() if len(tumour_rows) > 0 else 0
        result["confidence"] *= 0.5

    # Step 4: deepest tumour row = largest row index with any tumour pixel.
    tumour_rows = np.where(tumour_mask.sum(axis=1) > 0)[0]
    if len(tumour_rows) == 0:
        result["confidence"] = 0.0
        return result
    tumour_bottom = tumour_rows.max()

    # Step 5: pixel distance -> µm. Clamp at 0 (negative would mean reference below tumour).
    depth_pixels = max(0, tumour_bottom - reference_row)
    depth_um = depth_pixels * um_per_resized_pixel

    result["depth_pixels"] = float(depth_pixels)
    result["depth_um"] = float(depth_um)

    # Confidence penalty for tiny tumour predictions.
    if tumour_mask.sum() < 100:
        result["confidence"] *= 0.5
    elif tumour_mask.sum() < 500:
        result["confidence"] *= 0.8

    return result


def _calculate_breslow_depth_um_perpendicular(
    predicted_mask: np.ndarray,
    original_height: int,
    resized_height: int = 512,
    resolution_um_per_pixel: float = 4.0,
    min_component_size: int = 20,
) -> dict:
    """V3 ablation: perpendicular Breslow depth via Euclidean distance transform.

    Replaces the row-difference of the vertical method with the true
    point-to-surface distance. AJCC defines Breslow as the perpendicular
    measurement from the granular layer to the deepest tumour cell, so on
    tilted/curved sections the vertical proxy systematically overestimates.

    Algorithm:
        1. Same per-image scale factor as vertical.
        2. Morphological cleaning: drop epidermis components < `min_component_size`
           pixels (rejects stray pixels that would distort the surface).
        3. Surface mask: topmost epidermis row per column (vectorised via argmax).
        4. Euclidean distance transform from the surface — every pixel gets
           its distance to the nearest surface point.
        5. Perpendicular depth = max distance over tumour pixels.
        6. Scale to µm using the same factor as vertical.

    If cleaning removes all epidermis, falls back to the vertical calculator
    (its own confidence penalty handles the no-epidermis case).

    Example (sample Sample-15, V1 rerun):
        >>> # GT = 3883.5 µm, vertical predicts 3374.8 (-509 µm error)
        >>> out = _calculate_breslow_depth_um_perpendicular(mask, original_height=933)
        >>> out["method"]  # 'perpendicular'
    """
    scale_factor = original_height / resized_height
    um_per_resized_pixel = resolution_um_per_pixel * scale_factor

    tumour_mask = (predicted_mask == 1).astype(np.uint8)
    epidermis_mask = (predicted_mask == 2).astype(np.uint8)

    result = {
        "depth_um": 0.0,
        "depth_pixels": 0.0,
        "um_per_pixel": float(um_per_resized_pixel),
        "scale_factor": float(scale_factor),
        "tumour_pixels": int(tumour_mask.sum()),
        "epidermis_pixels": int(epidermis_mask.sum()),
        "confidence": 1.0,
        "method": "perpendicular",
    }

    if tumour_mask.sum() == 0:
        result["confidence"] = 0.0
        return result

    # Step 2: drop tiny epidermis blobs (8-connectivity).
    if epidermis_mask.sum() > 0:
        labelled, n_components = ndi.label(epidermis_mask)
        if n_components > 0:
            sizes = ndi.sum(epidermis_mask, labelled, range(1, n_components + 1))
            keep = np.where(sizes >= min_component_size)[0] + 1
            epidermis_mask = np.isin(labelled, keep).astype(np.uint8)

    # Fallback: cleaning destroyed all epidermis -> defer to vertical.
    if epidermis_mask.sum() == 0:
        fallback = _calculate_breslow_depth_um_vertical(
            predicted_mask, original_height, resized_height, resolution_um_per_pixel,
        )
        fallback["method"] = "perpendicular_fallback_vertical"
        return fallback

    # Step 3: top surface — for each column with epidermis, the topmost row.
    # `argmax` along axis=0 returns the index of the first 1 in each column.
    top_rows = np.argmax(epidermis_mask, axis=0)
    cols_with_epi = np.where(epidermis_mask.sum(axis=0) > 0)[0]
    surface_mask = np.zeros_like(epidermis_mask, dtype=np.uint8)
    surface_mask[top_rows[cols_with_epi], cols_with_epi] = 1

    # Step 4: distance transform — distance from each pixel to nearest surface pixel.
    # `distance_transform_edt` measures distance from foreground (1) to nearest
    # background (0) pixel, so we pass the complement of the surface mask.
    distance_to_surface = ndi.distance_transform_edt(1 - surface_mask)

    # Step 5: max distance over tumour pixels.
    tumour_distances = distance_to_surface[tumour_mask > 0]
    if len(tumour_distances) == 0:
        result["confidence"] = 0.0
        return result
    depth_pixels = float(tumour_distances.max())

    result["depth_pixels"] = depth_pixels
    result["depth_um"] = depth_pixels * um_per_resized_pixel

    # Confidence penalty for tiny tumour predictions (matches vertical method).
    if tumour_mask.sum() < 100:
        result["confidence"] *= 0.5
    elif tumour_mask.sum() < 500:
        result["confidence"] *= 0.8

    return result


_BRESLOW_CALCULATORS = {
    "vertical": _calculate_breslow_depth_um_vertical,
    "perpendicular": _calculate_breslow_depth_um_perpendicular,
}


def tta_forward(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """4-way test-time augmentation: identity + h-flip + v-flip + 180-rotation.

    Each augmentation is applied to the input, the model's logits are inverted
    back to the original orientation, softmax-normalised, then averaged over
    the four passes. Returns a (1, C, H, W) probability tensor that can be
    consumed exactly like raw logits (argmax over class dim still works).

    Augmentations chosen because semantic segmentation of histopathology is
    rotation/flip-invariant — there is no canonical "up" for a tissue slide.
    Rotating 90/270 also works in principle but doubles inference cost; the
    4-way scheme is the standard cheap choice.
    """
    aug_invs = [
        (lambda x: x,                       lambda y: y),                       # identity
        (lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),  # h-flip (W)
        (lambda x: torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),  # v-flip (H)
        (lambda x: torch.flip(x, dims=[-2, -1]), lambda y: torch.flip(y, dims=[-2, -1])),  # 180
    ]
    probs_sum = None
    for aug, inv in aug_invs:
        out = model(aug(image))               # may be tensor (single-task) or dict (V4 multi-task)
        logits = out["mask"] if isinstance(out, dict) else out
        logits = inv(logits)                  # back to original frame
        probs = logits.softmax(dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(aug_invs)


def postprocess_mask_for_breslow(
    mask: np.ndarray,
    min_tumour_size: int = 50,
    min_epidermis_size: int = 20,
    fill_tumour_holes: bool = False,
) -> np.ndarray:
    """Return a CLEANED COPY of the predicted class-index mask for the Breslow
    calculator. Caller keeps the original for segmentation metrics — that way
    post-processing can never inflate Dice/IoU.

    Cleaning steps:
      1. Drop tumour connected components smaller than `min_tumour_size`
         pixels (replaced with class 3 = dermis, the typical surrounding tissue).
      2. Drop epidermis connected components smaller than `min_epidermis_size`
         pixels (replaced with class 0 = background). Targets the vertical
         calculator's main failure mode — stray epidermis pixels skewing the
         10th-percentile reference row.
      3. Optionally fill small holes inside tumour blobs (off by default — can
         deepen the predicted tumour boundary and hurt the vertical calculator).

    Class mapping: 0=background, 1=tumour, 2=epidermis, 3=dermis.
    """
    cleaned = mask.copy()

    # Step 1: tumour CC cleaning.
    tumour = cleaned == 1
    if min_tumour_size > 0 and tumour.any():
        labelled, n = ndi.label(tumour)
        if n > 0:
            sizes = ndi.sum(tumour, labelled, range(1, n + 1))
            small = np.where(sizes < min_tumour_size)[0] + 1
            if len(small) > 0:
                cleaned[np.isin(labelled, small)] = 3  # surrounding tissue ≈ dermis
                tumour = cleaned == 1

    # Step 2: epidermis CC cleaning (perpendicular calc does this internally;
    # vertical doesn't — this is where the vertical calc's noise comes from).
    epidermis = cleaned == 2
    if min_epidermis_size > 0 and epidermis.any():
        labelled, n = ndi.label(epidermis)
        if n > 0:
            sizes = ndi.sum(epidermis, labelled, range(1, n + 1))
            small = np.where(sizes < min_epidermis_size)[0] + 1
            if len(small) > 0:
                # Stray epidermis blobs are usually surrounded by background, so
                # demoting them to background is the conservative choice.
                cleaned[np.isin(labelled, small)] = 0

    # Step 3: tumour hole filling (off by default — caused regression on V1 vert).
    if fill_tumour_holes and tumour.any():
        filled = ndi.binary_fill_holes(tumour)
        new_pixels = filled & ~tumour
        if new_pixels.any():
            cleaned[new_pixels] = 1

    return cleaned


def calculate_breslow_depth_um(
    predicted_mask: np.ndarray,
    original_height: int,
    resized_height: int = 512,
    resolution_um_per_pixel: float = 4.0,
    method: str = "vertical",
    reference_percentile: float = 10.0,
) -> dict:
    """Public Breslow calculator. Routes to the chosen algorithm.

    Args:
        predicted_mask:          (H, W) class-index mask (0=bg, 1=tumour, 2=epi, 3=dermis).
        original_height:         Original WSI PNG height in pixels.
        resized_height:          Mask height (matches config image_size[0]).
        resolution_um_per_pixel: Optical resolution of the original slide.
        method: "vertical" (V1/V2 baseline, row-distance) or "perpendicular"
                (V3 ablation, Euclidean distance from epidermis top surface).
        reference_percentile: For "vertical" only — percentile of epidermis
                rows used as the surface reference (default 10). Lower values
                place the surface higher and increase predicted depths.
                Ignored by the perpendicular method (which uses topmost-per-column).

    Returns:
        Result dict with depth_um, depth_pixels, um_per_pixel, scale_factor,
        tumour_pixels, epidermis_pixels, confidence, method.
    """
    if method not in _BRESLOW_CALCULATORS:
        raise ValueError(
            f"Unknown breslow method '{method}'. "
            f"Valid options: {sorted(_BRESLOW_CALCULATORS)}."
        )
    kwargs = dict(
        predicted_mask=predicted_mask,
        original_height=original_height,
        resized_height=resized_height,
        resolution_um_per_pixel=resolution_um_per_pixel,
    )
    # Only the vertical calculator uses reference_percentile.
    if method == "vertical":
        kwargs["reference_percentile"] = reference_percentile
    return _BRESLOW_CALCULATORS[method](**kwargs)


# === Logging =============================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure logging to terminal + timestamped file under `log_dir`."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


# === Core evaluation =====================================================

@torch.no_grad()
def run_evaluation(
    config,
    model,
    device,
    logger,
    breslow_method: str = "vertical",
    postprocess: bool = False,
    tta: bool = False,
    multitask_ensemble: str = "geometric",
    reference_percentile: float = 10.0,
):
    """Run the model on every test sample, accumulating segmentation + Breslow metrics.

    Args:
        breslow_method: "vertical" (V1/V2 baseline) or "perpendicular" (V3 ablation).
        postprocess: If True, clean tumour/epidermis CCs on the predicted mask
            BEFORE the Breslow calculator. Segmentation metrics still use the raw
            mask, so this can never inflate Dice/IoU.
        tta: If True, run 4-way test-time augmentation (identity + flips + 180)
            and average softmax probabilities before argmax. Affects BOTH the
            segmentation metrics and the Breslow calculation.
        multitask_ensemble: How to combine the geometric depth (from mask) with
            the regressed depth (from the V4 model's regression head):
            "geometric" (use only the geometric depth — backward-compatible
            default), "regressed" (use only the regression-head depth),
            "average" (mean of the two). Ignored for non-multitask models.

    Returns: (seg_results, breslow_results, sample_results, confusion, test_dataset).
    """
    data_config = config.get("data", {})
    split_config = config.get("split", {})
    image_size = tuple(data_config.get("image_size", [512, 512]))
    resolution = data_config.get("resolution_um_per_pixel", 4.0)
    num_classes = config.get("classes", {}).get("num_classes", 4)

    # Build test dataset with same seed + ratios as training -> identical test split.
    val_transform = get_val_transforms(config, image_size=image_size)

    # Match utils.create_dataloaders: pass coords_file/exclude_file from config so
    # evaluation uses the SAME CSV the model was trained on. Previously these
    # config fields were ignored, causing silent data leakage when the training
    # CSV differed from the default `breslow_depth_coords.csv`.
    data_dir = data_config.get("data_dir", "./data")
    coords_file = data_config.get("coords_file")
    coords_path = None
    if coords_file is not None:
        coords_path = Path(coords_file)
        if not coords_path.is_absolute():
            coords_path = Path(data_dir) / coords_path
    exclude_file = data_config.get("exclude_file")
    exclude_path = None
    if exclude_file is not None:
        exclude_path = Path(exclude_file)
        if not exclude_path.is_absolute():
            exclude_path = Path(data_dir) / exclude_path

    test_dataset = BreslowDataset(
        data_dir=data_dir,
        csv_path=coords_path,
        exclude_file=exclude_path,
        transform=val_transform,
        split="test",
        split_ratios=(
            split_config.get("train_ratio", 0.7),
            split_config.get("val_ratio", 0.15),
            split_config.get("test_ratio", 0.15),
        ),
        seed=split_config.get("random_seed", 42),
    )
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    logger.info(f"Breslow calculator: {breslow_method}")
    logger.info(f"Mask post-processing: {'on' if postprocess else 'off'}")
    logger.info(f"Test-time augmentation: {'on (4-way)' if tta else 'off'}")
    logger.info(f"Multi-task ensemble strategy: {multitask_ensemble}")
    if breslow_method == "vertical":
        logger.info(f"Vertical reference percentile: {reference_percentile}")

    class_names = ["Background", "Tumour", "Epidermis", "Dermis"]
    seg_metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)
    breslow_metrics = BreslowMetrics()

    sample_results = []
    all_preds = []
    all_masks = []
    model.eval()

    for idx in tqdm(range(len(test_dataset)), desc="Evaluating test set"):
        sample = test_dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)  # add batch dim
        mask = sample["mask"]
        gt_depth_um = sample["breslow_depth_um"]
        sample_id = sample["sample_id"]
        gt_t_cat = sample["t_category"]

        # Forward pass. Single-task models return logits (1, C, H, W);
        # the V4 multi-task model returns {"mask": logits, "depth": (1,) log-µm}.
        # `tta_forward` returns (1, C, H, W) softmax-averaged probs.
        if tta:
            mask_output = tta_forward(model, image)
            regressed_depth_um = None  # TTA wraps only the seg path
            # Detect if the underlying model is multi-task; if so, also run a
            # plain forward to grab the regression-head output (one extra fwd).
            with torch.no_grad():
                raw = model(image)
            if isinstance(raw, dict) and "depth" in raw:
                regressed_depth_um = float(torch.expm1(raw["depth"]).item())
        else:
            output = model(image)
            if isinstance(output, dict):
                mask_output = output["mask"]
                regressed_depth_um = float(torch.expm1(output["depth"]).item())
            else:
                mask_output = output
                regressed_depth_um = None

        pred_mask = mask_output.argmax(dim=1).squeeze(0).cpu()

        # Segmentation metrics use the RAW mask logits/probs.
        seg_metrics.update(mask_output.cpu(), mask.unsqueeze(0))
        per_sample = seg_metrics.compute_per_sample(pred_mask, mask)

        # Read original height for the per-image scale factor (one disk read per sample).
        orig_h, _ = get_original_image_size(test_dataset.samples[idx]["image_path"])

        # Optionally clean the mask (drop tiny tumour CCs, fill holes) before depth calc.
        breslow_input_mask = pred_mask.numpy()
        if postprocess:
            breslow_input_mask = postprocess_mask_for_breslow(breslow_input_mask)

        breslow_info = calculate_breslow_depth_um(
            breslow_input_mask,
            original_height=orig_h,
            resized_height=image_size[0],
            resolution_um_per_pixel=resolution,
            method=breslow_method,
            reference_percentile=reference_percentile,
        )
        geometric_depth_um = breslow_info["depth_um"]

        # Pick the headline depth based on the ensemble strategy.
        # Non-multitask models always use the geometric depth.
        if regressed_depth_um is None or multitask_ensemble == "geometric":
            pred_depth_um = geometric_depth_um
            ensemble_label = "geometric"
        elif multitask_ensemble == "regressed":
            pred_depth_um = regressed_depth_um
            ensemble_label = "regressed"
        elif multitask_ensemble == "average":
            pred_depth_um = 0.5 * (geometric_depth_um + regressed_depth_um)
            ensemble_label = "average"
        else:
            raise ValueError(
                f"Unknown multitask_ensemble='{multitask_ensemble}'. "
                f"Valid: geometric, regressed, average."
            )

        # Surface both heads' values in the per-sample record so post-hoc
        # analysis can re-aggregate without re-evaluating.
        breslow_info["geometric_depth_um"] = float(geometric_depth_um)
        breslow_info["regressed_depth_um"] = (
            float(regressed_depth_um) if regressed_depth_um is not None else None
        )
        breslow_info["ensemble_strategy"] = ensemble_label
        breslow_info["depth_um"] = float(pred_depth_um)
        pred_t_cat = get_t_category(pred_depth_um)
        breslow_metrics.update(pred_depth_um, gt_depth_um)

        sample_results.append({
            "sample_id": sample_id,
            "gt_breslow_um": gt_depth_um,
            "pred_breslow_um": pred_depth_um,
            "gt_t_category": gt_t_cat,
            "pred_t_category": pred_t_cat,
            "error_um": abs(pred_depth_um - gt_depth_um),
            "dice_mean": per_sample["dice_mean"],
            "dice_per_class": per_sample["dice_per_class"],
            "iou_mean": per_sample["iou_mean"],
            "pixel_accuracy": per_sample["pixel_accuracy"],
            "breslow_detail": breslow_info,
        })

        all_preds.append(pred_mask.numpy())
        all_masks.append(mask.numpy())

        logger.info(
            f"  {sample_id}: GT={gt_depth_um:.0f}um ({gt_t_cat}), "
            f"Pred={pred_depth_um:.0f}um ({pred_t_cat}), "
            f"Err={abs(pred_depth_um - gt_depth_um):.0f}um, "
            f"Dice={per_sample['dice_mean']:.4f}"
        )

    seg_results = seg_metrics.compute()
    breslow_results = breslow_metrics.compute()

    # Pixel-level confusion matrix across all samples.
    confusion = compute_confusion_matrix(
        torch.tensor(np.stack(all_preds)),
        torch.tensor(np.stack(all_masks)),
        num_classes,
    )

    return seg_results, breslow_results, sample_results, confusion, test_dataset


# === Visualisations ======================================================

def generate_visualisations(
    config, model, device, test_dataset, sample_results,
    seg_results, breslow_results, eval_dir, logger,
):
    """Save 6 plot families under `eval_dir`: per-sample overlays,
    breslow_scatter, bland_altman, error_distribution, dice_per_class,
    t_category_confusion."""
    vis_dir = eval_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    image_size = tuple(config.get("data", {}).get("image_size", [512, 512]))

    # 1. Per-sample prediction overlays.
    logger.info("Generating per-sample prediction overlays...")
    model.eval()
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        image_tensor = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy()
        sample_id = sample["sample_id"]
        sr = sample_results[idx]

        with torch.no_grad():
            pred_mask = model(image_tensor).argmax(dim=1).squeeze(0).cpu().numpy()

        # Un-normalise image (from ImageNet stats) and convert to uint8 for display.
        image_vis = denormalize(sample["image"]).permute(1, 2, 0).numpy()
        image_vis = (image_vis * 255).astype(np.uint8)

        # Scale GT Breslow coords from original to resized space.
        orig_h, orig_w = get_original_image_size(test_dataset.samples[idx]["image_path"])
        coords = sample["breslow_coords"]
        scaled_coords = {
            "x1": coords["x1"] * image_size[1] / orig_w,
            "y1": coords["y1"] * image_size[0] / orig_h,
            "x2": coords["x2"] * image_size[1] / orig_w,
            "y2": coords["y2"] * image_size[0] / orig_h,
        }

        fig = visualize_sample(
            image=image_vis,
            mask=mask,
            prediction=pred_mask,
            breslow_info=scaled_coords,
            title=(
                f"{sample_id}  |  GT: {sr['gt_breslow_um']:.0f}um ({sr['gt_t_category']})  |  "
                f"Pred: {sr['pred_breslow_um']:.0f}um ({sr['pred_t_category']})  |  "
                f"Dice: {sr['dice_mean']:.3f}"
            ),
            show=False,
        )
        save_figure(fig, vis_dir / f"{sample_id}_prediction.png")

    # 2. Predicted vs GT scatter, with diagonal + ±860 µm tolerance band.
    logger.info("Generating Breslow depth scatter plot...")
    gt = [r["gt_breslow_um"] for r in sample_results]
    pred = [r["pred_breslow_um"] for r in sample_results]
    max_val = max(max(gt), max(pred)) * 1.1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(gt, pred, s=80, alpha=0.7, edgecolors="k", linewidths=0.5, zorder=3)
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect prediction")
    x = np.linspace(0, max_val, 100)
    ax.fill_between(x, x - 860, x + 860, alpha=0.12, color="green",
                     label="±860 µm (inter-observer var.)")
    for thr, lab in [(800, "0.8 mm"), (1000, "1.0 mm"), (2000, "2.0 mm"), (4000, "4.0 mm")]:
        ax.axvline(thr, color="gray", ls=":", alpha=0.4)
        ax.axhline(thr, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel("Ground Truth Breslow Depth (µm)", fontsize=12)
    ax.set_ylabel("Predicted Breslow Depth (µm)", fontsize=12)
    ax.set_title("Predicted vs Ground Truth Breslow Depth", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    plt.tight_layout()
    save_figure(fig, eval_dir / "breslow_scatter.png")

    # 3. Bland-Altman plot (mean vs difference, with bias + 1.96 SD limits).
    logger.info("Generating Bland-Altman plot...")
    gt_arr, pred_arr = np.array(gt), np.array(pred)
    means = (gt_arr + pred_arr) / 2
    diffs = pred_arr - gt_arr
    md, sd = np.mean(diffs), np.std(diffs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(means, diffs, s=80, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.axhline(md, color="r", lw=1.5, label=f"Mean bias: {md:.0f} µm")
    ax.axhline(md + 1.96 * sd, color="r", ls="--", lw=1, label=f"+1.96 SD: {md + 1.96*sd:.0f} µm")
    ax.axhline(md - 1.96 * sd, color="r", ls="--", lw=1, label=f"‒1.96 SD: {md - 1.96*sd:.0f} µm")
    ax.axhline(860, color="green", ls=":", alpha=0.5, label="±860 µm")
    ax.axhline(-860, color="green", ls=":", alpha=0.5)
    ax.set_xlabel("Mean of GT and Predicted (µm)", fontsize=12)
    ax.set_ylabel("Predicted − Ground Truth (µm)", fontsize=12)
    ax.set_title("Bland-Altman Plot: Breslow Depth Agreement", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_figure(fig, eval_dir / "bland_altman.png")

    # 4. Absolute error histogram with 860 µm + MAE markers.
    logger.info("Generating error distribution histogram...")
    errors = [r["error_um"] for r in sample_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=12, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(860, color="red", ls="--", lw=2, label="Inter-observer var. (860 µm)")
    ax.axvline(np.mean(errors), color="orange", lw=2, label=f"MAE: {np.mean(errors):.0f} µm")
    ax.set_xlabel("Absolute Error (µm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Breslow Depth Prediction Errors", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_figure(fig, eval_dir / "error_distribution.png")

    # 5. Per-class Dice bar chart with 0.80 target line.
    logger.info("Generating per-class Dice chart...")
    names = seg_results["class_names"]
    dice = seg_results["dice_per_class"]
    colors = ["#555555", "#3498db", "#2ecc71", "#e74c3c"]  # bg/tumour/epi/dermis

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, dice, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.80, color="red", ls="--", lw=1, label="Target (0.80)")
    ax.set_ylabel("Dice Coefficient", fontsize=12)
    ax.set_title("Per-Class Segmentation Performance (Test Set)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    for bar, sc in zip(bars, dice):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{sc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, eval_dir / "dice_per_class.png")

    # 6. T-category confusion matrix (skipped if only one category present).
    logger.info("Generating T-category confusion matrix...")
    gt_cats = [r["gt_t_category"] for r in sample_results]
    pred_cats = [r["pred_t_category"] for r in sample_results]
    all_cats = ["T1a", "T1b", "T2", "T3", "T4"]
    present = sorted(set(gt_cats + pred_cats), key=lambda c: all_cats.index(c))

    if len(present) > 1:
        cm = sk_confusion_matrix(gt_cats, pred_cats, labels=present)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=present, yticklabels=present, ax=ax)
        ax.set_xlabel("Predicted T-Category", fontsize=12)
        ax.set_ylabel("True T-Category", fontsize=12)
        ax.set_title("T-Category Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_figure(fig, eval_dir / "t_category_confusion.png")

    logger.info(f"All visualisations saved to {eval_dir}")


# === Report ==============================================================

def save_report(
    seg_results, breslow_results, sample_results, confusion, eval_dir, logger,
    breslow_method: str = "vertical",
    postprocess: bool = False,
    tta: bool = False,
    multitask_ensemble: str = "geometric",
    reference_percentile: float = 10.0,
):
    """Write metrics.json, per_sample_results.json, segmentation_confusion_matrix.npy,
    and evaluation_report.txt under `eval_dir`."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    print_metrics_summary(seg_results, breslow_results, title="TEST SET EVALUATION")

    # metrics.json — machine-readable summary.
    metrics_json = {
        "breslow_method": breslow_method,
        "postprocess": postprocess,
        "tta": tta,
        "multitask_ensemble": multitask_ensemble,
        "reference_percentile": reference_percentile,
        "segmentation": {
            "dice_per_class": seg_results["dice_per_class"],
            "iou_per_class": seg_results["iou_per_class"],
            "precision_per_class": seg_results["precision_per_class"],
            "recall_per_class": seg_results["recall_per_class"],
            "dice_mean": seg_results["dice_mean"],
            "dice_mean_no_bg": seg_results["dice_mean_no_bg"],
            "iou_mean": seg_results["iou_mean"],
            "iou_mean_no_bg": seg_results["iou_mean_no_bg"],
            "pixel_accuracy": seg_results["pixel_accuracy"],
            "class_names": seg_results["class_names"],
        },
        "breslow": {k: v for k, v in breslow_results.items() if k != "t_category_confusion"},
        "t_category_confusion": breslow_results.get("t_category_confusion", {}),
    }
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)

    with open(eval_dir / "per_sample_results.json", "w") as f:
        json.dump(sample_results, f, indent=2, default=str)

    np.save(eval_dir / "segmentation_confusion_matrix.npy", confusion)

    # evaluation_report.txt — human-readable layout matches the original V1 report.
    with open(eval_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(" BRESLOW DEPTH PREDICTION - TEST SET EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f" Breslow calculator: {breslow_method}\n")
        f.write(f" Mask post-processing: {'on' if postprocess else 'off'}\n")
        f.write(f" Test-time augmentation: {'on (4-way)' if tta else 'off'}\n")
        f.write(f" Multi-task ensemble: {multitask_ensemble}\n")
        if breslow_method == "vertical":
            f.write(f" Vertical reference percentile: {reference_percentile}\n")
        f.write("=" * 70 + "\n\n")

        # Segmentation block.
        f.write("SEGMENTATION METRICS\n")
        f.write("-" * 70 + "\n")
        cnames = seg_results["class_names"]
        f.write(f"{'Class':<15} {'Dice':>10} {'IoU':>10} {'Precision':>10} {'Recall':>10}\n")
        f.write("-" * 57 + "\n")
        for i, name in enumerate(cnames):
            f.write(
                f"{name:<15} "
                f"{seg_results['dice_per_class'][i]:>10.4f} "
                f"{seg_results['iou_per_class'][i]:>10.4f} "
                f"{seg_results['precision_per_class'][i]:>10.4f} "
                f"{seg_results['recall_per_class'][i]:>10.4f}\n"
            )
        f.write("-" * 57 + "\n")
        f.write(f"\nMean Dice (all classes):         {seg_results['dice_mean']:.4f}\n")
        f.write(f"Mean Dice (excl. background):    {seg_results['dice_mean_no_bg']:.4f}\n")
        f.write(f"Mean IoU:                        {seg_results['iou_mean']:.4f}\n")
        f.write(f"Pixel Accuracy:                  {seg_results['pixel_accuracy']:.4f}\n")

        # Breslow block.
        f.write("\n\nBRESLOW DEPTH METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Mean Absolute Error (MAE):       {breslow_results['mae']:.2f} um\n")
        f.write(f"Root Mean Squared Error (RMSE):  {breslow_results['rmse']:.2f} um\n")
        f.write(f"R-squared:                       {breslow_results['r_squared']:.4f}\n")
        f.write(f"Pearson Correlation:             {breslow_results['pearson_r']:.4f}\n")
        f.write(f"\nT-category Accuracy:             {breslow_results['t_category_accuracy']*100:.1f}%\n")
        f.write(f"Adjacent T-cat Accuracy:         {breslow_results['t_category_adjacent_accuracy']*100:.1f}%\n")
        f.write(f"Binary Accuracy at 800um:        {breslow_results['accuracy_at_800um']*100:.1f}%\n")
        f.write(f"\nWithin 860um (inter-observer):   {breslow_results['within_observer_var']*100:.1f}%\n")
        f.write(f"Within 500um:                    {breslow_results['within_500um']*100:.1f}%\n")
        f.write(f"Within 200um:                    {breslow_results['within_200um']*100:.1f}%\n")
        f.write(f"\nNumber of test samples:          {breslow_results['n_samples']}\n")

        # Per-sample table.
        f.write("\n\nPER-SAMPLE RESULTS\n")
        f.write("-" * 90 + "\n")
        f.write(
            f"{'Sample ID':<25} {'GT (um)':>10} {'Pred (um)':>10} "
            f"{'Error':>10} {'GT Cat':>8} {'Pred Cat':>9} {'Dice':>8}\n"
        )
        f.write("-" * 82 + "\n")
        for r in sample_results:
            f.write(
                f"{r['sample_id']:<25} "
                f"{r['gt_breslow_um']:>10.0f} "
                f"{r['pred_breslow_um']:>10.0f} "
                f"{r['error_um']:>10.0f} "
                f"{r['gt_t_category']:>8} "
                f"{r['pred_t_category']:>9} "
                f"{r['dice_mean']:>8.4f}\n"
            )

        # Performance vs handbook targets.
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("PERFORMANCE vs TARGETS (from preliminary report)\n")
        f.write("-" * 70 + "\n")
        td = seg_results["dice_per_class"][1]
        ed = seg_results["dice_per_class"][2]
        dd = seg_results["dice_per_class"][3]
        md = seg_results["dice_mean_no_bg"]
        mae = breslow_results["mae"]
        t_acc = breslow_results.get("t_category_accuracy", 0.0)
        within_860 = breslow_results.get("within_observer_var", 0.0)
        f.write(f"Tumour Dice:        {td:.4f}  (target: >0.80)        {'PASS' if td > 0.80 else 'FAIL'}\n")
        f.write(f"Epidermis Dice:     {ed:.4f}  (target: >0.80)        {'PASS' if ed > 0.80 else 'FAIL'}\n")
        f.write(f"Dermis Dice:        {dd:.4f}  (target: >0.80)        {'PASS' if dd > 0.80 else 'FAIL'}\n")
        f.write(f"Mean Dice (no bg):  {md:.4f}  (target: >0.80)        {'PASS' if md > 0.80 else 'FAIL'}\n")
        f.write("-" * 70 + "\n")
        f.write(f"Breslow MAE:        {mae:>7.0f} um  (target: <500 um)    {'PASS' if mae < 500 else 'FAIL'}\n")
        f.write(f"MAE < 860 um (inter-observer var): {' '*9}{'PASS' if mae < 860 else 'FAIL'}\n")
        f.write(f"T-category accuracy:  {t_acc * 100:5.1f} %  (target: >75 %)     {'PASS' if t_acc > 0.75 else 'FAIL'}\n")
        f.write(f"Within 860 um (% samples):  {within_860 * 100:5.1f} %  (target: >50 %)  {'PASS' if within_860 > 0.5 else 'FAIL'}\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Saved evaluation report to {eval_dir / 'evaluation_report.txt'}")
    logger.info(f"Saved metrics JSON to {eval_dir / 'metrics.json'}")
    logger.info(f"Saved per-sample results to {eval_dir / 'per_sample_results.json'}")


# === Main ================================================================

def main():
    """CLI entry point. Parses args, loads model + config, runs evaluation, saves report."""
    parser = argparse.ArgumentParser(description="Evaluate Breslow depth prediction model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--no-visualize", action="store_true", help="Skip visualisation generation")
    parser.add_argument(
        "--breslow-method",
        type=str,
        default="vertical",
        choices=sorted(_BRESLOW_CALCULATORS),
        help="Breslow depth algorithm: vertical (V1/V2 baseline) or perpendicular (V3 ablation).",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Clean the predicted mask (drop tiny tumour CCs, fill holes) before "
             "the Breslow calculator. Segmentation metrics use the RAW mask, so "
             "this only affects depth numbers — never inflates Dice/IoU.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="4-way test-time augmentation: average softmax over identity + "
             "h-flip + v-flip + 180-rotation. Affects both segmentation and "
             "Breslow numbers. Inference cost: 4x.",
    )
    parser.add_argument(
        "--multitask-ensemble",
        type=str,
        default="geometric",
        choices=["geometric", "regressed", "average"],
        help="For V4 multi-task models, how to combine the geometric depth "
             "(from the mask + perpendicular calculator) with the regressed "
             "depth (from the model's regression head). Ignored for V1/V2/V3.",
    )
    parser.add_argument(
        "--reference-percentile",
        type=float,
        default=10.0,
        help="For --breslow-method vertical: percentile of epidermis rows "
             "used as the surface reference (default 10). Lower values place "
             "the surface higher and increase predicted depths. Ignored by "
             "perpendicular calculator.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report+plots (default: <results_dir>/evaluation). "
             "Use this to keep V1/V2/V3 ablation runs separate "
             "(e.g. results/evaluation_v1_perp).",
    )
    args = parser.parse_args()

    config_path = (
        Path(args.config) if args.config is not None
        else project_root / "breslow_depth_prediction" / "configs" / "config_v1.yaml"
    )
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint is not None
        else project_root / "checkpoints" / "best_model.pth"
    )

    config = load_config(str(config_path))
    # Resolve relative paths in config against project_root so the script
    # is cwd-agnostic.
    config = resolve_paths(config, project_root)
    paths_config = config.get("paths", {})
    results_dir = Path(paths_config.get("results_dir", project_root / "results"))
    log_dir = Path(paths_config.get("log_dir", project_root / "logs"))

    # eval_dir = explicit --output-dir, else <results_dir>/evaluation (legacy default).
    if args.output_dir is not None:
        eval_dir = Path(args.output_dir)
        if not eval_dir.is_absolute():
            eval_dir = project_root / eval_dir
    else:
        eval_dir = results_dir / "evaluation"

    logger = setup_logging(log_dir)
    logger.info(f"Config: {config_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Breslow method: {args.breslow_method}")
    logger.info(f"Output dir: {eval_dir}")

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model = get_model(config)
    model = load_model_checkpoint(model, str(checkpoint_path))
    model.eval()
    logger.info("Model loaded successfully")

    logger.info("Starting test set evaluation...")
    start = time.time()
    seg_results, breslow_results, sample_results, confusion, test_dataset = run_evaluation(
        config, model, device, logger,
        breslow_method=args.breslow_method,
        postprocess=args.postprocess,
        tta=args.tta,
        multitask_ensemble=args.multitask_ensemble,
        reference_percentile=args.reference_percentile,
    )
    logger.info(f"Evaluation completed in {time.time() - start:.1f} seconds")

    save_report(
        seg_results, breslow_results, sample_results, confusion, eval_dir, logger,
        breslow_method=args.breslow_method,
        postprocess=args.postprocess,
        tta=args.tta,
        multitask_ensemble=args.multitask_ensemble,
        reference_percentile=args.reference_percentile,
    )

    if not args.no_visualize:
        logger.info("Generating visualisations...")
        generate_visualisations(
            config, model, device, test_dataset, sample_results,
            seg_results, breslow_results, eval_dir, logger,
        )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
