#!/usr/bin/env python
"""Single-image inference: predict Breslow depth from a new H&E image.

Loads a trained checkpoint, runs it on one image you point at, and prints the
predicted Breslow depth (µm + mm), AJCC T-category, and confidence score.
Optionally saves the predicted mask, an overlay, and a JSON summary.

Uses the SAME `calculate_breslow_depth_um` function as `evaluate.py` —
single source of truth for the Breslow algorithm.

Usage (from project root):

    # Bare minimum
    python breslow_depth_prediction/scripts/predict_image.py \
        --image path/to/your_image.png \
        --checkpoint checkpoints/best_model_v1.pth

    # V2 checkpoint with V2 config (different image_size)
    python breslow_depth_prediction/scripts/predict_image.py \
        --image path/to/your_image.png \
        --checkpoint checkpoints/best_model_v2.pth \
        --config breslow_depth_prediction/configs/config_v2.yaml

    # Save mask + overlay + JSON to disk
    python breslow_depth_prediction/scripts/predict_image.py \
        --image path/to/your_image.png \
        --checkpoint checkpoints/best_model_v1.pth \
        --output-dir results/single_predictions
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive (works on headless / SSH)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path so `breslow_depth_prediction.*` imports work.
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from breslow_depth_prediction.src.config import load_config
from breslow_depth_prediction.src.data.transforms import get_val_transforms, denormalize
from breslow_depth_prediction.src.models import get_model
from breslow_depth_prediction.src.models.unet import load_model_checkpoint
from breslow_depth_prediction.src.training.metrics import get_t_category

# Re-use the canonical Breslow calculator + TTA helper from evaluate.py
# — single source of truth, so the demo number matches Table I.
from breslow_depth_prediction.scripts.evaluate import (
    _BRESLOW_CALCULATORS,
    calculate_breslow_depth_um,
    tta_forward,
)


# === Class metadata ======================================================
# Matches the configs and src/data/dataset.py.
CLASS_NAMES = ["Background", "Tumour", "Epidermis", "Dermis"]
# RGB colours for the overlay PNG: black / blue / green / red.
CLASS_COLOURS = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
], dtype=np.uint8)


def predict_one_image(
    image_path: Path,
    config: dict,
    model: torch.nn.Module,
    device: str,
    breslow_method: str = "vertical",
    use_tta: bool = False,
) -> dict:
    """Run the full inference pipeline on a single image. Returns a result dict.

    Steps:
        1. Read PNG with PIL.
        2. Note original image size for the per-image scale factor.
        3. Apply val transforms (resize + ImageNet normalise + to tensor).
        4. Forward pass -> logits -> argmax -> predicted mask.
        5. Hand the mask to `calculate_breslow_depth_um` to get depth in µm.
        6. Map depth -> AJCC T-category.
    """
    # 1-2. Read image and remember original height for scale factor.
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    original_height, original_width = image_np.shape[:2]

    # 3. Preprocess via the validation transforms (no augmentation, just resize + normalise).
    image_size = tuple(config.get("data", {}).get("image_size", [512, 512]))
    transform = get_val_transforms(config, image_size=image_size)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    # 4. Forward pass + argmax. Optional 4-way TTA matches evaluate.py's
    # canonical reporting path (identity + h-flip + v-flip + 180-rot,
    # softmax-averaged).
    with torch.no_grad():
        if use_tta:
            probs = tta_forward(model, image_tensor)            # (1, 4, H, W) probabilities
            pred_mask = probs.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            out = model(image_tensor)
            logits = out["mask"] if isinstance(out, dict) else out
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # 5. Breslow depth via the canonical calculator.
    resolution = config.get("data", {}).get("resolution_um_per_pixel", 4.0)
    breslow_info = calculate_breslow_depth_um(
        predicted_mask=pred_mask,
        original_height=original_height,
        resized_height=image_size[0],
        resolution_um_per_pixel=resolution,
        method=breslow_method,
    )

    # 6. Map depth -> T-category.
    pred_t_category = get_t_category(breslow_info["depth_um"])

    # Class-pixel breakdown — useful for sanity-checking ("did the model see any epidermis?").
    class_pixel_counts = {
        CLASS_NAMES[c]: int((pred_mask == c).sum()) for c in range(4)
    }

    return {
        "image_path": str(image_path),
        "original_size": (original_height, original_width),
        "resized_size": image_size,
        "predicted_mask": pred_mask,                 # (H, W) numpy array
        "image_for_overlay": (
            denormalize(transformed["image"]).permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8),
        "depth_um": breslow_info["depth_um"],
        "depth_mm": breslow_info["depth_um"] / 1000.0,
        "depth_pixels": breslow_info["depth_pixels"],
        "scale_factor": breslow_info["scale_factor"],
        "um_per_pixel": breslow_info["um_per_pixel"],
        "t_category": pred_t_category,
        "confidence": breslow_info["confidence"],
        "breslow_method": breslow_info["method"],
        "class_pixel_counts": class_pixel_counts,
        "tumour_pixels": breslow_info["tumour_pixels"],
        "epidermis_pixels": breslow_info["epidermis_pixels"],
    }


def save_outputs(result: dict, output_dir: Path, image_stem: str) -> dict:
    """Save predicted mask PNG, overlay PNG, and JSON summary into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    # Predicted mask (re-coloured class-index PNG).
    mask_rgb = CLASS_COLOURS[result["predicted_mask"]]
    mask_path = output_dir / f"{image_stem}_mask.png"
    Image.fromarray(mask_rgb).save(mask_path)
    saved["mask_png"] = str(mask_path)

    # Overlay: image + transparent mask.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(result["image_for_overlay"])
    ax.imshow(mask_rgb, alpha=0.40)
    ax.set_title(
        f"{image_stem}  |  Predicted Breslow: {result['depth_um']:.0f} µm "
        f"({result['depth_mm']:.2f} mm)  |  T-category: {result['t_category']}",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    overlay_path = output_dir / f"{image_stem}_overlay.png"
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["overlay_png"] = str(overlay_path)

    # JSON summary (drop the numpy arrays — keep just the scalars).
    json_summary = {k: v for k, v in result.items()
                    if k not in ("predicted_mask", "image_for_overlay")}
    json_path = output_dir / f"{image_stem}_prediction.json"
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)
    saved["json"] = str(json_path)

    return saved


def main():
    """CLI entry point. Parses args, loads model, runs inference, prints results."""
    parser = argparse.ArgumentParser(
        description="Predict Breslow depth from a single H&E whole-slide image."
    )
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (PNG / JPG).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model .pth checkpoint.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (default: configs/config_v1.yaml).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="If given, save mask PNG + overlay PNG + JSON there.")
    parser.add_argument(
        "--breslow-method",
        type=str,
        default="vertical",
        choices=sorted(_BRESLOW_CALCULATORS),
        help="Breslow depth algorithm: vertical (V1/V2 baseline) or perpendicular (V3).",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 4-way test-time augmentation (identity + h-flip + v-flip + 180-rot, "
             "softmax-averaged). Matches evaluate.py's canonical reporting path "
             "and produces ~4x slower but smoother predictions.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    config_path = (
        Path(args.config) if args.config is not None
        else project_root / "breslow_depth_prediction" / "configs" / "config_v1.yaml"
    )
    config = load_config(str(config_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Image:       {image_path}")
    print(f"Config:      {config_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Device:      {device}")
    print(f"Breslow:     {args.breslow_method}")
    print(f"TTA:         {'ON (4-way)' if args.tta else 'off'}")

    # Build model + load weights.
    print("\nLoading model...")
    model = get_model(config)
    model = load_model_checkpoint(model, str(checkpoint_path))
    model.eval()

    # Inference.
    print("Running inference...")
    start = time.time()
    result = predict_one_image(
        image_path, config, model, device,
        breslow_method=args.breslow_method,
        use_tta=args.tta,
    )
    elapsed = time.time() - start

    # Pretty-print headline results.
    print("\n" + "=" * 60)
    print(" PREDICTION")
    print("=" * 60)
    print(f"Breslow depth:   {result['depth_um']:.1f} µm  ({result['depth_mm']:.3f} mm)")
    print(f"T-category:      {result['t_category']}")
    print(f"Confidence:      {result['confidence']:.2f}")
    print(f"Scale factor:    {result['scale_factor']:.3f}  ({result['um_per_pixel']:.2f} µm/resized px)")
    print(f"Original size:   {result['original_size'][0]} × {result['original_size'][1]} px")
    print(f"Resized size:    {result['resized_size'][0]} × {result['resized_size'][1]} px")
    print(f"Inference time:  {elapsed:.2f} s ({device})")
    print("\nClass pixel breakdown (resized mask):")
    total = sum(result["class_pixel_counts"].values())
    for name, count in result["class_pixel_counts"].items():
        pct = 100 * count / total if total else 0
        print(f"  {name:<12s}  {count:>10,d}  ({pct:5.1f}%)")
    print("=" * 60)

    # Optional: save outputs.
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        saved = save_outputs(result, output_dir, image_path.stem)
        print(f"\nSaved outputs:")
        for kind, path in saved.items():
            print(f"  {kind:<14s}  {path}")


if __name__ == "__main__":
    main()
