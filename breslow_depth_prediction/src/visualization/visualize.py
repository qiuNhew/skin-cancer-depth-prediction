"""Visualization utilities for Breslow depth prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# Class Colours and Names
# =============================================================================

CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (0, 0, 255),      # Tumour - Blue
    2: (0, 255, 0),      # Epidermis - Green
    3: (255, 0, 0),      # Dermis - Red
}

CLASS_NAMES = {
    0: "Background",
    1: "Tumour",
    2: "Epidermis",
    3: "Dermis",
}

# Normalized colours for matplotlib (0-1 range)
CLASS_COLORS_NORM = {
    k: tuple(c / 255.0 for c in v) for k, v in CLASS_COLORS.items()
}

# T-category thresholds in micrometres
T_THRESHOLDS = {
    "T1a": 800,
    "T1b": 1000,
    "T2": 2000,
    "T3": 4000,
}

T_CATEGORY_COLORS = {
    "T1a": "#2ecc71",  # Green
    "T1b": "#f1c40f",  # Yellow
    "T2": "#e67e22",   # Orange
    "T3": "#e74c3c",   # Red
    "T4": "#9b59b6",   # Purple
}


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_hwc(image: np.ndarray) -> np.ndarray:
    """Ensure image is in (H, W, C) format."""
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.transpose(1, 2, 0)
    return image


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 with values in [0, 255]."""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class index mask to RGB image."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx, color in CLASS_COLORS.items():
        rgb[mask == class_idx] = color

    return rgb


def _create_class_legend() -> List[mpatches.Patch]:
    """Create legend patches for class colours."""
    return [
        mpatches.Patch(color=CLASS_COLORS_NORM[i], label=CLASS_NAMES[i])
        for i in range(4)
    ]


def save_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure to save.
        save_path: Output path.
        dpi: Resolution for the saved image.
        close: Whether to close the figure after saving.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)


# =============================================================================
# Main Visualization Functions
# =============================================================================

def visualize_sample(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: Optional[np.ndarray] = None,
    breslow_info: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
    alpha: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Visualize a single sample with image, mask, and optional prediction.

    Args:
        image: Original image (H, W, 3) or (3, H, W).
        mask: Ground truth mask with class indices (H, W).
        prediction: Optional predicted mask with class indices (H, W).
        breslow_info: Optional dict with 'x1', 'y1', 'x2', 'y2' coordinates.
        title: Optional figure title.
        figsize: Figure size.
        alpha: Overlay transparency.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    # Prepare image
    image = _ensure_hwc(image)
    image = _ensure_uint8(image)

    # Determine number of subplots
    ncols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    # 1. Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Draw Breslow measurement line if provided
    if breslow_info is not None:
        for ax in axes:
            ax.plot(
                [breslow_info["x1"], breslow_info["x2"]],
                [breslow_info["y1"], breslow_info["y2"]],
                "c-", linewidth=2, label="Breslow line"
            )
            ax.plot(
                [breslow_info["x1"], breslow_info["x2"]],
                [breslow_info["y1"], breslow_info["y2"]],
                "co", markersize=5
            )

    # 2. Ground truth mask overlay
    mask_rgb = _mask_to_rgb(mask)
    axes[1].imshow(image)
    axes[1].imshow(mask_rgb, alpha=alpha)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # 3. Prediction overlay (if provided)
    if prediction is not None:
        pred_rgb = _mask_to_rgb(prediction)
        axes[2].imshow(image)
        axes[2].imshow(pred_rgb, alpha=alpha)
        axes[2].set_title("Prediction", fontsize=12, fontweight="bold")
        axes[2].axis("off")

    # Add colour legend
    legend_patches = _create_class_legend()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Add title
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def visualize_batch(
    images: Union[np.ndarray, List[np.ndarray]],
    masks: Union[np.ndarray, List[np.ndarray]],
    predictions: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    sample_ids: Optional[List[str]] = None,
    num_show: int = 4,
    figsize_per_sample: Tuple[float, float] = (4, 4),
    alpha: float = 0.5,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Visualize a batch of samples in a grid layout.

    Args:
        images: Batch of images (B, C, H, W) or (B, H, W, C) or list.
        masks: Batch of masks (B, H, W) or list.
        predictions: Optional batch of predicted masks.
        sample_ids: Optional list of sample IDs for titles.
        num_show: Maximum number of samples to display.
        figsize_per_sample: Size per sample in the grid.
        alpha: Overlay transparency.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    # Convert to lists if needed
    if isinstance(images, np.ndarray) and images.ndim == 4:
        images = [images[i] for i in range(images.shape[0])]
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        masks = [masks[i] for i in range(masks.shape[0])]
    if predictions is not None:
        if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
            predictions = [predictions[i] for i in range(predictions.shape[0])]

    # Limit number of samples
    num_samples = min(len(images), num_show)
    images = images[:num_samples]
    masks = masks[:num_samples]
    if predictions is not None:
        predictions = predictions[:num_samples]

    # Determine grid layout
    nrows = 2 if predictions is None else 3
    ncols = num_samples

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_sample[0] * ncols, figsize_per_sample[1] * nrows)
    )

    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        # Prepare image
        img = _ensure_hwc(images[i])
        img = _ensure_uint8(img)
        mask = masks[i]
        mask_rgb = _mask_to_rgb(mask)

        # Row 0: Original image
        axes[0, i].imshow(img)
        title = f"Sample {i+1}" if sample_ids is None else sample_ids[i]
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis("off")

        # Row 1: Ground truth overlay
        axes[1, i].imshow(img)
        axes[1, i].imshow(mask_rgb, alpha=alpha)
        axes[1, i].set_title("Ground Truth", fontsize=10)
        axes[1, i].axis("off")

        # Row 2: Prediction overlay (if provided)
        if predictions is not None:
            pred = predictions[i]
            pred_rgb = _mask_to_rgb(pred)
            axes[2, i].imshow(img)
            axes[2, i].imshow(pred_rgb, alpha=alpha)
            axes[2, i].set_title("Prediction", fontsize=10)
            axes[2, i].axis("off")

    # Add legend
    legend_patches = _create_class_legend()
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_class_distribution(
    dataset,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot T-category and class pixel distribution.

    Args:
        dataset: BreslowDataset instance with samples attribute.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. T-category distribution (bar chart)
    t_cats = [s["t_category"] for s in dataset.samples]
    t_cat_counts = {}
    for cat in ["T1a", "T1b", "T2", "T3", "T4"]:
        t_cat_counts[cat] = t_cats.count(cat)

    categories = list(t_cat_counts.keys())
    counts = list(t_cat_counts.values())
    colors = [T_CATEGORY_COLORS[cat] for cat in categories]

    bars = axes[0].bar(categories, counts, color=colors, edgecolor="black")
    axes[0].set_xlabel("T-Category", fontsize=12)
    axes[0].set_ylabel("Number of Samples", fontsize=12)
    axes[0].set_title("T-Category Distribution", fontsize=14, fontweight="bold")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 2. Class pixel distribution (pie chart)
    # Sample a subset of masks for speed
    from PIL import Image
    from ..data.dataset import rgb_mask_to_class_indices

    class_counts = np.zeros(4)
    num_to_sample = min(len(dataset.samples), 30)

    for sample in dataset.samples[:num_to_sample]:
        mask_rgb = np.array(Image.open(sample["mask_path"]).convert("RGB"))
        mask = rgb_mask_to_class_indices(mask_rgb)
        for c in range(4):
            class_counts[c] += np.sum(mask == c)

    # Create pie chart
    class_labels = [CLASS_NAMES[i] for i in range(4)]
    class_colors_list = [CLASS_COLORS_NORM[i] for i in range(4)]

    # Filter out zero counts
    nonzero_mask = class_counts > 0
    pie_counts = class_counts[nonzero_mask]
    pie_labels = [class_labels[i] for i in range(4) if nonzero_mask[i]]
    pie_colors = [class_colors_list[i] for i in range(4) if nonzero_mask[i]]

    wedges, texts, autotexts = axes[1].pie(
        pie_counts,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=[0.02] * len(pie_counts),
    )
    axes[1].set_title(
        f"Class Pixel Distribution\n(sampled from {num_to_sample} masks)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_breslow_histogram(
    depths: Union[List[float], np.ndarray],
    title: str = "Breslow Depth Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot histogram of Breslow depths with T-category thresholds.

    Args:
        depths: List or array of Breslow depths in micrometres.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    depths = np.array(depths)
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    max_depth = max(depths.max(), 5000)
    bins = np.linspace(0, max_depth, 50)

    n, bins_out, patches = ax.hist(
        depths, bins=bins, edgecolor="black", alpha=0.7, color="#3498db"
    )

    # Add T-category threshold lines
    threshold_colors = {
        800: ("#2ecc71", "T1a/T1b (800 µm)"),
        1000: ("#f1c40f", "T1b/T2 (1000 µm)"),
        2000: ("#e67e22", "T2/T3 (2000 µm)"),
        4000: ("#e74c3c", "T3/T4 (4000 µm)"),
    }

    for threshold, (color, label) in threshold_colors.items():
        ax.axvline(
            x=threshold,
            color=color,
            linestyle="--",
            linewidth=2,
            label=label,
        )

    # Add shaded regions for T-categories
    ylim = ax.get_ylim()
    regions = [
        (0, 800, "#2ecc71", "T1a", 0.1),
        (800, 1000, "#f1c40f", "T1b", 0.1),
        (1000, 2000, "#e67e22", "T2", 0.1),
        (2000, 4000, "#e74c3c", "T3", 0.1),
        (4000, max_depth, "#9b59b6", "T4", 0.1),
    ]

    for start, end, color, label, alpha_val in regions:
        ax.axvspan(start, end, alpha=alpha_val, color=color)
        # Add label in region
        mid = (start + end) / 2
        ax.text(
            mid, ylim[1] * 0.95, label,
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Breslow Depth (µm)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, max_depth)

    # Add statistics text
    stats_text = (
        f"n = {len(depths)}\n"
        f"Mean: {depths.mean():.1f} µm\n"
        f"Std: {depths.std():.1f} µm\n"
        f"Median: {np.median(depths):.1f} µm"
    )
    ax.text(
        0.98, 0.65, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_training_history(
    history_dict: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training history with loss, dice score, and learning rate.

    Args:
        history_dict: Dictionary with keys like 'train_loss', 'val_loss',
                     'train_dice', 'val_dice', 'learning_rate'.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Determine epochs
    num_epochs = 0
    for key in ["train_loss", "val_loss", "train_dice", "val_dice"]:
        if key in history_dict:
            num_epochs = max(num_epochs, len(history_dict[key]))

    epochs = range(1, num_epochs + 1)

    # 1. Loss plot
    if "train_loss" in history_dict:
        axes[0].plot(epochs, history_dict["train_loss"], "b-", linewidth=2, label="Train")
    if "val_loss" in history_dict:
        axes[0].plot(epochs, history_dict["val_loss"], "r-", linewidth=2, label="Validation")

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Mark best validation loss
    if "val_loss" in history_dict:
        best_epoch = np.argmin(history_dict["val_loss"]) + 1
        best_loss = min(history_dict["val_loss"])
        axes[0].axvline(x=best_epoch, color="green", linestyle=":", alpha=0.7)
        axes[0].scatter([best_epoch], [best_loss], color="green", s=100, zorder=5)
        axes[0].annotate(
            f"Best: {best_loss:.4f}",
            (best_epoch, best_loss),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    # 2. Dice score plot
    if "train_dice" in history_dict:
        axes[1].plot(epochs, history_dict["train_dice"], "b-", linewidth=2, label="Train")
    if "val_dice" in history_dict:
        axes[1].plot(epochs, history_dict["val_dice"], "r-", linewidth=2, label="Validation")

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Dice Score", fontsize=12)
    axes[1].set_title("Dice Coefficient", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Mark best validation dice
    if "val_dice" in history_dict:
        best_epoch = np.argmax(history_dict["val_dice"]) + 1
        best_dice = max(history_dict["val_dice"])
        axes[1].axvline(x=best_epoch, color="green", linestyle=":", alpha=0.7)
        axes[1].scatter([best_epoch], [best_dice], color="green", s=100, zorder=5)
        axes[1].annotate(
            f"Best: {best_dice:.4f}",
            (best_epoch, best_dice),
            textcoords="offset points",
            xytext=(10, -15),
            fontsize=9,
        )

    # 3. Learning rate plot
    if "learning_rate" in history_dict:
        lr_epochs = range(1, len(history_dict["learning_rate"]) + 1)
        axes[2].plot(lr_epochs, history_dict["learning_rate"], "g-", linewidth=2)
        axes[2].set_xlabel("Epoch", fontsize=12)
        axes[2].set_ylabel("Learning Rate", fontsize=12)
        axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5, 0.5, "No learning rate data",
            ha="center", va="center", fontsize=12,
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    classes: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot confusion matrix for T-category classification.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Class names. Defaults to T-categories.
        normalize: Whether to normalize the matrix.
        figsize: Figure size.
        cmap: Colormap name.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    if classes is None:
        classes = ["T1a", "T1b", "T2", "T3", "T4"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, label="Proportion" if normalize else "Count")

    # Set ticks and labels
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True T-Category",
        xlabel="Predicted T-Category",
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i in range(len(classes)):
        for j in range(len(classes)):
            value = cm[i, j]
            ax.text(
                j, i, format(value, fmt),
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=11,
            )

    ax.set_title("T-Category Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_breslow_scatter(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot scatter plot of predicted vs actual Breslow depth.

    Args:
        y_true: True Breslow depths in micrometres.
        y_pred: Predicted Breslow depths in micrometres.
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Perfect prediction line (y=x)
    max_val = max(y_true.max(), y_pred.max()) * 1.1
    min_val = min(y_true.min(), y_pred.min()) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect prediction")

    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # Pearson correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(x_line, p(x_line), "b-", linewidth=1.5, alpha=0.7, label=f"Regression (slope={z[0]:.2f})")

    ax.set_xlabel("Actual Breslow Depth (µm)", fontsize=12)
    ax.set_ylabel("Predicted Breslow Depth (µm)", fontsize=12)
    ax.set_title("Predicted vs Actual Breslow Depth", fontsize=14, fontweight="bold")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f"R² = {r2:.4f}\n"
        f"r = {correlation:.4f}\n"
        f"MAE = {mae:.1f} µm\n"
        f"RMSE = {rmse:.1f} µm\n"
        f"n = {len(y_true)}"
    )
    ax.text(
        0.98, 0.02, metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


def plot_bland_altman(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot Bland-Altman plot for method comparison.

    Args:
        y_true: True values (reference method).
        y_pred: Predicted values (test method).
        figsize: Figure size.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate mean and difference
    mean_values = (y_true + y_pred) / 2
    diff_values = y_pred - y_true

    # Calculate statistics
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(mean_values, diff_values, alpha=0.6, edgecolors="black", linewidth=0.5)

    # Mean difference line
    ax.axhline(mean_diff, color="blue", linestyle="-", linewidth=2, label=f"Mean: {mean_diff:.1f} µm")

    # Limits of agreement
    ax.axhline(upper_loa, color="red", linestyle="--", linewidth=1.5, label=f"+1.96 SD: {upper_loa:.1f} µm")
    ax.axhline(lower_loa, color="red", linestyle="--", linewidth=1.5, label=f"-1.96 SD: {lower_loa:.1f} µm")

    # Zero line
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    # Add shaded region for limits of agreement
    ax.fill_between(
        ax.get_xlim(), lower_loa, upper_loa,
        alpha=0.1, color="red", label="95% Limits of Agreement"
    )

    ax.set_xlabel("Mean of Actual and Predicted (µm)", fontsize=12)
    ax.set_ylabel("Difference (Predicted - Actual) (µm)", fontsize=12)
    ax.set_title("Bland-Altman Plot", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"Mean Difference: {mean_diff:.1f} µm\n"
        f"SD of Differences: {std_diff:.1f} µm\n"
        f"95% LoA: [{lower_loa:.1f}, {upper_loa:.1f}] µm\n"
        f"n = {len(y_true)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, close=False)

    if show:
        plt.show()

    return fig


# =============================================================================
# Legacy/Utility Functions
# =============================================================================

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a class mask on an image with colours.

    Args:
        image: Original image (H, W, 3).
        mask: Mask with class indices (H, W).
        alpha: Transparency of the overlay.

    Returns:
        Image with overlay (H, W, 3) as uint8.
    """
    image = _ensure_hwc(image)
    image = _ensure_uint8(image)

    mask_rgb = _mask_to_rgb(mask)

    # Blend
    overlay = (alpha * mask_rgb + (1 - alpha) * image).astype(np.uint8)

    # Keep original where mask is background
    overlay[mask == 0] = image[mask == 0]

    return overlay


# Alias for backward compatibility
save_visualization = save_figure


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Visualization Module Examples")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)

    # Sample image and masks
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sample_mask = np.random.randint(0, 4, (256, 256), dtype=np.int64)
    sample_pred = np.random.randint(0, 4, (256, 256), dtype=np.int64)

    # Sample Breslow depths
    sample_depths = np.concatenate([
        np.random.normal(500, 200, 20),   # T1a
        np.random.normal(900, 50, 10),    # T1b
        np.random.normal(1500, 300, 25),  # T2
        np.random.normal(3000, 500, 15),  # T3
        np.random.normal(5000, 800, 5),   # T4
    ])
    sample_depths = np.clip(sample_depths, 100, 8000)

    # Sample training history
    epochs = 50
    sample_history = {
        "train_loss": [1.0 * np.exp(-0.05 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(epochs)],
        "val_loss": [1.0 * np.exp(-0.04 * i) + 0.15 + np.random.normal(0, 0.03) for i in range(epochs)],
        "train_dice": [1 - 0.9 * np.exp(-0.05 * i) + np.random.normal(0, 0.02) for i in range(epochs)],
        "val_dice": [1 - 0.95 * np.exp(-0.04 * i) + np.random.normal(0, 0.03) for i in range(epochs)],
        "learning_rate": [0.001 * (0.95 ** i) for i in range(epochs)],
    }

    print("\n1. Visualize single sample:")
    fig = visualize_sample(
        sample_image,
        sample_mask,
        prediction=sample_pred,
        breslow_info={"x1": 50, "y1": 100, "x2": 200, "y2": 150},
        title="Sample Visualization",
        show=False,
    )
    plt.close(fig)
    print("   Done!")

    print("\n2. Plot Breslow histogram:")
    fig = plot_breslow_histogram(sample_depths, show=False)
    plt.close(fig)
    print("   Done!")

    print("\n3. Plot training history:")
    fig = plot_training_history(sample_history, show=False)
    plt.close(fig)
    print("   Done!")

    print("\n4. Plot confusion matrix:")
    y_true = np.random.choice(["T1a", "T1b", "T2", "T3", "T4"], 100)
    y_pred = np.random.choice(["T1a", "T1b", "T2", "T3", "T4"], 100)
    fig = plot_confusion_matrix(y_true, y_pred, show=False)
    plt.close(fig)
    print("   Done!")

    print("\n5. Plot Breslow scatter:")
    y_true_depth = sample_depths
    y_pred_depth = y_true_depth + np.random.normal(0, 300, len(y_true_depth))
    fig = plot_breslow_scatter(y_true_depth, y_pred_depth, show=False)
    plt.close(fig)
    print("   Done!")

    print("\n6. Plot Bland-Altman:")
    fig = plot_bland_altman(y_true_depth, y_pred_depth, show=False)
    plt.close(fig)
    print("   Done!")

    print("\n" + "=" * 50)
    print("All examples completed!")
