#!/usr/bin/env python3
"""
Data Exploration Script for Breslow Depth Prediction

This script performs comprehensive data exploration including:
1. Dataset integrity verification
2. Dataset statistics calculation
3. Sample visualization with Breslow measurement lines
4. Class distribution analysis
5. Breslow depth analysis with histograms and box plots
6. Stratified train/val/test split creation
7. Summary report generation

Usage:
    python scripts/explore_data.py
    python scripts/explore_data.py --config configs/config_v1.yaml

Output:
    - Figures saved to results/figures/exploration/
    - Report saved to results/data_exploration_report.txt
    - Split info saved to results/data_split.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import (
    RGB_TO_CLASS,
    CLASS_NAMES,
    T_CATEGORY_THRESHOLDS,
    get_t_category,
    rgb_mask_to_class_indices,
    find_matching_files,
    extract_sample_id_from_svs,
)
from src.visualization import (
    CLASS_COLORS,
    CLASS_COLORS_NORM,
    T_CATEGORY_COLORS,
    visualize_sample,
    plot_breslow_histogram,
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DataExplorer:
    """Class to perform comprehensive data exploration."""

    def __init__(self, config_path: str = "configs/config_v1.yaml"):
        """Initialize the data explorer.

        Args:
            config_path: Path to the configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        # Extract paths from config
        data_config = self.config.get("data", {})
        self.data_dir = Path(data_config.get("data_dir", "./data"))
        self.images_dir = self.data_dir / data_config.get("images_dir", "ds_WSIs")
        self.csv_path = self.data_dir / data_config.get("coords_file", "measurements/breslow_depth_coords.csv")
        self.exclude_file = self.data_dir / data_config.get("exclude_file", "measurements/wrong_dir_coords.txt")
        self.resolution = data_config.get("resolution_um_per_pixel", 4.0)

        # Extract split config
        split_config = self.config.get("split", {})
        self.train_ratio = split_config.get("train_ratio", 0.7)
        self.val_ratio = split_config.get("val_ratio", 0.15)
        self.test_ratio = split_config.get("test_ratio", 0.15)
        self.random_seed = split_config.get("random_seed", 42)

        # Output directories
        paths_config = self.config.get("paths", {})
        self.results_dir = PROJECT_ROOT / paths_config.get("results_dir", "results")
        self.figures_dir = self.results_dir / "figures" / "exploration"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Storage for results
        self.report_lines: List[str] = []
        self.samples: List[Dict[str, Any]] = []
        self.excluded_samples: List[str] = []

    def log(self, message: str, print_msg: bool = True) -> None:
        """Log a message to both console and report."""
        if print_msg:
            print(message)
        self.report_lines.append(message)

    def run(self) -> None:
        """Run the complete data exploration pipeline."""
        self.log("=" * 70)
        self.log("BRESLOW DEPTH PREDICTION - DATA EXPLORATION REPORT")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

        # 1. Verify dataset integrity
        self.verify_dataset_integrity()

        # 2. Calculate dataset statistics
        self.calculate_dataset_statistics()

        # 3. Visualize samples
        self.visualize_samples()

        # 4. Class distribution analysis
        self.analyze_class_distribution()

        # 5. Breslow depth analysis
        self.analyze_breslow_depths()

        # 6. Create train/val/test split
        self.create_splits()

        # 7. Save summary report
        self.save_report()

        self.log("\n" + "=" * 70)
        self.log("DATA EXPLORATION COMPLETE")
        self.log("=" * 70)
        self.log(f"\nFigures saved to: {self.figures_dir}")
        self.log(f"Report saved to: {self.results_dir / 'data_exploration_report.txt'}")

    def verify_dataset_integrity(self) -> None:
        """Verify dataset integrity and list any issues."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 1: DATASET INTEGRITY VERIFICATION")
        self.log("=" * 70)

        # Check paths exist
        self.log(f"\nData directory: {self.data_dir}")
        self.log(f"  Exists: {self.data_dir.exists()}")

        self.log(f"\nImages directory: {self.images_dir}")
        self.log(f"  Exists: {self.images_dir.exists()}")

        self.log(f"\nCSV file: {self.csv_path}")
        self.log(f"  Exists: {self.csv_path.exists()}")

        self.log(f"\nExclude file: {self.exclude_file}")
        self.log(f"  Exists: {self.exclude_file.exists()}")

        if not self.data_dir.exists() or not self.images_dir.exists():
            self.log("\nERROR: Required directories not found!")
            return

        # Count files
        image_files = list(self.images_dir.glob("*_image.png"))
        mask_files = list(self.images_dir.glob("*_labels.png"))

        self.log(f"\nTotal image files found: {len(image_files)}")
        self.log(f"Total mask files found: {len(mask_files)}")

        # Load CSV
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(self.csv_path, encoding="latin-1")
            self.log(f"CSV entries count: {len(df)}")
            self.log(f"CSV columns: {list(df.columns)}")
        else:
            self.log("ERROR: CSV file not found!")
            return

        # Load exclusion list
        self.excluded_samples = []
        if self.exclude_file.exists():
            with open(self.exclude_file, "r") as f:
                for line in f:
                    line = line.strip().rstrip(",")
                    if line:
                        self.excluded_samples.append(line)
            self.log(f"\nExcluded samples (from wrong_dir_coords.txt): {len(self.excluded_samples)}")
            for sample in self.excluded_samples[:5]:
                self.log(f"  - {sample}")
            if len(self.excluded_samples) > 5:
                self.log(f"  ... and {len(self.excluded_samples) - 5} more")

        # Extract excluded sample IDs
        excluded_ids = set()
        for line in self.excluded_samples:
            match = re.match(r"^([^_\[]+(?:[-_][^_\[]+)*_\d+)", line)
            if match:
                excluded_ids.add(match.group(1))

        # Find Breslow depth column
        depth_col = None
        for col in df.columns:
            if "length" in col.lower() and ("m" in col.lower() or "µ" in col):
                if col != "length_ds":
                    depth_col = col
                    break
        if depth_col is None:
            depth_col = df.columns[-1]
        self.log(f"\nUsing Breslow depth column: '{depth_col}'")

        # Match CSV entries with files
        valid_samples = []
        missing_images = []
        missing_masks = []

        for _, row in df.iterrows():
            svs_filename = row["image"]
            sample_id = extract_sample_id_from_svs(svs_filename)

            # Skip excluded samples
            if sample_id in excluded_ids:
                continue

            # Find matching files
            image_path, mask_path = find_matching_files(sample_id, self.images_dir)

            if image_path is None:
                missing_images.append(sample_id)
            elif mask_path is None:
                missing_masks.append(sample_id)
            else:
                # Extract Breslow depth
                breslow_depth = float(row[depth_col])
                t_category = get_t_category(breslow_depth)

                valid_samples.append({
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "breslow_depth_um": breslow_depth,
                    "breslow_coords": {
                        "x1": float(row["x1_ds"]),
                        "y1": float(row["y1_ds"]),
                        "x2": float(row["x2_ds"]),
                        "y2": float(row["y2_ds"]),
                    },
                    "t_category": t_category,
                })

        self.samples = valid_samples

        self.log(f"\n{'='*50}")
        self.log("INTEGRITY SUMMARY")
        self.log(f"{'='*50}")
        self.log(f"Total valid samples: {len(valid_samples)}")
        self.log(f"Samples with missing images: {len(missing_images)}")
        self.log(f"Samples with missing masks: {len(missing_masks)}")
        self.log(f"Excluded samples: {len(excluded_ids)}")

        if missing_images:
            self.log(f"\nMissing images (first 5):")
            for sid in missing_images[:5]:
                self.log(f"  - {sid}")

        if missing_masks:
            self.log(f"\nMissing masks (first 5):")
            for sid in missing_masks[:5]:
                self.log(f"  - {sid}")

    def calculate_dataset_statistics(self) -> None:
        """Calculate and display dataset statistics."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 2: DATASET STATISTICS")
        self.log("=" * 70)

        if not self.samples:
            self.log("\nNo valid samples found!")
            return

        self.log(f"\nTotal valid samples: {len(self.samples)}")

        # Breslow depth statistics
        depths = np.array([s["breslow_depth_um"] for s in self.samples])
        self.log(f"\nBreslow Depth Statistics (in micrometres):")
        self.log(f"  Minimum: {depths.min():.2f} um")
        self.log(f"  Maximum: {depths.max():.2f} um")
        self.log(f"  Mean: {depths.mean():.2f} um")
        self.log(f"  Std: {depths.std():.2f} um")
        self.log(f"  Median: {np.median(depths):.2f} um")

        # T-category distribution
        t_cats = [s["t_category"] for s in self.samples]
        t_cat_counts = Counter(t_cats)

        self.log(f"\nT-Category Distribution:")
        for cat in ["T1a", "T1b", "T2", "T3", "T4"]:
            count = t_cat_counts.get(cat, 0)
            pct = 100 * count / len(t_cats) if len(t_cats) > 0 else 0
            self.log(f"  {cat}: {count:3d} ({pct:5.1f}%)")

        # Image dimensions - verify all same size
        self.log(f"\nImage Dimensions Check:")
        first_img = np.array(Image.open(self.samples[0]["image_path"]))
        first_shape = first_img.shape
        self.log(f"  First image shape: {first_shape}")

        # Check a random subset
        np.random.seed(self.random_seed)
        sample_indices = np.random.choice(len(self.samples), min(10, len(self.samples)), replace=False)
        all_same = True
        for idx in sample_indices:
            img = np.array(Image.open(self.samples[idx]["image_path"]))
            if img.shape != first_shape:
                self.log(f"  WARNING: {self.samples[idx]['sample_id']} has shape {img.shape}")
                all_same = False

        if all_same:
            self.log(f"  All sampled images have consistent dimensions: {first_shape}")

    def visualize_samples(self) -> None:
        """Visualize random samples with masks overlaid."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 3: SAMPLE VISUALIZATION")
        self.log("=" * 70)

        if not self.samples:
            self.log("\nNo valid samples to visualize!")
            return

        # Select 6 random samples
        np.random.seed(self.random_seed)
        num_samples = min(6, len(self.samples))
        sample_indices = np.random.choice(len(self.samples), num_samples, replace=False)

        self.log(f"\nVisualizing {num_samples} random samples...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, idx in enumerate(sample_indices):
            sample = self.samples[idx]

            # Load image and mask
            image = np.array(Image.open(sample["image_path"]).convert("RGB"))
            mask_rgb = np.array(Image.open(sample["mask_path"]).convert("RGB"))
            mask = rgb_mask_to_class_indices(mask_rgb)

            # Create overlay
            overlay = image.copy().astype(np.float32)
            for class_idx in range(1, 4):  # Skip background
                color = np.array(CLASS_COLORS[class_idx])
                mask_region = mask == class_idx
                overlay[mask_region] = 0.5 * overlay[mask_region] + 0.5 * color

            axes[i].imshow(overlay.astype(np.uint8))

            # Draw Breslow measurement line
            coords = sample["breslow_coords"]
            axes[i].plot(
                [coords["x1"], coords["x2"]],
                [coords["y1"], coords["y2"]],
                "c-", linewidth=2, label="Breslow line"
            )
            axes[i].plot(
                [coords["x1"], coords["x2"]],
                [coords["y1"], coords["y2"]],
                "co", markersize=5
            )

            # Title with sample info
            axes[i].set_title(
                f"{sample['sample_id']}\n"
                f"Breslow: {sample['breslow_depth_um']:.0f} um ({sample['t_category']})",
                fontsize=10
            )
            axes[i].axis("off")

        # Add legend
        import matplotlib.patches as mpatches
        legend_patches = [
            mpatches.Patch(color=np.array(CLASS_COLORS[i])/255, label=CLASS_NAMES[i])
            for i in range(4)
        ]
        legend_patches.append(mpatches.Patch(color="cyan", label="Breslow line"))
        fig.legend(handles=legend_patches, loc="lower center", ncol=5, fontsize=10)

        plt.suptitle("Random Sample Visualization with Breslow Measurement Lines", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

        # Save figure
        save_path = self.figures_dir / "sample_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        self.log(f"Saved sample visualization to: {save_path}")

    def analyze_class_distribution(self) -> None:
        """Analyze class pixel distribution across all masks."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 4: CLASS DISTRIBUTION ANALYSIS")
        self.log("=" * 70)

        if not self.samples:
            self.log("\nNo valid samples for class analysis!")
            return

        self.log(f"\nAnalyzing class distribution across all {len(self.samples)} masks...")

        # Calculate class pixel counts
        class_counts = np.zeros(4, dtype=np.int64)

        for i, sample in enumerate(self.samples):
            mask_rgb = np.array(Image.open(sample["mask_path"]).convert("RGB"))
            mask = rgb_mask_to_class_indices(mask_rgb)

            for class_idx in range(4):
                class_counts[class_idx] += np.sum(mask == class_idx)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(self.samples)} masks...")

        total_pixels = class_counts.sum()

        self.log(f"\nClass Pixel Distribution:")
        for class_idx in range(4):
            count = class_counts[class_idx]
            pct = 100 * count / total_pixels if total_pixels > 0 else 0
            self.log(f"  {CLASS_NAMES[class_idx]}: {count:,} pixels ({pct:.2f}%)")

        # Calculate class weights (inverse frequency)
        class_weights = total_pixels / (4 * class_counts + 1e-8)
        class_weights = class_weights / class_weights.sum() * 4  # Normalize

        self.log(f"\nInverse Frequency Class Weights:")
        for class_idx in range(4):
            self.log(f"  {CLASS_NAMES[class_idx]}: {class_weights[class_idx]:.4f}")

        # Create bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Pixel count bar chart
        class_labels = [CLASS_NAMES[i] for i in range(4)]
        class_percentages = [100 * class_counts[i] / total_pixels for i in range(4)]
        colors = [np.array(CLASS_COLORS[i])/255 for i in range(4)]

        bars = axes[0].bar(class_labels, class_percentages, color=colors, edgecolor="black")
        axes[0].set_xlabel("Class", fontsize=12)
        axes[0].set_ylabel("Percentage of Pixels", fontsize=12)
        axes[0].set_title("Class Pixel Distribution", fontsize=14, fontweight="bold")

        # Add percentage labels on bars
        for bar, pct in zip(bars, class_percentages):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        # Class weights bar chart
        bars2 = axes[1].bar(class_labels, class_weights, color=colors, edgecolor="black")
        axes[1].set_xlabel("Class", fontsize=12)
        axes[1].set_ylabel("Weight", fontsize=12)
        axes[1].set_title("Inverse Frequency Class Weights", fontsize=14, fontweight="bold")

        for bar, weight in zip(bars2, class_weights):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{weight:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        plt.tight_layout()

        # Save figure
        save_path = self.figures_dir / "class_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        self.log(f"\nSaved class distribution figure to: {save_path}")

    def analyze_breslow_depths(self) -> None:
        """Analyze Breslow depth distribution with histogram and box plot."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 5: BRESLOW DEPTH ANALYSIS")
        self.log("=" * 70)

        if not self.samples:
            self.log("\nNo valid samples for Breslow analysis!")
            return

        depths = np.array([s["breslow_depth_um"] for s in self.samples])
        t_cats = [s["t_category"] for s in self.samples]

        # Create figure with histogram and box plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Histogram with T-category regions
        ax1 = axes[0]
        max_depth = max(depths.max(), 5000)
        bins = np.linspace(0, max_depth, 50)

        n, bins_out, patches = ax1.hist(depths, bins=bins, edgecolor="black", alpha=0.7, color="#3498db")

        # Add T-category threshold lines
        threshold_info = [
            (800, "#2ecc71", "T1a/T1b (800 um)"),
            (1000, "#f1c40f", "T1b/T2 (1000 um)"),
            (2000, "#e67e22", "T2/T3 (2000 um)"),
            (4000, "#e74c3c", "T3/T4 (4000 um)"),
        ]

        for threshold, color, label in threshold_info:
            ax1.axvline(x=threshold, color=color, linestyle="--", linewidth=2, label=label)

        # Add shaded regions
        ylim = ax1.get_ylim()
        regions = [
            (0, 800, "#2ecc71", "T1a"),
            (800, 1000, "#f1c40f", "T1b"),
            (1000, 2000, "#e67e22", "T2"),
            (2000, 4000, "#e74c3c", "T3"),
            (4000, max_depth, "#9b59b6", "T4"),
        ]

        for start, end, color, label in regions:
            ax1.axvspan(start, end, alpha=0.1, color=color)
            mid = (start + end) / 2
            ax1.text(mid, ylim[1] * 0.95, label, ha="center", va="top",
                     fontsize=11, fontweight="bold", color=color)

        ax1.set_xlabel("Breslow Depth (um)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Breslow Depth Distribution with T-Category Thresholds", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_xlim(0, max_depth)

        # Add statistics text
        stats_text = (
            f"n = {len(depths)}\n"
            f"Mean: {depths.mean():.1f} um\n"
            f"Std: {depths.std():.1f} um\n"
            f"Median: {np.median(depths):.1f} um\n"
            f"Range: [{depths.min():.0f}, {depths.max():.0f}]"
        )
        ax1.text(0.98, 0.65, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # 2. Box plot by T-category
        ax2 = axes[1]

        # Prepare data for box plot
        t_cat_order = ["T1a", "T1b", "T2", "T3", "T4"]
        box_data = []
        box_labels = []
        box_colors = []

        for cat in t_cat_order:
            cat_depths = [s["breslow_depth_um"] for s in self.samples if s["t_category"] == cat]
            if cat_depths:
                box_data.append(cat_depths)
                box_labels.append(f"{cat}\n(n={len(cat_depths)})")
                box_colors.append(T_CATEGORY_COLORS[cat])

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)

            # Color the boxes
            for patch, color in zip(bp["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax2.set_xlabel("T-Category", fontsize=12)
            ax2.set_ylabel("Breslow Depth (um)", fontsize=12)
            ax2.set_title("Breslow Depth by T-Category", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)

            # Add threshold lines
            for threshold, _, label in threshold_info:
                ax2.axhline(y=threshold, color="gray", linestyle=":", alpha=0.5)

        plt.tight_layout()

        # Save figure
        save_path = self.figures_dir / "breslow_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        self.log(f"\nSaved Breslow depth analysis to: {save_path}")

        # Also save T-category distribution bar chart
        self._plot_t_category_distribution()

    def _plot_t_category_distribution(self) -> None:
        """Create T-category distribution bar chart."""
        t_cats = [s["t_category"] for s in self.samples]
        t_cat_counts = Counter(t_cats)

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ["T1a", "T1b", "T2", "T3", "T4"]
        counts = [t_cat_counts.get(cat, 0) for cat in categories]
        colors = [T_CATEGORY_COLORS[cat] for cat in categories]

        bars = ax.bar(categories, counts, color=colors, edgecolor="black")

        ax.set_xlabel("T-Category", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title("T-Category Distribution", fontsize=14, fontweight="bold")

        # Add count and percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            pct = 100 * count / total if total > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        plt.tight_layout()

        save_path = self.figures_dir / "t_category_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        self.log(f"Saved T-category distribution to: {save_path}")

    def create_splits(self) -> None:
        """Create stratified train/val/test splits."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 6: TRAIN/VAL/TEST SPLIT CREATION")
        self.log("=" * 70)

        if not self.samples:
            self.log("\nNo valid samples for splitting!")
            return

        self.log(f"\nSplit ratios:")
        self.log(f"  Train: {self.train_ratio:.0%}")
        self.log(f"  Val: {self.val_ratio:.0%}")
        self.log(f"  Test: {self.test_ratio:.0%}")
        self.log(f"  Random seed: {self.random_seed}")

        # Get T-categories for stratification
        t_categories = [s["t_category"] for s in self.samples]
        indices = list(range(len(self.samples)))

        # Check if stratification is possible
        category_counts = Counter(t_categories)
        min_count = min(category_counts.values())

        stratify = t_categories if min_count >= 2 else None
        if stratify is None:
            self.log("\nWARNING: Not enough samples per category for stratification, using random split")

        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            stratify=[t_categories[i] for i in indices] if stratify else None,
        )

        # Second split: val vs test
        relative_val_ratio = self.val_ratio / (self.val_ratio + self.test_ratio)
        temp_categories = [t_categories[i] for i in temp_indices]
        temp_counts = Counter(temp_categories)
        temp_stratify = temp_categories if min(temp_counts.values()) >= 2 else None

        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=relative_val_ratio,
            random_state=self.random_seed,
            stratify=temp_stratify,
        )

        # Get sample IDs
        train_ids = [self.samples[i]["sample_id"] for i in train_indices]
        val_ids = [self.samples[i]["sample_id"] for i in val_indices]
        test_ids = [self.samples[i]["sample_id"] for i in test_indices]

        # Print split distribution
        self.log(f"\nSplit Distribution:")
        self.log(f"{'Split':<10} {'Count':>8} {'Percentage':>12}")
        self.log("-" * 32)
        self.log(f"{'Train':<10} {len(train_ids):>8} {100*len(train_ids)/len(self.samples):>11.1f}%")
        self.log(f"{'Val':<10} {len(val_ids):>8} {100*len(val_ids)/len(self.samples):>11.1f}%")
        self.log(f"{'Test':<10} {len(test_ids):>8} {100*len(test_ids)/len(self.samples):>11.1f}%")
        self.log("-" * 32)
        self.log(f"{'Total':<10} {len(self.samples):>8} {100.0:>11.1f}%")

        # Print T-category distribution per split
        self.log(f"\nT-Category Distribution per Split:")

        def get_split_t_distribution(indices):
            cats = [t_categories[i] for i in indices]
            return Counter(cats)

        train_dist = get_split_t_distribution(train_indices)
        val_dist = get_split_t_distribution(val_indices)
        test_dist = get_split_t_distribution(test_indices)

        self.log(f"\n{'Category':<10} {'Train':>8} {'Val':>8} {'Test':>8}")
        self.log("-" * 36)
        for cat in ["T1a", "T1b", "T2", "T3", "T4"]:
            self.log(f"{cat:<10} {train_dist.get(cat, 0):>8} {val_dist.get(cat, 0):>8} {test_dist.get(cat, 0):>8}")

        # Save split info to JSON
        split_info = {
            "created_at": datetime.now().isoformat(),
            "config_path": str(self.config_path),
            "random_seed": self.random_seed,
            "split_ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "splits": {
                "train": {
                    "count": len(train_ids),
                    "sample_ids": sorted(train_ids),
                    "t_category_distribution": dict(train_dist),
                },
                "val": {
                    "count": len(val_ids),
                    "sample_ids": sorted(val_ids),
                    "t_category_distribution": dict(val_dist),
                },
                "test": {
                    "count": len(test_ids),
                    "sample_ids": sorted(test_ids),
                    "t_category_distribution": dict(test_dist),
                },
            },
            "total_samples": len(self.samples),
        }

        split_path = self.results_dir / "data_split.json"
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        self.log(f"\nSaved split info to: {split_path}")

        # Create split visualization
        self._plot_split_distribution(train_dist, val_dist, test_dist)

    def _plot_split_distribution(
        self,
        train_dist: Counter,
        val_dist: Counter,
        test_dist: Counter,
    ) -> None:
        """Create split distribution visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Overall split pie chart
        split_counts = [sum(train_dist.values()), sum(val_dist.values()), sum(test_dist.values())]
        split_labels = [f"Train\n({split_counts[0]})", f"Val\n({split_counts[1]})", f"Test\n({split_counts[2]})"]

        axes[0].pie(split_counts, labels=split_labels, autopct="%1.1f%%",
                    colors=["#3498db", "#2ecc71", "#e74c3c"],
                    explode=[0.02, 0.02, 0.02], startangle=90)
        axes[0].set_title("Overall Split Distribution", fontsize=14, fontweight="bold")

        # 2. T-category per split grouped bar chart
        categories = ["T1a", "T1b", "T2", "T3", "T4"]
        x = np.arange(len(categories))
        width = 0.25

        train_counts = [train_dist.get(cat, 0) for cat in categories]
        val_counts = [val_dist.get(cat, 0) for cat in categories]
        test_counts = [test_dist.get(cat, 0) for cat in categories]

        axes[1].bar(x - width, train_counts, width, label="Train", color="#3498db")
        axes[1].bar(x, val_counts, width, label="Val", color="#2ecc71")
        axes[1].bar(x + width, test_counts, width, label="Test", color="#e74c3c")

        axes[1].set_xlabel("T-Category", fontsize=12)
        axes[1].set_ylabel("Number of Samples", fontsize=12)
        axes[1].set_title("T-Category Distribution per Split", fontsize=14, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        save_path = self.figures_dir / "split_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        self.log(f"Saved split distribution figure to: {save_path}")

    def save_report(self) -> None:
        """Save the summary report to a text file."""
        self.log("\n" + "=" * 70)
        self.log("SECTION 7: SUMMARY REPORT")
        self.log("=" * 70)

        report_path = self.results_dir / "data_exploration_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        self.log(f"\nReport saved to: {report_path}")

        # Print summary of generated files
        self.log(f"\nGenerated Files:")
        self.log(f"  Figures:")
        for fig_file in self.figures_dir.glob("*.png"):
            self.log(f"    - {fig_file.name}")
        self.log(f"  Data:")
        self.log(f"    - data_split.json")
        self.log(f"  Report:")
        self.log(f"    - data_exploration_report.txt")


def main():
    """Main entry point for data exploration."""
    parser = argparse.ArgumentParser(
        description="Explore Breslow depth prediction dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_v1.yaml",
        help="Path to configuration file (default: configs/config_v1.yaml)"
    )
    args = parser.parse_args()

    # Change to project directory
    os.chdir(PROJECT_ROOT)

    # Run exploration
    explorer = DataExplorer(config_path=args.config)
    explorer.run()


if __name__ == "__main__":
    main()
