#!/usr/bin/env python3
"""Comprehensive data exploration script for Breslow depth prediction.

This script performs initial data exploration and generates:
- Dataset integrity verification
- Breslow depth statistics
- T-category distribution
- Class pixel distribution
- Train/val/test splits
- Visualization figures

Run with: python scripts/explore_data.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import yaml
try:
    import yaml
except ImportError:
    print("PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


class DataExplorer:
    """Comprehensive data exploration for Breslow depth prediction dataset."""

    # RGB to class mapping
    RGB_TO_CLASS = {
        (0, 0, 0): 0,       # Background - Black
        (0, 0, 255): 1,     # Tumour - Blue
        (0, 255, 0): 2,     # Epidermis - Green
        (255, 0, 0): 3,     # Dermis - Red
    }

    CLASS_NAMES = ['Background', 'Tumour', 'Epidermis', 'Dermis']

    # T-category thresholds (in micrometres)
    T_CATEGORIES = {
        'T1a': (0, 800),
        'T1b': (800, 1000),
        'T2': (1000, 2000),
        'T3': (2000, 4000),
        'T4': (4000, float('inf')),
    }

    def __init__(self, config_path: str):
        """Initialize with config file path."""
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data']['data_dir'])
        self.images_dir = self.data_dir / self.config['data']['images_dir']
        self.coords_file = self.data_dir / self.config['data']['coords_file']
        self.exclude_file = self.data_dir / self.config['data'].get('exclude_file', '')
        self.image_suffix = self.config['data'].get('image_suffix', '_image.png')
        self.mask_suffix = self.config['data'].get('mask_suffix', '_labels.png')
        self.resolution = self.config['data'].get('resolution_um_per_pixel', 4.0)

        # Results storage
        self.results = {}
        self.report_lines = []
        self.df = None  # Store CSV data
        self.sample_id_to_depth = {}  # Map sample IDs to Breslow depths

        # Create output directories
        self.results_dir = project_root / "results"
        self.figures_dir = self.results_dir / "figures" / "exploration"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def _extract_sample_id(self, filename: str) -> str:
        """Extract base sample ID from filename.

        Handles formats like:
        - Sample-26-A_001.svs -> Sample-26-A_001
        - Sample-26-A_001_[baseMPP=0.253_targetMPP=4.00]_image.png -> Sample-26-A_001
        """
        # Remove file extension
        name = Path(filename).stem

        # Remove _image or _labels suffix
        for suffix in ['_image', '_labels']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Remove [baseMPP=...] part if present
        if '[' in name:
            name = name.split('[')[0].rstrip('_')

        # Remove .svs if present (from CSV)
        if name.endswith('.svs'):
            name = name[:-4]

        return name

    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _log(self, message: str):
        """Add message to report and print."""
        print(message)
        self.report_lines.append(message)

    def _get_t_category(self, depth_um: float) -> str:
        """Get T-category from Breslow depth in micrometres."""
        for category, (min_val, max_val) in self.T_CATEGORIES.items():
            if min_val <= depth_um < max_val:
                return category
        return 'T4'

    def verify_dataset_integrity(self) -> Dict:
        """Verify dataset integrity - check all files exist and match."""
        self._log("\nDATASET INTEGRITY CHECK")
        self._log("-" * 23)

        # Load CSV
        if not self.coords_file.exists():
            self._log(f"ERROR: CSV file not found: {self.coords_file}")
            return {}

        self.df = pd.read_csv(self.coords_file, encoding='latin-1')
        csv_entries = len(self.df)

        # Load exclusion list and extract sample IDs
        excluded_samples = set()
        if self.exclude_file and Path(self.exclude_file).exists():
            with open(self.exclude_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Extract sample ID from exclusion entry
                        excluded_samples.add(self._extract_sample_id(line))

        # Find all image and mask files
        image_files = list(self.images_dir.glob(f"*{self.image_suffix}"))
        mask_files = list(self.images_dir.glob(f"*{self.mask_suffix}"))

        # Get sample IDs from filenames
        image_ids = {}  # sample_id -> full filename (without extension)
        for f in image_files:
            sample_id = self._extract_sample_id(f.name)
            image_ids[sample_id] = f.stem

        mask_ids = {}
        for f in mask_files:
            sample_id = self._extract_sample_id(f.name)
            mask_ids[sample_id] = f.stem

        # Get sample IDs from CSV
        csv_ids = set()
        id_col = 'image' if 'image' in self.df.columns else self.df.columns[0]
        for val in self.df[id_col]:
            csv_ids.add(self._extract_sample_id(str(val)))

        # Check for missing pairs
        image_set = set(image_ids.keys())
        mask_set = set(mask_ids.keys())
        missing_masks = image_set - mask_set
        missing_images = mask_set - image_set

        # Valid samples: have both image and mask, in CSV, not excluded
        valid_sample_ids = (image_set & mask_set & csv_ids) - excluded_samples

        # Store mapping of sample_id to full filenames for later use
        self.sample_to_files = {}
        for sample_id in valid_sample_ids:
            self.sample_to_files[sample_id] = {
                'image': image_ids.get(sample_id),
                'mask': mask_ids.get(sample_id),
            }

        results = {
            'total_images': len(image_files),
            'total_masks': len(mask_files),
            'csv_entries': csv_entries,
            'excluded_samples': len(excluded_samples),
            'valid_samples': len(valid_sample_ids),
            'missing_masks': list(missing_masks),
            'missing_images': list(missing_images),
            'valid_ids': list(valid_sample_ids),
        }

        self._log(f"Total image files found: {results['total_images']}")
        self._log(f"Total mask files found: {results['total_masks']}")
        self._log(f"CSV entries: {results['csv_entries']}")
        self._log(f"Excluded samples: {results['excluded_samples']} (from wrong_dir_coords.txt)")
        self._log(f"Valid samples: {results['valid_samples']}")

        if missing_masks:
            self._log(f"WARNING: {len(missing_masks)} images missing masks")
        if missing_images:
            self._log(f"WARNING: {len(missing_images)} masks missing images")

        self.results['integrity'] = results
        return results

    def calculate_breslow_statistics(self) -> Dict:
        """Calculate Breslow depth statistics from CSV."""
        self._log("\nBRESLOW DEPTH STATISTICS")
        self._log("-" * 24)

        if self.df is None:
            self.df = pd.read_csv(self.coords_file, encoding='latin-1')

        df = self.df.copy()

        # Determine the depth column name (handle special characters like µ)
        depth_col = None
        possible_depth_cols = [
            'length_µm', 'length_um', 'length_\u00b5m',  # µm variants
            'breslow_depth_um', 'breslow_depth', 'depth_um', 'depth',
            'Breslow_depth', 'Breslow'
        ]

        for col in df.columns:
            # Check exact match first
            if col in possible_depth_cols:
                depth_col = col
                break
            # Check if column contains 'length' and 'm' (for µm or um)
            if 'length' in col.lower() and 'm' in col.lower():
                depth_col = col
                break

        if depth_col is None:
            # Try to calculate from coordinates if available
            if 'y_epidermis' in df.columns and 'y_dermis' in df.columns:
                df['breslow_depth_um'] = abs(df['y_dermis'] - df['y_epidermis']) * self.resolution
                depth_col = 'breslow_depth_um'
            else:
                self._log(f"ERROR: No depth column found in CSV. Columns: {list(df.columns)}")
                return {}

        self._log(f"Using depth column: '{depth_col}'")

        # Get the ID column
        id_col = 'image' if 'image' in df.columns else df.columns[0]

        # Create mapping of sample_id to depth
        for _, row in df.iterrows():
            sample_id = self._extract_sample_id(str(row[id_col]))
            depth = row[depth_col]
            if pd.notna(depth):
                self.sample_id_to_depth[sample_id] = float(depth)

        # Filter to valid samples if integrity check was run
        valid_ids = set(self.results.get('integrity', {}).get('valid_ids', []))
        if valid_ids:
            depths = [self.sample_id_to_depth[sid] for sid in valid_ids
                     if sid in self.sample_id_to_depth]
        else:
            depths = list(self.sample_id_to_depth.values())

        if not depths:
            self._log("ERROR: No valid depth values found")
            return {}

        depths = np.array(depths)

        results = {
            'min': float(depths.min()),
            'max': float(depths.max()),
            'mean': float(depths.mean()),
            'std': float(depths.std()),
            'median': float(np.median(depths)),
            'count': len(depths),
        }

        self._log(f"Min: {results['min']:.2f} \u00b5m ({results['min']/1000:.2f} mm)")
        self._log(f"Max: {results['max']:.2f} \u00b5m ({results['max']/1000:.2f} mm)")
        self._log(f"Mean: {results['mean']:.2f} \u00b5m ({results['mean']/1000:.2f} mm)")
        self._log(f"Std: {results['std']:.2f} \u00b5m")

        self.results['breslow_stats'] = results
        self.results['depths'] = depths.tolist()
        return results

    def analyze_t_category_distribution(self) -> Dict:
        """Analyze T-category distribution."""
        self._log("\nT-CATEGORY DISTRIBUTION")
        self._log("-" * 23)

        if 'depths' not in self.results:
            if not self.sample_id_to_depth:
                self._log("ERROR: Run calculate_breslow_statistics first")
                return {}
            # Use depths from valid samples
            valid_ids = self.results.get('integrity', {}).get('valid_ids', [])
            depths = [self.sample_id_to_depth[sid] for sid in valid_ids
                     if sid in self.sample_id_to_depth]
            self.results['depths'] = depths
        else:
            depths = self.results['depths']

        if not depths:
            self._log("ERROR: No depth data available")
            return {}

        t_categories = [self._get_t_category(d) for d in depths]

        distribution = {}
        for cat in ['T1a', 'T1b', 'T2', 'T3', 'T4']:
            count = t_categories.count(cat)
            pct = count / len(t_categories) * 100 if t_categories else 0
            distribution[cat] = {'count': count, 'percentage': pct}

        # Print with threshold info
        thresholds = {
            'T1a': '< 0.8mm',
            'T1b': '0.8-1.0mm',
            'T2': '1.0-2.0mm',
            'T3': '2.0-4.0mm',
            'T4': '> 4.0mm',
        }

        for cat, info in distribution.items():
            self._log(f"{cat} ({thresholds[cat]}): {info['count']:>5} samples ({info['percentage']:>5.2f}%)")

        self.results['t_category_distribution'] = distribution
        self.results['t_categories'] = t_categories
        return distribution

    def analyze_class_distribution(self) -> Dict:
        """Analyze pixel class distribution across all masks."""
        self._log("\nCLASS PIXEL DISTRIBUTION")
        self._log("-" * 24)

        if 'integrity' not in self.results:
            self.verify_dataset_integrity()

        valid_ids = self.results['integrity'].get('valid_ids', [])
        if not valid_ids:
            self._log("ERROR: No valid samples found")
            return {}

        # Count pixels per class
        class_pixels = defaultdict(int)
        total_pixels = 0
        samples_processed = 0

        for sample_id in valid_ids:
            # Get the full mask filename from our mapping
            if hasattr(self, 'sample_to_files') and sample_id in self.sample_to_files:
                mask_stem = self.sample_to_files[sample_id]['mask']
                mask_path = self.images_dir / f"{mask_stem}.png"
            else:
                # Fallback: try to find mask file matching sample_id
                mask_files = list(self.images_dir.glob(f"*{sample_id}*{self.mask_suffix}"))
                if not mask_files:
                    continue
                mask_path = mask_files[0]

            if not mask_path.exists():
                continue

            try:
                mask = np.array(Image.open(mask_path).convert('RGB'))
                h, w = mask.shape[:2]
                total_pixels += h * w

                # Convert RGB to class indices
                for rgb, class_idx in self.RGB_TO_CLASS.items():
                    match = np.all(mask == rgb, axis=-1)
                    class_pixels[class_idx] += np.sum(match)

                samples_processed += 1
            except Exception as e:
                print(f"  Warning: Could not process {mask_path}: {e}")

        self._log(f"Processed {samples_processed} mask files")

        if total_pixels == 0:
            self._log("ERROR: No pixels processed")
            return {}

        # Calculate percentages
        distribution = {}
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            count = class_pixels[class_idx]
            pct = count / total_pixels * 100
            distribution[class_name] = {'pixels': count, 'percentage': pct}
            self._log(f"{class_name}: {pct:.1f}%")

        # Calculate suggested class weights (inverse frequency)
        percentages = [distribution[name]['percentage'] for name in self.CLASS_NAMES]
        if all(p > 0 for p in percentages):
            inv_freq = [100 / p for p in percentages]
            # Normalize to have max weight of 1.0
            max_weight = max(inv_freq)
            weights = [round(w / max_weight, 2) for w in inv_freq]
            self._log(f"\nSuggested class weights: {weights}")
            distribution['suggested_weights'] = weights

        self.results['class_distribution'] = distribution
        return distribution

    def create_stratified_split(self) -> Dict:
        """Create stratified train/val/test split by T-category."""
        self._log("\nTRAIN/VAL/TEST SPLIT (Stratified)")
        self._log("-" * 33)

        valid_ids = self.results['integrity'].get('valid_ids', [])

        if not valid_ids:
            self._log("ERROR: No valid samples to split")
            return {}

        # Get T-categories for valid samples
        valid_ids_with_depth = []
        t_categories = []
        for sid in valid_ids:
            if sid in self.sample_id_to_depth:
                valid_ids_with_depth.append(sid)
                depth = self.sample_id_to_depth[sid]
                t_categories.append(self._get_t_category(depth))

        valid_ids = valid_ids_with_depth

        if len(valid_ids) < 10:
            self._log("WARNING: Too few samples for stratified split, using simple split")
            np.random.seed(self.config['split'].get('random_seed', 42))
            indices = np.random.permutation(len(valid_ids))

            train_ratio = self.config['split'].get('train_ratio', 0.7)
            val_ratio = self.config['split'].get('val_ratio', 0.15)

            n_train = int(len(valid_ids) * train_ratio)
            n_val = int(len(valid_ids) * val_ratio)

            train_ids = [valid_ids[i] for i in indices[:n_train]]
            val_ids = [valid_ids[i] for i in indices[n_train:n_train + n_val]]
            test_ids = [valid_ids[i] for i in indices[n_train + n_val:]]
        else:
            # Stratified split
            try:
                from sklearn.model_selection import train_test_split

                train_ratio = self.config['split'].get('train_ratio', 0.7)
                val_ratio = self.config['split'].get('val_ratio', 0.15)
                test_ratio = self.config['split'].get('test_ratio', 0.15)
                seed = self.config['split'].get('random_seed', 42)

                # First split: train vs (val + test)
                train_ids, temp_ids, train_cats, temp_cats = train_test_split(
                    valid_ids, t_categories,
                    train_size=train_ratio,
                    stratify=t_categories,
                    random_state=seed
                )

                # Second split: val vs test
                val_size = val_ratio / (val_ratio + test_ratio)
                val_ids, test_ids = train_test_split(
                    temp_ids,
                    train_size=val_size,
                    stratify=temp_cats,
                    random_state=seed
                )
            except Exception as e:
                self._log(f"WARNING: Stratified split failed ({e}), using simple split")
                np.random.seed(self.config['split'].get('random_seed', 42))
                indices = np.random.permutation(len(valid_ids))

                train_ratio = self.config['split'].get('train_ratio', 0.7)
                val_ratio = self.config['split'].get('val_ratio', 0.15)

                n_train = int(len(valid_ids) * train_ratio)
                n_val = int(len(valid_ids) * val_ratio)

                train_ids = [valid_ids[i] for i in indices[:n_train]]
                val_ids = [valid_ids[i] for i in indices[n_train:n_train + n_val]]
                test_ids = [valid_ids[i] for i in indices[n_train + n_val:]]

        split = {
            'train': list(train_ids),
            'val': list(val_ids),
            'test': list(test_ids),
        }

        total = len(valid_ids)
        self._log(f"Train: {len(train_ids)} samples ({len(train_ids)/total*100:.0f}%)")
        self._log(f"Val: {len(val_ids)} samples ({len(val_ids)/total*100:.0f}%)")
        self._log(f"Test: {len(test_ids)} samples ({len(test_ids)/total*100:.0f}%)")

        # Save split to JSON
        split_path = self.results_dir / "data_split.json"
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        self._log(f"\nSplit saved to: {split_path}")

        self.results['split'] = split
        self.results['t_categories'] = t_categories
        return split

    def visualize_samples(self, n_samples: int = 6):
        """Visualize random samples from the dataset."""
        if 'integrity' not in self.results:
            self.verify_dataset_integrity()

        valid_ids = self.results['integrity'].get('valid_ids', [])
        if not valid_ids:
            return

        # Select random samples
        np.random.seed(42)
        sample_ids = np.random.choice(valid_ids, min(n_samples, len(valid_ids)), replace=False)

        fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
        if n_samples == 1:
            axes = axes.reshape(2, 1)

        for i, sample_id in enumerate(sample_ids):
            # Get full filenames from mapping
            if hasattr(self, 'sample_to_files') and sample_id in self.sample_to_files:
                image_stem = self.sample_to_files[sample_id]['image']
                mask_stem = self.sample_to_files[sample_id]['mask']
                image_path = self.images_dir / f"{image_stem}.png"
                mask_path = self.images_dir / f"{mask_stem}.png"
            else:
                # Fallback
                image_files = list(self.images_dir.glob(f"*{sample_id}*{self.image_suffix}"))
                mask_files = list(self.images_dir.glob(f"*{sample_id}*{self.mask_suffix}"))
                image_path = image_files[0] if image_files else None
                mask_path = mask_files[0] if mask_files else None

            # Get Breslow depth for title
            depth = self.sample_id_to_depth.get(sample_id, 0)
            t_cat = self._get_t_category(depth) if depth > 0 else "?"

            if image_path and image_path.exists():
                img = Image.open(image_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"{sample_id[:15]}...\n{depth:.0f}\u00b5m ({t_cat})", fontsize=8)
                axes[0, i].axis('off')

            if mask_path and mask_path.exists():
                mask = Image.open(mask_path)
                axes[1, i].imshow(mask)
                axes[1, i].set_title("Mask", fontsize=8)
                axes[1, i].axis('off')

        plt.suptitle("Sample Images and Masks", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.figures_dir / "sample_images.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_breslow_distribution(self):
        """Plot Breslow depth distribution."""
        if 'depths' not in self.results:
            return

        depths = np.array(self.results['depths'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        axes[0].hist(depths, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Breslow Depth (\u00b5m)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Breslow Depth Distribution')
        axes[0].axvline(depths.mean(), color='r', linestyle='--', label=f'Mean: {depths.mean():.0f}\u00b5m')
        axes[0].legend()

        # T-category bar plot
        if 't_category_distribution' in self.results:
            dist = self.results['t_category_distribution']
            categories = list(dist.keys())
            counts = [dist[c]['count'] for c in categories]

            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
            axes[1].bar(categories, counts, color=colors, edgecolor='black')
            axes[1].set_xlabel('T-Category')
            axes[1].set_ylabel('Count')
            axes[1].set_title('T-Category Distribution')

            # Add count labels on bars
            for i, (cat, count) in enumerate(zip(categories, counts)):
                axes[1].text(i, count + 0.5, str(count), ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "breslow_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_class_distribution(self):
        """Plot class pixel distribution."""
        if 'class_distribution' not in self.results:
            return

        dist = self.results['class_distribution']

        fig, ax = plt.subplots(figsize=(8, 6))

        classes = self.CLASS_NAMES
        percentages = [dist[c]['percentage'] for c in classes]
        colors = ['#95a5a6', '#3498db', '#2ecc71', '#e74c3c']

        bars = ax.bar(classes, percentages, color=colors, edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Percentage of Pixels')
        ax.set_title('Class Pixel Distribution')

        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f'{pct:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / "class_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    def save_report(self):
        """Save the exploration report to file."""
        report_path = self.results_dir / "data_exploration_report.txt"

        header = [
            "=" * 80,
            "BRESLOW DEPTH PREDICTION - DATA EXPLORATION REPORT",
            "=" * 80,
            "",
            f"Dataset Location: {self.images_dir}",
            f"CSV File: {self.coords_file}",
        ]

        footer = [
            "",
            f"Figures saved to: {self.figures_dir}",
            "",
            "=" * 80,
        ]

        full_report = header + self.report_lines + footer

        with open(report_path, 'w') as f:
            f.write('\n'.join(full_report))

        print(f"\nReport saved to: {report_path}")

    def run_full_exploration(self):
        """Run complete data exploration pipeline."""
        print("=" * 80)
        print("BRESLOW DEPTH PREDICTION - DATA EXPLORATION REPORT")
        print("=" * 80)
        print(f"\nDataset Location: {self.images_dir}")
        print(f"CSV File: {self.coords_file}")

        # Run all analyses
        self.verify_dataset_integrity()
        self.calculate_breslow_statistics()
        self.analyze_t_category_distribution()
        self.analyze_class_distribution()
        self.create_stratified_split()

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_samples()
        self.plot_breslow_distribution()
        self.plot_class_distribution()

        self._log(f"\nFigures saved to: {self.figures_dir}")

        # Save report
        self.save_report()

        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    # Find config file
    config_path = project_root / "breslow_depth_prediction" / "configs" / "config_v1.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print("Please create the config file first.")
        sys.exit(1)

    # Run exploration
    explorer = DataExplorer(str(config_path))
    explorer.run_full_exploration()


if __name__ == "__main__":
    main()
