#!/usr/bin/env python3
"""Verify the entire Breslow depth prediction pipeline setup.

This script tests all components of the pipeline to ensure
everything is properly configured and working.

Run with: python scripts/verify_setup.py
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Use non-interactive backend to prevent plt.show() from blocking
import matplotlib
matplotlib.use("Agg")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestResult:
    """Container for test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results: List[Tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, message: str = ""):
        self.results.append((name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_result(self, name: str, passed: bool, message: str = ""):
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  [{status}] {name}")
        if message and not passed:
            print(f"         |- {message}")

    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for name, passed, message in self.results:
            self.print_result(name, passed, message)

        print("\n" + "-" * 60)
        if self.failed == 0:
            print(f"\033[92m All {total} tests passed!\033[0m")
        else:
            print(f"\033[93m{self.passed}/{total} tests passed\033[0m")
            print(f"\033[91m{self.failed} tests failed\033[0m")
        print("=" * 60)


def test_imports(results: TestResult) -> Dict[str, Any]:
    """Test 1: Test importing all modules."""
    print("\n" + "=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    modules = {}

    # Test config module
    try:
        from breslow_depth_prediction.src.config import load_config, resolve_paths
        modules['config'] = {'load_config': load_config, 'resolve_paths': resolve_paths}
        results.add("Import config module", True)
        print("  [PASS] config module")
    except ImportError as e:
        results.add("Import config module", False, str(e))
        print(f"  [FAIL] config module: {e}")

    # Test data module
    try:
        from breslow_depth_prediction.src.data import (
            BreslowDataset, create_dataloaders,
            get_train_transforms, get_val_transforms,
        )
        modules['data'] = {
            'BreslowDataset': BreslowDataset,
            'create_dataloaders': create_dataloaders,
            'get_train_transforms': get_train_transforms,
            'get_val_transforms': get_val_transforms,
        }
        results.add("Import data module", True)
        print("  [PASS] data module")
    except ImportError as e:
        results.add("Import data module", False, str(e))
        print(f"  [FAIL] data module: {e}")

    # Test models module
    try:
        from breslow_depth_prediction.src.models import (
            get_model, get_loss_function, DiceLoss, FocalLoss, CombinedLoss
        )
        modules['models'] = {
            'get_model': get_model,
            'get_loss_function': get_loss_function,
            'DiceLoss': DiceLoss,
            'FocalLoss': FocalLoss,
            'CombinedLoss': CombinedLoss,
        }
        results.add("Import models module", True)
        print("  [PASS] models module")
    except ImportError as e:
        results.add("Import models module", False, str(e))
        print(f"  [FAIL] models module: {e}")

    # Test training module
    try:
        from breslow_depth_prediction.src.training import (
            SegmentationMetrics, BreslowMetrics, compute_confusion_matrix, print_metrics_summary
        )
        modules['training'] = {
            'SegmentationMetrics': SegmentationMetrics,
            'BreslowMetrics': BreslowMetrics,
            'compute_confusion_matrix': compute_confusion_matrix,
            'print_metrics_summary': print_metrics_summary,
        }
        results.add("Import training module", True)
        print("  [PASS] training module")
    except ImportError as e:
        results.add("Import training module", False, str(e))
        print(f"  [FAIL] training module: {e}")

    # Test visualization module
    try:
        from breslow_depth_prediction.src.visualization import (
            visualize_sample, visualize_batch, plot_training_history
        )
        modules['visualization'] = {
            'visualize_sample': visualize_sample,
            'visualize_batch': visualize_batch,
            'plot_training_history': plot_training_history,
        }
        results.add("Import visualization module", True)
        print("  [PASS] visualization module")
    except ImportError as e:
        results.add("Import visualization module", False, str(e))
        print(f"  [FAIL] visualization module: {e}")

    # Test PyTorch
    try:
        import torch
        modules['torch'] = torch
        cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
        results.add("Import PyTorch", True)
        print(f"  [PASS] PyTorch ({cuda_status})")
    except ImportError as e:
        results.add("Import PyTorch", False, str(e))
        print(f"  [FAIL] PyTorch: {e}")

    # Test segmentation-models-pytorch
    try:
        import segmentation_models_pytorch as smp
        modules['smp'] = smp
        results.add("Import segmentation-models-pytorch", True)
        print("  [PASS] segmentation-models-pytorch")
    except ImportError as e:
        results.add("Import segmentation-models-pytorch", False, str(e))
        print(f"  [FAIL] segmentation-models-pytorch: {e}")

    return modules


def test_config_loading(results: TestResult, modules: Dict[str, Any]) -> Dict[str, Any]:
    """Test 2: Test config loading."""
    print("\n" + "=" * 60)
    print("TEST 2: Config Loading")
    print("=" * 60)

    config = None

    if 'config' not in modules:
        results.add("Load config_v1.yaml", False, "Config module not imported")
        print("  [SKIP] Config module not available")
        return config

    try:
        load_config = modules['config']['load_config']
        resolve_paths = modules['config']['resolve_paths']
        config_path = project_root / "breslow_depth_prediction" / "configs" / "config_v1.yaml"
        # Resolve relative paths against project_root so this verifier works from any cwd.
        config = resolve_paths(load_config(str(config_path)), project_root)
        results.add("Load config_v1.yaml", True)
        print("  [PASS] Config loaded successfully")
    except Exception as e:
        results.add("Load config_v1.yaml", False, str(e))
        print(f"  [FAIL] Failed to load config: {e}")
        return config

    # Verify required keys
    required_keys = ['data', 'model', 'training', 'classes']
    missing_keys = [k for k in required_keys if k not in config]

    if not missing_keys:
        results.add("Verify required config keys", True)
        print(f"  [PASS] All required keys present: {required_keys}")
    else:
        results.add("Verify required config keys", False, f"Missing: {missing_keys}")
        print(f"  [FAIL] Missing keys: {missing_keys}")

    # Print config summary
    print("\n  Config summary:")
    if 'data' in config:
        print(f"    - Data dir: {config['data'].get('data_dir', 'N/A')}")
    if 'model' in config:
        print(f"    - Architecture: {config['model'].get('architecture', 'N/A')}")
        print(f"    - Encoder: {config['model'].get('encoder', 'N/A')}")
    if 'classes' in config:
        print(f"    - Num classes: {config['classes'].get('num_classes', 'N/A')}")

    return config


def test_dataset(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any]) -> Any:
    """Test 3: Test dataset functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Dataset")
    print("=" * 60)

    dataset = None

    if 'data' not in modules or config is None:
        results.add("Create dataset", False, "Data module or config not available")
        print("  [SKIP] Dependencies not available")
        return dataset

    try:
        BreslowDataset = modules['data']['BreslowDataset']
        get_val_transforms = modules['data']['get_val_transforms']

        # Get transforms
        transform = None
        try:
            transform = get_val_transforms(config)
        except Exception:
            pass

        dataset = BreslowDataset(
            data_dir=config['data']['data_dir'],
            split='train',
            transform=transform,
        )
        results.add("Create dataset instance", True)
        print("  [PASS] Dataset created")
    except Exception as e:
        results.add("Create dataset instance", False, str(e))
        print(f"  [FAIL] Failed to create dataset: {e}")
        return dataset

    # Verify __len__ > 0
    try:
        length = len(dataset)
        if length > 0:
            results.add("Dataset __len__ > 0", True)
            print(f"  [PASS] Dataset length: {length}")
        else:
            results.add("Dataset __len__ > 0", False, "Dataset is empty")
            print("  [FAIL] Dataset is empty")
            return dataset
    except Exception as e:
        results.add("Dataset __len__ > 0", False, str(e))
        print(f"  [FAIL] Failed to get dataset length: {e}")
        return dataset

    # Get one sample
    try:
        sample = dataset[0]
        required_keys = ['image', 'mask']
        missing_keys = [k for k in required_keys if k not in sample]

        if not missing_keys:
            results.add("Sample has required keys", True)
            print(f"  [PASS] Sample keys: {list(sample.keys())}")
        else:
            results.add("Sample has required keys", False, f"Missing: {missing_keys}")
            print(f"  [FAIL] Missing keys: {missing_keys}")
            return dataset
    except Exception as e:
        results.add("Sample has required keys", False, str(e))
        print(f"  [FAIL] Failed to get sample: {e}")
        return dataset

    # Verify image shape (3, H, W)
    try:
        image = sample['image']
        if hasattr(image, 'shape'):
            if len(image.shape) == 3 and image.shape[0] == 3:
                results.add("Image shape (3, H, W)", True)
                print(f"  [PASS] Image shape: {tuple(image.shape)}")
            else:
                results.add("Image shape (3, H, W)", False, f"Got shape: {tuple(image.shape)}")
                print(f"  [FAIL] Expected (3, H, W), got: {tuple(image.shape)}")
        else:
            results.add("Image shape (3, H, W)", False, "Image is not a tensor")
            print("  [FAIL] Image is not a tensor")
    except Exception as e:
        results.add("Image shape (3, H, W)", False, str(e))
        print(f"  [FAIL] Failed to verify image shape: {e}")

    # Verify mask shape (H, W)
    try:
        mask = sample['mask']
        if hasattr(mask, 'shape'):
            if len(mask.shape) == 2:
                results.add("Mask shape (H, W)", True)
                print(f"  [PASS] Mask shape: {tuple(mask.shape)}")
            else:
                results.add("Mask shape (H, W)", False, f"Got shape: {tuple(mask.shape)}")
                print(f"  [FAIL] Expected (H, W), got: {tuple(mask.shape)}")
        else:
            results.add("Mask shape (H, W)", False, "Mask is not a tensor")
            print("  [FAIL] Mask is not a tensor")
    except Exception as e:
        results.add("Mask shape (H, W)", False, str(e))
        print(f"  [FAIL] Failed to verify mask shape: {e}")

    # Verify mask values in [0, 1, 2, 3]
    try:
        mask = sample['mask']
        torch = modules['torch']
        unique_values = torch.unique(mask).tolist()
        valid_values = all(v in [0, 1, 2, 3] for v in unique_values)

        if valid_values:
            results.add("Mask values in [0,1,2,3]", True)
            print(f"  [PASS] Mask unique values: {unique_values}")
        else:
            results.add("Mask values in [0,1,2,3]", False, f"Got values: {unique_values}")
            print(f"  [FAIL] Invalid mask values: {unique_values}")
    except Exception as e:
        results.add("Mask values in [0,1,2,3]", False, str(e))
        print(f"  [FAIL] Failed to verify mask values: {e}")

    # Verify breslow_depth_um is positive float
    try:
        if 'breslow_depth_um' in sample:
            depth = sample['breslow_depth_um']
            if isinstance(depth, (int, float)) and depth > 0:
                results.add("breslow_depth_um positive", True)
                print(f"  [PASS] Breslow depth: {depth:.2f} \u00b5m")
            else:
                results.add("breslow_depth_um positive", False, f"Got: {depth}")
                print(f"  [FAIL] Invalid breslow depth: {depth}")
        else:
            results.add("breslow_depth_um positive", True)
            print("  [PASS] breslow_depth_um not in sample (optional)")
    except Exception as e:
        results.add("breslow_depth_um positive", False, str(e))
        print(f"  [FAIL] Failed to verify breslow depth: {e}")

    return dataset


def test_transforms(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any], dataset: Any):
    """Test 4: Test transforms."""
    print("\n" + "=" * 60)
    print("TEST 4: Transforms")
    print("=" * 60)

    if 'data' not in modules or config is None:
        results.add("Apply train transforms", False, "Data module not available")
        print("  [SKIP] Dependencies not available")
        return

    get_train_transforms = modules['data']['get_train_transforms']
    get_val_transforms = modules['data']['get_val_transforms']

    # Test train transforms
    try:
        train_transforms = get_train_transforms(config)
        results.add("Create train transforms", True)
        print(f"  [PASS] Train transforms created: {type(train_transforms).__name__}")
    except Exception as e:
        results.add("Create train transforms", False, str(e))
        print(f"  [FAIL] Failed to create train transforms: {e}")

    # Test val transforms
    try:
        val_transforms = get_val_transforms(config)
        results.add("Create val transforms", True)
        print(f"  [PASS] Val transforms created: {type(val_transforms).__name__}")
    except Exception as e:
        results.add("Create val transforms", False, str(e))
        print(f"  [FAIL] Failed to create val transforms: {e}")


def test_dataloader(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any]):
    """Test 5: Test dataloader."""
    print("\n" + "=" * 60)
    print("TEST 5: DataLoader")
    print("=" * 60)

    if 'data' not in modules or config is None:
        results.add("Create dataloader", False, "Data module or config not available")
        print("  [SKIP] Dependencies not available")
        return

    try:
        create_dataloaders = modules['data']['create_dataloaders']

        # Create dataloaders with small batch size for testing
        import copy
        test_config = copy.deepcopy(config)
        test_config['training']['batch_size'] = 2

        train_loader, val_loader, test_loader = create_dataloaders(test_config)

        results.add("Create dataloaders", True)
        print(f"  [PASS] Dataloaders created (train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches)")
    except Exception as e:
        results.add("Create dataloaders", False, str(e))
        print(f"  [FAIL] Failed to create dataloaders: {e}")
        return

    # Get one batch
    try:
        batch = next(iter(train_loader))

        # batch could be a tuple (images, masks) or a dict
        if isinstance(batch, (list, tuple)):
            images, masks = batch[0], batch[1]
        elif isinstance(batch, dict):
            images, masks = batch['image'], batch['mask']
        else:
            results.add("Get batch from dataloader", False, f"Unexpected batch type: {type(batch)}")
            return

        results.add("Get batch from dataloader", True)
        print("  [PASS] Got batch from dataloader")

        # Verify batch dimensions
        if len(images.shape) == 4 and images.shape[1] == 3:
            results.add("Batch image dimensions", True)
            print(f"  [PASS] Batch images shape: {tuple(images.shape)}")
        else:
            results.add("Batch image dimensions", False, f"Got: {tuple(images.shape)}")
            print(f"  [FAIL] Expected (B, 3, H, W), got: {tuple(images.shape)}")

        if len(masks.shape) in [2, 3]:
            results.add("Batch mask dimensions", True)
            print(f"  [PASS] Batch masks shape: {tuple(masks.shape)}")
        else:
            results.add("Batch mask dimensions", False, f"Got: {tuple(masks.shape)}")
            print(f"  [FAIL] Expected (B, H, W), got: {tuple(masks.shape)}")

    except Exception as e:
        results.add("Get batch from dataloader", False, str(e))
        print(f"  [FAIL] Failed to get batch: {e}")


def test_model(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any]):
    """Test 6: Test model creation and forward pass."""
    print("\n" + "=" * 60)
    print("TEST 6: Model")
    print("=" * 60)

    if 'models' not in modules or 'torch' not in modules or config is None:
        results.add("Create model", False, "Models module or config not available")
        print("  [SKIP] Dependencies not available")
        return

    torch = modules['torch']
    get_model = modules['models']['get_model']

    # Create model
    try:
        model = get_model(config)
        results.add("Create model from config", True)
        print("  [PASS] Model created successfully")
    except Exception as e:
        results.add("Create model from config", False, str(e))
        print(f"  [FAIL] Failed to create model: {e}")
        return

    # Forward pass with dummy input (use small size to save memory)
    try:
        device = next(model.parameters()).device
        num_classes = config['classes'].get('num_classes', 4)
        test_size = 256  # Use smaller size for testing

        dummy_input = torch.randn(1, 3, test_size, test_size).to(device)

        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        results.add("Model forward pass", True)
        print(f"  [PASS] Forward pass successful")
        print(f"    - Input shape: {tuple(dummy_input.shape)}")
        print(f"    - Output shape: {tuple(output.shape)}")
    except Exception as e:
        results.add("Model forward pass", False, str(e))
        print(f"  [FAIL] Forward pass failed: {e}")
        return

    # Verify output shape (B, num_classes, H, W)
    try:
        expected_shape = (1, num_classes, test_size, test_size)

        if output.shape == torch.Size(expected_shape):
            results.add("Output shape (B, C, H, W)", True)
            print(f"  [PASS] Output shape matches expected: {expected_shape}")
        else:
            results.add("Output shape (B, C, H, W)", False,
                       f"Expected {expected_shape}, got {tuple(output.shape)}")
            print(f"  [FAIL] Expected {expected_shape}, got {tuple(output.shape)}")
    except Exception as e:
        results.add("Output shape (B, C, H, W)", False, str(e))
        print(f"  [FAIL] Failed to verify output shape: {e}")


def test_loss_function(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any]):
    """Test 7: Test loss function."""
    print("\n" + "=" * 60)
    print("TEST 7: Loss Function")
    print("=" * 60)

    if 'models' not in modules or 'torch' not in modules or config is None:
        results.add("Create loss function", False, "Models module or config not available")
        print("  [SKIP] Dependencies not available")
        return

    torch = modules['torch']
    get_loss_function = modules['models']['get_loss_function']

    # Create loss function
    try:
        loss_fn = get_loss_function(config)
        results.add("Create loss from config", True)
        print(f"  [PASS] Loss function created: {type(loss_fn).__name__}")
    except Exception as e:
        results.add("Create loss from config", False, str(e))
        print(f"  [FAIL] Failed to create loss function: {e}")
        return

    # Compute loss on dummy data
    try:
        num_classes = config['classes'].get('num_classes', 4)
        batch_size = 2
        height, width = 64, 64

        predictions = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        loss = loss_fn(predictions, targets)

        results.add("Compute loss on dummy data", True)
        print(f"  [PASS] Loss computed: {loss.item():.4f}")
    except Exception as e:
        results.add("Compute loss on dummy data", False, str(e))
        print(f"  [FAIL] Failed to compute loss: {e}")
        return

    # Verify loss is scalar tensor
    try:
        if loss.dim() == 0 and loss.numel() == 1:
            results.add("Loss is scalar tensor", True)
            print("  [PASS] Loss is scalar tensor")
        else:
            results.add("Loss is scalar tensor", False, f"Loss shape: {loss.shape}")
            print(f"  [FAIL] Loss is not scalar, shape: {loss.shape}")
    except Exception as e:
        results.add("Loss is scalar tensor", False, str(e))
        print(f"  [FAIL] Failed to verify loss shape: {e}")


def test_metrics(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any]):
    """Test 8: Test metrics."""
    print("\n" + "=" * 60)
    print("TEST 8: Metrics")
    print("=" * 60)

    if 'training' not in modules or 'torch' not in modules:
        results.add("Create metrics objects", False, "Training module not available")
        print("  [SKIP] Dependencies not available")
        return

    torch = modules['torch']
    SegmentationMetrics = modules['training']['SegmentationMetrics']
    BreslowMetrics = modules['training']['BreslowMetrics']

    # Create segmentation metrics
    try:
        num_classes = config['classes'].get('num_classes', 4) if config else 4
        seg_metrics = SegmentationMetrics(num_classes=num_classes)
        results.add("Create SegmentationMetrics", True)
        print("  [PASS] SegmentationMetrics created")
    except Exception as e:
        results.add("Create SegmentationMetrics", False, str(e))
        print(f"  [FAIL] Failed to create SegmentationMetrics: {e}")
        return

    # Create Breslow metrics
    try:
        breslow_metrics = BreslowMetrics()
        results.add("Create BreslowMetrics", True)
        print("  [PASS] BreslowMetrics created")
    except Exception as e:
        results.add("Create BreslowMetrics", False, str(e))
        print(f"  [FAIL] Failed to create BreslowMetrics: {e}")

    # Update with dummy predictions
    try:
        batch_size = 4
        height, width = 64, 64

        predictions = torch.randint(0, num_classes, (batch_size, height, width))
        targets = torch.randint(0, num_classes, (batch_size, height, width))

        seg_metrics.update(predictions, targets)
        results.add("Update SegmentationMetrics", True)
        print("  [PASS] SegmentationMetrics updated")
    except Exception as e:
        results.add("Update SegmentationMetrics", False, str(e))
        print(f"  [FAIL] Failed to update SegmentationMetrics: {e}")

    # Update Breslow metrics
    try:
        pred_depths = torch.rand(batch_size) * 2000 + 500
        true_depths = torch.rand(batch_size) * 2000 + 500

        breslow_metrics.update(pred_depths.tolist(), true_depths.tolist())
        results.add("Update BreslowMetrics", True)
        print("  [PASS] BreslowMetrics updated")
    except Exception as e:
        results.add("Update BreslowMetrics", False, str(e))
        print(f"  [FAIL] Failed to update BreslowMetrics: {e}")

    # Compute segmentation metrics
    try:
        seg_results = seg_metrics.compute()

        # Check actual keys returned
        expected_keys = ['dice_per_class', 'iou_per_class', 'dice_mean', 'iou_mean', 'pixel_accuracy']
        missing_keys = [k for k in expected_keys if k not in seg_results]

        if not missing_keys:
            results.add("Compute SegmentationMetrics", True)
            print("  [PASS] SegmentationMetrics computed")
            print(f"    - Mean Dice: {seg_results['dice_mean']:.4f}")
            print(f"    - Mean IoU: {seg_results['iou_mean']:.4f}")
            print(f"    - Pixel Accuracy: {seg_results['pixel_accuracy']:.4f}")
        else:
            # Show what keys we actually got for debugging
            results.add("Compute SegmentationMetrics", False,
                        f"Missing keys: {missing_keys}. Available: {list(seg_results.keys())}")
            print(f"  [FAIL] Missing keys: {missing_keys}")
            print(f"         Available keys: {list(seg_results.keys())}")
    except Exception as e:
        results.add("Compute SegmentationMetrics", False, str(e))
        print(f"  [FAIL] Failed to compute SegmentationMetrics: {e}")

    # Compute Breslow metrics
    try:
        breslow_results = breslow_metrics.compute()

        expected_keys = ['mae', 'rmse', 't_category_accuracy']
        missing_keys = [k for k in expected_keys if k not in breslow_results]

        if not missing_keys:
            results.add("Compute BreslowMetrics", True)
            print("  [PASS] BreslowMetrics computed")
            print(f"    - MAE: {breslow_results['mae']:.2f} \u00b5m")
            print(f"    - RMSE: {breslow_results['rmse']:.2f} \u00b5m")
            print(f"    - T-category accuracy: {breslow_results['t_category_accuracy']:.2%}")
        else:
            results.add("Compute BreslowMetrics", False,
                        f"Missing keys: {missing_keys}. Available: {list(breslow_results.keys())}")
            print(f"  [FAIL] Missing keys: {missing_keys}")
    except Exception as e:
        results.add("Compute BreslowMetrics", False, str(e))
        print(f"  [FAIL] Failed to compute BreslowMetrics: {e}")


def test_visualization(results: TestResult, modules: Dict[str, Any], config: Dict[str, Any], dataset: Any):
    """Test 9: Test visualization."""
    print("\n" + "=" * 60)
    print("TEST 9: Visualization")
    print("=" * 60)

    if 'visualization' not in modules or dataset is None:
        results.add("Create visualization", False, "Visualization module or dataset not available")
        print("  [SKIP] Dependencies not available")
        return

    visualize_sample = modules['visualization']['visualize_sample']

    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_visualization.png"

    try:
        sample = dataset[0]
        import numpy as np

        image = sample['image']
        mask = sample['mask']

        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()

        # (C, H, W) -> (H, W, C)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        elif image.min() < 0:
            image = (image - image.min()) / (image.max() - image.min())

        fig = visualize_sample(
            image=image,
            mask=mask,
            title="Test Visualization",
            save_path=str(output_path),
            show=False,
        )

        results.add("Create sample visualization", True)
        print("  [PASS] Visualization created")
    except Exception as e:
        results.add("Create sample visualization", False, str(e))
        print(f"  [FAIL] Failed to create visualization: {e}")
        return

    # Verify file created
    try:
        if output_path.exists():
            file_size = output_path.stat().st_size
            results.add("Save visualization file", True)
            print(f"  [PASS] File saved: {output_path}")
            print(f"    - File size: {file_size / 1024:.1f} KB")
        else:
            results.add("Save visualization file", False, "File not created")
            print("  [FAIL] File was not created")
    except Exception as e:
        results.add("Save visualization file", False, str(e))
        print(f"  [FAIL] Failed to verify file: {e}")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("BRESLOW DEPTH PREDICTION - PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Project root: {project_root}")

    results = TestResult()

    modules = test_imports(results)
    config = test_config_loading(results, modules)
    dataset = test_dataset(results, modules, config)
    test_transforms(results, modules, config, dataset)
    test_dataloader(results, modules, config)
    test_model(results, modules, config)
    test_loss_function(results, modules, config)
    test_metrics(results, modules, config)
    test_visualization(results, modules, config, dataset)

    results.print_summary()

    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
