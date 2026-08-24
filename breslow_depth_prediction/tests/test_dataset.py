"""Unit tests for dataset and data utilities."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import BreslowDataset, BreslowInferenceDataset
from data.transforms import get_train_transforms, get_val_transforms
from data.utils import load_image, save_mask, split_dataset


class TestBreslowDataset:
    """Tests for BreslowDataset class."""

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create a temporary dataset for testing."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Create sample images and masks
        for i in range(5):
            # Create random image
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(img).save(image_dir / f"sample_{i}.png")

            # Create random binary mask
            mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
            Image.fromarray(mask).save(mask_dir / f"sample_{i}.png")

        return image_dir, mask_dir

    def test_dataset_creation(self, temp_dataset):
        """Test that dataset can be created successfully."""
        image_dir, mask_dir = temp_dataset
        dataset = BreslowDataset(str(image_dir), str(mask_dir))
        assert len(dataset) == 5

    def test_dataset_getitem(self, temp_dataset):
        """Test that dataset returns correct items."""
        image_dir, mask_dir = temp_dataset
        dataset = BreslowDataset(str(image_dir), str(mask_dir))

        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels
        assert len(mask.shape) == 2  # 2D mask

    def test_dataset_with_transforms(self, temp_dataset):
        """Test dataset with data augmentation transforms."""
        image_dir, mask_dir = temp_dataset
        transform = get_train_transforms(image_size=(128, 128))

        dataset = BreslowDataset(
            str(image_dir), str(mask_dir), transform=transform
        )

        image, mask = dataset[0]

        assert image.shape == (3, 128, 128)
        assert mask.shape == (128, 128)

    def test_missing_mask_raises_error(self, tmp_path):
        """Test that missing mask raises FileNotFoundError."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Create image without corresponding mask
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(image_dir / "orphan.png")

        with pytest.raises(FileNotFoundError):
            BreslowDataset(str(image_dir), str(mask_dir))


class TestInferenceDataset:
    """Tests for BreslowInferenceDataset class."""

    @pytest.fixture
    def temp_images(self, tmp_path):
        """Create temporary images for testing."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        for i in range(3):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(img).save(image_dir / f"test_{i}.png")

        return image_dir

    def test_inference_dataset(self, temp_images):
        """Test inference dataset returns image and filename."""
        dataset = BreslowInferenceDataset(str(temp_images))
        assert len(dataset) == 3

        image, filename = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(filename, str)
        assert filename.endswith(".png")


class TestTransforms:
    """Tests for data transforms."""

    def test_train_transforms_output_shape(self):
        """Test that train transforms produce correct output shape."""
        transform = get_train_transforms(image_size=(256, 256))

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

        result = transform(image=image, mask=mask)

        assert result["image"].shape == (3, 256, 256)
        assert result["mask"].shape == (256, 256)

    def test_val_transforms_deterministic(self):
        """Test that validation transforms are deterministic."""
        transform = get_val_transforms(image_size=(256, 256))

        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

        result1 = transform(image=image, mask=mask)
        result2 = transform(image=image, mask=mask)

        assert torch.allclose(result1["image"], result2["image"])
        assert torch.equal(result1["mask"], result2["mask"])


class TestDataUtils:
    """Tests for data utility functions."""

    def test_load_image(self, tmp_path):
        """Test image loading utility."""
        img_path = tmp_path / "test.png"
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(original).save(img_path)

        loaded = load_image(str(img_path))

        assert loaded.shape == (100, 100, 3)
        assert np.array_equal(loaded, original)

    def test_save_mask(self, tmp_path):
        """Test mask saving utility."""
        mask_path = tmp_path / "mask.png"
        mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255

        save_mask(mask, str(mask_path))

        assert mask_path.exists()
        loaded = np.array(Image.open(mask_path))
        assert loaded.shape == (100, 100)

    def test_split_dataset(self, tmp_path):
        """Test dataset splitting utility."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        # Create 10 sample images
        for i in range(10):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            Image.fromarray(img).save(image_dir / f"img_{i}.png")

        train, val, test = split_dataset(
            str(image_dir),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2

        # Check no overlap
        all_files = set(train + val + test)
        assert len(all_files) == 10


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_dice_coefficient(self):
        """Test Dice coefficient calculation."""
        from training.metrics import dice_coefficient

        # Perfect prediction
        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 10, 10)
        dice = dice_coefficient(pred, target)
        assert abs(dice.item() - 1.0) < 0.01

        # No overlap
        pred = torch.ones(1, 1, 10, 10)
        target = torch.zeros(1, 10, 10)
        dice = dice_coefficient(pred, target)
        assert dice.item() < 0.01

    def test_iou_score(self):
        """Test IoU score calculation."""
        from training.metrics import iou_score

        # Perfect prediction
        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 10, 10)
        iou = iou_score(pred, target)
        assert abs(iou.item() - 1.0) < 0.01

    def test_pixel_accuracy(self):
        """Test pixel accuracy calculation."""
        from training.metrics import pixel_accuracy

        pred = torch.ones(1, 1, 10, 10)
        target = torch.ones(1, 10, 10)
        acc = pixel_accuracy(pred, target)
        assert abs(acc.item() - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
