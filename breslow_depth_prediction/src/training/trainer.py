"""Training orchestrator: encapsulates the epoch loop, validation, checkpointing, early stopping.

`Trainer.fit(...)` is the entry point called by `train.py`. Tracks per-epoch
history (loss / Dice / IoU / lr) for the training-history plot, and saves the
best checkpoint by validation Dice. Handles mixed precision (disabled in V1/V2
because fp16 spatial sums in the Dice loss overflow on a randomly-init decoder).
"""

import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_metrics


def _unpack_batch(batch, device):
    """Pull (images, masks) out of a batch, accepting dict or tuple format."""
    if isinstance(batch, dict):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
    else:
        images, masks = batch[0], batch[1]
        images = images.to(device)
        masks = masks.to(device)
    return images, masks


class Trainer:
    """Training loop wrapper for segmentation models.

    After construction, call `.fit(train_loader, val_loader, epochs)`. Returns
    a history dict consumed by `plot_training_history`.

    Args:
        model:           Segmentation model.
        criterion:       Loss function with signature (logits, targets).
        optimizer:       Usually AdamW from `train.py`.
        device:          'cuda' or 'cpu'.
        scheduler:       Optional LR scheduler (handled type-generically).
        checkpoint_dir:  Where to save best_model.pth.
        mixed_precision: Enable fp16 autocast + scaling. Disabled for V1/V2.
        num_classes:     4 for this project.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: str = "cuda",
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_dir: str = "checkpoints",
        mixed_precision: bool = True,
        num_classes: int = 4,
    ):
        self.model = model.to(device)
        # `criterion.to(device)` matters when the loss has registered class_weights buffer.
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision only on CUDA.
        self.mixed_precision = mixed_precision and device == "cuda"
        # GradScaler prevents fp16 gradient underflow.
        self.scaler = torch.amp.GradScaler("cuda") if self.mixed_precision else None

        # Per-epoch history. `plot_training_history` consumes these lists.
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "train_iou": [], "val_iou": [],
            "learning_rate": [],
        }
        self.best_val_loss = float("inf")
        self.best_val_dice = 0.0  # primary signal — matches the headline metric

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """One training epoch: forward, loss, backward, optimiser step. Returns mean metrics."""
        self.model.train()  # enables dropout, BN updates
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images, masks = _unpack_batch(batch, self.device)

            self.optimizer.zero_grad()  # PyTorch accumulates gradients by default

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                # Loss in fp32 — Dice spatial sums overflow in fp16 with random decoder init.
                loss = self.criterion(outputs, masks)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

            # Per-batch metrics (no_grad — these don't go into the training graph).
            with torch.no_grad():
                batch_metrics = compute_all_metrics(
                    outputs, masks, num_classes=self.num_classes,
                )
                dice = batch_metrics["dice_mean"]
                iou = batch_metrics["iou_mean"]

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.4f}"})

        return {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """One validation pass. Same logic as train_epoch minus the backward step."""
        self.model.eval()  # disables dropout; BN uses running stats
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images, masks = _unpack_batch(batch, self.device)

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            batch_metrics = compute_all_metrics(
                outputs, masks, num_classes=self.num_classes,
            )

            total_loss += loss.item()
            total_dice += batch_metrics["dice_mean"]
            total_iou += batch_metrics["iou_mean"]
            total_accuracy += batch_metrics["pixel_accuracy"]

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_metrics['dice_mean']:.4f}"})

        return {
            "loss": total_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_accuracy / num_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
    ) -> Dict[str, list]:
        """Run the full epoch loop. Saves best checkpoint by val Dice, applies early stopping.

        Args:
            train_loader, val_loader: DataLoaders.
            epochs:                   Max epochs.
            early_stopping_patience:  Epochs without val Dice improvement before bail.
            save_best_only:           True saves only best_model.pth; False saves every epoch.

        Returns:
            self.history (also stored on the instance).
        """
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # Train pass.
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])

            # Val pass.
            val_metrics = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["learning_rate"].append(current_lr)

            # LR scheduler step. ReduceLROnPlateau needs the metric; others don't.
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Epoch summary -> stdout (NOT logger; tqdm already clutters that channel).
            print(f"Train Loss: {train_metrics['loss']:.4f} | Train Dice: {train_metrics['dice']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")

            # Best-model check. Val Dice is the signal (matches Research Article).
            is_best = val_metrics["dice"] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics["dice"]
                self.best_val_loss = val_metrics["loss"]
                patience_counter = 0
                print(f"New best model! Dice: {self.best_val_dice:.4f}")
            else:
                patience_counter += 1

            if save_best_only:
                if is_best:
                    self.save_checkpoint(epoch, "best_model.pth")
            else:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}.pth")

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        return self.history

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        """Serialise model + optimiser + scheduler + scaler + history (everything resume needs)."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_dice": self.best_val_dice,
            "history": self.history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str) -> int:
        """Restore everything `save_checkpoint` saved. Returns the epoch the checkpoint was saved at."""
        # weights_only=False because our checkpoint contains optimiser state,
        # scheduler state, history dict, etc. — not just tensor weights.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # `.get(..., default)` tolerates checkpoints from older trainer versions.
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
        self.history = checkpoint.get("history", self.history)
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint["epoch"]
