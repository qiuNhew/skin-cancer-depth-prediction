"""Trainer for the V4 multi-task model.

Subclasses `Trainer` and overrides only the per-batch logic so the
multi-task model's dict outputs and the depth target are handled correctly.
The fit/checkpoint/early-stopping plumbing is inherited unchanged — same
behaviour, same checkpoint format, same `history` dict shape.
"""

from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_all_metrics
from .trainer import Trainer, _unpack_batch


def _unpack_multitask_batch(batch, device):
    """Pull (images, masks, depths) out of a batch dict from BreslowDataset."""
    if not isinstance(batch, dict):
        raise TypeError(
            "MultiTaskTrainer requires dict-format batches "
            "(BreslowDataset returns dicts by default)."
        )
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)
    # `breslow_depth_um` is a float on each sample; default_collate stacks them
    # into a (B,) tensor.
    depths = batch["breslow_depth_um"].to(device).float()
    return images, masks, depths


class MultiTaskTrainer(Trainer):
    """Trainer for `MultiTaskUnetPlusPlus`.

    The model outputs `{"mask": logits, "depth": log_um}`; the loss expects
    `(outputs_dict, {"mask": gt_mask, "depth_um": gt_depth_um})`. Segmentation
    metrics use only the `mask` head, so Dice/IoU stay comparable to V1/V2/V3.

    Side effect: the per-batch `seg_loss` / `depth_loss` components are
    captured from the loss module's `last_components` and printed via tqdm.
    """

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_depth_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images, masks, depths = _unpack_multitask_batch(batch, self.device)
            targets = {"mask": masks, "depth_um": depths}

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                # Compute loss in fp32 — same rationale as the seg-only Trainer.
                loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                # Seg metrics use only the mask head.
                batch_metrics = compute_all_metrics(
                    outputs["mask"], masks, num_classes=self.num_classes,
                )
                dice = batch_metrics["dice_mean"]
                iou = batch_metrics["iou_mean"]

            total_loss += loss.item()
            seg_l = self.criterion.last_components.get("seg_loss", 0.0)
            depth_l = self.criterion.last_components.get("depth_loss", 0.0)
            total_seg_loss += seg_l
            total_depth_loss += depth_l
            total_dice += dice
            total_iou += iou

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{seg_l:.4f}",
                "depth": f"{depth_l:.4f}",
                "dice": f"{dice:.4f}",
            })

        return {
            "loss": total_loss / num_batches,
            "seg_loss": total_seg_loss / num_batches,
            "depth_loss": total_depth_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_depth_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images, masks, depths = _unpack_multitask_batch(batch, self.device)
            targets = {"mask": masks, "depth_um": depths}

            if self.mixed_precision:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            batch_metrics = compute_all_metrics(
                outputs["mask"], masks, num_classes=self.num_classes,
            )

            total_loss += loss.item()
            total_seg_loss += self.criterion.last_components.get("seg_loss", 0.0)
            total_depth_loss += self.criterion.last_components.get("depth_loss", 0.0)
            total_dice += batch_metrics["dice_mean"]
            total_iou += batch_metrics["iou_mean"]
            total_accuracy += batch_metrics["pixel_accuracy"]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{batch_metrics['dice_mean']:.4f}",
            })

        return {
            "loss": total_loss / num_batches,
            "seg_loss": total_seg_loss / num_batches,
            "depth_loss": total_depth_loss / num_batches,
            "dice": total_dice / num_batches,
            "iou": total_iou / num_batches,
            "accuracy": total_accuracy / num_batches,
        }
