"""Loss functions for multi-class semantic segmentation.

`get_loss_function(config)` reads `config["loss"]["type"]` and returns the
matching module. V1/V2 use `combined` (CombinedLoss = weighted Dice + weighted CE).

Available: DiceLoss, FocalLoss, CombinedLoss (V1/V2 default), TverskyLoss, BoundaryLoss.

All losses take logits `(B, C, H, W)` and integer targets `(B, H, W)`,
return a scalar loss tensor.

Project class imbalance — drives the choice of Dice over plain CE:
    Background ~60%, Tumour ~25%, Dermis ~13%, Epidermis ~2% (the rare,
    clinically critical class for Breslow measurement).
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss = 1 - mean over classes of 2*|X∩Y| / (|X| + |Y|).

    Scale-invariant by construction -> robust to class imbalance.

    Args:
        num_classes:   Number of segmentation classes (4 for this project).
        smooth:        Numerical-stability constant.
        class_weights: Optional per-class weights (shape (C,)). V2 uses [0.05, 1.0, 3.0, 0.5].
        ignore_index:  Pixels with this label are excluded.
        reduction:     'mean', 'sum', or 'none' over the batch.

    Example:
        >>> loss_fn = DiceLoss(num_classes=4, class_weights=torch.tensor([0.1, 1.0, 1.0, 0.5]))
        >>> loss = loss_fn(predictions, targets)  # (B, 4, H, W) and (B, H, W)
    """

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-6,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction
        # Register class_weights as a buffer (moves with .to(device), no gradient).
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted Dice loss for a batch."""
        # Upcast to fp32 — spatial sums over 512x512 pixels overflow in fp16.
        predictions = predictions.float()

        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]

        probs = F.softmax(predictions, dim=1)

        # One-hot encode targets, masking out ignore_index.
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone().long()
        targets_safe[~valid_mask] = 0  # placeholder; we'll mask back out below
        targets_one_hot = F.one_hot(targets_safe, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, H, W, C) -> (B, C, H, W)
        valid_mask = valid_mask.unsqueeze(1).expand_as(targets_one_hot).float()
        targets_one_hot = targets_one_hot * valid_mask
        probs = probs * valid_mask

        # Per-class Dice over the batch.
        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets_one_hot.view(batch_size, num_classes, -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Apply class weights if provided.
        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1).expand(batch_size, -1)
            dice_per_class = dice_per_class * weights
            dice = dice_per_class.sum(dim=1) / weights.sum()
        else:
            dice = dice_per_class.mean(dim=1)

        loss = 1.0 - dice  # convert coefficient (maximise) to loss (minimise)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss (Lin 2017): FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t).

    Down-weights easy examples, focuses on hard ones. gamma=0 reduces to weighted CE.

    Args:
        alpha:         Class-balancing factor (scalar or per-class tensor).
        gamma:         Focusing parameter (typical 2.0).
        class_weights: Per-class weights (overrides alpha if set).
        ignore_index:  Pixels to exclude.
        reduction:     'mean', 'sum', 'none'.
    """

    def __init__(
        self,
        alpha: Union[float, torch.Tensor] = 1.0,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        # Resolve alpha source: class_weights > tensor alpha > scalar alpha.
        if class_weights is not None:
            self.register_buffer("alpha", class_weights.float())
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", torch.tensor(alpha).float())

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for a batch."""
        predictions = predictions.float()
        targets = targets.long()
        num_classes = predictions.shape[1]

        probs = F.softmax(predictions, dim=1)
        # CE without reduction so we can apply per-pixel focal weighting before reducing.
        ce_loss = F.cross_entropy(
            predictions, targets, weight=None, ignore_index=self.ignore_index, reduction="none",
        )

        # Extract p_t = predicted probability of the TRUE class per pixel.
        targets_for_gather = targets.clone()
        targets_for_gather[targets == self.ignore_index] = 0  # placeholder for gather
        pt = probs.gather(1, targets_for_gather.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # Per-class alpha lookup (or broadcast scalar).
        if self.alpha.numel() > 1:
            alpha_t = self.alpha.gather(0, targets_for_gather.view(-1)).view_as(targets)
            alpha_t[targets == self.ignore_index] = 0
        else:
            alpha_t = self.alpha

        focal_loss = alpha_t * focal_weight * ce_loss
        valid_mask = targets != self.ignore_index

        if self.reduction == "mean":
            return focal_loss[valid_mask].mean() if valid_mask.any() else focal_loss.sum() * 0
        elif self.reduction == "sum":
            return focal_loss[valid_mask].sum()
        return focal_loss


class CombinedLoss(nn.Module):
    """V1/V2 default: ``dice_weight * DiceLoss + ce_weight * CrossEntropyLoss``.
    V3 adds optional ``boundary_weight * BoundaryLoss`` for thin-structure boundaries.

    Dice for class-imbalance robustness, CE for early-training gradient stability,
    Boundary for fine epidermis edges. ``boundary_weight=0`` disables the boundary
    term, preserving V1/V2 behaviour exactly.

    Args:
        dice_weight, ce_weight: Component multipliers (V1: 0.5/0.5; V2: 0.6/0.4).
        boundary_weight:        V3 boundary-loss weight (V1/V2: 0; V3: typically 0.1-0.5).
        num_classes, class_weights, smooth, ignore_index: as DiceLoss.

    Example:
        >>> loss_fn = CombinedLoss(0.5, 0.4, boundary_weight=0.1,
        ...                        class_weights=torch.tensor([0.05, 1.0, 3.0, 0.5]))
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        boundary_weight: float = 0.0,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        smooth: float = 1e-6,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(
            num_classes=num_classes, smooth=smooth,
            class_weights=class_weights, ignore_index=ignore_index,
        )
        # Lazy: only build BoundaryLoss if it'll actually be used.
        if boundary_weight > 0:
            self.boundary_loss = BoundaryLoss(
                num_classes=num_classes, class_weights=class_weights,
                ignore_index=ignore_index,
            )
        else:
            self.boundary_loss = None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sum of Dice + CE (+ optional Boundary). Any term can be disabled with weight=0."""
        predictions = predictions.float()
        targets = targets.long()
        loss = 0.0
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice_loss(predictions, targets)
        if self.ce_weight > 0:
            ce = F.cross_entropy(
                predictions, targets,
                weight=self.class_weights, ignore_index=self.ignore_index,
            )
            loss = loss + self.ce_weight * ce
        if self.boundary_loss is not None and self.boundary_weight > 0:
            loss = loss + self.boundary_weight * self.boundary_loss(predictions, targets)
        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss: 1 - TP / (TP + alpha*FP + beta*FN). Generalises Dice (alpha=beta=0.5).

    alpha < beta -> higher recall (penalises FN more); alpha > beta -> higher precision.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        num_classes: int = 4,
        smooth: float = 1e-6,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute 1 - weighted_mean(Tversky_index)."""
        predictions = predictions.float()  # fp16 spatial sums overflow
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]

        probs = F.softmax(predictions, dim=1)
        valid_mask = targets != self.ignore_index
        targets_safe = targets.clone().long()
        targets_safe[~valid_mask] = 0
        targets_one_hot = F.one_hot(targets_safe, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1).expand_as(targets_one_hot).float()
        targets_one_hot = targets_one_hot * valid_mask
        probs = probs * valid_mask

        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets_one_hot.view(batch_size, num_classes, -1)

        # TP / FP / FN per (batch, class).
        tp = (probs_flat * targets_flat).sum(dim=2)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=2)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=2)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1).expand(batch_size, -1)
            tversky = tversky * weights
            tversky = tversky.sum(dim=1) / weights.sum()
        else:
            tversky = tversky.mean(dim=1)
        return (1 - tversky).mean()


class BoundaryLoss(nn.Module):
    """Multi-class boundary loss (Kervadec MIDL 2019).

    Computes per-class signed distance maps from the integer targets on the fly
    (via scipy EDT — adds ~300-500ms/step at 768², acceptable for ablation),
    then returns ``mean(class_weight * softmax(predictions) * sdt)``.

    Inside the true class, sdt is negative -> minimising the product pushes
    probability toward 1. Outside, sdt is positive -> pushes probability to 0.
    Especially helpful for thin/fragmented regions like epidermis where Dice
    saturates but boundaries remain noisy.

    Same `(predictions, targets)` interface as Dice/CE — the trainer doesn't
    need any changes.

    Args:
        num_classes:   Segmentation classes (4 here).
        class_weights: Per-class weights (typically same as DiceLoss). When
                       set, V3's [0.05, 1.0, 3.0, 0.5] keeps the 3× epidermis
                       emphasis on the boundary term too.
        ignore_index:  Pixels with this label are excluded from both the SDT
                       and the loss.
        reduction:     'mean' or 'sum'.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduction = reduction
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", None)

    @staticmethod
    def _compute_sdt(targets_np: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
        """Per-class signed Euclidean distance transform.

        Returns (B, C, H, W) float32. Negative inside the class, positive
        outside. Classes that are entirely absent (or entirely fill the image)
        contribute zeros — no gradient signal, harmless.
        """
        B, H, W = targets_np.shape
        sdt = np.zeros((B, num_classes, H, W), dtype=np.float32)
        valid = targets_np != ignore_index
        for b in range(B):
            for c in range(num_classes):
                posmask = (targets_np[b] == c) & valid[b]
                if posmask.any() and not posmask.all():
                    negmask = ~posmask
                    sdt[b, c] = (
                        ndi.distance_transform_edt(negmask)
                        - ndi.distance_transform_edt(posmask)
                    )
        return sdt

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss for a batch."""
        predictions = predictions.float()
        targets = targets.long()
        device = predictions.device

        # SDT computed on CPU via scipy — small batches (≤4) keep this affordable.
        sdt_np = self._compute_sdt(
            targets.detach().cpu().numpy(), self.num_classes, self.ignore_index,
        )
        sdt = torch.from_numpy(sdt_np).to(device)  # (B, C, H, W)

        probs = F.softmax(predictions, dim=1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
        loss_map = probs * sdt * valid_mask

        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1, 1, 1)
            loss_map = loss_map * weights

        if self.reduction == "mean":
            return loss_map.mean()
        return loss_map.sum()


class MultiTaskLoss(nn.Module):
    """Joint segmentation + depth-regression loss for the V4 multi-task model.

    Combines a `CombinedLoss` (Dice + CE [+ optional boundary]) on the predicted
    mask with an L1 loss on the predicted depth in **log-µm space**.

    Forward signature differs from the seg-only losses: takes `(outputs, targets)`
    where `outputs` is the dict produced by `MultiTaskUnetPlusPlus` and `targets`
    is a dict `{"mask": (B, H, W), "depth_um": (B,)}`. The depth target is
    log-transformed inside this module (so callers pass raw µm).

    Args:
        num_classes:     4 for this project.
        dice_weight:     Weight on the Dice term inside CombinedLoss.
        ce_weight:       Weight on the CE term inside CombinedLoss.
        boundary_weight: Weight on the boundary term inside CombinedLoss (V3-style).
        depth_weight:    Weight on the depth-regression term in the joint loss.
        class_weights:   Per-class weights for seg loss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.4,
        boundary_weight: float = 0.0,
        depth_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.seg_loss = CombinedLoss(
            num_classes=num_classes,
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            boundary_weight=boundary_weight,
            class_weights=class_weights,
        )
        self.depth_weight = float(depth_weight)
        # Cache last-computed components for optional logging by the trainer.
        self.last_components: Dict[str, float] = {}

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(outputs, dict) or "mask" not in outputs or "depth" not in outputs:
            raise TypeError(
                "MultiTaskLoss expects outputs as {'mask': ..., 'depth': ...}; "
                f"got {type(outputs).__name__}"
            )
        mask_logits = outputs["mask"]
        depth_log_pred = outputs["depth"]
        gt_mask = targets["mask"]
        gt_depth_um = targets["depth_um"].to(depth_log_pred.dtype)

        # Seg term — same loss as V1/V2/V3.
        seg_loss = self.seg_loss(mask_logits, gt_mask)

        # Depth term — predict log1p(µm), regress with L1.
        # `clamp_min(0)` guards against the (impossible) negative GT depths.
        gt_depth_log = torch.log1p(gt_depth_um.clamp_min(0.0))
        depth_loss = F.l1_loss(depth_log_pred, gt_depth_log)

        total = seg_loss + self.depth_weight * depth_loss

        # Stash for logging.
        self.last_components = {
            "seg_loss": float(seg_loss.detach().item()),
            "depth_loss": float(depth_loss.detach().item()),
            "total_loss": float(total.detach().item()),
        }
        return total


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Factory: build a loss module from a YAML config.

    Routes on `config["loss"]["type"]`. Supported: dice, ce, combined, focal, tversky.
    V1/V2 use 'combined'.

    Args:
        config: dict with `loss.{type, dice_weight, ce_weight, class_weights, ...}`
                and `classes.num_classes`.

    Returns:
        Configured loss module ready for `(logits, targets)` calls.
    """
    loss_config = config.get("loss", {})
    classes_config = config.get("classes", {})
    loss_type = loss_config.get("type", "combined").lower()
    num_classes = classes_config.get("num_classes", 4)

    class_weights = loss_config.get("class_weights")
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f"\nCreating loss function:")
    print(f"  Type: {loss_type}")
    print(f"  Num classes: {num_classes}")
    if class_weights is not None:
        print(f"  Class weights: {class_weights.tolist()}")

    if loss_type == "dice":
        loss_fn = DiceLoss(num_classes=num_classes, class_weights=class_weights)

    elif loss_type in ("ce", "crossentropy"):
        loss_fn = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None \
            else nn.CrossEntropyLoss()

    elif loss_type == "combined":
        # V1/V2 default; V3 may add boundary_weight > 0.
        dice_weight = loss_config.get("dice_weight", 0.5)
        ce_weight = loss_config.get("ce_weight", 0.5)
        boundary_weight = loss_config.get("boundary_weight", 0.0)
        print(f"  Dice weight: {dice_weight}")
        print(f"  CE weight: {ce_weight}")
        if boundary_weight > 0:
            print(f"  Boundary weight: {boundary_weight}")
        loss_fn = CombinedLoss(
            dice_weight=dice_weight, ce_weight=ce_weight, boundary_weight=boundary_weight,
            num_classes=num_classes, class_weights=class_weights,
        )

    elif loss_type == "focal":
        gamma = loss_config.get("focal_gamma", 2.0)
        alpha = loss_config.get("focal_alpha", 1.0)
        print(f"  Gamma: {gamma}")
        print(f"  Alpha: {alpha}")
        loss_fn = FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)

    elif loss_type == "multitask":
        # V4: joint segmentation + depth-regression.
        dice_weight = loss_config.get("dice_weight", 0.5)
        ce_weight = loss_config.get("ce_weight", 0.4)
        boundary_weight = loss_config.get("boundary_weight", 0.0)
        depth_weight = loss_config.get("depth_weight", 0.1)
        print(f"  Dice weight: {dice_weight}")
        print(f"  CE weight: {ce_weight}")
        if boundary_weight > 0:
            print(f"  Boundary weight: {boundary_weight}")
        print(f"  Depth weight: {depth_weight}")
        loss_fn = MultiTaskLoss(
            num_classes=num_classes,
            dice_weight=dice_weight, ce_weight=ce_weight,
            boundary_weight=boundary_weight, depth_weight=depth_weight,
            class_weights=class_weights,
        )

    elif loss_type == "tversky":
        alpha = loss_config.get("tversky_alpha", 0.5)
        beta = loss_config.get("tversky_beta", 0.5)
        print(f"  Alpha (FP weight): {alpha}")
        print(f"  Beta (FN weight): {beta}")
        loss_fn = TverskyLoss(
            alpha=alpha, beta=beta,
            num_classes=num_classes, class_weights=class_weights,
        )

    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Available: dice, ce, combined, focal, tversky, multitask"
        )

    return loss_fn


# === Smoke test ==========================================================

if __name__ == "__main__":
    print("Loss Function Examples")
    print("=" * 60)

    batch_size = 4
    num_classes = 4
    height, width = 256, 256
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    class_weights = torch.tensor([0.1, 1.0, 1.0, 0.5])

    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")

    print("\n1. Dice Loss:")
    loss = DiceLoss(num_classes=num_classes, class_weights=class_weights)(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")

    print("\n2. Focal Loss (gamma=2):")
    loss = FocalLoss(gamma=2.0, class_weights=class_weights)(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")

    print("\n3. Combined Dice + CE Loss:")
    combined_loss = CombinedLoss(
        dice_weight=0.5, ce_weight=0.5, num_classes=num_classes, class_weights=class_weights,
    )
    loss = combined_loss(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")

    print("\n4. Tversky Loss (recall-focused, alpha=0.3, beta=0.7):")
    loss = TverskyLoss(
        alpha=0.3, beta=0.7, num_classes=num_classes, class_weights=class_weights,
    )(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")

    print("\n5. Using get_loss_function:")
    config = {
        "loss": {
            "type": "combined", "dice_weight": 0.5, "ce_weight": 0.5,
            "class_weights": [0.1, 1.0, 1.0, 0.5],
        },
        "classes": {"num_classes": 4},
    }
    loss = get_loss_function(config)(predictions, targets)
    print(f"   Loss value: {loss.item():.4f}")

    print("\n6. Gradient check:")
    predictions.requires_grad = True
    loss = combined_loss(predictions, targets)
    loss.backward()
    print(f"   Gradient computed successfully")
    print(f"   Gradient shape: {predictions.grad.shape}")
    print(f"   Gradient mean: {predictions.grad.mean().item():.6f}")

    print("\n" + "=" * 60)
    print("All examples completed!")
