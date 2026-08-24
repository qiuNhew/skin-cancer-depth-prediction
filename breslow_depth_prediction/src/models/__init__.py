"""Model architectures and loss functions."""

from .unet import (
    # Custom architectures
    UNet,
    UNetEncoder,
    UNetDecoder,
    AttentionUNet,
    AttentionBlock,
    ConvBlock,
    # SMP integration
    get_model,
    get_model_summary,
    get_device,
    freeze_encoder,
    unfreeze_encoder,
    load_model_checkpoint,
    # Constants
    ARCHITECTURE_MAP,
    ENCODER_OPTIONS,
)
from .losses import (
    DiceLoss,
    FocalLoss,
    CombinedLoss,
    TverskyLoss,
    BoundaryLoss,
    get_loss_function,
)

__all__ = [
    # Custom architectures
    "UNet",
    "UNetEncoder",
    "UNetDecoder",
    "AttentionUNet",
    "AttentionBlock",
    "ConvBlock",
    # SMP integration functions
    "get_model",
    "get_model_summary",
    "get_device",
    "freeze_encoder",
    "unfreeze_encoder",
    "load_model_checkpoint",
    # Constants
    "ARCHITECTURE_MAP",
    "ENCODER_OPTIONS",
    # Loss functions
    "DiceLoss",
    "FocalLoss",
    "CombinedLoss",
    "TverskyLoss",
    "BoundaryLoss",
    "get_loss_function",
]
