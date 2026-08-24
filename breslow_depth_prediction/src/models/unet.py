"""Model architectures: custom U-Net / AttentionUNet, SMP factory, and utilities.

The V1/V2 pipeline uses `get_model(config)` which builds an SMP UnetPlusPlus
with an EfficientNet-B4 encoder. The custom U-Net classes are kept as
reference implementations.

Supported SMP architectures: Unet, UnetPlusPlus (V1/V2 default), DeepLabV3Plus,
FPN, PSPNet, PAN, LinkNet, MAnet.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# SMP is required for V1/V2 but optional at import time (custom classes still work without it).
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation-models-pytorch not installed. "
          "Install with: pip install segmentation-models-pytorch")


# === Custom U-Net building blocks ========================================
# Reference implementations — V1/V2 use SMP via `get_model`.

class ConvBlock(nn.Module):
    """Double-conv block (conv -> BN -> ReLU -> conv -> BN -> ReLU). U-Net's basic unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """Contracting path. Each level: ConvBlock + MaxPool. Saves pre-pool features as skips."""

    def __init__(self, in_channels: int = 3, features: List[int] = [64, 128, 256, 512],
                 use_batchnorm: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.encoders.append(ConvBlock(in_channels, feature, use_batchnorm=use_batchnorm))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns (skip_connections shallow->deep, last_pooled_feature_map)."""
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)  # save before pooling
            x = pool(x)
        return skip_connections, x


class UNetDecoder(nn.Module):
    """Expanding path. Each level: upsample, concat with skip, ConvBlock."""

    def __init__(self, features: List[int] = [512, 256, 128, 64], use_batchnorm: bool = True):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1):
            self.upconvs.append(
                nn.ConvTranspose2d(features[i], features[i + 1], kernel_size=2, stride=2)
            )
            self.decoders.append(
                ConvBlock(features[i], features[i + 1], use_batchnorm=use_batchnorm)
            )

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        """Reverses skip order to deep->shallow, then applies upconv+concat+conv per level."""
        skip_connections = skip_connections[::-1]
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]
            # Align spatial dims if odd input dimensions caused size mismatch.
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        return x


class UNet(nn.Module):
    """Vanilla U-Net (Ronneberger 2015): encoder + bottleneck + decoder + 1x1 conv."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: List[int] = [64, 128, 256, 512], use_batchnorm: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, features, use_batchnorm)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, use_batchnorm=use_batchnorm)
        decoder_features = [features[-1] * 2] + features[::-1]
        self.decoder = UNetDecoder(decoder_features, use_batchnorm)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, out_channels, H, W) raw logits."""
        skip_connections, x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        return self.final_conv(x)


class AttentionBlock(nn.Module):
    """Attention gate (Oktay 2018). Down-weights skip-connection regions deemed irrelevant."""

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_g(gate)
        x = self.W_x(skip)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        attention = self.sigmoid(self.psi(self.relu(g + x)))
        return skip * attention


class AttentionUNet(nn.Module):
    """U-Net with attention gates on every skip connection."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 features: List[int] = [64, 128, 256, 512], use_batchnorm: bool = True):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, features, use_batchnorm)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, use_batchnorm=use_batchnorm)
        self.attention_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_features = [features[-1] * 2] + features[::-1]

        for i in range(len(features)):
            self.upconvs.append(
                nn.ConvTranspose2d(decoder_features[i], decoder_features[i + 1],
                                   kernel_size=2, stride=2)
            )
            # inter_channels = skip_channels // 2 (standard recipe from the paper)
            self.attention_blocks.append(
                AttentionBlock(decoder_features[i + 1], decoder_features[i + 1],
                               decoder_features[i + 1] // 2)
            )
            self.decoders.append(
                ConvBlock(decoder_features[i], decoder_features[i + 1],
                          use_batchnorm=use_batchnorm)
            )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections, x = self.encoder(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i, (upconv, attention, decoder) in enumerate(
            zip(self.upconvs, self.attention_blocks, self.decoders)
        ):
            x = upconv(x)
            skip = attention(x, skip_connections[i])
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        return self.final_conv(x)


# === SMP integration =====================================================
# V1/V2 path: `get_model(config)` returns an SMP UnetPlusPlus with EfficientNet-B4.

# Maps config "architecture" string -> SMP class name.
ARCHITECTURE_MAP = {
    "Unet": "Unet",
    "UnetPlusPlus": "UnetPlusPlus",     # V1/V2 default
    "DeepLabV3Plus": "DeepLabV3Plus",
    "FPN": "FPN",
    "PSPNet": "PSPNet",
    "PAN": "PAN",
    "LinkNet": "Linknet",
    "MAnet": "MAnet",
}

# Multi-task architecture (V4): SMP backbone + parallel regression head.
MULTITASK_ARCHITECTURE = "MultiTaskUnetPlusPlus"


class MultiTaskUnetPlusPlus(nn.Module):
    """UnetPlusPlus + a parallel depth-regression head sharing the encoder.

    Forward returns a dict `{"mask": (B, C, H, W) logits, "depth": (B,) log-depth}`.
    The depth output is in **log-µm space** (log1p(depth_um)) — chosen because the
    target Breslow depth ranges 50–50 000 µm and a linear regressor would collapse
    to predicting the dataset mean. At inference, recover µm via `expm1(pred)`.

    Architecturally identical to the V1/V2/V3 UnetPlusPlus for the segmentation
    head; the regression head is a small MLP on the encoder bottleneck.

    Args:
        encoder_name:    SMP encoder backbone (default: efficientnet-b4).
        encoder_weights: Pretrained weights (default: imagenet).
        in_channels:     Input channels (3 for RGB).
        classes:         Segmentation classes (4 for this project).
        head_hidden:     Hidden width of the regression MLP.
        head_dropout:    Dropout in the regression MLP.
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 4,
        head_hidden: int = 256,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is required for MultiTaskUnetPlusPlus."
            )
        # Standard SMP UNet++. Identical seg head behaviour to V3.
        self.seg_net = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        # Capture encoder feature list via forward hook so we can run the
        # regression head off the deepest features without touching SMP's
        # internal API or running the encoder twice.
        self._encoder_features: Optional[List[torch.Tensor]] = None
        self.seg_net.encoder.register_forward_hook(self._encoder_hook)

        # SMP encoders expose `.out_channels` as a list of channel counts at
        # each scale; the last entry is the bottleneck.
        bottleneck_channels = int(self.seg_net.encoder.out_channels[-1])

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encoder_hook(self, _module, _inp, output):
        # SMP's encoder forward returns a list of feature tensors at increasing depth.
        self._encoder_features = output

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns `{"mask": (B, C, H, W), "depth": (B,) log-µm}`."""
        mask_logits = self.seg_net(x)
        bottleneck = self._encoder_features[-1]  # (B, bottleneck_channels, H/32, W/32)
        depth_log = self.regression_head(bottleneck).squeeze(-1)  # (B,)
        return {"mask": mask_logits, "depth": depth_log}

# Available encoder backbones via SMP (V1/V2 use efficientnet-b4).
ENCODER_OPTIONS = [
    # EfficientNet — V1/V2 use 'efficientnet-b4'
    "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3",
    "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7",
    # ResNet
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    # ResNeXt
    "resnext50_32x4d", "resnext101_32x8d",
    # SE-ResNet
    "se_resnet50", "se_resnet101", "se_resnet152",
    # DenseNet
    "densenet121", "densenet169", "densenet201",
    # VGG
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    # MobileNet (lightweight)
    "mobilenet_v2", "timm-mobilenetv3_large_100",
]


def get_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(config: Dict[str, Any]) -> nn.Module:
    """Factory: build a segmentation model from a YAML config dict.

    V1/V2 use architecture='UnetPlusPlus', encoder='efficientnet-b4',
    encoder_weights='imagenet', in_channels=3, classes=4.

    Args:
        config: YAML config with `model.{architecture, encoder, encoder_weights,
                in_channels}` and `classes.num_classes`.

    Returns:
        Constructed model on the best available device.

    Raises:
        ValueError: unknown architecture.
        ImportError: SMP requested but not installed.
    """
    model_config = config.get("model", {})
    classes_config = config.get("classes", {})

    architecture = model_config.get("architecture", "UnetPlusPlus")
    encoder_name = model_config.get("encoder", "efficientnet-b4")
    encoder_weights = model_config.get("encoder_weights", "imagenet")
    in_channels = model_config.get("in_channels", 3)
    num_classes = model_config.get("classes", classes_config.get("num_classes", 4))

    device = get_device()
    print(f"\nCreating model:")
    print(f"  Architecture: {architecture}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Encoder weights: {encoder_weights}")
    print(f"  In channels: {in_channels}")
    print(f"  Num classes: {num_classes}")
    print(f"  Device: {device}")

    if architecture.lower() == "custom_unet":
        model = UNet(in_channels=in_channels, out_channels=num_classes,
                     features=[64, 128, 256, 512], use_batchnorm=True)
        print("  Using custom UNet implementation")

    elif architecture.lower() == "attention_unet":
        model = AttentionUNet(in_channels=in_channels, out_channels=num_classes,
                              features=[64, 128, 256, 512], use_batchnorm=True)
        print("  Using custom AttentionUNet implementation")

    elif architecture == MULTITASK_ARCHITECTURE:
        # V4: shared encoder + parallel mask & depth heads.
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is required for MultiTaskUnetPlusPlus. "
                "Install with: pip install segmentation-models-pytorch"
            )
        head_hidden = model_config.get("head_hidden", 256)
        head_dropout = model_config.get("head_dropout", 0.3)
        model = MultiTaskUnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            head_hidden=head_hidden,
            head_dropout=head_dropout,
        )
        print(f"  Using MultiTaskUnetPlusPlus (head_hidden={head_hidden}, dropout={head_dropout})")

    elif architecture in ARCHITECTURE_MAP:
        # SMP path — V1/V2 default.
        if not SMP_AVAILABLE:
            raise ImportError(
                "segmentation-models-pytorch is required for this architecture. "
                "Install with: pip install segmentation-models-pytorch"
            )
        smp_arch = ARCHITECTURE_MAP[architecture]
        model_class = getattr(smp, smp_arch)
        model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        print(f"  Using smp.{smp_arch}")

    else:
        available = list(ARCHITECTURE_MAP.keys()) + ["custom_unet", "attention_unet"]
        raise ValueError(f"Unknown architecture: {architecture}. Available options: {available}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    return model


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 512, 512),
    verbose: bool = True,
) -> Dict[str, Any]:
    """Parameter counts + layer breakdown for the SDR / training log.

    Args:
        model: PyTorch model.
        input_size: (C, H, W) for the forward pass / torchinfo.
        verbose: Print summary to stdout.

    Returns:
        Dict with total/trainable/non_trainable params, estimated_size_mb, layer_counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    param_size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes

    summary_dict = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "input_size": input_size,
        "estimated_size_mb": param_size_mb,
        "layer_info": [],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Input size: {input_size}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        print(f"Estimated size: {param_size_mb:.2f} MB")
        print("=" * 60)

    # Optional torchinfo per-layer breakdown.
    try:
        from torchinfo import summary as torchinfo_summary
        device = next(model.parameters()).device
        full_input_size = (1,) + input_size
        if verbose:
            print("\nDetailed Layer Information:")
            torchinfo_summary(
                model, input_size=full_input_size, device=device,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20, row_settings=["var_names"],
            )
    except ImportError:
        if verbose:
            print("\nNote: Install torchinfo for detailed layer summary: pip install torchinfo")

    # Layer-type histogram.
    layer_types = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        layer_types[module_type] = layer_types.get(module_type, 0) + 1
    summary_dict["layer_counts"] = layer_types

    if verbose:
        print("\nLayer Type Counts:")
        for layer_type, count in sorted(layer_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {layer_type}: {count}")

    return summary_dict


def freeze_encoder(model: nn.Module) -> nn.Module:
    """Freeze encoder params (sets requires_grad=False) for two-stage transfer learning.

    Not currently used by V1/V2 (both train end-to-end) but available for ablation.
    """
    frozen_count = 0
    if hasattr(model, "encoder"):
        # SMP and custom models both expose `.encoder`.
        for param in model.encoder.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        print(f"Frozen encoder: {frozen_count:,} parameters")
    else:
        # Fallback: match by parameter name.
        for name, param in model.named_parameters():
            if "encoder" in name.lower() or "down" in name.lower():
                param.requires_grad = False
                frozen_count += param.numel()
        if frozen_count > 0:
            print(f"Frozen encoder layers: {frozen_count:,} parameters")
        else:
            print("Warning: Could not identify encoder to freeze")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params:,} / {total_params:,}")
    return model


def unfreeze_encoder(model: nn.Module) -> nn.Module:
    """Inverse of `freeze_encoder` — set requires_grad=True on encoder params."""
    unfrozen_count = 0
    if hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        print(f"Unfrozen encoder: {unfrozen_count:,} parameters")
    else:
        for name, param in model.named_parameters():
            if "encoder" in name.lower() or "down" in name.lower():
                param.requires_grad = True
                unfrozen_count += param.numel()
        if unfrozen_count > 0:
            print(f"Unfrozen encoder layers: {unfrozen_count:,} parameters")
        else:
            print("Warning: Could not identify encoder to unfreeze")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_params:,} / {total_params:,}")
    return model


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """Load weights from a .pth file. Handles 3 on-disk formats (model_state_dict /
    state_dict / raw)."""
    if device is None:
        device = get_device()

    # weights_only=False because our checkpoints contain optimiser state too.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # raw state dict

    model.load_state_dict(state_dict, strict=strict)
    model = model.to(device)
    print(f"Loaded model checkpoint from: {checkpoint_path}")
    return model


# === Smoke test ==========================================================

if __name__ == "__main__":
    print("Model Architecture Examples")
    print("=" * 60)

    print("\n1. Custom UNet:")
    custom_unet = UNet(in_channels=3, out_channels=4)
    x = torch.randn(1, 3, 256, 256)
    out = custom_unet(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    print("\n2. Attention UNet:")
    attention_unet = AttentionUNet(in_channels=3, out_channels=4)
    out = attention_unet(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    if SMP_AVAILABLE:
        print("\n3. SMP UNet++ with EfficientNet-B4 encoder:")
        config = {
            "model": {
                "architecture": "UnetPlusPlus",
                "encoder": "efficientnet-b4",
                "encoder_weights": "imagenet",
                "in_channels": 3,
            },
            "classes": {"num_classes": 4},
        }
        model = get_model(config)
        device = get_device()
        x = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {out.shape}")

        print("\n4. Model Summary:")
        summary = get_model_summary(model, input_size=(3, 512, 512), verbose=True)

        print("\n5. Freeze/Unfreeze Encoder:")
        model = freeze_encoder(model)
        model = unfreeze_encoder(model)

        print("\n6. Available architectures:")
        for arch in ARCHITECTURE_MAP.keys():
            print(f"   - {arch}")
    else:
        print("\n3-6. Skipped (segmentation-models-pytorch not installed)")

    print("\n" + "=" * 60)
    print("Examples completed!")
