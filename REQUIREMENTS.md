# Requirements

This document covers the hardware and software environment needed to run this project end-to-end (training, evaluation, and inference).

## Hardware

### Minimum (inference / evaluation of pre-trained checkpoints)

| Component | Minimum |
|---|---|
| CPU | Any x86-64 with AVX2 (Intel Core i5 / AMD Ryzen 5 class or newer) |
| RAM | 8 GB |
| GPU | Not required — inference runs on CPU in ~30 seconds per image |
| Disk | 5 GB free (for venv + checkpoints + results; data is external) |

### Recommended (training from scratch)

| Component | Recommended | Actually used in this project |
|---|---|---|
| CPU | 8-core x86-64 | Intel / AMD 8-core+ |
| RAM | 16 GB | 16 GB |
| GPU | NVIDIA CUDA GPU with ≥ 10 GB VRAM | NVIDIA GeForce RTX 3080 (10 GB VRAM) |
| Disk | 15 GB free | — |

A training run at 768 × 768 (V2 config) uses ~9.96 GB / 10.24 GB VRAM on the RTX 3080. Smaller GPUs must drop to 512 × 512 (V1 config) or reduce batch size.

**Non-commodity peripherals:** none. Whole-slide imagery is pre-rendered to PNG before entering this pipeline.

## Software

### Operating system

Developed and tested on:
- Windows 11 Pro (primary development environment)
- Should work on Linux and macOS without changes (uses only cross-platform Python libraries); install steps in `INSTALL.md` have PowerShell variants — bash equivalents are one-line substitutions (`source .venv/bin/activate` instead of `.\.venv\Scripts\activate`, etc.)

### CUDA / drivers (GPU training only)

- NVIDIA driver 535 or newer
- CUDA 11.8 or 12.1 (PyTorch's cu118 or cu121 wheels)
- Verified with: driver 551.78, CUDA 12.1

### Python

- **Python 3.10** (required — not 3.11+, not 3.9).
  PyTorch wheels, segmentation-models-pytorch, and some pinned transitive deps are tied to Python 3.10 at the versions listed below.

### Python packages

Version-pinned requirements live in [`breslow_depth_prediction/requirements.txt`](breslow_depth_prediction/requirements.txt). Headline dependencies:

| Package | Version | Role |
|---|---|---|
| `torch` | ≥ 2.0 | Deep-learning framework |
| `torchvision` | ≥ 0.15 | Pretrained encoders |
| `segmentation-models-pytorch` | ≥ 0.3.3 | UNet++ + EfficientNet-B4 |
| `albumentations` | ≥ 1.3.1 | Data augmentation |
| `opencv-python` | ≥ 4.8 | Image I/O and geometric ops |
| `Pillow` | ≥ 10.0 | PNG I/O |
| `numpy` / `pandas` / `scikit-learn` | modern | Data handling |
| `matplotlib` / `seaborn` | modern | Plots for the Research Article |
| `tqdm` / `PyYAML` / `tensorboard` | modern | Utilities, config loading, training logs |
| `scipy` | ≥ 1.10 | Distance transforms (inference) |
| `pypdf` | modern | Handbook / template parsing (dev only) |

Install instructions are in `INSTALL.md`.

## Data

The patient histopathology dataset (whole-slide images + masks + Breslow ground-truth CSV) is **not redistributed** with this repository. See `REPLICATION.md` for data access terms, provenance, and ethics.

Expected layout if you have access:

```
<project_root>/data/
├── ds_WSIs/                              # 89 image/mask PNG pairs (~800 MB)
│   ├── <sample_id>_image.png
│   └── <sample_id>_labels.png
└── measurements/
    ├── breslow_depth_coords_210725.csv   # Ground-truth Breslow depths (µm), current canonical
    └── wrong_dir_coords.txt              # 10 samples to exclude
```

10 samples are excluded by `wrong_dir_coords.txt`, leaving 75 valid samples (stratified 52 train / 11 val / 12 test by T-category, seed 42).

## Environment file

A `.env` file is not required. All configuration is in YAML under `breslow_depth_prediction/configs/` — one file per experimental run (V1, V2, V3, V4, V5). V5 is the canonical headline configuration.
