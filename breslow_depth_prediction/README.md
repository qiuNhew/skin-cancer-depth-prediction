# Breslow Depth Prediction

A deep learning pipeline for predicting melanoma **Breslow depth** from histopathology whole-slide images (WSIs) using semantic segmentation followed by a geometric perpendicular-distance calculation.

---

## Table of Contents

1. [Clinical Context](#1-clinical-context)
2. [How It Works — The Full Pipeline](#2-how-it-works--the-full-pipeline)
3. [Project Structure](#3-project-structure)
4. [How Files Connect to Each Other](#4-how-files-connect-to-each-other)
5. [Setup — Step by Step](#5-setup--step-by-step)
6. [Data Format](#6-data-format)
7. [Configuration Reference](#7-configuration-reference)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Understanding the Model](#9-understanding-the-model)
10. [Understanding the Loss Functions](#10-understanding-the-loss-functions)
11. [Understanding the Metrics](#11-understanding-the-metrics)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Clinical Context

**Breslow depth** measures the thickness of a melanoma tumour in micrometres (µm) — from the top of the granular layer of the epidermis to the deepest tumour cell, measured perpendicular to the skin surface. It is the single most important prognostic factor for melanoma staging.

### T-Category Staging (AJCC)

| Category | Breslow Depth | Meaning |
|----------|--------------|---------|
| T1a | < 800 µm | Very thin, excellent prognosis |
| T1b | 800 – 1000 µm | Thin |
| T2 | 1000 – 2000 µm | Intermediate |
| T3 | 2000 – 4000 µm | Thick |
| T4 | > 4000 µm | Very thick, poor prognosis |

Currently Breslow depth is measured manually by pathologists under a microscope — a time-consuming and subjective process with inter-observer variability of ~860 µm. This project automates that measurement using deep learning.

---

## 2. How It Works — The Full Pipeline

```
Raw WSI Image (PNG)
        |
        v
+----------------------------------+
|  BreslowDataset (dataset.py)     |
|  . Loads image + RGB mask        |
|  . Converts RGB -> class indices |
|  . Reads Breslow depth from CSV  |
+---------------+------------------+
                |  dict: {image, mask, breslow_depth_um, ...}
                v
+----------------------------------+
|  Transforms (transforms.py)      |
|  . Resize to 768x768             |
|  . Augmentation (train only)     |
|  . Normalise (ImageNet stats)    |
|  . -> PyTorch tensor             |
+---------------+------------------+
                |  Tensor (3, 768, 768)
                v
+----------------------------------+
|  DataLoader (utils.py)           |
|  . Batches of 2 images           |
|  . 52 train / 11 val / 12 test   |
|  . Stratified by T-category      |
+---------------+------------------+
                |  Batch (2, 3, 768, 768)
                v
+----------------------------------+
|  UNet++ Model (unet.py)          |
|  . Encoder: EfficientNet-B4      |
|  . Decoder: UNet++ nested blocks |
|  . Output: (2, 4, 768, 768)      |
|    4 class scores per pixel      |
+---------------+------------------+
                |  Logits (4 channels)
                v
+----------------------------------+
|  CombinedLoss (losses.py)        |
|  . Dice Loss (overlap)           |
|  . CrossEntropy (pixel-wise)     |
|  . Boundary loss (edge fidelity) |
|  . Class weights [0.05,1,3,0.5]  |
+---------------+------------------+
                |  Scalar loss -> backprop
                v
+----------------------------------+
|  Trainer (trainer.py)            |
|  . AdamW optimiser               |
|  . Cosine LR annealing           |
|  . Early stopping (patience=15)  |
|  . Saves best_model.pth          |
+---------------+------------------+
                |  Trained model
                v
+----------------------------------+
|  Inference / Prediction          |
|  . (Optional) 4-way TTA          |
|  . Argmax -> segmentation mask   |
|  . Perpendicular depth from mask |
|  . Convert pixels -> um          |
|  . Assign T-category             |
+----------------------------------+
```

### What the model learns to segment

Each pixel is classified into one of 4 classes:

| Class | Index | Colour in mask | Meaning |
|-------|-------|---------------|---------|
| Background | 0 | Black (0,0,0) | Non-tissue / slide background |
| Tumour | 1 | Blue (0,0,255) | Melanoma tumour cells |
| Epidermis | 2 | Green (0,255,0) | Outer skin layer |
| Dermis | 3 | Red (255,0,0) | Inner skin layer below epidermis |

Breslow depth = the maximum perpendicular distance from any tumour pixel to the predicted epidermis surface, multiplied by the known pixel resolution (4.0 µm/pixel at the canonical downsample).

---

## 3. Project Structure

```
skin-cancer-depth-prediction/          <- Project root (run all commands from here)
|
+-- breslow_depth_prediction/          <- Main Python package
|   +-- configs/
|   |   +-- config_v1.yaml             <- V1 baseline (512x512, vertical Breslow)
|   |   +-- config_v2.yaml             <- V2 (768x768, epidermis weight=3)
|   |   +-- config_v3.yaml             <- V3 (boundary loss + perpendicular)
|   |   +-- config_v4.yaml             <- V4 (multi-task: seg + regression head)
|   |   +-- config_v5.yaml             <- V5 (canonical; expanded CSV manifest)
|   |
|   +-- scripts/                       <- Runnable scripts
|   |   +-- train.py                   <- Train a single-task model
|   |   +-- train_multitask.py         <- Train the V4 multi-task variant
|   |   +-- evaluate.py                <- Evaluate on test set (supports --tta)
|   |   +-- predict_image.py           <- Single-image inference + overlay
|   |
|   +-- src/                           <- Library source code
|       +-- config.py                  <- Config loading + path resolution
|       |
|       +-- data/                      <- Everything about data
|       |   +-- dataset.py             <- BreslowDataset (loads images+masks+CSV)
|       |   +-- transforms.py          <- Resize + augmentation + normalise
|       |   +-- utils.py               <- create_dataloaders(), split logic
|       |
|       +-- models/                    <- Neural networks
|       |   +-- unet.py                <- UnetPlusPlus + MultiTaskUnetPlusPlus
|       |   +-- losses.py              <- DiceLoss, CombinedLoss, MultiTaskLoss
|       |
|       +-- training/                  <- Training loops
|       |   +-- trainer.py             <- Trainer class (fit, validate, checkpoint)
|       |   +-- multitask_trainer.py   <- V4 subclass for dict-output models
|       |   +-- metrics.py             <- SegmentationMetrics, BreslowMetrics
|       |
|       +-- inference/                 <- Post-processing
|       |   +-- breslow_calculator.py  <- Pixel mask -> Breslow depth in um
|       |
|       +-- visualization/             <- Plotting
|           +-- visualize.py           <- visualize_sample, plot_training_history
|
+-- scripts/                           <- Utility scripts (project root level)
|   +-- verify_setup.py                <- Tests all 36 pipeline components
|   +-- explore_data.py                <- Data statistics and visualisation
|
+-- checkpoints/                       <- Saved model weights (.pth gitignored, SHA-256 manifest tracked)
+-- results/                           <- Training plots, evaluation reports per iteration
+-- logs/                              <- Training log files
```

The actual data lives outside the repository (gitignored). Default path is `./data/` at the project root — symlink or copy your local copy there:

```
<project_root>/data/
+-- ds_WSIs/                           <- Image + mask PNG files (89 pairs)
+-- measurements/
    +-- breslow_depth_coords_210725.csv  <- Ground truth Breslow depths (current)
    +-- wrong_dir_coords.txt             <- 10 samples to exclude
```

---

## 4. How Files Connect to Each Other

### Import chain (what imports what)

```
config_v5.yaml                  <- (and config_v1..v4.yaml for prior iterations)
    |
    +-> config.py              load_config() reads the YAML file
              |                resolve_paths() makes paths cwd-independent
              |
              +-> Used by EVERY script as the single source of truth


dataset.py  <-- Reads: ds_WSIs/*.png + measurements/*.csv
    |
    |  Provides: BreslowDataset.__getitem__() returns:
    |  {image: Tensor(3,H,W), mask: Tensor(H,W), breslow_depth_um: float, ...}
    |
    +-> transforms.py          get_train_transforms(config, image_size)
              |                get_val_transforms(config, image_size)
              |                Pipeline: Resize -> Augment -> Normalize -> ToTensor
              |
              +-> utils.py     create_dataloaders(config)
                       |       Creates BreslowDataset + transforms + DataLoader
                       |       Returns: (train_loader, val_loader, test_loader)
                       |
                       +-> train.py  (consumes the dataloaders)


unet.py
    |  get_model(config) -> smp.UnetPlusPlus(encoder="efficientnet-b4", classes=4)
    |  20.8 million parameters, pretrained on ImageNet
    |  MultiTaskUnetPlusPlus (V4) adds a regression head on the bottleneck features

losses.py
    |  get_loss_function(config) -> CombinedLoss(dice_weight, ce_weight, boundary_weight)
    |  MultiTaskLoss (V4) combines CombinedLoss with L1 on log-µm depth

metrics.py
    |  SegmentationMetrics  -- tracks Dice, IoU, pixel accuracy per class
    |  BreslowMetrics       -- tracks MAE, RMSE, T-category accuracy

trainer.py
    |  Trainer(model, criterion, optimizer, device, ...)
    |   .train_epoch(dataloader) -> loss + metrics
    |   .validate(dataloader)    -> loss + metrics
    |   .fit(train_loader, val_loader, epochs)
    |   .save_checkpoint("best_model.pth")
    |
multitask_trainer.py
    |  MultiTaskTrainer(Trainer)  -- overrides train_epoch and validate to handle dict outputs

evaluate.py
    |  Runs the trained model over the test set; supports:
    |    --tta                      4-way test-time augmentation
    |    --breslow-method           {vertical | perpendicular}
    |    --postprocess              CC cleaning + hole filling
    |    --reference-percentile     vertical-calc tuning knob
    |    --multitask-ensemble       V4-specific ensemble strategy
    |
predict_image.py
    |  Single-image inference. Same model and Breslow calculator as evaluate.py,
    |  exposed via a single --image flag. Supports --tta and --breslow-method.


visualize.py
    |  visualize_sample(image, mask, ...)
    |  plot_training_history(history)
    |  save_figure(fig, path)
```

### The role of each file in one sentence

| File | Role |
|------|------|
| `config_v5.yaml` | All hyperparameters for the canonical V5 run — the single source of truth |
| `config.py` | Reads the YAML into a Python dict; resolves relative paths against the project root |
| `dataset.py` | Loads one (image, mask) pair from disk per `__getitem__` call |
| `transforms.py` | Resizes, augments, and normalises images/masks into tensors |
| `utils.py` | Creates stratified train/val/test DataLoaders from the dataset |
| `unet.py` | Defines the UNet++ segmentation model (and V4's multi-task variant) using SMP |
| `losses.py` | Computes Dice + CrossEntropy + boundary training loss with class weights |
| `metrics.py` | Evaluates predictions — Dice, IoU, MAE, T-category accuracy |
| `trainer.py` | Runs the training loop, validates, checkpoints the best model |
| `multitask_trainer.py` | V4 subclass for the dict-output multi-task model |
| `train.py` | Entry point — wires everything together and starts training |
| `evaluate.py` | Runs trained model on the test set with TTA / post-processing options |
| `predict_image.py` | Runs trained model on a single image for live inference / demo |
| `verify_setup.py` | Runs 36 sanity checks across the entire pipeline |

---

## 5. Setup — Step by Step

### Prerequisites

- Python 3.10+
- 8 GB RAM minimum (16 GB recommended for training)
- NVIDIA GPU with ≥ 10 GB VRAM recommended for training (CPU training works but is ~3 min/epoch and may stall at 768²)

### Step 1: Navigate to the project root

All commands must be run from the project root:

```powershell
cd <path-to-skin-cancer-depth-prediction>
```

### Step 2: Activate the virtual environment

```powershell
.\.venv\Scripts\activate
```

You will see `(.venv)` at the start of your prompt when it is active.

### Step 3: Verify data is in place

The data must be at `<project_root>/data/` (the default, gitignored). Check with:

```powershell
dir ".\data\ds_WSIs" | measure
dir ".\data\measurements"
```

You should see 178 files (89 images + 89 masks) and the CSV/TXT files including `breslow_depth_coords_210725.csv`.

If your data is in a different location, open `breslow_depth_prediction/configs/config_v5.yaml` and change:

```yaml
data:
  data_dir: "./data"   # <- edit this line (relative to project root)
```

### Step 4: Verify the entire pipeline (36 tests)

```powershell
python scripts/verify_setup.py
```

Expected final output:

```
All 36 tests passed!
```

This confirms imports, config loading, dataset (75 valid samples), transforms, dataloaders, model creation (20.8M params), loss computation, metrics, and visualisation all work correctly.

### Step 5: (Optional) Explore the dataset

```powershell
python scripts/explore_data.py
```

Generates a statistics report, depth histogram, and T-category distribution in `results/`.

### Step 6: Train the model

```powershell
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml
```

Training will:
1. Load config from the YAML file
2. Create datasets — 52 train / 11 val / 12 test (75 valid samples, stratified by T-category)
3. Build UNet++ with EfficientNet-B4 encoder (ImageNet pretrained)
4. Train for up to 100 epochs with AdamW + Cosine LR schedule
5. Stop early if val Dice does not improve for 15 consecutive epochs
6. Save the best checkpoint to `checkpoints/best_model.pth` (rename to `best_model_v5.pth` to keep history)
7. Save training history plot to `results/training_history.png`
8. Log everything to `logs/train_YYYYMMDD_HHMMSS.log`

### Step 7: Resume interrupted training

```powershell
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --resume checkpoints/best_model.pth
```

---

## 6. Data Format

### Image files (`_image.png`)

- RGB PNG tiles downsampled from full-resolution WSIs
- Variable original sizes (e.g., 2496×1433, 1450×1673 pixels)
- Each pixel represents 4.0 µm of real tissue (at the canonical `targetMPP=4.00` downsample)
- Automatically resized to 768×768 during training via Albumentations

### Mask files (`_labels.png`)

- RGB PNG at the same resolution as the corresponding image
- Each pixel colour encodes the tissue class:

```
(0, 0, 0)     Black  -> Background  (class 0)
(0, 0, 255)   Blue   -> Tumour      (class 1)
(0, 255, 0)   Green  -> Epidermis   (class 2)
(255, 0, 0)   Red    -> Dermis      (class 3)
```

The dataset converts these RGB colours to integer class indices (0-3) for training.

### Filename convention

Images and masks share the same sample ID prefix:

```
Sample-26-A_001_[baseMPP=0.253_targetMPP=4.00]_image.png
Sample-26-A_001_[baseMPP=0.253_targetMPP=4.00]_labels.png
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 sample ID: Sample-26-A_001
```

The CSV references the original SVS filename — the `.svs` extension is stripped to match.

### CSV file (`breslow_depth_coords_210725.csv`)

- Encoding: **Latin-1** (not UTF-8, because the column name contains the µ symbol)
- Key columns:
  - `image` — original SVS filename (e.g., `Sample-26-A_001.svs`)
  - `length_µm` (or `length_um` — both supported) — ground truth Breslow depth in micrometres
  - `x1`, `y1`, `x2`, `y2` (or `x1_ds`, `y1_ds`, `x2_ds`, `y2_ds` — both column-naming conventions handled) — coordinates of the measurement line

### Dataset split

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | 52 | Model learns from these |
| Val | 11 | Early stopping, hyperparameter tuning |
| Test | 12 | Final unbiased evaluation |

The split is **stratified by T-category** using random seed 42 to ensure all stages appear in each split proportionally. 10 samples are excluded due to annotation issues (`wrong_dir_coords.txt`), leaving 75 valid out of 89.

---

## 7. Configuration Reference

All settings live in `breslow_depth_prediction/configs/config_v5.yaml` (and `config_v1..v4.yaml` for prior iterations). Edit this file to change any aspect of the pipeline.

```yaml
# WHERE IS YOUR DATA?
data:
  data_dir: "../data"                                    # Resolved relative to the package root
  images_dir: "ds_WSIs"                                  # Subfolder with PNG files
  coords_file: "measurements/breslow_depth_coords_210725.csv"
  exclude_file: "measurements/wrong_dir_coords.txt"
  image_size: [768, 768]                                 # All images resized to this before training
  resolution_um_per_pixel: 4.0                           # Physical scale of one pixel

# SEGMENTATION CLASSES
classes:
  num_classes: 4                                         # background, tumour, epidermis, dermis

# T-STAGING THRESHOLDS (micrometres)
breslow_thresholds:
  T1a_max: 800
  T1b_max: 1000
  T2_max: 2000
  T3_max: 4000

# DATASET SPLIT
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42                                        # Change for a different random split

# TRAINING HYPERPARAMETERS
training:
  batch_size: 2                                          # Reduce to 1 if you run out of GPU memory
  num_epochs: 100
  learning_rate: 0.0001                                  # AdamW optimiser
  weight_decay: 0.0001
  early_stopping_patience: 15

# MODEL ARCHITECTURE
model:
  architecture: "UnetPlusPlus"                           # Best for medical segmentation
  encoder: "efficientnet-b4"                             # ImageNet pretrained
  encoder_weights: "imagenet"
  in_channels: 3                                         # RGB input
  classes: 4                                             # 4 tissue classes

# LOSS FUNCTION
loss:
  type: "combined"                                       # Dice + CrossEntropy + Boundary
  dice_weight: 0.5
  ce_weight: 0.4
  boundary_weight: 0.1
  class_weights: [0.05, 1.0, 3.0, 0.5]                   # 3x weight on rare epidermis class

# OUTPUT PATHS
paths:
  checkpoint_dir: "checkpoints"
  results_dir: "results"
  log_dir: "logs"
```

---

## 8. Running the Pipeline

### Verify everything is set up

```powershell
python scripts/verify_setup.py
```

### Explore the dataset

```powershell
python scripts/explore_data.py
```

### Train the model

```powershell
# V5 (canonical)
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml

# Resume from checkpoint
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --resume checkpoints/best_model.pth

# V4 multi-task ablation (documented negative result)
python breslow_depth_prediction/scripts/train_multitask.py `
    --config breslow_depth_prediction/configs/config_v4.yaml
```

Example training output (one epoch on an RTX 3080, ~45 seconds):

```
Epoch 1/100
----------------------------------------
Training: 100%|**********| 26/26 [00:45, loss=1.06, dice=0.32]
Validation: 100%|**********| 11/11 [00:09, loss=1.10, dice=0.23]
Train Loss: 1.1554 | Train Dice: 0.2466
Val Loss:   1.1169 | Val Dice:   0.2326
Learning Rate: 0.000100
New best model! Dice: 0.2326
```

### Evaluate on the test set

```powershell
# V5 + perpendicular + TTA (canonical headline configuration)
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --checkpoint checkpoints/best_model_v5.pth `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/evaluation_v5_perpendicular_tta
```

### Single-image inference

```powershell
python breslow_depth_prediction/scripts/predict_image.py `
    --image path/to/slide.png `
    --checkpoint checkpoints/best_model_v5.pth `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/single_predictions
```

---

## 9. Understanding the Model

### Why UNet++?

Standard U-Net copies encoder features directly to the decoder via skip connections. **UNet++** replaces these with **nested dense sub-networks** — additional convolutional layers between encoder and decoder that progressively re-process and blend features at each scale. This produces sharper segmentation boundaries, which is critical when measuring the thin epidermis layer needed for Breslow depth.

```
Encoder stage      Skip connection path          Decoder stage
--------------     --------------------          -------------

stride-2 conv  -->  x_0_0  -->  x_0_1  -->  x_0_2  -->  x_0_3
                                  |                         ^
stride-2 conv  -->  x_1_0  -->  x_1_1  -->  x_1_2  -------+
                                  |                 ^
stride-2 conv  -->  x_2_0  -->  x_2_1  -----------+
                                  |        ^
stride-2 conv  -->  x_3_0  -->  x_3_1 ---+
                                  |
               Bottleneck x_4_0 -+
```

Each `x_i_j` node receives input from the node to its left AND all previous nodes in the same row. This dense connectivity lets the decoder see features processed at multiple levels of abstraction.

### Why EfficientNet-B4?

- **ImageNet pretrained** — already understands textures, edges, colour gradients
- **Compound scaling** — balances network depth, width, and resolution simultaneously
- **B4 variant** — good accuracy/speed trade-off (~19M encoder parameters)
- **Transfer learning** — histopathology staining patterns share low-level texture cues with natural images, so ImageNet weights give a strong starting point

### Model output

The model outputs a `(B, 4, 768, 768)` tensor — 4 class scores per pixel.

- During **training**: raw logits fed directly to the loss function.
- During **inference**: `argmax(dim=1)` picks the highest-scoring class → produces a `(B, 768, 768)` integer mask.
- With **`--tta`**: four model forward passes (identity, h-flip, v-flip, 180° rotation) are averaged in softmax space before the argmax, reducing test-set variance.

### V4 multi-task variant

`MultiTaskUnetPlusPlus` adds a regression head on top of the encoder bottleneck — same segmentation output, plus a single scalar predicting log-µm depth. The intent was to let the regression signal sharpen the segmentation. The V4 ablation showed no improvement over V3/V5; it is retained as a documented negative result.

---

## 10. Understanding the Loss Functions

### Why class weights `[0.05, 1.0, 3.0, 0.5]`?

In histopathology tiles, background pixels can make up ~60 % of the image, and epidermis is typically only ~2 %. Without class weighting, the model learns to ignore epidermis. The weights force the model to focus on:
- **Epidermis** (weight **3.0**) — the rare reference surface for the perpendicular Breslow calculator
- **Tumour** (weight 1.0) — what we measure
- **Dermis** (weight 0.5) — important tissue context
- **Background** (weight 0.05) — easy to predict, deliberately down-weighted

V1 used `[0.1, 1.0, 1.0, 0.5]` (epidermis weight 1); V2 onwards uses `[0.05, 1.0, 3.0, 0.5]` to push the model harder on the rare epidermis class.

### Why Dice + CrossEntropy + Boundary loss?

| Loss component | Strength | Role in this project |
|---------------|---------|---------------------|
| **Dice Loss** | Optimises pixel overlap; robust to class imbalance | Ensures tumour and epidermis regions are well-segmented |
| **CrossEntropy** | Stable gradient; sharpens per-pixel boundaries | Keeps edge detail accurate for the perpendicular calculator |
| **Boundary Loss** (V3+) | Penalises errors near class boundaries | Tightens the epidermis–dermis edge that the calculator depends on |
| **Combined (with weights)** | Gets benefits of all three | V3+ uses `L = 0.5·L_dice + 0.4·L_ce + 0.1·L_boundary` |

---

## 11. Understanding the Metrics

### Segmentation quality (reported each epoch)

| Metric | Formula | Good value |
|--------|---------|-----------|
| **Dice** | 2|A∩B| / (|A|+|B|) | > 0.7 for clinical usefulness |
| **IoU (mIoU)** | |A∩B| / |A∪B| | Stricter than Dice; > 0.5 is reasonable |
| **Pixel Accuracy** | correct pixels / total pixels | Can be misleading with class imbalance |

Dice and IoU are computed **per class** then averaged, with an option to exclude background (class 0) since it is trivially easy to predict.

### Clinical relevance (after test evaluation)

| Metric | Clinical meaning | Target |
|--------|-----------------|--------|
| **MAE** | Average error in µm | < 860 µm (inter-observer variability baseline) |
| **Within 860 µm** | % predictions within clinical baseline | > 50% |
| **T-category accuracy** | % of cases in the correct T-stage | > 75% |
| **T-category adjacent accuracy** | % within 1 T-stage | > 90% |

A MAE below 860 µm means the model performs at least as well as the variation between human pathologists — this is the clinical success criterion.

---

## 12. Troubleshooting

### `FileNotFoundError: No such file or directory: 'configs/config_v5.yaml'`

Always run scripts from the **project root** (the `skin-cancer-depth-prediction/` directory), not from inside a subdirectory.

```powershell
# Wrong - run from inside a subfolder
cd breslow_depth_prediction
python scripts/train.py     # fails

# Correct - always from project root
cd <path-to-skin-cancer-depth-prediction>
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml
```

### `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb5`

The CSV uses Latin-1 encoding due to the µ symbol in `length_µm`. This is already handled in `dataset.py`. If you encounter this opening the CSV manually, use:
```python
pd.read_csv("breslow_depth_coords_210725.csv", encoding="latin-1")
```

### `RuntimeError: stack expects each tensor to be equal size`

`image_size` is missing from config or the transforms are not receiving it. Ensure the config has:
```yaml
data:
  image_size: [768, 768]
```

### `RuntimeError: one_hot is only applicable to index tensor of type LongTensor`

The mask tensor must be `torch.int64`. Already fixed in `losses.py` with `.long()` casts.

### `pickle.UnpicklingError: Weights only load failed`

PyTorch ≥ 2.5 defaults `torch.load(weights_only=True)`. The trainer uses `weights_only=False` to load the full optimiser/scheduler state alongside the model weights — this is intentional and safe for project-controlled checkpoints.

### Training is very slow

On CPU, each epoch can take several minutes. Options to speed up:

| Option | How |
|--------|-----|
| Use a GPU | Training auto-detects CUDA |
| Reduce image size | Set `image_size: [512, 512]` (matches V1 baseline) |
| Reduce batch size | Set `batch_size: 1` |

### Out of GPU memory

Drop `batch_size` to 1 in the config. If still failing, also reduce `image_size` to `[512, 512]`.

### Persistent DataLoader stalls on Windows

The training pipeline uses `num_workers=2` and **does not** set `persistent_workers=True` — testing showed `persistent_workers` caused silent mid-training stalls on Windows. Validation uses `num_workers=0`. If you tweak these, expect to re-validate stability over a full run.

### Early stopping triggers before convergence

Increase `early_stopping_patience` in the config (default is 15 epochs). With only 11 validation samples the loss can be noisy — 20–25 may be more appropriate.

---

## Appendix: Key API Reference

| Symbol | Location | Signature |
|--------|---------|-----------|
| `load_config` | `src/config.py` | `load_config(path) -> dict` |
| `resolve_paths` | `src/config.py` | `resolve_paths(config, project_root) -> dict` |
| `BreslowDataset` | `src/data/dataset.py` | `BreslowDataset(data_dir, split="train")` |
| `create_dataloaders` | `src/data/utils.py` | `create_dataloaders(config) -> (train, val, test)` |
| `get_train_transforms` | `src/data/transforms.py` | `get_train_transforms(config, image_size)` |
| `get_val_transforms` | `src/data/transforms.py` | `get_val_transforms(config, image_size)` |
| `get_model` | `src/models/unet.py` | `get_model(config) -> nn.Module` |
| `MultiTaskUnetPlusPlus` | `src/models/unet.py` | V4 model class with regression head |
| `get_loss_function` | `src/models/losses.py` | `get_loss_function(config) -> nn.Module` |
| `Trainer` | `src/training/trainer.py` | `Trainer(model, criterion, optimizer, device, ...)` |
| `MultiTaskTrainer` | `src/training/multitask_trainer.py` | V4 subclass of Trainer |
| `SegmentationMetrics` | `src/training/metrics.py` | `SegmentationMetrics(num_classes=4)` |
| `BreslowMetrics` | `src/training/metrics.py` | `BreslowMetrics()` |
| `tta_forward` | `scripts/evaluate.py` | `tta_forward(model, image) -> probs` |
| `calculate_breslow_depth_um` | `scripts/evaluate.py` | Canonical Breslow calculator shared with `predict_image.py` |
| `visualize_sample` | `src/visualization/visualize.py` | `visualize_sample(image, mask, show=False)` |
