# Installation

This guide gets the project running on a fresh machine in about 10 minutes (excluding the dataset, which is not redistributed — see `REPLICATION.md`).

## Prerequisites

See `REQUIREMENTS.md` for the full spec. The short version:

- **Python 3.10** (exactly — not 3.11+, not 3.9)
- **Git**
- **NVIDIA GPU with ≥ 10 GB VRAM + CUDA 11.8/12.1** — only needed if you want to train from scratch; evaluation and inference run fine on CPU.

## 1. Clone the repository

```powershell
git clone https://github.com/qiuNhew/skin-cancer-depth-prediction.git
cd skin-cancer-depth-prediction
```

(Replace the URL if you're cloning a fork or mirror.)

## 2. Create and activate the virtual environment

The project uses a standard `venv` at the project root, named `.venv`.

### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS (bash)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your prompt when it's active. Deactivate with `deactivate`.

## 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r breslow_depth_prediction/requirements.txt
```

**GPU users** — the `torch>=2.0` in `requirements.txt` pulls the default CUDA build from PyPI. If you need a specific CUDA version, install PyTorch *first* from the appropriate index, then the rest of the requirements:

```powershell
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r breslow_depth_prediction/requirements.txt
```

The install takes 2–5 minutes depending on your connection and whether PyTorch wheels are cached.

## 4. (Optional) Link your dataset

The patient dataset is not distributed with this repository (see `REPLICATION.md`). If you have access via the QUB data agreement, either copy or symlink it so the default config path works:

### Windows

```powershell
# From the project root, with data located at D:\melanoma-data\
mklink /D data D:\melanoma-data
```

### Linux / macOS

```bash
ln -s /path/to/melanoma-data data
```

The expected layout inside `./data/` is documented in `REQUIREMENTS.md`.

If you don't have the dataset, you can still run the segmentation model on **your own PNG images** via the inference entry-point (step 6).

## 5. Smoke test — verify the installation

The project ships with a 36-step verification script that checks imports, config loading, dataset plumbing, model construction, loss computation, metrics, and visualisation:

```powershell
python scripts/verify_setup.py
```

**Expected final output:**

```
All 36 tests passed!
```

If any test fails, the script prints the failing module and exception traceback. Most common causes: wrong Python version, missed an activation step, or running from a subdirectory instead of the project root.

Some tests require the dataset to be present. On a machine without data, the data-related tests will fail but the import/model/loss/metrics tests (around 20 of the 36) should still pass — enough to confirm the code itself installed correctly.

## 6. Basic usage examples

**Reproduce the canonical V5 evaluation** (dataset + checkpoint required):

```powershell
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --checkpoint checkpoints/best_model_v5.pth `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/evaluation_v5_perpendicular_tta
```

Writes test-set metrics + plots under the chosen output directory. See `REPLICATION.md` for the full V5 reproduction procedure (and historical V1 / V2 iterations).

**Train from scratch** (GPU recommended, ~1 hour for V1 at 512², ~1–2 hours for V5 at 768²):

```powershell
$env:PYTHONUNBUFFERED=1
python -u breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml
```

## Troubleshooting

- **`ModuleNotFoundError: breslow_depth_prediction`** — you're not running from the project root, or `.venv` isn't activated. Always run commands from `<project_root>`, not from a subdirectory.
- **`CUDA out of memory`** — drop to the V1 config (512 × 512) or reduce `batch_size` from 2 to 1; you can also add `$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"`.
- **`UnicodeDecodeError: 0xb5`** on CSV load — the length column has a µ character. The dataset loader handles this, but if you're reading the CSV manually pass `encoding="latin-1"` or `encoding="cp1252"`.
- **`FileNotFoundError: configs/config_v5.yaml`** — you're running from a subdirectory. Use the full path `breslow_depth_prediction/configs/config_v5.yaml` as shown above.
