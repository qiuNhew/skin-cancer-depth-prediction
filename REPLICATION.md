# Replication Guide

This document describes how to reproduce the results reported in this project — and, crucially, what can and cannot be redistributed with the repository.

---

## 1. Data provenance

The histopathology dataset used in this project was provided by **Cirdan Imaging Limited** for the purpose of the CSC4006 Research & Development Project at Queen's University Belfast.

**Dataset characteristics:**
- 89 H&E-stained whole-slide image downsamples of melanoma biopsies (PNG format)
- Paired pixel-level segmentation masks (4 classes: background, tumour, epidermis, dermis)
- Ground-truth Breslow depth measurements in micrometres (pathologist-annotated, CSV)
- 10 samples flagged for exclusion (malformed coordinates), leaving 75 valid samples
- Samples are pseudonymised at source (opaque sample identifiers); the link between pseudonym and patient is held by the data custodian, not the student

**Data is not distributed with this repository.** The `data/` directory is gitignored. An expected-layout stub lives at [`breslow_depth_prediction/data/README.md`](breslow_depth_prediction/data/README.md).

---

## 2. Ethical and legal statement

Access to and use of the dataset is governed by a **mutual non-disclosure agreement** between the student and **Cirdan Imaging Limited**. Under that agreement:

- **Use is restricted to the CSC4006 project Purpose.** The dataset and any Confidential Information derived from it may not be used for any other reason.
- **No copies without prior written consent.** Forking or redistributing the raw data — including pseudonymised patient IDs, masks, or image files — is not permitted.
- **Return or destruction on completion / request.** On completion of the Purpose, or on Cirdan's request, the data must be returned or certified destroyed.
- **Ongoing confidentiality for 5 years** after return / destruction.
- **Governing law:** Northern Ireland.

The NDA itself is not reproduced in this repository. Assessors / supervisors / future researchers wishing to see the operative clauses should contact the student and Cirdan Imaging.

**In practical terms for the CSC4006 submission:**
- The code, configs, and aggregate metrics in this repository are **MIT-licensed** and can be read / cloned / forked freely (see [`LICENSE`](LICENSE)).
- The aggregate per-iteration result folders under [`results/`](results/) (metrics JSON, scatter plots, Bland–Altman, confusion matrices) contain no patient identifiers and are safe to share through the access-limited QUB GitLab.
- Per-sample artefacts (per-sample JSON tables, per-slide visualisation PNGs) contain pseudonymised patient IDs and are not redistributed via the repository; they are submitted to the assessor through the QUB submission channel (Canvas / OneDrive).

**This work has not yet been published externally.** Any future publication (journal, conference, preprint) would require review against the NDA and any institutional ethics requirements before distribution.

---

## 3. What is reproducible without dataset access

The following artefacts are in the repository and inspectable with only a Git clone:

| Artefact | Where | Purpose |
|---|---|---|
| Source code | [`breslow_depth_prediction/src/`](breslow_depth_prediction/src/) | Full pipeline — dataset, transforms, model, losses, trainer, metrics, visualisation, inference |
| Configs | [`breslow_depth_prediction/configs/`](breslow_depth_prediction/configs/) | V1 through V5 ([`config_v1.yaml`](breslow_depth_prediction/configs/config_v1.yaml) ... [`config_v5.yaml`](breslow_depth_prediction/configs/config_v5.yaml)) |
| Aggregate metrics (JSON) | `results/evaluation_*/metrics.json` | All segmentation + Breslow metrics per iteration / variant |
| Aggregate plots | `results/evaluation_*/*.png` | Scatter plots, Bland–Altman, Dice bar charts, T-cat confusion |
| Training history | `results/training_history*.png` | Loss / Dice curves per epoch |
| V1 reproducibility rerun | [`results/v1_rerun/`](results/v1_rerun/) | Full bundle (log, plot, summary, evaluation) of the April 2026 V1 retraining run |
| Checkpoint SHA-256 manifest | [`checkpoints/README.md`](checkpoints/README.md) | Chain-of-custody for the `.pth` files (which are themselves gitignored) |

The assessor can verify the reported metrics by reading the JSON metric files directly — no data access required.

---

## 4. What requires dataset access

Re-running **training** or **evaluation** from scratch requires:
- A copy of the Cirdan dataset at `<project_root>/data/` in the layout documented in [`REQUIREMENTS.md`](REQUIREMENTS.md)
- The corresponding model checkpoint (also not redistributed — trained weights are derived from the dataset and are covered by the same NDA)

Checkpoints (`.pth` files) are gitignored by default. They are stored locally by the student at [`checkpoints/`](checkpoints/); the SHA-256 manifest in [`checkpoints/README.md`](checkpoints/README.md) allows a reviewer who has received the binaries through a separate channel to confirm they match the published artefacts.

---

## 5. Reproducing the V5 result (canonical headline)

The headline figures reported in the Research Article and the Software Development Report come from V5 with the perpendicular Breslow calculator and 4-way test-time augmentation.

### 5.1 Re-evaluate the published V5 checkpoint

**Prerequisites:** completed `INSTALL.md` (including the 36-test `verify_setup.py` passing on your machine), plus dataset and V5 checkpoint access.

```powershell
.\.venv\Scripts\activate
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --checkpoint checkpoints/best_model_v5.pth `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/evaluation_v5_perpendicular_tta
```

**Expected output** (matches [`results/evaluation_v5_perpendicular_tta/evaluation_report.txt`](results/evaluation_v5_perpendicular_tta/evaluation_report.txt)):

| Metric | Value | Target | Status |
|---|---|---|---|
| Tumour Dice | 0.937 | > 0.80 | ✓ PASS |
| Epidermis Dice | 0.803 | > 0.80 | ✓ PASS |
| Dermis Dice | 0.923 | > 0.80 | ✓ PASS |
| Mean Dice (no background) | 0.888 | > 0.80 | ✓ PASS |
| Breslow MAE | 3 561 µm | < 500 µm | FAIL (48 % lower than V3) |
| T-category accuracy | 41.7 % | > 75 % | FAIL |
| Adjacent T-cat accuracy | 91.7 % | > 90 % | ✓ PASS |

### 5.2 Retrain V5 from scratch

```powershell
$env:PYTHONUNBUFFERED=1
python -u breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v5.yaml
```

Wall-clock on an NVIDIA RTX 3080 (10 GB VRAM): approximately one to two hours (early-stopping driven, patience = 15).

Random seed 42 is fixed in the config — this produces the same 52 / 11 / 12 train / val / test split as the published runs, so results are reproducible modulo CUDA non-determinism (expect val Dice within ±0.005 of the reported number).

After training, rename `checkpoints/best_model.pth` to `checkpoints/best_model_v5.pth` to preserve the published comparison and avoid clobbering on the next training run.

### 5.3 Single-image inference (for live demo or ad-hoc evaluation)

```powershell
python breslow_depth_prediction/scripts/predict_image.py `
    --image path/to/slide.png `
    --checkpoint checkpoints/best_model_v5.pth `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/single_predictions
```

Produces a class-coloured mask PNG, an overlay PNG, and a `prediction.json` summary with the predicted depth (µm and mm) and T-category. The `--tta` flag is optional; omitting it gives a single forward-pass prediction in ~0.2 s on the RTX 3080.

---

## 6. Historical iterations

V1 through V4 are retained as the development trajectory leading to V5. They are reproducible with their own configs; we summarise each below.

### 6.1 V1 — 512² baseline (Dice + CE, vertical Breslow calculator)

```powershell
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v1.yaml `
    --checkpoint checkpoints/best_model_v1.pth `
    --output-dir results/evaluation_v1
```

Headline V1 numbers (see [`results/evaluation_v1/metrics.json`](results/evaluation_v1/metrics.json)):

| Metric | Value |
|---|---|
| Tumour Dice | 0.879 |
| Epidermis Dice | 0.683 |
| Dermis Dice | 0.905 |
| Breslow MAE | 6 503 µm |
| T-category accuracy | 63.6 % |

Training wall-clock on RTX 3080: ~51–96 minutes (early-stopping driven).

### 6.2 V2 — 768², epidermis weight ×3

The V2 checkpoint `best_model_v2.pth` was **overwritten on disk during a V1 rerun** in April 2026 and is not currently recoverable. The aggregate V2 metrics in [`results/evaluation_v2_original/`](results/evaluation_v2_original/) remain (metrics JSON, plots, per-sample JSON, reports). To reproduce V2 numerically you must retrain:

```powershell
python breslow_depth_prediction/scripts/train.py `
    --config breslow_depth_prediction/configs/config_v2.yaml
```

Training wall-clock: ~118 minutes on an RTX 3080. Known V2 outcome (see [`results/evaluation_v2_original/metrics.json`](results/evaluation_v2_original/metrics.json)) showed improved segmentation but worse Breslow MAE than V1 — the finding that motivated the V3 perpendicular calculator.

### 6.3 V3 — boundary loss + perpendicular calculator

```powershell
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v3.yaml `
    --checkpoint checkpoints/best_model_v3.pth `
    --breslow-method perpendicular `
    --tta `
    --output-dir results/evaluation_v3_perpendicular_tta
```

V3 introduced the perpendicular Breslow calculator and the boundary loss. MAE dropped from V2's 6 959 µm to ~6 836 µm; V5 builds on this baseline.

### 6.4 V4 — multi-task ablation (documented negative result)

```powershell
python breslow_depth_prediction/scripts/train_multitask.py `
    --config breslow_depth_prediction/configs/config_v4.yaml
```

V4 adds a depth-regression head to the encoder bottleneck on top of the V3 segmentation model. The hypothesis — that the regression signal would sharpen segmentation — did not bear out: V4 did not improve on V3 / V5. Retained as a documented negative result; the architecture (`MultiTaskUnetPlusPlus`, `MultiTaskLoss`, `MultiTaskTrainer`) is in the source for completeness.

### 6.5 V1 reproducibility rerun (April 2026)

After the V2 checkpoint was overwritten, V1 was retrained using the same seed and config to verify reproducibility. The bundled run lives at [`results/v1_rerun/`](results/v1_rerun/) and includes the training log, training-history plot, training summary, and full evaluation subfolder. See [`results/v1_rerun/README.md`](results/v1_rerun/README.md) for the manifest and headline numbers.

The rerun confirms the V1 recipe is deterministic at the ~0.01 Dice level — the small differences (Tumour Dice 0.8789 → 0.8899, Epidermis 0.683 → 0.707, MAE 6 503 → 6 675 µm) are within run-to-run CUDA non-determinism on a 52-sample training set.

---

## 7. Preparing the results for public release

*Only relevant if and when the repository is ever published beyond QUB GitLab.* The following steps would be required under the NDA (§2, §3):

1. **Obtain Cirdan's prior written consent** for external release of any derivative of the dataset (metrics, per-sample results, model weights).
2. **Strip patient pseudonyms** from any per-sample artefacts that have been delivered through the QUB submission channel before any wider release: replace `Sample-15_001`, `Sample-09-A_001`, etc. with opaque identifiers (`Sample-01`, `Sample-02`, …).
3. **Remove the pseudonym → index mapping** from any artefact once substitution is complete.
4. **Re-check aggregate metrics** — these are already non-identifying and pose no additional risk.

The repository currently tracks only aggregate plots and metrics; per-sample identifying artefacts are kept out of git by `.gitignore` and submitted separately.

---

## 8. Contact

For questions about replicating this work, or to request dataset / checkpoint access, contact the student via the supervisor. For the Cirdan NDA itself, contact Cirdan Imaging Limited directly.
