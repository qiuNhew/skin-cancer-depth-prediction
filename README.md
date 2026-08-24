# Automated Breslow Depth Prediction for Melanoma

CSC4006 Research & Development Project, Queen's University Belfast, 2025–2026.

Predicts **Breslow thickness** — the primary prognostic biomarker for cutaneous melanoma — from H&E histopathology whole-slide images (WSIs), using semantic segmentation of four tissue classes (Background / Tumour / Epidermis / Dermis) followed by geometric depth calculation perpendicular to the predicted skin surface.

---

## Repository map

| File / folder | Purpose |
|---|---|
| [`README.md`](README.md) | You are here — project overview and index |
| [`REQUIREMENTS.md`](REQUIREMENTS.md) | Hardware, OS, CUDA, Python, and package requirements |
| [`INSTALL.md`](INSTALL.md) | Step-by-step installation + a 36-test smoke-check |
| [`REPLICATION.md`](REPLICATION.md) | How to reproduce the V5 headline result, plus data provenance and ethics |
| [`LICENSE`](LICENSE) | MIT licence (source code only — patient data is separately governed) |
| [`breslow_depth_prediction/`](breslow_depth_prediction/) | Python package: `src/`, `configs/`, `scripts/`, `tests/`, plus a detailed internal [`README.md`](breslow_depth_prediction/README.md) |
| [`results/`](results/) | Evaluation metrics, plots, per-sample JSONs for V1 through V5 (plus a reference-percentile sweep for V5) |
| [`scripts/`](scripts/) | `verify_setup.py` — the smoke-test that runs after install |
| [`checkpoints/`](checkpoints/) | SHA-256 manifest of trained weights (`.pth` files gitignored — see [`checkpoints/README.md`](checkpoints/README.md)) |
| `data/` | Patient data — **gitignored**, external to the repository |
| [`logs/`](logs/) | Training / evaluation run logs |

---

## Quick start

```powershell
# 1. Clone and enter
git clone https://github.com/qiuNhew/skin-cancer-depth-prediction.git
cd skin-cancer-depth-prediction

# 2. Environment
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r breslow_depth_prediction/requirements.txt

# 3. Verify the install
python scripts/verify_setup.py

# 4. (If you have the dataset and the V5 checkpoint) reproduce the headline V5 evaluation
python breslow_depth_prediction/scripts/evaluate.py `
    --config breslow_depth_prediction/configs/config_v5.yaml `
    --checkpoint checkpoints/best_model_v5.pth `
    --breslow-method perpendicular `
    --tta
```

Full details in [`INSTALL.md`](INSTALL.md) and [`REPLICATION.md`](REPLICATION.md).

---

## Method at a glance

1. **Semantic segmentation** with UNet++ + EfficientNet-B4 encoder (via `segmentation-models-pytorch`), trained at 768 × 768.
2. **Loss**: weighted Dice + weighted Cross-Entropy + boundary loss, with a 10× class weight on the rare epidermis class.
3. **Breslow calculation** from the predicted mask: the perpendicular distance from each tumour pixel to the predicted epidermis surface, scaled by the per-image resolution factor; the maximum is reported as the per-slide depth.
4. **Test-time augmentation** (4-way: identity + h-flip + v-flip + 180-rot, softmax-averaged) reduces variance on the test set.
5. **Evaluation**: Dice / IoU per class, plus MAE / T-category accuracy / within-inter-observer for Breslow depth.

## Headline results — V5 + perpendicular + TTA

| Metric | Value | Target | Status |
|---|---|---|---|
| Tumour Dice | **0.937** | > 0.80 | ✓ PASS |
| Epidermis Dice | **0.803** | > 0.80 | ✓ PASS |
| Dermis Dice | **0.923** | > 0.80 | ✓ PASS |
| Mean Dice (no background) | **0.888** | > 0.80 | ✓ PASS |
| Breslow MAE (µm) | 3 561 | < 500 | FAIL (48 % lower than V3 baseline) |
| T-category accuracy | 41.7 % | > 75 % | FAIL |

Full per-sample tables, scatter plots, Bland–Altman analyses, and the T-category confusion matrix are in [`results/evaluation_v5_perpendicular_tta/`](results/evaluation_v5_perpendicular_tta/). The full iteration trajectory (V1 → V5, including the V4 multi-task ablation) is documented in the Research Article.

## Honest summary of failure modes

- **The T-category plateau at ~42 %** is data-limited rather than algorithm-limited. A reference-percentile sweep on V5 (p ∈ {10, 5, 3, 0}) all hit the same accuracy, confirming the segmentation model is not the bottleneck on this 75-slide dataset.
- **Breslow MAE remains 7× the < 500 µm target.** The dominant error mode is correctly-segmented deep T4 tumours where the geometric perpendicular drop misses the true clinical-measurement path. This is documented in the Research Article and is the most promising direction for future iterations.
- **A V4 multi-task ablation (segmentation head + log-µm regression head)** did not improve on V3/V5 and was retained as a documented negative result.

---

## Documentation

### Software engineering
- [`breslow_depth_prediction/README.md`](breslow_depth_prediction/README.md) — internal package documentation (architecture, file-by-file walk-through, config reference, troubleshooting)
- [`INSTALL.md`](INSTALL.md) / [`REQUIREMENTS.md`](REQUIREMENTS.md) / [`REPLICATION.md`](REPLICATION.md) — assessor-facing operational docs

### Research write-ups (separately submitted, not in repo)
- Research Article — IEEE-style four-page paper covering V1 through V5
- Software Development Report — 20-page report on the engineering, per CSC4006 handbook §6.3

## Access

Per the CSC4006 handbook (§6.3.3), the module coordinator, supervisor, and assessor have been added to this GitLab project as members. See the project's **Settings → Members** page.

## Author

**Hao Hew**
MEng Computer Science, Queen's University Belfast

## Licence

Source code is MIT-licensed. The patient dataset is governed by a separate Queen's University Belfast data access agreement and is not distributed with this repository. See [`LICENSE`](LICENSE) and [`REPLICATION.md`](REPLICATION.md).
