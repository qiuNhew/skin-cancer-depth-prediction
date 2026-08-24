# V1 Rerun — 2026-04-20

Full bundle for the V1 rerun. Everything downstream of the V1 config (`breslow_depth_prediction/configs/config_v1.yaml`, 512×512, batch 4, patience 15, class weights `[0.1, 1.0, 1.0, 0.5]`, Dice 0.5 / CE 0.5) is collected here.

## Contents

| File | What |
|---|---|
| `training.log` | Training log (stdout copy) — config, dataset split, GPU, timings |
| `training_history.png` | Loss + Dice curves over epochs |
| `training_summary.txt` | One-line summary: epochs, best val Dice, wall-clock |
| `evaluation/evaluation_report.txt` | Segmentation + Breslow metrics on test set |
| `evaluation/metrics.json` | Same metrics, machine-readable |
| `evaluation/per_sample_results.json` | Per-sample predictions vs ground truth |
| `evaluation/*.png` | Bland–Altman, scatter, per-class Dice, T-category confusion |
| `evaluation/visualisations/` | Per-sample mask overlays |

## Recipe

- **Config:** `breslow_depth_prediction/configs/config_v1.yaml`
- **Checkpoint:** `checkpoints/best_model_v1_rerun.pth` (244 MB, gitignored)
- **Training command:** `$env:PYTHONUNBUFFERED=1; .venv\Scripts\python -u breslow_depth_prediction\scripts\train.py --config breslow_depth_prediction/configs/config_v1.yaml`
- **Evaluation command:** `.venv\Scripts\python breslow_depth_prediction\scripts\evaluate.py --config breslow_depth_prediction/configs/config_v1.yaml --checkpoint checkpoints/best_model_v1_rerun.pth`

## Headline numbers

| Metric | Value | Target | |
|---|---|---|---|
| Epochs trained | 50 | — | |
| Training time | 51.2 min (RTX 3080) | — | |
| Best val Dice | 0.8776 | — | |
| Tumour Dice (test) | 0.8899 | >0.80 | PASS |
| Epidermis Dice (test) | 0.7069 | >0.80 | FAIL |
| Dermis Dice (test) | 0.9144 | — | |
| Breslow MAE (test) | 6675 µm | <500 µm | FAIL |
| T-category accuracy | 54.5% | — | |
| Adjacent T-cat accuracy | 81.8% | — | |
| Within inter-observer (860 µm) | 18.2% | — | |

## Notes

- Test set: 11 samples (P16 and P15 pseudonymised IDs — see `evaluation/per_sample_results.json`).
- This rerun reproduced the original V1 (same config, same seed) after the V2 checkpoint was overwritten on disk; the original V1 bundle lives at `results/evaluation_v1/` for cross-check.
- Epidermis remains the weak class (1.9% pixel share) — the motivation for V2's higher epidermis class weight and larger 768² input.
