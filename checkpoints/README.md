# Checkpoints

Trained UNet++ / EfficientNet-B4 weights for each iteration of the project.

**The `.pth` files themselves are gitignored** — they are derivatives of the
patient dataset and fall under the same Cirdan / QUB data-access agreement
that governs the source imagery. They are not redistributed via this
repository. If you need a specific checkpoint for re-evaluation, request it
through the channels described in [`REPLICATION.md`](../REPLICATION.md).

This README is committed so the chain-of-custody is verifiable: a reviewer
who receives a checkpoint via a side channel can run
`sha256sum best_model_v5.pth` and confirm the binary matches what was used
to produce the reported results.

## Manifest

| File                          | Iteration                                                  | Size (bytes)  | Trained      | SHA-256                                                            |
| ----------------------------- | ---------------------------------------------------------- | ------------- | ------------ | ------------------------------------------------------------------ |
| `best_model_v1.pth`           | V1 — 512², Dice + CE baseline                              | 244,550,120   | 2026-04-09   | `b030606603ab34ab7336e0ee1e689c3a56397339efd58a5079d31751ff3a5a19` |
| `best_model_v1_rerun.pth`     | V1 reproducibility rerun (April 2026, after V2 overwrite)  | 244,553,211   | 2026-04-20   | `3a818b8f2b1b34463a15b937fdebca936461f322d73af16195eca1ac321c4f10` |
| `best_model_v3.pth`           | V3 — 768², boundary loss, perpendicular Breslow calculator | 244,571,771   | 2026-05-01   | `aaaabb9c98023a2c0108f4226d7f0e5c8d4204aa080c024dc5b3f640c9dc64f2` |
| `best_model_v4.pth`           | V4 — multi-task variant (segmentation + depth regression)  | 245,965,656   | 2026-05-03   | `0c37d03f4f259c09333b7f2cb206c3a37373fb4eceafc71825e2b54987b07cd1` |
| `best_model_v5.pth`           | V5 — expanded CSV manifest, canonical headline result      | 244,565,755   | 2026-05-03   | `113eee00d5f9b96cf9a8f15febb922779ed44ce2fa089417e723b0864d6398c9` |

The canonical reported configuration is **V5 + perpendicular Breslow
calculator + 4-way TTA**. See
[`results/evaluation_v5_perpendicular_tta/`](../results/evaluation_v5_perpendicular_tta/)
for the matching aggregate metrics.

## Verifying a checkpoint locally

```powershell
# PowerShell
Get-FileHash -Algorithm SHA256 .\checkpoints\best_model_v5.pth

# bash / git-bash
sha256sum checkpoints/best_model_v5.pth
```

Either should match the SHA-256 above, byte-for-byte.

## Regenerating any checkpoint

The training command for V5 is documented in
[`REPLICATION.md`](../REPLICATION.md) (Appendix D of the Software Development
Report). Wall-clock on an NVIDIA RTX 3080 is approximately one to two hours
per checkpoint at 768², batch size 2.

Random seed 42 is fixed in every config; results are reproducible up to
CUDA non-determinism (typically ±0.005 Dice).
