---
experiment: "exp-018"
evidence: "validated-local"
summary: "Batch64 ten-epoch training and large-batch capacity; reference cancelled"
verdicts: {"H-012": "inconclusive"}
---
# Ours batch-size comparison

| Batch | Outcome | Epoch10 accuracy | Mean epoch2–10 seconds | Peak GiB |
|---|---|---:|---:|---:|
| 64 | Complete on fifi | 73.94% | 754.3 | 21.13 |
| 128 | CUDA OOM | — | — | — |
| 256 | CUDA OOM | — | — | — |

Filip cancelled batch32 and retained only batch64. H-012 remains inconclusive because no complete matched batch32 reference exists. Report batch64 accuracy, runtime and memory descriptively; do not claim a measured speedup or accuracy change. Batch32 partial artifacts are preserved and excluded from the ten-epoch comparison. No replacement control is authorized.

Validated fresh Adam, epoch1–10 data coverage, final optimizer/scheduler and BN steps. Raw native bundles and summary: `results/cifar10-ours-batch-size-5090-20260929-v1/`. Preserve original V100 failure evidence in v1; v2 was cancelled pending without training.
