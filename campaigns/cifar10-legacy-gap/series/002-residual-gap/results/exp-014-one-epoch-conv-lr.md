---
experiment: "exp-014"
evidence: "validated-local"
summary: "Tripling legacy convolution LR at epoch30 worsens fixed-cohort fitting and validation within100 updates and fails the one-epoch screen; no extension warranted."
verdicts: {}
---
# One-epoch convolution LR screen

Both branches completed 1,407 matched updates from the same epoch-30 model/Adam
state. Eight fixed-cohort measurements validated; initial CE/accuracy match exactly.
No official test read. Native epoch-31 model, Adam, scheduler and RNG checkpoints
are preserved for a separately assigned continuation. No extension launched.

Root: `results/cifar10-conv-lr-one-epoch-20260928-v1/`; bundle `probe/`;
full arithmetic `analysis/summary.json`; figure `analysis/one_epoch_response.jpg`.

| Updates | Training CE, original | Training CE, 3× | Training CE improvement | Validation CE change | Validation accuracy change |
|---|---:|---:|---:|---:|---:|
| 0 | 0.147438 | 0.147438 | +0.00% | +0.00% | +0.000 pp |
| 100 | 0.133340 | 0.245941 | -84.45% | +37.99% | -4.590 pp |
| 400 | 0.139837 | 0.277773 | -98.64% | +30.44% | -3.516 pp |
| 1407 | 0.135835 | 0.269630 | -98.50% | +28.43% | -4.199 pp |

Predeclared screen pass: False. All checkpoints shown; no endpoint selection.
These fixed evaluation cohorts and one seed do not establish a long-run accuracy
gain, an optimal LR, or the cause of the historical legacy/ours gap.

Scientific review: the 3× branch is worse at every measured post-update checkpoint.
At 100 updates its fixed training CE is 84.45% higher than the control and validation
accuracy is 4.59 pp lower. At the complete epoch, training CE is 98.50% higher
(0.269630 versus 0.135835), validation CE is 28.43% higher (0.475926 versus
0.370573), and validation accuracy is 82.715% versus 86.914% (-4.199 pp).
Both branches began at exactly the same fixed training CE 0.147438 and validation
accuracy 86.719%. The control improves relative to that start while the 3× branch
remains worse at the end. Initial model/Adam equality, matching training-order
hashes and the intended convolution-only LR ratio were validated.

This is not solely a saved-stat BN evaluation observation: the augmented online
training stream also has higher CE under 3× (0.376966 versus 0.231565) and lower
accuracy (86.802% versus 92.042%). That corroborates impaired fitting during this
screen, without separating the underlying weight/BN/Adam dynamics.

Decision: reject this particular abrupt 3× convolution-LR increase as a promising
short-horizon intervention from the epoch-30 checkpoint; do not extend either
branch automatically. The adverse response is already visible at 100 updates,
so ten epochs were unnecessary for this screen. This does not prove that the
original LR is optimal or too large, rule out every milder LR increase or warmup,
or explain the historical legacy/ours gap. Long-run recovery was not tested.

All coverage and native epoch-31 checkpoint validation passed. Production took
31.0 minutes; total GPU execution/smoke/validation took 31.3 minutes, within the
90-minute cap. Peak reserved memory was 11,124 MiB. No failures or further runs.
