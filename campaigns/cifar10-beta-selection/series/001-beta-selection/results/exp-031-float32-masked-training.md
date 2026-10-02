---
experiment: "exp-031"
evidence: "validated-local"
summary: "Mask and unmasked float32 hybrid EqProp finish epoch5 level: 60.88%/CE1.098 versus 60.68%/CE1.089 (+0.20 points, higher CE); H-026 contradicted. Both arms exceed earlier fixed-beta float32 ours (54.12%) but remain 5.8 points below native BPTT (66.68%)"
verdicts: {"H-026": "contradicts"}
---
# Result: the force mask does not change five-epoch float32 learning

## Evidence and coverage

Both arms completed all 5 epochs (7035 updates, 20 calibrations each) with exit
0 and passed the queue collection hook (`analysis/{mask,nomask}-collection.json`,
`validated: true`). The run bundles show identical initializer, data order,
targets and calibration schedule, float32 throughout, and no official test read.
Root `results/cifar-float32-masked-training-20261001-v1/`. Placement: mask on
nom-cool-1 RTX3090 (shared with Ben), nomask on the local nom-cool-2 RTX3090,
pinned PyTorch2.11.0+cu128. Runtime 3.80h / 3.54h (2725 / 2550s per epoch;
the difference is confounded with GPU sharing), peak reserved memory 11.06GiB
in both arms, about 7.3 GPU-hours in total against a 20-hour ceiling. No Jean Zay
job was submitted.

## Measurements

Validation (5k held-out training images), every epoch:

| Epoch | mask acc / CE | nomask acc / CE | Δacc (points) |
|---:|---|---|---:|
| 1 | 32.40% / 1.806 | 31.06% / 1.856 | +1.34 |
| 2 | 44.42% / 1.548 | 46.04% / 1.504 | −1.62 |
| 3 | 51.42% / 1.355 | 53.16% / 1.307 | −1.74 |
| 4 | 54.52% / 1.307 | 58.32% / 1.175 | −3.80 |
| **5** | **60.88% / 1.0982** | **60.68% / 1.0890** | **+0.20** |

Training accuracy at epoch 5: mask 57.07%, nomask 58.57%.

**H-026's frozen rule:** +0.20 points ≤ 0.5, and the mask arm's CE is higher, so
the result **contradicts** the hypothesis.

Reported without grading:

- **Gap to native BPTT** (ours epoch5, 66.68%/CE0.942): −5.8 / −6.0 points.
- **Inactive block3 channels (mask arm).** Mean per update by epoch:
  10.6, 2.0, 2.0, 3.1, 6.0 (max 32, 3, 3, 5, 12). They are present in almost
  every update and carry 97–99% of block3's Σf² in every epoch. Block2 had
  inactive channels in only 6 updates in epoch 1; block1 had none.
- **Betas** at the last calibration of each epoch (blocks 1/2/3):
  mask [1.53,105,151] → [0.13,65.4,71.6]; nomask [2.16,185,61.7] →
  [0.12,37.4,88.4]. Both arms grew from the initial [2.9e-4,2.2e-4,5.5e-4]
  by roughly 10³–10⁵ within the first 352 updates. At matched RMS the arms'
  betas differ by up to 2.5× in block3 (mask larger in epoch 1) and 1.7× in
  block2 (mask larger at epoch 5).
- **Calibration accuracy:** every calibration hit all three targets within 3%
  relative error.
- **Training-time RMS** (median of 19 per-epoch observations, blocks 1/2/3):
  epochs 2–5 are 0.005–0.006 / 0.019–0.022 / 0.0022–0.0027 in both arms, on
  target. Block3 sits below target in epoch 1 (0.0015 mask, 0.0007 nomask),
  before the betas had grown. Occasional block3 excursions reach 0.012 (mask)
  and 0.021 (nomask).

## Interpretation

Measured: at one seed, the mask changes the block3 estimator in nearly every
update, removing almost all of its injected force, yet the two arms' learning
curves differ by at most 3.8 points at any epoch and end 0.2 points apart. The
per-epoch sign alternates, which is consistent with seed-level variation rather
than a systematic effect. On the replayed ours fixed-0.3× checkpoints (exp-030),
the 0.0025 block3 target lies above the unmasked window (3e-5 at epoch1, 2.5e-4
at epoch5) and inside the masked one. If these trajectories behave similarly, the
unmasked arm trained with degraded block3 gradients for most updates; its smaller
early block3 beta at matched RMS fits dead units switching and contributing
displacement. Inferred, untested: block3 gradient quality at this level does not
limit five-epoch learning. Gradient agreement was not measured on these
trajectories, so this remains a hypothesis.

Uncontrolled observation: both arms reach about 60.8%, 6.7 points above the
earlier float32 ours fixed-0.3× run (54.12%, [exp-013](exp-013-all-scheme-long-eqprop.md))
and 7–9 points above both exp-010 arms (51.82% once-per-epoch recalibration,
53.56% fixed). exp-013 used the same
T/K and schedule but a fixed beta. The improvement therefore most plausibly comes from
four-per-epoch recalibration to the exp-030 targets [0.005,0.02,0.0025], not
from the mask. This is a different contract, not a matched control, so it is a
candidate for a direct test, not a finding.

## Decision

Do not adopt the force mask as a training intervention on the strength of this
evidence; it stays available as an exact estimator option. The more promising
lever is the periodic-calibration policy with these small block targets. A
matched float32 comparison against a fixed-beta arm at the same initializer
would test it directly, along with a second seed for the arm difference. The
remaining 5.8-point gap to native BPTT is unexplained. Candidates are block2's
float32 floor at initialization (exp-030) and the early epoch-1 betas, which
undershoot block3's target.
