# RESET-trained cohort-A MNIST DRN

This exploratory experiment tests whether the original dual-rail DRN can learn
from raw measured ReRAM trajectories without first mapping a digital teacher
into conductances. It is registered as `mnist_relu_drn_reset.v1`.

The frozen teacher is the bias-free `784 -> 50 -> 10` ReLU network. The student
is the bias-free `1568 -> 100 -> 20` DRN with one non-negative memristor on each
physical edge. Its ten logits are the differences between paired output rails.
The fixed readout gain is exactly one.

Every assigned cohort-A virtual device starts at measured pulse index zero. No
teacher weight, affine conductance mapping, initial target write, deadband, or
probabilistic write is used. Training retains the established per-device
global-nearest projection of each digital-shadow update onto its raw measured
pulse-conductance graph. The circuit uses `voltage_amp = 4` and
`current_amp = 0.25`.

Two arms use the same MNIST split, device split, virtual-device assignments,
RESET state, and training budget:

- `teacher_kl` minimizes `KL(softmax(teacher) || softmax(DRN))`; labels are used
  only for metrics.
- `cross_entropy` is the matched hard-label control.

Each arm independently selects its two conductance learning rates from RESET
using the bounded relative-update probe, safety canaries, and 3-by-3 validation
grid. Every cell is restored to the exact pre-probe state before it runs, and
the production training starts from that same state.

The canonical configs are
`examples/mnist_relu_drn_reset/cohort_a_kl.json` and
`examples/mnist_relu_drn_reset/cohort_a_cross_entropy.json`. The campaign is
`campaigns/manifests/mnist_relu_drn_reset.json`. Raw outputs belong below
`results/mnist-relu-drn-reset-exploratory-20260816/`.

The hypothesis is supported when the KL arm reaches at least 94% held-out test
accuracy and is no more than one percentage point below the matched
cross-entropy arm. Test data are read only after validation-based learning-rate
and checkpoint selection. A post-hoc positive logit scale is fitted on the
validation split only to report calibrated test KL; it never changes training,
checkpoint selection, or accuracy.

## Exploratory result

The six-stage local-GPU campaign completed successfully. The teacher reached
`97.36%` held-out test accuracy. Both DRN arms began at `9.96%` validation
accuracy from the identical pulse-zero assignments, and all nine learning-rate
cells passed their safety canaries.

| Objective | Selected learning rates (hidden, output) | Selected validation accuracy | Test accuracy | Test teacher agreement | Raw test KL | Report-only calibrated test KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `teacher_kl` | `(6.824e-7, 8.823e-8)` | `83.60%` | `85.52%` | `86.35%` | `0.68973` | `0.38257` |
| `cross_entropy` | `(2.260e-7, 8.707e-8)` | `85.44%` | `86.88%` | `87.61%` | `0.67903` | `0.35383` |

The predefined hypothesis gate did not pass: KL accuracy is below `94%` and
is `1.36` percentage points below the cross-entropy control, just outside the
one-point tolerance. This is nevertheless real learning from RESET rather than
an unchanged initialization: both layers changed pulse states, and KL improved
validation accuracy by `73.64` points over the RESET operating point.
Training was strongly non-monotonic under global-nearest raw-trace projection,
so validation-selected checkpoints were essential.

The matched control indicates that KL is not the main source of the shortfall:
hard-label cross-entropy improves test accuracy by only `1.36` points. This
experiment is also not an exact reproduction of the earlier approximately
`95%` measured-cohort-A from-scratch run. That historical `small_drn.v1` run
trained `base.bias.0` and used `SquaredErrorPairedOutputs`; the present student
is deliberately bias-free and uses KL or cross-entropy. A completed exact
historical-control replay subsequently found an additional confound: production
training used process-global amplifier indices 3/4/5, while LR selection and
fresh-process testing used 0/1/2. That replay recovered the archived `95.97%`
only with paired squared error; cross-entropy reached `92.37%` and KL `83.72%`.
See [`mnist_relu_drn_reset_bias.md`](mnist_relu_drn_reset_bias.md). The archived
result is therefore not a valid target for the corrected logical-index circuit.

Machine-readable metrics and the matplotlib comparison are in
`results/mnist-relu-drn-reset-exploratory-20260816/analysis/summary.json` and
`validation_comparison.png`.
