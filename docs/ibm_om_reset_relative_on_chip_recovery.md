# RESET-relative deployed-array pulse recovery

This protocol tests whether training the already programmed IBM OM array can
recover error that off-chip RESET-relative QAT did not anticipate. It starts
from one immutable pulse-resolved deployment, not from the clean checkpoint
or from newly reprogrammed weights.

## Source state

The source is the selected nine-level RESET-relative QAT checkpoint (selection
epoch 4, SHA-256
`537681598b887d4d63c6429c7f7bee3a7082041215c09d6cb3c25947452f311a`).
It is freshly deployed once on repaired assignment seed `84501` with endpoint
seed `84601`. The sidecar contains both states needed for continuation:

- `apparent_endpoint`: clipped conductances used by the network forward pass;
- `persistent_endpoint`: the physical simulator state from which the next
  SET or RESET pulse evolves; and
- `generator_state_after_programming`: the exact continuation of pulse and
  write-noise randomness.

Persistent accuracy is a hidden-state diagnostic. It is not a later read,
retention measurement, or noisy reread. Apparent-forward accuracy remains the
deployment result.

The runner validates the checkpoint, teacher, device model, assignment,
endpoint seed, target mapper, RESET commissioning, rail layouts, codebook,
population fingerprint, and full deployment hash before applying a pulse.
An arm cannot remap targets, inspect a hidden bound to choose an update, or
replace an unsupported cell.

## Physical scale

The nine-level codebook has logical contrast spacing `delta = 0.095849`.
Because one signed contrast is split across two same-sign cells, adjacent
per-cell offsets are `delta / 2 = 0.0479245` in the public normalized
conductance coordinate. The OM preset's nominal minimum one-pulse change in
that coordinate is `0.04745`. Thus one nominal pulse is approximately one
adjacent cell-offset step, but an actual pulse is nonlinear, stochastic, and
state dependent; it is not a guaranteed transition between QAT codes.

Recovery deliberately allows conductances to leave the original five cell
offsets. It never rounds a programmed endpoint back to a code or projects the
persistent state to the public range.

## Matched update paths

All gradient-directed paths use teacher-KL BPTT on the host. Only the physical
weight mutation differs.

### Rail refresh

This no-gradient control spreads a fixed number of single pulses across the
slow array. A cell originally above its quad RESET baseline receives SET; a
baseline cell receives RESET. It asks whether merely refreshing the intended
rail orientation helps, independently of learning.

### Direct pulses

For each physical conductance `G_i`, host BPTT computes its own gradient. The
desired update is `-learning_rate_i * gradient_i`. Its magnitude, divided by
`conductance_span * 0.04745`, becomes an unscaled one-pulse probability. One
positive scale is fitted on 64 replayed source-state minibatches so the
expected full-run requests equal the declared cap. A sampled positive update
issues one SET; a sampled negative update issues one RESET.

The calibration restores the training dataloader and global RNG states, so it
does not consume the first epoch's minibatch order.

### Ideal-fast Tiki-Taka

Every minibatch adds the normalized continuous desired update to a zeroed
host-side fast matrix. For each slow layer, a cyclic column schedule makes
`budget * number_of_columns` transfer visits over the full ten epochs. At a
visit, the accumulated row signal is stochastically converted to at most one
slow SET or RESET per cell and that fast column is cleared.

This separates the value of accumulation and sparse transfer from physical
fast-array imperfections.

### Physical-fast Tiki-Taka

Each slow conductance receives a repaired-OM fast differential pair. The fast
assignment is `85501` and endpoint seed is `85601`. Commissioning applies
eight RESET/read samples, takes the guarded maximum of each pair, and
equalizes both rails to that observed baseline by adaptive verify. Gradient
sign selects the fast plus or minus rail for a stochastic SET pulse. A cyclic
column transfer reads the apparent pair contrast, requests slow pulses, and
adaptively RESETs the transferred pair slice.

Fast SET/RESET pulses, commissioning, equalization, verify reads, reversals,
failures, and unique cells are reported separately from the matched slow
budget.

## Budgets and selection

The slow array has `158,800` conductances. The three whole-run caps are:

| Budget per slow cell | Maximum requested slow pulses |
| ---: | ---: |
| 0.1 | 15,880 |
| 1 | 158,800 |
| 4 | 635,200 |

Every arm runs exactly ten epochs (`34,380` minibatches) and retains epoch 10.
There is no validation-based checkpoint selection. All twelve arms use one
seed initially; the pilot can check mechanism and implementation, but cannot
rank methods statistically.

## Gain and evaluation

The original deployed gain is `562.341325190349`. Before training and after
each epoch, the runner also fits one positive gain on the fixed 1,024-example
training calibration cohort. It evaluates apparent and persistent states at
both gains on all 5,000 validation examples and proves that positive gain
does not change argmax accuracy. Gain can improve KL and gradient scale; it
cannot itself explain an accuracy change.

Each epoch checkpoint retains every slow and fast state, RNG stream, direct
probability calibration, transfer position, pulse ledger, logit gain,
dataloader generator, and metric history. Resume therefore continues the
same physical trajectory rather than starting a statistically similar one.

## Evidence boundary

This is a one-seed development pilot. It uses the fitted AIHWKit OM preset and
host-computed gradients, not fabricated on-chip gradient circuitry. No test
array is released. A viable method and budget must be frozen before two more
development repeats and a separately declared formal deployment study.

The fixed p90/p95/p99 absolute-gradient follow-up is specified separately in
[`ibm_om_reset_relative_direct_gradient_thresholds.md`](ibm_om_reset_relative_direct_gradient_thresholds.md).
It reuses this pilot's exact direct-pulse scales and does not redistribute
suppressed writes.
