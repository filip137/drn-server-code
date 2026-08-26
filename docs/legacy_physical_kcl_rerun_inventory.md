# Legacy physical-KCL rerun inventory

Updated: 2026-08-26

## Scope and reason

This inventory covers active ordinary-MNIST manuscript evidence for the
`legacy` amplification setting `(voltage_amp, current_amp) = (4, 0.25)`.
The former interaction energy multiplied the receiving-node voltage by
`current_amp`.  It therefore minimized a coherent but differently
parameterized energy, rather than the physical KCL energy stated in the
manuscript.  Baseline `(1, 1)` and ours `(4, 1)` are unchanged because their
`current_amp` is one.

Historical medium-affine, hard-sigmoid, pooling, and superseded exploratory
studies are excluded unless the manuscript later elects to cite them.  The
official MNIST test split remains sealed.  The active compute priority is Adam
only.  The three clean-wide SGD arms remain mathematically affected, but are
deferred and are not part of the scheduled replacement queue.

## Active Adam training arms to replace

The Adam-first manuscript scope contains 18 affected legacy training arms.  Each
replacement starts from fresh matched initialization and retains its recorded
seed, data order, epoch budget, and parameter-wise learning-rate
vector unless a preceding corrected-model selection gate changes the final
paper contract.

| Block | Architecture | Optimizer/condition | Affected arms | Source authority |
|---|---|---|---:|---|
| Clean wide BPTT | Conv1 | Adam | 1 | `configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv1/05_legacy_adam_bias_zero_seed0.json` |
| Clean wide BPTT | Conv2 | Adam | 1 | Same study, Conv2 legacy config `05` |
| Clean wide BPTT | Conv3 | Adam | 1 | Same study, Conv3 legacy config `05` |
| Clean centered EqProp | Conv1 | Adam, no endpoint noise | 1 | `perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json` |
| Clean centered EqProp | Conv2 | Adam, no endpoint noise | 1 | Same study authority |
| Clean centered EqProp | Conv3 | Adam, no endpoint noise | 1 | Same study authority |
| Noisy centered EqProp | Conv1 | Adam, `sigma_v=5e-4` | 1 | `perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json` |
| Noisy centered EqProp | Conv2 | Adam, `sigma_v=5e-4` | 1 | `perfectdiode_conv2_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_30ep_seed0_20260817_v1.json` |
| Noisy centered EqProp | Conv3 | Adam, `sigma_v=5e-4` | 1 | Conv1/Conv3 noisy study authority above |
| Fixed-initialization bounded BPTT | Conv1 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/*legacy*.json` |
| Fixed-initialization bounded BPTT | Conv2 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | Same study, Conv2 legacy configs |
| Fixed-initialization bounded BPTT | Conv3 | Adam, `Gmax={1e-4,5e-4,1e-3}` | 3 | Same study, Conv3 legacy configs |
| **Active total** |  |  | **18** |  |

The bounded-Kaiming raw-LR transfer adds two supporting legacy arms (Conv1 and
Conv2) if that diagnostic remains in the manuscript discussion.  They are not
part of the 18 primary Adam table cells.

The deferred clean-wide SGD authorities are the `04_legacy_sgd` config in each
of the Conv1, Conv2, and Conv3 directories of the ordinary-MNIST zero-bias
study.  Conv1 already demonstrates that this is not a benign omission: its
corrected exact-rate retrain changed best validation accuracy by `-4.14 pp`.
These three arms must not be presented as corrected physical-KCL evidence; they
are simply outside the current compute priority.

### Exact 18-arm Adam execution checklist

Paths below are relative to `configs/conv/`.  A completed same-LR diagnostic
does not mark the corresponding paper replacement complete because the
corrected-model selection gate is still pending.

| # | Legacy arm authority | Status |
|---:|---|---|
| 1 | `perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv1/05_legacy_adam_bias_zero_seed0.json` | same-LR diagnostic complete; old rate is a viable corrected-model candidate; replacement pending |
| 2 | `perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv2/05_legacy_adam_bias_zero_seed0.json` | same-LR diagnostic complete and locally validated as Jean Zay job `1398318`, exit `0:0`, elapsed `01:44:46`, best/final validation `98.30/98.08%`; replacement selection pending |
| 3 | `perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv3/05_legacy_adam_bias_zero_seed0.json` | same-LR diagnostic complete and locally validated as Jean Zay job `1398320`, exit `0:0`, elapsed `03:16:51`, best/final validation `98.84/98.78%`; replacement selection pending |
| 4 | clean centered-EqProp authority `perfectdiode_conv123_zero_bias_adam_eqprop_one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json`, Conv1 legacy logical case `4` in pack `2` | pending |
| 5 | same clean centered-EqProp authority, Conv2 legacy logical case `8` in pack `4` | complete on `umg@v100` as `1416203`, exit `0:0`, elapsed `02:04:11`, best/final validation `98.30/98.06%`; injected beta `.03`, `T=K=6`, every guard and remote canonical validation passed, official test disabled; local collection pending the matched noisy case |
| 6 | same clean centered-EqProp authority, Conv3 legacy logical case `5` in pack `2` | complete and locally validated on `umg@v100` as `1399200`; exit `0:0`, elapsed `04:18:39`, best/final validation `98.84/98.78%`, official test disabled |
| 7 | noisy centered-EqProp authority `perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json`, Conv1 legacy logical case `4` in pack `2` | pending |
| 8 | noisy centered-EqProp authority `perfectdiode_conv2_zero_bias_adam_eqprop_one_decade_sigma_5em4_ordinary_mnist_30ep_seed0_20260817_v1.json`, Conv2 legacy logical case `2` in pack `1` | queued on `umg@v100` as `1416204`, released from its dependency and `PENDING (Priority)` at `2026-08-26 21:22 CEST`; non-binding Slurm start estimate `2026-08-27 02:18:48 CEST`; injected beta `.03`, `T=K=6`, `sigma_v=5e-4`, source/config and smoke gates passed |
| 9 | Conv1/Conv3 noisy authority above, Conv3 legacy logical case `5` in pack `2` | complete and locally validated on `umg@v100` as `1399201`; exit `0:0`, elapsed `04:20:39`, best/final validation `78.64/74.94%`, official test disabled |
| 10 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/02_wmax_1em4_legacy_adam.json` | pending |
| 11 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/05_wmax_5em4_legacy_adam.json` | pending |
| 12 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/08_wmax_1em3_legacy_adam.json` | pending |
| 13 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv2/02_wmax_1em4_legacy_adam.json` | pending |
| 14 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv2/05_wmax_5em4_legacy_adam.json` | pending |
| 15 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv2/08_wmax_1em3_legacy_adam.json` | pending |
| 16 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/02_wmax_1em4_legacy_adam.json` | queued on `umg@v100` as `1399202`, now `afterok:1416204` following the inserted Conv2 clean/noisy pair |
| 17 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/05_wmax_5em4_legacy_adam.json` | queued on `umg@v100` as `1399203`, `afterok:1399202` |
| 18 | `perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv3/08_wmax_1em3_legacy_adam.json` | queued on `umg@v100` as `1399204`, `afterok:1399203` |

If retained, the two supporting bounded-Kaiming authorities are
`perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/conv1_legacy_adam.json`
and
`perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/conv2_legacy_adam.json`.

## Corrected-model gates and derived analyses to repeat

The following legacy-only portions must be recomputed before corrected results
can replace manuscript evidence:

1. Conv1/2/3 legacy operational `T/K` checks and the three wide legacy Adam LR
   selection surfaces.  The immediate same-LR
   repeats below are controlled diagnostics, not a claim that the old rates
   remain selected for the corrected model.
2. Conv1/2/3 legacy EqProp beta-gradient qualification, including the clean
   tiers, endpoint-noise tiers, and the one-decade state-displacement rows.
3. The legacy columns of the manuscript beta-cosine and displacement figures.
   Baseline and ours source rows can be retained.
4. The nine legacy bounded final-weight distributions and all table deltas or
   narrative statements derived from them.  Baseline and ours checkpoints can
   be retained.
5. Any official-test evaluation only after the corrected inclusion set,
   selected checkpoints, and all preceding gates are frozen.  No official-test
   replay is authorized by this inventory.

## Conv1 first-wave diagnostic — complete

The first wave isolates two questions in order:

1. Replay the old Conv1 legacy SGD and Adam best/final checkpoints unchanged
   under the corrected physical equilibrium.  Compare the full 5,000-example
   validation accuracy with the recorded source accuracy and record the
   corrected voltage/score scales.  A descriptive absolute change of `0.5`
   percentage points or more is predeclared as material.
2. Retrain the clean wide Conv1 legacy SGD and Adam arms for 10 epochs with
   their exact existing rates:
   - SGD: `[2.05753e-4, 4.76259e-5, 0]`;
   - Adam: `[2.88917e-4, 3.64618e-5, 0]`.

Both use ordinary MNIST, seed/order `0`, exact-zero bias, `T=K=4`, batch size
16, validation batch size 64, and no official-test access.  The only intended
model change is the physical-KCL interaction energy.

### Results

All four full checkpoint replays and both 10-epoch exact-LR repeats completed
on 2026-08-25 and validate canonically.  The comparison uses the predeclared
`0.5 pp` materiality threshold.

| Optimizer | Test | Old validation | Corrected validation | Change | Material? |
|---|---|---:|---:|---:|---|
| SGD | unchanged best checkpoint | 95.92% | 93.84% | -2.08 pp | yes |
| SGD | unchanged final checkpoint | 95.60% | 93.88% | -1.72 pp | yes |
| Adam | unchanged best checkpoint | 96.44% | 96.12% | -0.32 pp | no |
| Adam | unchanged final checkpoint | 96.38% | 96.24% | -0.14 pp | no |
| SGD | exact-LR retrain, best | 95.92% | 91.78% | -4.14 pp | yes |
| SGD | exact-LR retrain, final | 95.60% | 91.78% | -3.82 pp | yes |
| Adam | exact-LR retrain, best | 96.44% | 96.50% | +0.06 pp | no |
| Adam | exact-LR retrain, final | 96.38% | 96.44% | +0.06 pp | no |

Measured conclusion: compatibility is optimizer-dependent.  The corrected
Conv1 legacy model preserves the old Adam result at this resolution, including
when trained from scratch with the same rate, but the old SGD checkpoint and
training rate both change materially.  The immediate next selection task is
corrected-model compatibility for the remaining Adam architectures.  The old
Conv1 Adam rate can remain a candidate, but this one-seed diagnostic does not
replace the remaining Conv2/Conv3, EqProp-beta, noisy, or bounded gates.  The
SGD result is retained as evidence of material sensitivity, but further SGD
work, including its LR qualification, is deferred under the Adam-only compute
priority.

Detailed Conv1 evidence is in the
[study report](../results/perfectdiode-conv1-legacy-physical-kcl-same-weights-and-bptt-rerun-seed0-20260825-v1/analysis/report.md).

## Immediate Conv2/Conv3 Adam wave

The next controlled diagnostic uses the unchanged clean-wide Adam configs and
the corrected physical-KCL source.  Both runs use ordinary MNIST, seed/order
`0`, exact-zero hidden biases, 30 epochs, batch size 16, validation batch size
64, and no official-test access.

| Order | Architecture | Exact weight learning rates | `T=K` | Jean Zay budget | Status |
|---:|---|---|---:|---|---|
| 1 | Conv2 legacy Adam | `[8.66750e-4, 1.52841e-4, 5.16982e-5]` | 6 | one V100, 3 h ceiling | completed as `1398318`, exit `0:0`, elapsed `01:44:46`, best/final validation `98.30/98.08%`; locally validated |
| 2 | Conv3 legacy Adam | `[2.60030e-3, 4.58737e-4, 3.25132e-4, 3.64861e-5]` | 8 | one V100, 5 h ceiling, `afterok` Conv2 | completed as `1398320`, exit `0:0`, elapsed `03:16:51`, best/final validation `98.84/98.78%`; locally validated |

The shared study root is
`perfectdiode-conv23-legacy-adam-physical-kcl-same-lr-bptt-rerun-seed0-20260826-v1`.
The two jobs are separate for failure isolation but dependency-serialized, so
the study can occupy at most one GPU at a time.  The authoritative frozen source
is commit `0efbfc4a` (archive SHA-256 `09439c41...ce8503`); both exact-config
local CPU smokes, the 115-test staged-source gate, and both scheduler test-only
requests passed before submission.  The topology-preserving local collection
matches Jean Zay exactly: result-archive SHA-256 `15776b9a...01770a`, 42 regular
files, 28 symlinks, regular-file digest `0a598a6e...d89e9`, and symlink digest
`cba1ad9a...2912d`; both canonical bundles validate remotely and locally.
