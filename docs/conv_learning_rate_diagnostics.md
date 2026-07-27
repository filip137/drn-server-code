# Conv Learning-Rate Diagnostic Ledger

Updated: 2026-07-27

Status: diagnostic curation only. None of the studies in this file replaces
the deterministic-medium-affine Conv1/Conv2 v1/v3 handoff, resolves the
medium-affine Conv3 learning rate, selects the paper optimizer, or authorizes
final paper training.

The canonical paper handoff is
[`conv_paper_learning_rate_protocol.md`](conv_paper_learning_rate_protocol.md).
This ledger preserves the important identities, outcomes, and stop reasons
from follow-up optimizer investigations without repeating launcher,
scheduler-accounting, or wrapper-debug narration.

## Curation Rules

- Validation metrics are from MNIST-train validation splits, not the official
  test split.
- Ordinary-MNIST results are optimization diagnostics, not paper-facing
  deterministic-medium-affine accuracy evidence.
- `unbracketed` means the passing plateau touched a declared search boundary;
  the diagnostic best is not a frozen LR.
- A failed safety gate is valid negative evidence only for the exact attempted
  row; it is not a low-accuracy completed candidate.
- Partial studies retain their immutable config and evidence identity but
  freeze no selection.
- Configs and source files that participate in content-addressed provenance
  remain tracked even after a diagnostic is retired. Obsolete job narration
  and duplicate artifacts do not.

## Ordinary-MNIST learning protocols

The design is separated by nonlinearity:

- [`Hard-Sigmoid Conv1/2/3 Ordinary-MNIST Learning Protocol`](hardsigmoid_learning_protocol.md);
- [`Perfect-Diode Conv1/Conv2 Ordinary-MNIST Learning Protocol`](perfectdiode_learning_protocol.md).

Each defines its own identity namespace, restarted 640-step canaries, exact
three-epoch candidates, an inclusive 90% validation-accuracy gate, and at
most one factor-of-three expansion in either direction. The perfect-diode v1
pipeline contains 12 Conv1/Conv2 scheme x optimizer surfaces and requires a
fixed-`T/K` gradient security comparison with `(T=64,K=64)` before probing.

The user-fixed perfect-diode operating points are diagnostic choices shared
across schemes:

| Architecture | Input gain | Operational `T/K` |
|---|---:|---:|
| Conv1 | `40` | `4/4` |
| Conv2 | `100` | `6/6` |

The canonical Conv1/Conv2 config is
[`perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json`](../configs/conv/perfectdiode_conv12_sgd_adam_hparam_ordinary_mnist_v1.json),
with study identity
`lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46`.

### Perfect-diode v1 core execution — incomplete boundary study

The fixed-`T/K` security checks, probes, core canaries, and promoted
three-epoch candidates completed. The diagnostic winners were generally
strong, but the tested rho range was not sufficient to bracket eleven of the
twelve surfaces. The five Conv2 surfaces requested for the next continuation
are:

| Scheme | Optimizer | Passing / trained | Provisional core winner `(rho_conv,rho_dense)` | Final validation accuracy | Core decision |
|---|---|---:|---:|---:|---|
| baseline | SGD | `9/9` | `(0.009,0.03)` | `95.70%` | expand both |
| baseline | Adam | `8/9` | `(0.009,0.03)` | `95.90%` | expand both |
| proposed/ours | SGD | `9/9` | `(0.009,0.03)` | `97.18%` | expand both |
| proposed/ours | Adam | `9/9` | `(0.009,0.03)` | `97.18%` | expand both |
| legacy | SGD | `8/8` | `(0.003,0.03)` | `96.78%` | expand both |

The complete passing plateaus, rather than only the displayed winner cells,
implicate both upper edges. The prescribed one-wave continuation adds
`rho_conv=0.027` and `rho_dense=0.09`; each surface adds seven new Cartesian
cells, including the corner, and reuses all existing cells by hash. These
results suggest testing larger rho targets, not that a final optimum has
already been found.

Candidate measurements are intact, but `select_core` publication stopped on
a structured safety-failure/schema mismatch. Only the Conv1 baseline-SGD and
Conv2 baseline-SGD `needs_expansion` entries were published before the
failure. The table applies the intended selector read-only to all surfaces;
the other ten rows are reconstructions. None is a frozen LR handoff. No
extension, `select_final`, or long confirmation has completed or been
launched. The full twelve-surface, full-precision table is
[`conv_perfectdiode_lr_core_screen_20260727.csv`](conv_perfectdiode_lr_core_screen_20260727.csv);
the protocol-level interpretation is in
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md).

These values do not replace the pending medium-affine paper calibration or
handoff. Perfect-diode Conv3 is outside this v1 execution pipeline and remains
unresolved.

## V4 Medium-Affine Layer-Wise Diagnostic — `retired_incomplete`

Canonical config:
[`hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json`](../configs/conv/hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json)

Study identity:
`lrstudy_b345477de5ff64c828a9d31602314a00046bc9728039562d3e5011132805e14d`

V4 tested whether independent per-weight constant LRs improved on a shared
scalar LR. Each bounded weight used its own initialization-scale Q90
coefficient; each Conv bias inherited its attached Conv-weight LR. It covered
the six medium-affine Conv1/Conv2 rows and was diagnostic from the outset.

An earlier ignored local root,
`simulation_results/conv_lr_layerwise_v4_20260720`, reached the probe stage
only. The most complete partial attempt is under
`simulation_results/conv_lr_layerwise_v4_20260720_r2`. Its exact state is:

- probe: `6/6` rows complete; completion SHA-256
  `0deb18cc65204a8c259a1a6104d6a0227e4465a3c6173fc0c79fe722073b8295`;
- range: `6/6` rows complete; completion SHA-256
  `2750551b05112ebf46dff67f6e91c2620c1e374f5b757e4224f5887811ae5485`;
- candidate manifest: 18 entries; SHA-256
  `0edff50911c1e2b54a0b3ad5b7b5bafcffe3f4ec37fddea6e2fe388c85e9d9b2`;
- candidates: `9/18` complete, exactly the three roles for each Conv1 row;
- candidate-stage completion: absent;
- selection-stage completion: absent;
- frozen v4 LR vectors: none.

All nine completed Conv1 candidates were admissible and ran 17,190 constant-LR
steps:

| Scheme | Role | Target | Conv-weight LR | Dense-weight LR | Final validation loss | Final validation accuracy |
|---|---|---:|---:|---:|---:|---:|
| baseline | fast | `0.003` | `0.03683882748751181` | `0.0010351090187622855` | `0.3469991213798523` | `56.18%` |
| baseline | middle | `0.03` | `0.3683882748751181` | `0.010351090187622855` | `0.3308339047431946` | `59.40%` |
| baseline | high | `1` | `12.279609162503936` | `0.3450363395874285` | `0.3614371646881103` | `50.10%` |
| proposed/ours | fast | `0.001` | `0.00227998854222303` | `0.00006128047453430722` | `0.35977697191238406` | `52.22%` |
| proposed/ours | middle | `0.01` | `0.0227998854222303` | `0.0006128047453430722` | `0.31869862146377564` | `63.64%` |
| proposed/ours | high | `0.1` | `0.227998854222303` | `0.006128047453430722` | `0.37209602184295654` | `47.94%` |
| legacy | fast | `0.001` | `0.00018109659132678927` | `0.0000046391141584621624` | `0.4518673370361328` | `35.92%` |
| legacy | middle | `0.01` | `0.0018109659132678926` | `0.00004639114158462162` | `0.3432171362876892` | `57.34%` |
| legacy | high | `0.1` | `0.018109659132678927` | `0.00046391141584621625` | `0.38923702583312986` | `40.94%` |

V4 is retired operationally because the active v1/v3 scalar handoff is already
complete and no current paper gate depends on finishing this diagnostic. Later
v5-v7 and boundary studies overtook the optimizer investigation. V4 is not
scientifically equivalent to v5: it used medium-affine data and a different
normalization. Preserve its config and this tombstone because real partial
measurements exist; do not describe it as wholly pending, complete, or
selected.

## V5 Ordinary-MNIST Architecture-Level Median Diagnostic

Canonical config:
[`hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json`](../configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json)

Study identity:
`lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2`

V5 disabled the affine transform while retaining the six frozen rows, gains,
and row-specific `T/K`. It compared strict per-weight equalization against a
historical-profile arm using 36 five-epoch constant-vector candidates.

The historical anchors used incompatible padding-0 models without verified
initialization bytes. Therefore the immutable audit correctly records
`derivation_status=fallback_predeclared`; the alpha centers were not
empirically recovered from those runs.

| Architecture | Selected arm / alpha | Worst-row validation accuracy | Worst-row validation loss | Comparator arm / alpha | Comparator worst loss |
|---|---:|---:|---:|---:|---:|
| Conv1 | historical profile / `1e-3` | `94.82%` | `0.11973580062389373` | strict / `1e-2` | `0.12696153950691222` |
| Conv2 | strict / `3e-3` | `93.18%` | `0.1334704913377762` | historical profile / `5e-4` | `0.15356838121414185` |

All 36 candidate artifacts were published; 32 candidates completed the full
17,190-step schedule and four stopped on the loss-EMA gate. Selection records
`official_test_read=false` and `final_paper_training_authorized=false`.

## V6 Conv2 Baseline Two-Rho Diagnostic

Canonical config:
[`hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json`](../configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json)

Study identity:
`lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f`

V6 swept the ordinary-MNIST Conv2 baseline over:

```text
rho_conv  = {5e-4, 1e-3, 3e-3, 1e-2}
rho_dense = {3e-3, 1e-2, 3e-2, 1e-1}
```

All 16 baseline candidates completed. Fifteen passed the inclusive 90%
accuracy gate. The diagnostic best was `(1e-2,3e-2)` at validation loss
`0.1215850609` and accuracy `94.28%`; `(1e-2,1e-2)` was also in the 2%
plateau. Because the complete plateau touched the maximum `rho_conv` edge, the
selector returned `unbracketed` and froze no pair.

A separately authorized transfer used identity
`lrtransfer_b75712190eb0ae6ea818fd6b54f335ee22171d34aeed2c9b82de9bbd454d7d14`.
Applying `(1e-2,3e-2)` to ours and legacy with scheme-specific probe medians
made both rows hit the loss-EMA gate at onset step 33 / confirmation step 40.
Neither reached an epoch or validation evaluation. This does not select the
v6 boundary pair.

## Conv1 Amplified Scheme-Specific Two-Rho Follow-Up

Sweep identity:
`lrsweep_77b1676e0d417242638c34901e2ec247210993a578fdf6c621df660ed54e84ae`

Manifest SHA-256:
`c338c434c7cfa26bdb69e5c8e24d955a06f10a3d83d2cc4a09041ce070abac01`

The ordinary-MNIST sweep selected ours and legacy independently over:

```text
rho_conv  = {1/3000, 1e-3, 3e-3, 1e-2}
rho_dense = {3e-3, 1e-2, 3e-2}
```

Both lowest-loss cells occurred at the outward corner `(1e-2,3e-2)`: ours
reached `95.76%` / `0.104888`, and legacy reached `95.22%` / `0.113515`.
Both selectors returned `unbracketed`; no pair is frozen.

## Conv2 Amplified Scheme-Specific Two-Rho Follow-Up

Sweep identity:
`lrsweep_07eed332608b684ff77bcf994071b90230fd877345b2e5132d9144cc4fa40435`

Manifest SHA-256:
`6a220748c1d133c5bcfa33dad7de9c9965c76d3ac0f9883afe11577f1adf6b0d`

The ordinary-MNIST sweep selected ours and legacy independently over:

```text
rho_conv  = {5e-4, 1.5e-3, 3e-3, 1e-2}
rho_dense = {3e-3, 1e-2, 3e-2}
```

Both diagnostic bests were `(1e-2,1e-2)`. Ours reached validation accuracy
`96.26%` and loss `0.0936492513`; legacy reached `95.84%` and
`0.0986378435`. Both passing plateaus touched the maximum `rho_conv` edge, so
both selectors returned `unbracketed` and froze no pair. Ours passed 11/12
cells; legacy passed 8/12 and failed every `rho_dense=3e-2` cell on the
loss-EMA gate.

Final summary JSON / CSV SHA-256:
`de02400c87af51b6d320da04bcc717f73572b7dc2a413c8c974398d6f39bb0fb`
/ `be92b24b6192eb2debf45e54414315db1f5a96598aa8a1eb2d870a89fa8825b6`.

## Conv2 SGD/Adam Boundary Extension

Focused protocol:
[`conv2_sgd_adam_boundary_diagnostic_protocol.md`](conv2_sgd_adam_boundary_diagnostic_protocol.md)

Canonical config:
[`hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json`](../configs/conv/hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json)

Study identity:
`lrstudy_87864fb9a23c07e5761d61bdbc9104b869d01fd5b9697aa768a3c5a7f588f931`

The attached-bias ordinary-MNIST surface is complete: 36 main cells plus ten
triggered `rho_conv=0.1` sentinels. Adam was tested only on this declared local
high-rho surface after optimizer-specific initial update normalization.

| Scheme | Optimizer | Selected `(rho_conv,rho_dense)` | Final validation loss / accuracy | Boundary status |
|---|---|---:|---:|---|
| baseline | SGD | `(0.1,0.01)` | `0.11471955 / 94.56%` | `unbracketed_high` |
| baseline | Adam | `(0.1,0.01)` | `0.08561958 / 96.10%` | `unbracketed_high` |
| ours | SGD | `(0.03,0.01)` | `0.08794243 / 96.66%` | `unbracketed_high` |
| ours | Adam | `(0.03,0.003)` | `0.07726470 / 96.88%` | `unbracketed_high` |
| legacy | SGD | `(0.03,0.01)` | `0.08837134 / 96.28%` | `bracketed` |
| legacy | Adam | `(0.015,0.01)` | `0.05492024 / 97.86%` | `unbracketed_low` |

The epochwise replay supplies the useful mechanistic result. Legacy SGD's
achieved Conv/Dense rho fell from approximately `3%/1%` at initialization to
`0.058--0.142% / 0.026%` at epoch five. Legacy Adam retained approximately
`0.301--0.331% / 0.211%`. Adam reduced fixed-SGD underexposure but did not
resolve amplification-induced gradient rotation or whole-network calibration.

Aggregate / selection SHA-256:
`c089fdc5e931f393bbcbfb842be5ecce833ca57b5a9338269848255383e5ce44`
/ `2f1812ca5f8e8afa3ea03f1f262f933e6662bafd90661d0cc45c1d960614666a`.

The original design included six separately bias-capped confirmations. The
user stopped that branch after selection to keep the conclusion focused on
learning rates. No partial confirmation is scientific evidence. Therefore the
attached-bias surface is complete, while the original capped-bias branch is
explicitly canceled rather than completed.

## V7 Conv3 Ordinary-MNIST Study — Canonical Execution Incomplete

Canonical config:
[`hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json`](../configs/conv/hardsigmoid_lr_conv3_scheme_two_rho_constant_sgd_bs16_v7.json)

Study identity:
`lrstudy_6e56b399cf3e522a2c3f39f66a51611147b6171dd3b9e223f20791bc6275caf0`

V7 intentionally reuses the medium-affine-calibrated Conv3 gains and `T/K`
while disabling the affine transform:

| Scheme | Frozen `input_gain` | Operational `T/K` |
|---|---:|---:|
| baseline `v1/c1` | `251.3061370850` | `24/8` |
| proposed/ours `v4/c1` | `744.7390747070` | `32/8` |
| legacy `v4/c0.25` | `665.0302124023` | `8/6` |

Its per-scheme core is a `3 x 3` grid of three-epoch constant-vector
candidates, with an inclusive 90% validation-accuracy gate and at most one
upper-edge extension. All nine legacy core cells failed safety. The canonical
baseline/ours work and full three-row handoff remain incomplete. Any eventual
selection is ordinary-MNIST optimization evidence only.

### Legacy lower-target rescue

Config:
[`hardsigmoid_lr_conv3_legacy_low_rho_rescue_bs16_v1.json`](../configs/conv/hardsigmoid_lr_conv3_legacy_low_rho_rescue_bs16_v1.json)

Identity:
`lrrescue_e9c1d08dc339f723f61d3d73b8059fe5f82fe95952a5d1fa06107ed877141a04`

| Target `(rho_conv,rho_dense)` | 256-step validation accuracy | Epoch 1/2/3 validation accuracy | Final loss | Result |
|---|---:|---:|---:|---|
| `(5e-5,3e-4)` | `53.88%` | `64.80 / 72.16 / 76.38%` | `0.24017` | below 90% |
| `(1.5e-5,1e-4)` | `25.48%` | `40.24 / 61.50 / 72.46%` | `0.27001` | below 90% |
| `(5e-6,3e-5)` | `19.36%` | `36.88 / 52.46 / 57.70%` | `0.32409` | below 90% |

All three promotions completed 10,314 steps safely, but none met the accuracy
gate. No LR is frozen. Completion SHA-256:
`f5064dc0639dabf6edeb51f3e216fa5bb1629ab7813773a756c06899546e0c53`.

### Gain-200 sensitivity

Config:
[`hardsigmoid_lr_conv3_legacy_gain200_rho5e5_3e4_bs16_v1.json`](../configs/conv/hardsigmoid_lr_conv3_legacy_gain200_rho5e5_3e4_bs16_v1.json)

Identity:
`lrgain_0dc9dc42f5cb2044a9c4d7e4d4a1ab8fab4e3f4b4f356837fde4de1c3f74e303`

This control changed only legacy gain from `665.0302124023` to `200.0` while
retaining the raw LR vector of rescue source `(5e-5,3e-4)`. It did not re-probe
rho. It hit the loss-EMA gate at onset step 33 / confirmation step 40, before
validation, and froze no LR. Completion SHA-256:
`2a243b7cf6ab3c886c3d61fcdc1843251dd155db3b19a840641a9273ef1db8ed`.

### Gain-200 `Bias_2`/10 control

Config:
[`hardsigmoid_lr_conv3_legacy_gain200_bias2_0p1x_bs16_v1.json`](../configs/conv/hardsigmoid_lr_conv3_legacy_gain200_bias2_0p1x_bs16_v1.json)

Identity:
`lrbias2_0135c2403354462edfe121ed198c0ff973132303ff4f28ed7ca5d13bda29ef81`

This control changed only `LR(Bias_2)` from `3.27987781947773e-4` to
`3.27987781947773e-5`. It completed all 10,314 steps safely. Validation
accuracy was `48.00%`, `66.58%`, and `73.26%`; final loss was
`0.2937524602`. It failed the 90% gate and underperformed the gain-665 rescue
source at `76.38%`. It freezes no LR. Result SHA-256:
`0d4f8bd7b305d44d629129f02c66894a5d9de8407394df6e6ffc89afb8fd6b79`.

## Retained Checkpoint-Replay Tooling — No Curated Result

The saturation and gradient replay CLIs are retained as read-only
ordinary-MNIST diagnostic tooling. They verify run-spec and checkpoint hashes,
leave checkpoint bytes and parameter tensors unchanged, use deterministic
MNIST-train validation cohorts, and never read the official test split.

No saturation- or gradient-replay result has been generated or curated for
this ledger. The current implementations accept the matched ordinary-MNIST
v5/v6 checkpoint contract only; they do not analyze the active
deterministic-medium-affine v1/v3 handoff unchanged. Their presence therefore
does not add paper evidence or alter any frozen LR.

## Diagnostic Bottom Line

- V4 is a retired incomplete medium-affine diagnostic with real partial
  measurements and no selection.
- V5 selected diagnostic architecture-level settings on ordinary MNIST.
- V6 and both scheme-specific follow-ups were unbracketed.
- The Conv2 boundary study shows that Adam better maintains achieved updates
  on its tested local surface, but it is not a global optimizer selection.
- V7 and its controlled legacy diagnostics freeze no medium-affine Conv3 LR.

None changes the canonical v1/v3 paper handoff or the final-training gate.
