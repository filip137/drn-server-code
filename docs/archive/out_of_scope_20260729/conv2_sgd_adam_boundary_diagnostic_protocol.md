# Conv2 SGD/Adam Boundary-Extension Diagnostic

Updated: 2026-07-25

Status: the attached-bias LR surface is complete with 36 main cells and ten
triggered upper sentinels. The user stopped the separately capped-bias
confirmations to keep the conclusion focused on learning rates. No result from
this study is part of the frozen deterministic-medium-affine learning-rate
handoff.

Canonical config:
[`hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json`](../../../configs/conv/hardsigmoid_lr_conv2_sgd_adam_boundary_constant_bs16_v1.json)

- Study ID:
  `lrstudy_87864fb9a23c07e5761d61bdbc9104b869d01fd5b9697aa768a3c5a7f588f931`
- Config SHA-256:
  `1093e198acb0bd770fce07bb3efdaba769d9585f669d1f80043d0f7fc9b1a30e`

## Scope

This is an ordinary-MNIST, seed-0 optimization diagnostic for the frozen Conv2
hard-sigmoid rows:

| Scheme | Amplification | Input gain | Operational `T/K` |
|---|---:|---:|---:|
| baseline | `v1/c1` | `253.302230835` | `16/6` |
| proposed/ours | `v4/c1` | `716.3439331055` | `24/6` |
| legacy | `v4/c0.25` | `661.4369506836` | `8/4` |

It compares constant plain SGD with constant Adam after matching each
optimizer's initial proposed weight-update size. Adam is evaluated only on the
predeclared local high-rho surface below. It is not globally tuned, does not
replace SGD in the paper protocol, and cannot replace the frozen
medium-affine Conv1/Conv2 handoff.

The study schema is `mnist-conv-lr-optimizer-boundary/v1`. Candidate bundles
use the existing `mnist-conv-run/v7` envelope under the distinct protocol ID
`conv-hardsigmoid-lr-conv2-sgd-adam-boundary-constant-bs16-v1`. The earlier
ordinary-MNIST Conv3 v7 protocol retains its original normalization and
identity; v7 dispatch is strict on `protocol_id`.

## Frozen Runtime

- Ordinary MNIST only, using the deterministic 55,000/5,000 stratified split.
- Official MNIST test data is neither instantiated nor read.
- Model seed `0`, train-loader shuffle seed `0`, batch size `16`.
- One hash-verified shared Conv2 initialization.
- Five epochs, 3,438 optimizer steps per epoch, 17,190 steps total.
- Constant explicit LR for every scientific parameter; no scheduler.
- One optimizer group per `ConvWeight_i`, `Bias_i`, and `DenseWeight_0`.
- Existing medium-affine-calibrated input gains and row-specific `T/K` are
  reused without recalibration.

The exact optimizer union is:

- SGD: momentum `0`, weight decay `0`.
- Adam: betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0`,
  `amsgrad=false`, `foreach=false`, `fused=false`, `maximize=false`,
  `capturable=false`, and `differentiable=false`.

No AdamW, adaptive-rho controller, multi-seed training, or final-paper
optimizer change is in scope.

## Frozen Minibatches and Optimizer Normalization

The deterministic ordinary-MNIST provenance is:

| Object | SHA-256 |
|---|---|
| validation indices | `4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4` |
| first 32 normalization minibatches | `1c65076487a87e0badfe19b80edc8263f50c4b765dc317299dec4a8f1fad460b` |
| first 8 replay minibatches | `1d67f374dea1d58ed7d779ed5eb14471fce3b6c64e614c47f47fda5526bc5673` |
| complete five-epoch batch order | `e774de39fe6f9529dafded0b14356727b3e8510bd42f3a92f7416dcab0592d14` |

For each row and optimizer, 32 nominal-LR shadow proposals are computed. Every
proposal restores the exact parameter tensors, gradients, optimizer parameter
groups, moments, variances, and step counters before the next minibatch.

For a bounded weight \(W_i\),

\[
u_i=\operatorname{median}_b
\frac{\operatorname{RMS}(\Delta W_{i,b}^{\eta=1})}
     {\operatorname{RMS}(W_{i,0})},
\qquad
\eta_i=\frac{\rho_i}{u_i}.
\]

Adam's \(u_i\) values are measured from fresh Adam proposals. The authoritative
SGD weight units and weight-probe hashes are instead imported from the two
hash-verified v5/v6 source slices for each scheme, which must agree exactly.
Fresh SGD proposals are verification diagnostics and supply only the new Q90
bias units. They never redefine SGD weight units, raw weight LRs, or reused
weight-probe provenance.

For a hidden bias, the separately recorded unit is the linear Q90 across the
same 32 optimizer proposals, normalized by its attached weight's initial RMS.

## Main Grid and Reuse

| Scheme | `rho_conv` | `rho_dense` |
|---|---|---|
| baseline | `0.01, 0.015, 0.03` | `0.01, 0.03` |
| proposed/ours | `0.01, 0.015, 0.03` | `0.003, 0.01` |
| legacy | `0.01, 0.015, 0.03` | `0.003, 0.01` |

The main grid contains 36 cells. Six completed SGD cells at
`rho_conv=0.01` are reused only after verifying their content-addressed parent,
completion record, run bundle, summary, checkpoints, minibatches, diagnostics,
and validation artifacts by SHA-256 and size. A mismatch aborts the study; it
must not trigger a fallback rerun. The remaining main work is 12 new SGD and
18 new Adam runs.

The main grid uses the attached-bias rule:

\[
\eta_{b_i}^{\rm attached}=\eta_{W_i}.
\]

## Selection, Boundary Status, and Canceled Bias Follow-Up

Selection is independent for every scheme × optimizer. It first finds the
minimum final validation loss, forms the inclusive 2% loss plateau, and then
uses:

1. higher final validation accuracy;
2. higher median projection efficiency;
3. lower \(\max(\rho_{\rm conv},\rho_{\rm dense})\);
4. lower \(\rho_{\rm conv}+\rho_{\rm dense}\);
5. deterministic entry identity.

If a plateau reaches `rho_conv=0.03`, both dense slices are evaluated at the
predeclared `rho_conv=0.1` sentinel. A plateau that still reaches `0.1` is
`unbracketed_high`; the grid is not extended again. An Adam plateau touching
the lower `rho_conv=0.01` edge is `unbracketed_low`.

The original design predeclared one bias-capped confirmation per
scheme × optimizer after the sentinel decision and re-selection. Weight LRs
would have remained unchanged, with

\[
\eta_{b_i}^{\rm cap}
=\min\left(\eta_{b_i}^{\rm attached},\frac{10^{-3}}{u_{b_i}^{Q90}}\right).
\]

The cap targets at most a `0.1%` initial bias proposal relative to the attached
weight's initial RMS and can never increase a bias LR. After the attached-bias
surface was complete, the user removed this follow-up from scope to keep the
conclusion focused on learning rates. The capped-bias jobs were canceled, no
completed capped-bias result is scientific evidence, and the final selection
is the attached-bias selection below.

## Epochwise Replay and Safety

At initialization and after every epoch, the same eight frozen minibatches are
replayed with a shadow step from the current optimizer state. Every replay
must restore parameters and complete optimizer state exactly. The bundle
records:

- achieved weight rho and bias rho;
- bias-step/attached-weight-step ratio;
- projection efficiency;
- parameter displacement relative to initial attached-weight RMS;
- both hidden-layer hard-sigmoid saturation fractions;
- full validation loss and accuracy.

Real Adam training retains moments, variances, and step counters across all
17,190 steps. Any non-finite parameter or optimizer state is an immediate,
recorded safety failure. A completed entry must contain exactly 17,190
successful steps.

Completed artifacts are immutable and resumable by hash-verified skip.
Incomplete Adam entries restart from the shared initialization in a new
attempt because model-only checkpoints do not contain optimizer state.

## Execution

Production measurements used one Jean Zay V100-32GB allocation per pack after
a measured concurrency benchmark. The immutable manifests and completion
records, rather than the retired launch commands, are the execution authority.
The final report uses matplotlib heatmaps and epochwise achieved-rho plots and
states explicitly that Adam was tested only on this local high-rho surface and
that no result changes the frozen medium-affine handoff.

### Execution record (2026-07-24)

The accepted implementation uses canonical study
`lrstudy_87864fb9a23c07e5761d61bdbc9104b869d01fd5b9697aa768a3c5a7f588f931`
and effective code fingerprint
`f05a9c18445b492621b21833416de7d7b4ac13aec8d103db698d83340b7997b3`.
The canonical config, probe bundle, and main manifest SHA-256 values are,
respectively:

- `1093e198acb0bd770fce07bb3efdaba769d9585f669d1f80043d0f7fc9b1a30e`;
- `b191f3e4664862f49863fd302de2e22740172eeac19bd0d939e0d71028477a33`;
- `21fa7501774d1f5b037d982935d7265b4a4a1512119733785cad9c84a40c189a`.

The main manifest contains 36 cells: six immutable SGD reuses and 30 new
five-epoch runs. A local RTX 3090 smoke completed 288/288 ours-Adam steps with
finite optimizer state.

Benchmark job `160333` selected width 16 with `7496 MiB` peak GPU memory and
94.52% of the best safe aggregate throughput; its report SHA-256 is
`2bbca0de71da003fce749c4ad056df3e9e27a97dd8fbd7c367978e5a56e34f46`.
The three main packs completed all 30 new runs, and the six immutable SGD
reuses produced a complete 36-cell main surface. All new main runs were
admissible, safety-clean, and completed exactly 17,190 steps.

The base selection triggered ten predeclared `rho_conv=0.1` sentinels: four
baseline, four ours, and two legacy-SGD cells. All ten were admissible,
safety-clean, and completed exactly 17,190 steps. The final attached-bias
selection is:

| Scheme | Optimizer | Selected `(rho_conv,rho_dense)` | Final validation loss / accuracy | Boundary status |
|---|---|---:|---:|---|
| baseline | SGD | `(0.1,0.01)` | `0.11471955 / 94.56%` | `unbracketed_high` |
| baseline | Adam | `(0.1,0.01)` | `0.08561958 / 96.10%` | `unbracketed_high` |
| ours | SGD | `(0.03,0.01)` | `0.08794243 / 96.66%` | `unbracketed_high` |
| ours | Adam | `(0.03,0.003)` | `0.07726470 / 96.88%` | `unbracketed_high` |
| legacy | SGD | `(0.03,0.01)` | `0.08837134 / 96.28%` | `bracketed` |
| legacy | Adam | `(0.015,0.01)` | `0.05492024 / 97.86%` | `unbracketed_low` |

The final 46-record aggregate and selection SHA-256 values are
`c089fdc5e931f393bbcbfb842be5ecce833ca57b5a9338269848255383e5ce44`
and
`2f1812ca5f8e8afa3ea03f1f262f933e6662bafd90661d0cc45c1d960614666a`.
The selected epochwise replay shows the LR mechanism directly. Legacy SGD
falls from approximately `3%/1%` Conv/Dense achieved rho at initialization to
only `0.058--0.142% / 0.026%` at epoch five. Legacy Adam starts at the
requested `1.5%/1%` and still supplies approximately
`0.301--0.331% / 0.211%` at epoch five. Adam therefore substantially reduces
legacy's fixed-SGD underexposure, but it does not undo the scheme's gradient
rotation or whole-network calibration mismatch.

Six capped-bias plan entries were published after selection under the original
design. The user then explicitly removed that question from scope. Jobs
`205406`, `205407`, and `205408` were canceled on 2026-07-25; no capped-bias
completion is treated as scientific evidence. No completed main or sentinel
artifact was changed or removed.

The LR-only local report binds the 46 completed records and contains validation
loss/accuracy heatmaps plus selected-run epochwise effective-rho plots. Its
summary and completion SHA-256 values are
`52b54cb0f1816f06b7cbca2bacf78b5447e625363a733c4922c7f953de80ba5d`
and
`ef8422b22bf81e0c345f3f3976f6faf0cb8610bb08bf7f005137d506f111a480`.
