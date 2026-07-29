# Conv Paper Hard-Sigmoid Learning-Rate Protocol

Updated: 2026-07-26

Status: **the deterministic-medium-affine Conv1/Conv2 seed-0 handoff is
complete. The two baseline peak learning rates come from v1 and the four
amplified peak learning rates come from v3. All six have status
`frozen_seed0_screen`. Final paper training remains gated.**

This document is the canonical learning-rate protocol and handoff for the six
hard-sigmoid Conv1/Conv2 paper rows. Ordinary-MNIST optimizer studies, the
incomplete v4 layer-wise diagnostic, and the separate Conv3 diagnostics are
curated in [`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md).

## Scope And Gate

The protocol covers deterministic medium affine MNIST for:

- Conv1 and Conv2;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- hard sigmoid only;
- model seed `0`;
- batch size `16` and plain SGD without momentum or weight decay.

The selected values are **peak learning rates** for the exact five-epoch
warm-up/cosine candidate schedule below. They are not constant-LR values.
Later row-matched training may reuse them only under an explicitly reviewed
training contract. Changing optimizer, batch size, parameter grouping, or the
meaning of the LR requires a new review and may require a new LR screen.

This protocol does not authorize final paper training. It does not select a
final epoch budget, final model seeds, a final-paper checkpoint rule, or
inclusion categories. Perfect diode and deterministic-medium-affine Conv3
learning rates remain outside this completed handoff.

## Frozen Rows

Raw gain remains the value calibrated at `T=64` for 30% first-hidden
saturation. It must not be recalibrated at the operational `T`.

| Architecture | Scheme | Voltage/current amp | Frozen `input_gain` | Operational `T/K` |
|---|---|---:|---:|---:|
| Conv1 | baseline `v1/c1` | `1/1` | `75.6030807495` | `4/4` |
| Conv1 | proposed/ours `v4/c1` | `4/1` | `84.8402175903` | `4/4` |
| Conv1 | legacy `v4/c0.25` | `4/0.25` | `31.8188591003` | `4/4` |
| Conv2 | baseline `v1/c1` | `1/1` | `253.3022308350` | `16/6` |
| Conv2 | proposed/ours `v4/c1` | `4/1` | `716.3439331055` | `24/6` |
| Conv2 | legacy `v4/c0.25` | `4/0.25` | `661.4369506836` | `8/4` |

Every row uses:

- hard sigmoid with `g_on=100`, `g_off=0`, and `v_off=4`;
- explicit empty quadratic- and exponential-diode parameter dictionaries;
- fixed amplification and paired 20-output squared-error loss;
- fixed-step asynchronous minimization with
  `adaptive_equilibrium=false`;
- reset state at every batch;
- conductance bounds `[0,100]`, Kaiming-uniform initialization, and weight
  gain `1`;
- model seed `0`.

The complete minimizer settings are explicit in each study JSON so simulator
defaults cannot change a run.

## Dataset And Reproducibility

Use the frozen deterministic medium-affine MNIST train split. Select validation
indices independently of model RNG consumption with a dedicated seed-0
generator and class-wise `torch.randperm`: exactly 500 examples from each of
the ten classes. The remaining 55,000 examples form the training split. Store
the exact indices and SHA-256 digests. Validation batch size is `128`.

Training uses batch size `16`, no dropped final batch, and a dedicated seed-0
shuffle generator reset identically for every candidate. Store and hash the
exact minibatch-index order. The official MNIST test dataset must not be
constructed or read during the screen.

Reset global layer and parameter name counters before every independent model
build. Create one seed-0 initialization checkpoint for Conv1 and one for
Conv2. Within an architecture, every scheme, probe, range run, and candidate
starts from the same tensor values and records checkpoint and tensor digests.

Common frozen provenance:

- train indices:
  `c0940cfdde9fb2a87f846a4dd234a6ff634eed32a98b9f5d72dfed3e2119a809`;
- validation indices:
  `4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4`;
- probe minibatches:
  `1c65076487a87e0badfe19b80edc8263f50c4b765dc317299dec4a8f1fad460b`;
- candidate minibatches:
  `e774de39fe6f9529dafded0b14356727b3e8510bd42f3a92f7416dcab0592d14`;
- Conv1 checkpoint/tensors:
  `204282e999a10306cc15f6095d1ce6971dcc137f8a473ff8f6481eed5921177c`
  / `99f6ee133f72aae48205e58bb5d9a5568f6d7a2f3d3a801a0f4e07d16463fc1d`;
- Conv2 checkpoint/tensors:
  `dc16ef2bebd2c9e6afe307b41555683c3eec29b9481ab64ea3f412382c837c7e`
  / `e9aa471dc5558e525758b345de12c53f0a086c5a1b022024c2fc349ec138e2e3`.

## Immutable Study Identities

| Version | Declarative source | Study identity | Role | Executed result |
|---|---|---|---|---|
| v1 | [`hardsigmoid_lr_study_sgd_bs16_v1.json`](../configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json) | `lrstudy_2103c850c10540d5628cd20cbebc0af519bd871116062e8b80e50ad0c068a9a4` | conductance-span screen for all six rows | two baseline LRs frozen; four amplified rows unresolved |
| v2 | [`hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json`](../configs/conv/hardsigmoid_lr_rescue_warmin_sgd_bs16_v2.json) | `lrstudy_aa0a5d954056594099cf76ed49a8ef159f9e707d207b0784fc11f44212f9b0a6` | lower-rho warm-in rescue for Conv1 amplified rows | both rows unresolved; no candidate training |
| v3 | [`hardsigmoid_lr_relative_rho_sgd_bs16_v3.json`](../configs/conv/hardsigmoid_lr_relative_rho_sgd_bs16_v3.json) | `lrstudy_b2cffbb9c9976c58338141fc179fba2ff4f37b3d02009ef9255e3d5a21d650d2` | parameter-relative screen for the four amplified rows | all four LRs frozen |

The immutable resolved v1 and v3 sources intentionally retain their
pre-execution `status.measurements=pending` values. Completion markers,
selection artifacts, and the tracked handoff CSVs are the post-execution
authority; editing a resolved source in place would create a different study
identity.

The historical staged runner used to produce these measurements has been
removed from the active tree. The frozen JSON files and recorded outputs
remain provenance for this completed handoff. A future repetition should use
a focused scientific script that calls `labs.mnist_train` and can be launched
with `python -m experiments.launch`; it must preserve the scientific choices
in the relevant JSON rather than editing a completed result in place.

## Shared Parameter Diagnostics And Safety Gates

Every proposal exposes and records:

1. pre-update parameters;
2. post-optimizer, pre-projection parameters;
3. post-projection parameters.

For every parameter, log gradient RMS, proposed-update RMS, normalized update,
proposed bound crossings, boundary occupancy, and projection efficiency.
`ConvWeight_*` and `DenseWeight_*` are bounded and determine gates. Biases use
the same scalar LR, are reported separately, and never set a gate.

After the first 32 reference steps, failure onset is the first step of the
earliest sustained event:

- any non-finite loss, gradient, update, state, or diagnostic: immediate;
- loss EMA greater than four times its prior minimum for 8 consecutive steps;
- any bounded tensor's gradient RMS greater than 100 times its first-32-step
  median for 8 steps;
- any bounded tensor's combined-bound occupancy more than `0.20` above its
  initial value for 16 steps;
- any bounded tensor's projection efficiency below `0.50` for 16 steps,
  ignoring numerically zero proposals.

Proposed-bound crossing fraction is logged but is not a gate.

## V1 Conductance-Span Screen

### Probe coordinate

For each row, use the same 32 training minibatches and its architecture's
shared checkpoint. Make a real SGD proposal with nominal LR `1`, record the
pre-projection update, and restore all parameters before the next minibatch.

For bounded weight tensor `l` and minibatch `t`:

```text
rho_span[l,t] = RMS(delta_W_pre[l,t]) / 100.
```

`rho_unit` is the unweighted linear-interpolated Q90 over all bounded-weight
tensor × minibatch values. Convert each target in
`{1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2}` using
`eta = rho_target / rho_unit`.

### Range and candidates

Start every row from its shared checkpoint. Run 600 optimizer steps while the
LR increases geometrically, with both endpoints included, from `eta(1e-5)` to
`eta(1e-2)`. Evaluate all 5,000 validation examples at step 0 and every 100
steps. If and only if all 600 steps are safe, append 128 geometric steps from
target `1e-2` through `3e-2`.

At each stable target crossing, summarize the following eight steps. `fast` is
the first target whose median smoothed loss is at least 10% below the
first-32-step median. `high` is the highest stable target before failure.
`middle` is the stable target nearest the geometric midpoint of `fast` and
`high`; fill duplicates with the nearest unused stable target. A row is
unresolved if it lacks three distinct stable targets or never shows a
decreasing region. No fallback is allowed.

Each resolved role trains for five epochs on all 55,000 training examples:
3,438 steps per epoch and 17,190 total. The scalar peak LR is broadcast to all
trainable weights and biases.

- Steps 1-860 linearly increase from `eta_peak/860` to `eta_peak`.
- Steps 861-17,190 use cosine decay from `eta_peak` to exactly zero.
- There are no restarts.

Evaluate the full validation split after every epoch. Save both the
minimum-validation-loss and final checkpoints. A candidate is inadmissible
after numerical failure or sustained projection domination in the first 20%
of training.

Among admissible candidates:

1. find the minimum final validation loss;
2. form the plateau within 2% of that loss;
3. choose the candidate nearest the plateau's log-space LR center;
4. break ties by higher final validation accuracy, higher median projection
   efficiency, then lower raw LR.

### Executed v1 result

| Architecture | Scheme | `rho_unit` | Range result | Selected role / peak LR | Final validation loss / accuracy | Row status |
|---|---|---:|---|---|---:|---|
| Conv1 | baseline `v1/c1` | `9.9423320586e-5` | safe through extension | fast / `0.30174007288404703` | `0.3232406965 / 0.6344` | `frozen_seed0_screen` |
| Conv1 | proposed/ours `v4/c1` | `5.4257199188e-4` | loss EMA, onset 33 / confirmed 40 | — | — | `unresolved` |
| Conv1 | legacy `v4/c0.25` | `7.3453733418e-3` | loss EMA, onset 33 / confirmed 40 | — | — | `unresolved` |
| Conv2 | baseline `v1/c1` | `1.7293739938e-5` | loss EMA 579/586; candidates resolved first | fast / `1.7347317646208873` | `0.3425347980 / 0.5778` | `frozen_seed0_screen` |
| Conv2 | proposed/ours `v4/c1` | `5.9204244898e-4` | loss EMA, onset 33 / confirmed 40 | — | — | `unresolved` |
| Conv2 | legacy `v4/c0.25` | `1.4717254442e-2` | loss EMA, onset 33 / confirmed 40 | — | — | `unresolved` |

The candidate stage contained six runs: three roles for each resolved baseline
row. All six completed the exact schedule and were admissible. Validation
metrics are LR-selection diagnostics, not official-test or paper-facing
accuracy.

Evidence:

- tracked v1 handoff:
  [`conv_hardsigmoid_lr_seed0_selection_20260719.csv`](conv_hardsigmoid_lr_seed0_selection_20260719.csv);
- v1 resolved-config SHA-256:
  `8bbe06ca62ed622233ab467cc1eb1124a85e4081eeea8c2e2cf0ee33bc7cb896`;
- v1 selection SHA-256:
  `fe6fa3b744d48c9de0951bd2e71d0d09887c9972cc6006d3d72874888d3d08a7`.

## V2 Conv1 Warm-In Rescue

V2 changed only the range schedule for Conv1 ours and legacy. It inherited the
v1 split, Conv1 initialization, probes, gains, `T/K=4/4`, optimizer, gates,
candidate schedule, and selection rule. Conv1 baseline and all Conv2 rows were
not rerun.

The 1,199-step base schedule is:

```text
concat(geomspace(1e-8, 1e-5, 600)[:-1],
       geomspace(1e-5, 1e-2, 600))
```

Step 600 is the exact shared `1e-5` anchor. Points below `1e-5` are not
candidate-eligible. The optional unchanged extension reaches `3e-2` only when
the complete base is safe.

| Row | Completed steps | Earliest sustained gate | Maximum admissible rho / raw LR | Stable eligible targets | Result |
|---|---:|---|---:|---|---|
| Conv1 ours | 772 | loss EMA, onset 765 / confirmed 772 | `6.6277931865e-5 / 0.1221550925` | `1e-5`, `3e-5` | unresolved |
| Conv1 legacy | 403 | loss EMA, onset 396 / confirmed 403 | `9.4034845358e-7 / 1.2801915026e-4` | none | unresolved |

The candidate stage executed a zero-work handoff because neither row had three
stable eligible targets. No LR was selected or invented.

Evidence:

- tracked rescue:
  [`conv_hardsigmoid_lr_conv1_rescue_20260719.csv`](conv_hardsigmoid_lr_conv1_rescue_20260719.csv);
- v2 resolved-config SHA-256:
  `25c9832e2a55992ab2fe19d3833967f49a173c092534350242a45bfda9fc98af`;
- parent-import receipt SHA-256:
  `106368c276773b29fd8838193b47b5b8b086aba3d73c3dbc8a43429e9d254508`;
- ours / legacy range-summary SHA-256:
  `a3383bf58dd9ccc79857f81adbc99dd89d420747b07515702ec0096949b90594`
  / `0461c3648891ee10a54c6e9ad4531a70dca8552b4c66250c523139d2954dc7c7`;
- v2 selection SHA-256:
  `6833a026fe41c8d1a3f8d0ce3147f5ec3f258bf4441dcb6b6747d654fa551202`.

## V3 Parameter-Relative Screen

V3 covers exactly the four amplified rows left unresolved by v1 and v2. The
two baseline rows are not rerun or reselected.

### Parameter-relative coordinate

For each bounded tensor `l`, freeze:

```text
S[l] = RMS(W_initial[l]).
rho_relative[l,t] = RMS(delta_W_pre[l,t]) / S[l].
q90_relative[l] = linear_Q90_t(rho_relative[l,t]).
rho_unit_relative = max_l(q90_relative[l]).
eta_target = rho_target_relative / rho_unit_relative.
```

The Q90 is computed independently per bounded tensor before taking the
maximum. The initialization denominator is positive, finite, and never
recomputed from an evolving parameter. Biases are excluded from aggregation
and gates.

Conductance-span rho remains a separate safety diagnostic:

```text
rho_span[l,t] =
    RMS(delta_W_pre[l,t]) / (upper_bound[l] - lower_bound[l]).
```

V3 copies and verifies the v1 split and Conv1/Conv2 initialization assets,
verifies the v1/v2 evidence chain, and recomputes all four probes under the new
definition. V1/v2 span-normalized probe measurements are not reused as v3
measurements.

### Range and candidates

The base range contains 800 geometric steps:

```text
rho_relative = geomspace(3e-4, 3, 800).
```

Candidate targets are
`{1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1}`. If and only if the base is safe and no
gate is active at its boundary, append 128 geometric steps from `3` through
`9`. Gates, candidate extraction, the five-epoch candidate schedule, and the
selection rule remain those defined above.

### Executed v3 result

Every probe identified `DenseWeight_0` as the limiting bounded tensor. Every
range run reached the loss-EMA gate only after resolving three roles. All
twelve candidate runs completed exactly 17,190 steps and were admissible.

| Architecture | Scheme | Selected role / target | Frozen peak LR | Observed relative rho | Observed span rho | Final validation loss / accuracy |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | proposed/ours `v4/c1` | middle / `3e-2` | `0.001838414260143094` | `0.017006431029241174` | `6.018252270345785e-7` | `0.3125611982345581 / 0.6638` |
| Conv1 | legacy `v4/c0.25` | high / `1e-2` | `4.639114646319357e-5` | `0.0023581856036333252` | `7.892850980966992e-8` | `0.367435618686676 / 0.5438` |
| Conv2 | proposed/ours `v4/c1` | high / `1e-1` | `0.011865709277518458` | `0.019843829220147238` | `7.2741440137604796e-6` | `0.30687670884132384 / 0.6794` |
| Conv2 | legacy `v4/c0.25` | high / `3e-1` | `0.0015293192084949393` | `0.009222524758373938` | `9.038794657063601e-7` | `0.2184314489364624 / 0.8466` |

Validation metrics come only from the frozen MNIST-train validation split and
are not paper-facing accuracy measurements.

V3 evidence:

- four-row handoff:
  [`conv_hardsigmoid_lr_relative_rho_v3_20260719.csv`](conv_hardsigmoid_lr_relative_rho_v3_20260719.csv),
  SHA-256
  `ae4ad79329614b9682282ca09c96aace11835da99ff13517f2136abe02b749c4`;
- v3 resolved-config / import-receipt SHA-256:
  `7982adfb4e55ccf2da3a7e75a3faa2bfe1a08ac3224c1ac017e6c547b579b736`
  / `c39f57c2db8c2f8995b9d534df1adeef692d4f9d819799726265b73402188c5c`;
- effective code fingerprint:
  `2efe660e439d9a4a3fd1abbe6c8fb60488a11160dbf5814d79b57ae07ac1f55a`;
- probe manifest/completion:
  `1514d8bce059918ea471dadc606c0645d9d98e0e9c47650e6de2bb62d0811bf8`
  / `f40c5fbc232eef1e65de66c0745e52c7fbf43518471a5b2ceb81bc55558c6c12`;
- range manifest/completion:
  `3885cdf47032b9f04eaf30c0802b514a8f776c897a92f994cb5927478118a847`
  / `c1b007f540bb63775e0882b22a582b5cf826f6fc9d87fb631524228ac503e87d`;
- candidate manifest/completion:
  `1fcfff23cb5a63bb9c83b30d82fc6d760413974a964ab0c32d7a93f9eb583cb6`
  / `f1707db7596900e75cf5f1f2b9b5d2a550594dba7742fe7732050e3795e780d5`;
- selection manifest/completion/JSON:
  `476617fcf466ab85c4a012a2055b2836ad52e18cedb377ea4acd61f49a3bcce1`
  / `4b1560af8b6486c87e5e1b2382e5c2b51abf29dd27f1d41e5aca6a456bb2f06e`
  / `28cc8980496ae3ae7aac91566bc5a56937101e281dd04dc03e3592e52b3e56da`.

## Active Six-Row Handoff

The active handoff combines the two untouched v1 baselines with the four v3
amplified selections. Baseline relative-rho values below were rederived from
the immutable v1 candidate logs using the v3 per-tensor Q90-then-maximum rule;
their native selection coordinate remains conductance-span rho.

| Architecture | Scheme | Source / native coordinate | Frozen peak LR | Observed relative rho | Observed span rho | Status |
|---|---|---|---:|---:|---:|---|
| Conv1 | baseline `v1/c1` | v1 / conductance span | `0.30174007288404703` | `0.12322412433401704` | `2.417930612133236e-5` | `frozen_seed0_screen` |
| Conv1 | proposed/ours `v4/c1` | v3 / initial parameter RMS | `0.001838414260143094` | `0.017006431029241174` | `6.018252270345785e-7` | `frozen_seed0_screen` |
| Conv1 | legacy `v4/c0.25` | v3 / initial parameter RMS | `4.639114646319357e-5` | `0.0023581856036333252` | `7.892850980966992e-8` | `frozen_seed0_screen` |
| Conv2 | baseline `v1/c1` | v1 / conductance span | `1.7347317646208873` | `0.17286832710411465` | `7.20551922793696e-5` | `frozen_seed0_screen` |
| Conv2 | proposed/ours `v4/c1` | v3 / initial parameter RMS | `0.011865709277518458` | `0.019843829220147238` | `7.2741440137604796e-6` | `frozen_seed0_screen` |
| Conv2 | legacy `v4/c0.25` | v3 / initial parameter RMS | `0.0015293192084949393` | `0.009222524758373938` | `9.038794657063601e-7` | `frozen_seed0_screen` |

The tracked full-precision authority is
[`conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv`](conv_hardsigmoid_lr_active_handoff_relative_rho_v3.csv),
SHA-256
`9eefa3a901be8ca049fb9060744ba99d5a8c28ac6dc4de36e1fc745f147aeed7`.
Its curation provenance SHA-256 is
`1b12c58e0887d35ab2a253033125ae5959ce27f7e0267842975dcd64e928ef37`.

All split, import, candidate, and selection records state
`official_test_read=false`. The v3 selection states
`final_paper_training_authorized=false`.

## Required Artifacts

Probe, range, candidate, and selection artifacts are SHA-256 checksummed. They
must retain split, checkpoint, and minibatch hashes; the exact rho definition;
target and raw peak LRs; step logs; full validation metrics; per-parameter
projection diagnostics; selected LR and observed peak rho; and explicit
failure or unresolved reasons.

V3 artifacts additionally preserve each initial bounded-parameter RMS,
per-parameter and worst-parameter `rho_unit_relative`, and `rho_span` as a
separate safety diagnostic. A v1/v2 span-normalized measurement must never be
relabelled as parameter-relative.

Any tracked plot of these selections must use matplotlib, identify peak LR
rather than constant LR, use separate Conv1 and Conv2 panels, and keep
consistent baseline/ours/legacy colors.

## Current Gate

The six Conv1/Conv2 hard-sigmoid seed-0 peak LRs above are frozen for
row-matched reuse under a reviewed compatible training contract. V1 amplified
failures and the exhausted v2 rescue remain immutable historical evidence; no
fallback was assigned.

Final paper training remains blocked. A later protocol must still define or
resolve:

- deterministic-medium-affine Conv3 learning rates;
- all perfect-diode calibration, operational `T/K`, and learning-rate rules;
- final epoch budget and model seeds;
- checkpoint selection;
- paper inclusion categories.

Ordinary-MNIST and incomplete optimizer diagnostics are not substitutes for
these missing decisions; see
[`conv_learning_rate_diagnostics.md`](conv_learning_rate_diagnostics.md).
