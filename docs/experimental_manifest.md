# Experimental Manifest

Updated: 2026-07-29

This is the manually curated ledger of analyzed studies. It is not generated
from raw metrics and does not track transient job state.

Each finished entry records:

- scientific question and evidence class;
- frozen setup, arms, seeds, and inclusion rule;
- outcome: `positive`, `negative`, `mixed`, or `inconclusive`;
- headline measurements;
- interpretation;
- limitations and protocol deviations;
- raw run and study-analysis locations; and
- links to detailed tables, plots, or reviewed historical result cards.

Ordinary-MNIST selection accuracy must never be presented as paper evidence.
An operational failure needs an entry only when it materially affects the
scientific conclusion.

## Published Learning-Rate Handoffs

These are the current parameter-wise optimizer inputs for downstream
unbounded/wide-range work. Publishing a handoff does not promote its
ordinary-MNIST measurements to paper evidence or erase its stated review and
confirmation limits.

| Handoff | Scope | Evidence status | Machine-readable authority |
|---|---|---|---|
| `perfectdiode-conv12-unbounded-fixed-lr-20260729-v1` | Conv1/Conv2, `[0,100]` | Conv1 ten-epoch confirmations complete, review pending; Conv2 three-epoch evidence only | [`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json) |
| `perfectdiode-conv3-unbounded-selected-lr-20260729-v1` | Conv3, downstream `[0,100]` authorized from a `weight_max=null` source study | six-surface rho selection complete; source-contract mismatch recorded; long confirmation pending | [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json) |

### Conv1/Conv2 current fixed vectors

Notation is `C_i=ConvWeight_i`, `B_i=Bias_i`, and `D=DenseWeight_0`.
Runtime order is `[C0,D,B0]` for Conv1 and `[C0,C1,D,B0,B1]` for
Conv2. The machine-readable handoff derives each ordered vector from the named
mapping in that runtime order.

| Architecture | Scheme | Optimizer | Parameter-wise LR vector |
|---|---|---|---|
| Conv1 | baseline | SGD | `C0=B0=0.142696, D=0.0120238` |
| Conv1 | baseline | Adam | `C0=B0=8.66763e-4, D=1.09401e-4` |
| Conv1 | ours | SGD | `C0=B0=0.0375994, D=0.00311655` |
| Conv1 | ours | Adam | `C0=B0=8.66753e-4, D=1.09388e-4` |
| Conv1 | legacy | SGD | `C0=B0=2.05753e-4, D=4.76259e-5` |
| Conv1 | legacy | Adam | `C0=B0=2.88917e-4, D=3.64618e-5` |
| Conv2 | baseline | SGD | `C0=B0=7.90864, C1=B1=3.81476, D=0.820630` |
| Conv2 | baseline | Adam | `C0=B0=2.60078e-3, C1=B1=4.59053e-4, D=4.67660e-4` |
| Conv2 | ours | SGD | `C0=B0=0.523358, C1=B1=0.246043, D=0.0520201` |
| Conv2 | ours | Adam | `C0=B0=2.60036e-3, C1=B1=4.58651e-4, D=1.55136e-4` |
| Conv2 | legacy | SGD | `C0=B0=0.00496733, C1=0.00367839, B1=6.74726e-4, D=0.00236044` |
| Conv2 | legacy | Adam | `C0=B0=8.66750e-4, C1=B1=1.52841e-4, D=5.16982e-5` |

## Finished Studies

## `conv3_pd_unbounded_rho_t8k8_20260729T125734Z` — Conv3 perfect-diode rho selection

- Analyzed: 2026-07-29
- Completed: 2026-07-29 19:48 CEST
- Evidence class: `ordinary_mnist_selection`
- Outcome: `positive`
- Scientific question: Which `(rho_conv, rho_dense)` pairs minimize the
  three-epoch validation loss for Conv3 perfect-diode networks with `T=K=8`
  and nonnegative weights without an upper bound, for SGD and Adam across the
  baseline, ours, and legacy amplification schemes? Are the initial rho bounds
  sufficient?
- Runs and seeds: Seed 0; six optimizer/scheme surfaces; all 91 semantically
  complete cells included (54 initial cells and 37 boundary-expansion cells).
  All six final search bounds are closed, the retained cells share one cohort
  contract, and the final audit found zero semantic issues.
  Stability probes and canaries were operational gates and were excluded from
  the scientific comparison.
- Setup: Ordinary MNIST 55,000/5,000 deterministic train/validation split;
  official test split not read; input gain 360; Conv3 channels
  `[64, 128, 256]`; kernels 3; strides `[2, 2, 1]`; padding 1; output width
  20; Kaiming-uniform initialization; asynchronous minimizer; batch size 16;
  validation batch size 64; three epochs per cell. The schemes were baseline
  `(voltage_amp=1, current_amp=1)`, ours `(4, 1)`, and legacy `(4, 0.25)`.
  Weights used `weight_min=0` and `weight_max=null`. Runs used one
  `v100-16g` GPU each on Jean Zay under `fmu@v100`.
- Headline measurements: Selection used minimum final validation loss. Every
  selected point is interior to the final explored bounds. Accuracy is shown
  only as an ordinary-MNIST selection diagnostic.

  | Scheme | Optimizer | `rho_conv` | `rho_dense` | Validation loss | Validation accuracy |
  |---|---|---:|---:|---:|---:|
  | baseline | SGD | 0.001 | 0.03 | 0.09482917 | 94.74% |
  | baseline | Adam | 0.027 | 0.27 | 0.06915623 | 96.20% |
  | ours | SGD | 0.009 | 0.03 | 0.05100432 | 97.18% |
  | ours | Adam | 0.081 | 0.09 | 0.04339848 | 97.70% |
  | legacy | SGD | 0.003 | 0.01 | 0.04871860 | 97.44% |
  | legacy | Adam | 0.027 | 0.01 | 0.03983585 | 97.96% |

- Interpretation: Adam lowered the selected validation loss relative to SGD
  by `0.02567295` for baseline, `0.00760584` for ours, and `0.00888274`
  for legacy. Both amplified schemes improved on the baseline for both
  optimizers in this selection study. Legacy Adam produced the lowest
  three-epoch validation loss. The boundary expansions turned back around all
  six selected points, so no further rho expansion is indicated for this
  protocol.
- Limitations: This is a one-seed, three-epoch operating-point selection
  study, not final-training or paper-test evidence. Ordinary-MNIST selection
  accuracy is diagnostic only. “Unbounded” means nonnegative weights without
  an upper cap, not signed unconstrained weights. The run predates
  `docs/experiment_reporting.md`, so its raw bundle uses the reviewed
  rho-search layout rather than per-cell
  `manifest.json`/`status.json`/`result.json`; the frozen source hashes,
  semantic cell records, scheduler logs, final receipt, and 1,817-file
  checksum manifest provide the retained provenance. An initial canary failed
  because `git` was absent from the compute-node module path; the corrected
  canary passed before any scientific batch was released and does not affect
  the conclusion.
- Raw results:
  [validated local copy](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/remote_full/conv3_pd_unbounded_rho_t8k8_20260729T125734Z)
  and
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z`
  on Jean Zay.
- Analysis:
  [`results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis`](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis)
- Selected LR handoff:
  [`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json).
  It copies the six exact named vectors from the selected cells, including
  bias rates modified by the `q90_cap` policy; rho is not recomputed
  downstream.
- Provenance: frozen source commit
  `20b495ec8f3bf4c6179480e2686bde1984b643c9`; source archive SHA-256
  `23ea58b784d189dd2e373e1e019af3b53bd6638a7d51390e654e19f1d2e9b9ac`;
  run-input archive SHA-256
  `43b8114218c566f88ab2454d82915b245ed9b8c9089254d8e2d70e0184ae62cd`;
  1,817-file manifest SHA-256
  `07bc63932bbe3df5dbc9d0013432315392fed3feca3555b16ca7cf8403120190`,
  verified locally with exit code 0.
- Detailed evidence:
  [summary](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/study_summary.md),
  [selected-point table](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_by_surface.csv),
  [rho surfaces](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/rho_validation_surfaces.png),
  [optimizer comparison](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/best_validation_loss_by_surface.png),
  and
  [final receipt](../results/conv3_pd_unbounded_rho_t8k8_20260729T125734Z/analysis/final_receipt.json).

Existing reviewed JSON cards under `result_registry/` remain historical
evidence and need not be rewritten.

<!--
## <study-id> — <title>

- Analyzed: YYYY-MM-DD
- Evidence class:
- Outcome:
- Runs and seeds:
- Setup:
- Headline measurements:
- Interpretation:
- Limitations:
- Raw results:
- Analysis:
- Detailed evidence:
-->
