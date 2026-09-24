# How the problematic phase displacements change with T

Completed 2026-09-16. This read-only sweep varies only the free-phase length T
for the affected cases from the earlier nine-case displacement study. The
original checkpoint, K, beta, initialization, cohort, numerical precision and
source training settings remain fixed along each curve.

**Increasing T makes residual drift negligible in every included case, while
the controlled nudging-response RMS changes by less than0.001% throughout
the tested grids relative to the largest T.** Thus the large changes in raw
free-to-nudged displacement are attributable to continued relaxation, rather
than to a meaningful change in the nudging-response magnitude.

For trained Conv3 baseline H1, pooled drift/response falls from224.456 at T8
to15.107 at T10,1.0673 at T12,.005871 at T16 and2.451e-7 at T24. T12 passes
the separate equilibrium-residual gate but is still insufficient relative to
this small nudging signal. T16 meets the pooled1% target; its worst batch is
still1.30%, so the every-batch target first passes at the tested T24 point.

Fifteen configurations and 900 checkpoint/batch replays are complete and
validated. The repeated native-T cases reproduce every previous shared
gradient, residual and displacement measurement exactly. The added observable
directly measures continued relaxation, including cases where the old raw/control
RMS ratio was close to one.

## Smallest tested T with small drift

The descriptive target below is **zero-nudge drift <=1% of the controlled
nudging-response RMS**, in every layer and both nudge signs. A listed T must
also satisfy the target at every larger tested T. The pooled column combines
all 576 examples by element count; the stricter column requires every one of
the 36 batches to satisfy the same criterion. This is not a new beta or
training qualification gate.

| Model | Scheme | Checkpoint | Smallest tested T, pooled | Smallest tested T, every batch |
|---|---|---|---:|---:|
| Conv3 | baseline | initialization | 12 | 12 |
| Conv3 | baseline | best BPTT checkpoint | 16 | 24 |
| Conv3 | legacy | initialization | 16 | 16 |
| Conv3 | legacy | best BPTT checkpoint | 10 | 12 |
| Conv2 | legacy | initialization | 10 | 10 |

The grid does not determine the exact minimum between tested values. All
threshold results concern seed0 and these particular saved checkpoints.
Longer T is a replay diagnostic here; no training configuration was changed.
Conv3 baseline beta10 still carries its previous seed1 confirmation failure.

![Direct drift relative to nudging response versus T](phase_t_drift_fraction_20260916.png)

## Per-layer drift relative to the nudging response

Each entry is RMS(Z−F)/RMS(P−Z) for the **positive nudge**, where F is the
post-T free state, Z the endpoint after K additional zero-nudge steps, and P
the endpoint after K positive-nudge steps from F. For example, 0.01 means
drift RMS is 1% of controlled-response RMS; 10 means it is ten times larger.
The final column retains the earlier raw/control ratio RMS(P−F)/RMS(P−Z).
Both signs are present in the figures and complete CSV.

### Conv3 baseline: initialization

| T | H1 drift/response | H2 | H3 | Output | H1 raw/control |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.31731 | 0.038827 | 0.001563 | 1.0847e-06 | 1.05929 |
| 10 | 0.01358 | 0.001669 | 6.7163e-05 | 4.7106e-08 | 1.00045 |
| 12 | 0.0005843 | 7.2004e-05 | 2.8986e-06 | 2.042e-09 | 1.00001 |
| 16 | 1.09e-06 | 1.3468e-07 | 5.4252e-09 | 3.8332e-12 | 1 |
| 24 | 7.29e-11 | 1.9237e-12 | 2.916e-14 | 1.763e-17 | 1 |

### Conv3 baseline: best BPTT checkpoint

| T | H1 drift/response | H2 | H3 | Output | H1 raw/control |
|---:|---:|---:|---:|---:|---:|
| 8 | 224.46 | 33.157 | 1.121 | 0.0082582 | 224.482 |
| 10 | 15.107 | 2.3012 | 0.076672 | 0.00056148 | 15.1622 |
| 12 | 1.0673 | 0.16562 | 0.0053987 | 3.8724e-05 | 1.47807 |
| 16 | 0.0058712 | 0.0009361 | 2.8669e-05 | 1.8857e-07 | 1.00013 |
| 24 | 2.4512e-07 | 4.0856e-08 | 1.0672e-09 | 5.2451e-12 | 1 |

### Conv3 legacy: initialization

| T | H1 drift/response | H2 | H3 | Output | H1 raw/control |
|---:|---:|---:|---:|---:|---:|
| 8 | 49.678 | 6.0847 | 0.24508 | 0.00017014 | 49.7069 |
| 10 | 2.126 | 0.26155 | 0.010531 | 7.3888e-06 | 2.36154 |
| 12 | 0.091478 | 0.011284 | 0.00045452 | 3.203e-07 | 1.00506 |
| 16 | 0.00017064 | 2.1107e-05 | 8.507e-07 | 6.0126e-10 | 1 |
| 24 | 1.1413e-08 | 3.0147e-10 | 4.5724e-12 | 2.7654e-15 | 1 |

### Conv3 legacy: best BPTT checkpoint

| T | H1 drift/response | H2 | H3 | Output | H1 raw/control |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.0234 | 0.047512 | 0.0039446 | 8.5667e-07 | 1.40956 |
| 10 | 0.0074855 | 0.00036874 | 3.051e-05 | 6.8368e-09 | 0.999659 |
| 12 | 6.2052e-05 | 3.1449e-06 | 2.5931e-07 | 5.9101e-11 | 0.999996 |
| 16 | 6.9545e-08 | 1.8418e-09 | 3.0391e-11 | 5.5765e-15 | 1 |
| 24 | 0 | 0 | 0 | 0 | 1 |

### Conv2 legacy: initialization

| T | H1 drift/response | H2 | H3 | Output | H1 raw/control |
|---:|---:|---:|---:|---:|---:|
| 6 | 2.7824 | 0.074057 | — | 0.00038338 | 2.98248 |
| 8 | 0.083784 | 0.0022828 | — | 1.3575e-05 | 1.00546 |
| 10 | 0.0026984 | 7.4644e-05 | — | 4.806e-07 | 1.00005 |
| 12 | 9.0665e-05 | 2.53e-06 | — | 1.7034e-08 | 1 |
| 16 | 1.0923e-07 | 3.0718e-09 | — | 2.1531e-11 | 1 |

## Does the controlled nudging response itself change?

Increasing T can change the free operating point as well as reducing its
remaining relaxation. The table reports the maximum absolute relative change
in controlled-response RMS from the largest tested T, across all layers and
both nudge signs. It compares RMS magnitudes, not vector directions.

| Model / scheme | Checkpoint | T | Maximum response change from largest T |
|---|---|---:|---:|
| Conv3 baseline | initialization | 8 | 3.5395e-08% |
| Conv3 baseline | initialization | 10 | 1.259e-09% |
| Conv3 baseline | initialization | 12 | 8.4066e-11% |
| Conv3 baseline | initialization | 16 | 4.774e-13% |
| Conv3 baseline | initialization | 24 | 0% |
| Conv3 baseline | best BPTT checkpoint | 8 | 0.0009151% |
| Conv3 baseline | best BPTT checkpoint | 10 | 5.3313e-05% |
| Conv3 baseline | best BPTT checkpoint | 12 | 3.2903e-06% |
| Conv3 baseline | best BPTT checkpoint | 16 | 1.4645e-08% |
| Conv3 baseline | best BPTT checkpoint | 24 | 0% |
| Conv3 legacy | initialization | 8 | 1.88e-06% |
| Conv3 legacy | initialization | 10 | 5.9846e-08% |
| Conv3 legacy | initialization | 12 | 1.9825e-08% |
| Conv3 legacy | initialization | 16 | 7.5057e-10% |
| Conv3 legacy | initialization | 24 | 0% |
| Conv3 legacy | best BPTT checkpoint | 8 | 6.2515e-08% |
| Conv3 legacy | best BPTT checkpoint | 10 | 2.2655e-09% |
| Conv3 legacy | best BPTT checkpoint | 12 | 2.3448e-11% |
| Conv3 legacy | best BPTT checkpoint | 16 | 2.2204e-14% |
| Conv3 legacy | best BPTT checkpoint | 24 | 0% |
| Conv2 legacy | initialization | 6 | 6.6257e-05% |
| Conv2 legacy | initialization | 8 | 1.6977e-06% |
| Conv2 legacy | initialization | 10 | 4.2133e-08% |
| Conv2 legacy | initialization | 12 | 9.8372e-10% |
| Conv2 legacy | initialization | 16 | 0% |

![Controlled nudging response versus T](phase_t_controlled_response_20260916.png)

## Raw/control ratio and direct drift

![Raw/control RMS ratio versus T](phase_t_raw_control_20260916.png)

![Absolute zero-nudge drift versus T](phase_t_zero_drift_20260916.png)

The exact vector identity is P−F = (P−Z) + (Z−F). Its two terms need not point
in the same direction; their RMS values therefore do not add as scalars.
A raw/control RMS ratio near one alone does not prove negligible relaxation.
The direct drift/response ratio answers that question more directly. Measuring
small drift across K steps still does not establish arbitrary-time convergence;
the separate projected-residual diagnostics are retained.

## Contract and coverage

| Model/scheme | Observed checkpoint roles | T grid | K | Injected beta |
|---|---|---|---:|---:|
| Conv3 baseline | initialization, best BPTT | 8,10,12,16,24 | 8 | 10, unconfirmed |
| Conv3 legacy | initialization, best BPTT | 8,10,12,16,24 | 8 | .001 |
| Conv2 legacy | initialization | 6,8,10,12,16 | 6 | .03 |

Cases were selected from the previous study when the positive-nudge raw/control
RMS ratio differed from one by at least1% in any hidden layer. Conv1, ours,
and the trained Conv2 legacy checkpoint were omitted as requested. This does
not assign them a passing result on the newly measured direct-drift criterion.

All curves use seed0, the same36 validation batches of16, original wide [0,100]
weights and Kaiming initializers, exact-zero biases, original input gains,
float64 centered frozen-current EqProp, and zero endpoint read noise. Base
beta remains injected beta divided by the scheme's output-row scale. Trained
checkpoints retain their original Adam learning histories and best epochs;
there is no retraining, intermediate-epoch reconstruction or optimizer step.
No official-test data are read and no new confirmation cohort is consumed.

All15 formal bundles and both smoke attempts validate:900 included replays,3,420 gradient
comparisons,20,520 individual-layer displacement rows,570 pooled layer/context
groups and190 sign-specific drift/response comparisons. No formal case is
missing or excluded. Source bytes and cohort payload guards pass. Canonical
gradient/residual gate failures remain in the evidence and do not prevent
the requested sensitivity diagnostic from being analyzed.

| Model / scheme | T | Gradient gate over included roles | Equilibrium gate over included roles |
|---|---:|---|---|
| Conv3 baseline | 8 | pass | fail |
| Conv3 legacy | 8 | pass | pass |
| Conv2 legacy | 6 | pass | pass |
| Conv3 baseline | 10 | pass | fail |
| Conv3 legacy | 10 | pass | pass |
| Conv2 legacy | 8 | pass | pass |
| Conv3 baseline | 12 | pass | pass |
| Conv3 legacy | 12 | pass | pass |
| Conv2 legacy | 10 | pass | pass |
| Conv3 baseline | 16 | pass | pass |
| Conv3 legacy | 16 | pass | pass |
| Conv2 legacy | 12 | pass | pass |
| Conv3 baseline | 24 | pass | pass |
| Conv3 legacy | 24 | pass | pass |
| Conv2 legacy | 16 | pass | pass |

Execution: one local RTX3090, **27.63 minutes,
0.460455/1 physical GPU-hours**, including smoke. Eight
focused iteration/measurement tests passed before launch; an additional
coverage-count regression test passed before recovery. One operational failure
occurred: the initialization-only Conv2 legacy config retained completion
counts for two checkpoint roles. Its36 replays finished but the final count
check correctly failed before a successful result bundle was written.
The failed attempt is preserved at `runs/conv2_legacy_seed0_T6`; the unchanged
scientific case was rerun with corrected completion metadata at
`runs/conv2_legacy_seed0_T6_v2`. The other four unstarted Conv2 configs received
the same correction in `recovery_v2/`. Both completed Conv3 cases were retained.
Original/recovery inputs and source identities are preserved; the failed
attempt and both smokes are included in the GPU cost above. The GPU was released at
completion. Differences between schemes retain their beta/training confounds;
the T curves themselves hold those factors fixed. These are deterministic
single-seed mechanism measurements, not noisy-training or paper accuracy.

Files:

- [Drift/response comparisons, both signs](phase_t_sweep_20260916_drift_response_comparison.csv).
- [All pooled per-layer measurements](phase_t_sweep_20260916_layer_displacement_summary.csv).
- [Source checkpoints and run provenance](phase_t_sweep_20260916_sources.csv).
- [Structured results and descriptive T thresholds](phase_t_sweep_20260916_summary.json).
- PDF figures: [drift fraction](phase_t_drift_fraction_20260916.pdf),
  [raw/control ratio](phase_t_raw_control_20260916.pdf),
  [controlled response](phase_t_controlled_response_20260916.pdf),
  [absolute drift](phase_t_zero_drift_20260916.pdf).
- [All local bundles and raw batch measurements](../results/eqprop-phase-displacement-t-sweep-20260916-v1/).
- [Execution plan](../docs/eqprop_phase_displacement_t_sweep_plan_20260916.md).
- [Previous all-scheme displacement comparison](phase_displacement_20260916.md).
