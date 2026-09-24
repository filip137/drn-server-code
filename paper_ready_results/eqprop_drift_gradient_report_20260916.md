# Residual voltage drift and centered EP–BPTT gradient agreement

Date: 2026-09-16. Evidence: read-only ordinary-MNIST validation diagnostics.

**The large residual-drift ratios in the affected cases do not translate into
large EP–BPTT gradient discrepancies. Increasing T removes most of that drift,
while the measured gradient agreement changes little. This supports retaining
the native iteration counts for now; it does not establish that incomplete
relaxation can never affect the gradient.**

## Recorded iteration decision

After reviewing these results, Filip instructed: “for now we keep 4 6 8
iterations for all.” The retained settings are:

| Architecture | Free-phase T | Gradient-phase K | Amplification schemes |
|---|---:|---:|---|
| Conv1 | 4 | 4 | baseline, legacy, ours |
| Conv2 | 6 | 6 | baseline, legacy, ours |
| Conv3 | 8 | 8 | baseline, legacy, ours |

These are the common counts for the present wide-weight EP/BPTT comparisons
and read-noise discussion. The tentative increase to Conv3 baseline T=12/K=8
is not adopted. The 1% drift-to-response threshold remains descriptive and
is not a new training requirement. The separate gradient and equilibrium
checks retain their measured outcomes and qualification caveats.

This report records the decision without launching training. Historical
bounded-weight runs with longer T remain separately identified by their
actual configurations; their results are not relabeled or newly qualified
by this wide-weight diagnostic. The paused training ledger is unchanged.

## What the drift quantity measures

For each input batch, let F_T be the free voltage state after T iterations.
From that same state, run another K iterations with three nudging settings:
zero, positive beta, and negative beta. Denote the endpoints by Z_K, V_K^+
and V_K^−. Inputs and weights remain fixed.

For layer ℓ, the zero-nudge drift is d_ℓ = Z_K^ℓ − F_T^ℓ. It measures how much
the voltages would continue to move over the next K iterations without a
training nudge. The controlled nudging response is
r_ℓ^± = V_K^{ℓ,±} − Z_K^ℓ. The raw displacement decomposes exactly as

\[
V_K^{\ell,\pm}-F_T^\ell=d_\ell+r_\ell^\pm.
\]

The pooled drift RMS and the dimensionless drift-to-response ratio are

\[
D_\ell=\sqrt{\frac{1}{N_\ell}\sum_{b,i}d_{\ell,b,i}^{\,2}},
\qquad
R_\ell^\pm=
\sqrt{\frac{\sum_{b,i}d_{\ell,b,i}^{\,2}}
{\sum_{b,i}(r_{\ell,b,i}^\pm)^2}}.
\]

The sum includes all voltage elements in that layer across all 36 batches
of 16 examples. “Pooled” means combining sums of squares before taking the
RMS, rather than averaging batchwise ratios. Layers, model schemes, checkpoint
roles and nudging signs remain separate. A reported **worst pooled ratio**
is the maximum across layers and signs for one configuration and checkpoint.
Percentages are 100 × R; the tables below report R as a multiplier.

For example, R=2.78 means that continued relaxation has 2.78 times the RMS
magnitude of the controlled nudging response. It is not a 278% gradient error,
nor a fraction of the raw displacement. RMS magnitudes do not generally add
because the drift and response vectors can point in different directions.
This finite-K drift also does not measure distance to the exact equilibrium.

## Why a large ratio need not imply a gradient mismatch

The implementation uses centered EP. With Q denoting the energy derivative
with respect to a weight, its estimator is proportional to

\[
\widehat g_{\mathrm{EP}}\propto
\frac{Q(V_K^+)-Q(V_K^-)}{2\beta},
\]

with the declared scheme-dependent normalization applied by the runner.
It does not estimate the gradient from the raw free-to-positive voltage
displacement alone. The beta-independent component of Q cancels in the
positive-minus-negative subtraction. Within a smooth regime, the centered
difference also removes the leading finite-beta bias. The standard theoretical
EP–BPTT equivalence depends on equilibrium assumptions and the small-beta
limit; finite-K matching requires the corresponding dynamics and truncation
conditions. [Laborieux et al., *Scaling Equilibrium Propagation to Deep
ConvNets by Drastically Reducing Its Gradient Estimator Bias*, sections 2.2.2
and 3.1](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.633674/full).

Cancellation of the sign-independent term is not a proof that residual
relaxation is irrelevant. Changing the free state can change the nudging
response, the frozen output force, and the parameter derivatives. A shared
voltage shift need not cancel from every nonlinear product in Q. Perfect-diode
active-set changes also limit a smooth Taylor argument. The empirical question
must therefore be answered using the gradients themselves.

In this implementation, BPTT starts from the identical post-T state and uses
K gradient iterations. Positive and negative EP branches each restore that
same state and reuse the same output force, frozen at the post-T state.
The comparison is against this matched, finite-K BPTT reference; it is not a
proof of equality to fully converged implicit differentiation or to an
untruncated gradient through all free-phase iterations.

## Controlled experiment and direct gradient results

The T-only study uses seed 0, the same 576 validation examples, wide [0,100]
weights, exact-zero biases, float64, and saved initialization and best BPTT
checkpoints. Within each sweep, only T changes. The cases are:

| Case | Checkpoints | T values | Fixed K | Injected beta |
|---|---|---|---:|---:|
| Conv3 baseline | initialization and best | 8, 10, 12, 16, 24 | 8 | 10, unconfirmed candidate |
| Conv3 legacy | initialization and best | 8, 10, 12, 16, 24 | 8 | 0.001 |
| Conv2 legacy | initialization | 6, 8, 10, 12, 16 | 6 | 0.03 |

No optimizer steps, endpoint read noise, or official-test evaluation are part
of this diagnostic. The T sweep includes only the affected cases identified
by the preceding nine-case displacement study.

For each checkpoint/T pair, the gradient columns below are the minimum cosine
and maximum symmetric norm discrepancy over batches and scored weight tensors.
The latter is

\[
\delta_{\rm norm}=
\frac{2\left|\|g_{\rm EP}\|_2-\|g_{\rm BPTT}\|_2\right|}
{\|g_{\rm EP}\|_2+\|g_{\rm BPTT}\|_2},
\]

with a denominator floor in the implementation. It compares magnitudes, not
the norm of the vector difference. The existing gates require cosine ≥0.99
and norm discrepancy ≤0.10. Bias tensors are excluded from gradient scoring.

| Case | Checkpoint | T | Worst pooled drift/response | Minimum gradient cosine | Maximum norm discrepancy | Endpoint residual gate |
|---|---|---:|---:|---:|---:|---|
| Conv2 legacy | initialization | 6 | 2.78264 | 0.999883062 | 0.005051517 | pass |
| Conv2 legacy | initialization | 16 | 1.09238e-7 | 0.999882398 | 0.005068553 | pass |
| Conv3 baseline | initialization | 8 | 0.317312 | 0.994886405 | 0.057005854 | pass |
| Conv3 baseline | initialization | 24 | 7.28998e-11 | 0.994886408 | 0.057005865 | pass |
| Conv3 baseline | best | 8 | 224.456 | 0.999943525 | 0.001378272 | fail |
| Conv3 baseline | best | 24 | 2.45117e-7 | 0.999943521 | 0.001275815 | pass |
| Conv3 legacy | initialization | 8 | 49.6781 | 0.999960925 | 0.003262729 | pass |
| Conv3 legacy | initialization | 24 | 1.14132e-8 | 0.999961000 | 0.003263364 | pass |
| Conv3 legacy | best | 8 | 1.02339 | 0.999990491 | 0.000637698 | pass |
| Conv3 legacy | best | 24 | 0 | 0.999990491 | 0.000637698 | pass |

All 15 included configurations pass the direct gradient gates on this seed-0
cohort, including every intermediate T. The [complete 25-row checkpoint/T
table](eqprop_drift_gradient_comparison_20260916.csv) retains the source
summary paths and result hashes. Zero in the last row is measured float64
stationarity over K steps, not a claim of exact real-number convergence.

The trained Conv3 baseline illustrates the distinction: reducing the drift
ratio from 224.456 to 2.45e-7 barely changes the gradient cosine and reduces
the norm discrepancy only from 0.001378 to 0.001276. At T=12, its residual
gate already passes, although drift/response is still 1.067. Conversely,
Conv2 legacy's gradient metrics become slightly worse as T increases, while
remaining comfortably within the gates. Gradient agreement does not improve
monotonically with this drift diagnostic.

Across the entire sweep, controlled nudging-response RMS differs from its
largest-T value by less than 0.001% in every measured layer and sign; the
maximum is 0.000915105%. This is a magnitude result. It alone would not prove
unchanged response direction or gradients; the direct comparisons above
provide the relevant additional evidence.

## Native settings and remaining qualification limits

The earlier nine-case study provides the following seed-0 outcomes at the
retained counts, combining initialization and best-checkpoint checks:

| Architecture | Scheme | T/K | Injected beta tested | Gradient gate | Equilibrium gate |
|---|---|---:|---:|---|---|
| Conv1 | baseline | 4/4 | 100 | pass | pass |
| Conv1 | legacy | 4/4 | 3 | pass | pass |
| Conv1 | ours | 4/4 | 30 | pass | pass |
| Conv2 | baseline | 6/6 | 100 | pass | pass |
| Conv2 | legacy | 6/6 | 0.03 | pass | pass |
| Conv2 | ours | 6/6 | 10 | pass | pass |
| Conv3 | baseline | 8/8 | 10 | pass on this cohort | fail at best checkpoint |
| Conv3 | legacy | 8/8 | 0.001 | pass | pass |
| Conv3 | ours | 8/8 | 3 | fail | pass |

These tested betas are diagnostic settings, not a new set of promoted values.
Conv3 baseline beta 10 failed two initialization comparisons on confirmation
seed 1 (cosine 0.968773 and norm discrepancy 0.101208); the present seed-0
sweep does not resolve that failure. Conv3 ours retains its known gradient
exception (minimum cosine 0.939168, maximum norm discrepancy 0.278178 across
the earlier diagnostic). Keeping 4/6/8 does not assert that every case passes.

The supported interpretation is that the 1% relative-drift target is too
restrictive to use as a proxy for EP–BPTT agreement in these measured cases.
Direct gradient agreement is the more relevant evidence for that specific
question, while residual convergence remains separately reported. The user
decision retains native T/K; it does not waive beta qualification, establish
read-noise robustness, or provide evidence across all training epochs/seeds.
Further beta work, if undertaken, should respect the retained T/K and the
already-consumed confirmation cohort. No follow-up experiment is launched by
this report.

## Evidence and verification

- [Nine-case displacement study](phase_displacement_20260916.md) and
  [source/gate table](phase_displacement_20260916_sources.csv).
- [T-sweep report and figures](phase_t_sweep_20260916.md),
  [per-layer drift/response data](phase_t_sweep_20260916_drift_response_comparison.csv),
  [source/gate table](phase_t_sweep_20260916_sources.csv), and
  [new joined gradient/drift table](eqprop_drift_gradient_comparison_20260916.csv).
- [Fixed-T8/K8 beta selection and confirmation](conv3_baseline_beta_tk8_20260916.md).
- [Local T-sweep evidence](../results/eqprop-phase-displacement-t-sweep-20260916-v1/)
  and [integrity record](../results/eqprop-phase-displacement-t-sweep-20260916-v1/analysis/integrity.json):
  15 included configurations, 900 replays and 3,420 gradient comparisons.
  The failed Conv2 completion-metadata attempt is preserved as
  `runs/conv2_legacy_seed0_T6`; the included replacement is
  `runs/conv2_legacy_seed0_T6_v2`. The failed attempt is excluded from the tables.
- Before writing this report, the 15 included result hashes, their 165 declared
  artifact hashes/sizes, all 14 previously promoted T-sweep artifact hashes,
  and the nine native-case result hashes were checked. Existing measured
  reports and experiment artifacts remain unchanged. This report and its
  joined CSV add interpretation and traceability; they do not add simulations.

Implementation references: [pooled RMS aggregation](../experiments/analyze_phase_displacement.py),
[drift/response and worst-case aggregation](../experiments/analyze_phase_displacement_t_sweep.py),
[centered estimator and matched BPTT replay](../experiments/audit_eqprop_float64_shadow.py),
and [gradient scoring](../experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py).
