# Proposed diagnostic tables for Sections 4.2 and 4.3

Proposal dated 2026-09-18, updated to use a figure for Section 4.2. The
bound-occupancy measurements are now extracted from the saved checkpoints;
the Section 4.3 replay remains a proposal. No new training was performed.
The manuscript reviewed is
[`bidir_paper_theory_revised.tex`](../papers/amplification_overleaf/bidir_paper_theory_revised.tex).

Add a diagnostic figure for Section 4.2 and a diagnostic table for Section
4.3, alongside the existing accuracy tables. The accuracy tables describe
performance; these additions test possible explanations.

## Section 4.2: conductance occupancy at the bounds

Generated figure: **Fraction of learned conductances at the hardware bounds.**

Use six panels: columns for Conv1, Conv2 and Conv3; rows for the percentage
at G_min and G_max. Each panel shows the three schemes as colored lines
with distinct markers, across the three ceilings on a logarithmic x-axis.
Points and error bars are mean +/- sample SD across three seeds. Each row
shares its y-axis across architectures; rows have separate scales because
upper-bound occupancy is much smaller.

[Figure and measurements](bounded_conductance_occupancy_best.md) ·
[PDF](figures/bounded_conductance_occupancy_best.pdf)

Measurement rules:

- Use the same best-validation checkpoints and original shared T/K as the
  existing bounded BPTT accuracy table: 4/4, 6/6 and 8/8. Do not substitute
  the longer-T baseline repeats.
- Count only `ConvWeight_*` and `DenseWeight_*`. Each stored trainable
  coefficient counts once; do not multiply convolution coefficients by
  their spatial reuse. Exclude biases and checkpoint metadata.
- Within a seed, pool counts across weight tensors:
  `p_min = 100 * sum_l count(G_l == G_min) / sum_l numel(G_l)`, and likewise
  for `p_max`. This is a parameter-count-weighted fraction, not an average
  of layer percentages. Use G_min = 1e-5 and each run's recorded G_max.
- Compare with bounds cast to the checkpoint tensor's dtype. Exact endpoint
  equality is the primary statistic because projection can produce exact
  endpoints. Keep any near-bound tolerance audit separate and explicit.
- Report mean +/- sample SD across model seeds 0, 1 and 2, preferably to one
  decimal place in percent. Preserve per-seed counts in the underlying CSV.
- In the supplement, provide the same measurements separately for each
  convolutional tensor and the dense readout, plus the final checkpoints.
  Record best-checkpoint epochs so selection timing remains visible. The
  current extraction includes best-checkpoint per-layer data; final-checkpoint
  extraction remains optional follow-up.

This tests whether the observed accuracy ordering accompanies different
use of the allowed conductance range, and whether occupancy changes with
depth or ceiling. Lower-bound and upper-bound occupancy should remain
separate: they describe different learned distributions.

Use **"at the bound"**, rather than **"stuck"**. A checkpoint does not show
persistence or a blocked optimizer proposal. Establishing persistent
sticking requires repeated parameter observations or saved projection
traces. High occupancy alone does not establish failed learning: boundary
solutions can be useful, and the schemes have different learning rates.

Availability: all 81 expected run directories under
[`bundles/table3_bounded_bptt/`](bundles/table3_bounded_bptt/) have both
`weights_best.npz` and `weights_final.npz` locally (27 runs per architecture).
The completed figure extraction additionally verifies all 81 selected
checkpoint hashes, metadata and numerical bounds, as recorded in the
[measurement report](bounded_conductance_occupancy_best.md). The
[reproduction script](plot_bounded_conductance_occupancy.py) uses the same
exact dtype-aware endpoint-counting definition as the existing
[final-clipping analyzer](../experiments/analyze_bounded_wmax_final_clipping.py).
No retraining or MNIST data access was needed.

## Section 4.3: gradient direction under endpoint read noise

Suggested title: **EqProp gradient alignment under voltage read noise.**

Primary metric for layer l, minibatch b and noise draw r:

`c_task(l,b,r,sigma) = cos(g_EP(sigma), g_BPTT)`.

Both gradients must be evaluated at identical weights and on the same
minibatch. BPTT is the matched finite-K reference, not a claim about an
exact fully converged gradient. Use gradients with the implemented
scheme-dependent scaling, before Adam or learning-rate multiplication.

Use nine primary rows, one per architecture and scheme, with the injected
beta explicit. Include clean readout and representative noise levels:

| Architecture | Scheme | Injected beta | sigma = 0 | 1e-5 | 1e-4 | 5e-4 |
|---|---|---:|---:|---:|---:|---:|
| Conv1 | Baseline / Ours / Legacy, separate rows | recorded | pending | pending | pending | pending |
| Conv2 | Baseline / Ours / Legacy, separate rows | recorded | pending | pending | pending | pending |
| Conv3 | Baseline / Ours / Legacy, separate rows | recorded | pending | pending | pending | pending |

For a compact all-layer summary, each cell can report the **lowest layer
median cosine**: compute the median across minibatches and noise draws for
each weight tensor, then take the minimum across tensors. Identify the
limiting tensor in the underlying data. This avoids a concatenated gradient
being dominated by a large or high-norm tensor. It is not the minimum
individual cosine and is not an all-batch qualification gate.

The supplement should show every layer separately at all five nonzero
noise levels, with the median and 10th/90th percentiles. Compute each cosine
before summarizing across draws: averaging noisy gradients first would
measure a different estimator with less noise.

Include a second comparison in the supplement:

`c_noise(l,b,r,sigma) = cos(g_EP(sigma), g_EP(0))`.

This directly isolates the directional distortion introduced by readout
noise. The primary metric assesses task-gradient agreement; the secondary
metric separates noise effects from an existing clean EqProp/BPTT mismatch.
If only one cosine table fits, retain the primary metric with its sigma-zero
column and discuss the second comparison in the text.

Suggested replay design:

- Main table: model seed 0, each scheme's clean EqProp best checkpoint,
  including the matching-beta baseline clean controls. Freeze that checkpoint
  across all noise columns. Replay the saved common initialization as a
  supplementary control before training trajectories diverge.
- Use one frozen validation cohort, initially 32 batches of 16 examples,
  with eight independent endpoint-noise realizations per nonzero sigma.
  These counts are a proposed diagnostic budget, not a completed analysis.
  Batch/noise variability is not uncertainty across training seeds.
- Keep T/K = 4/4, 6/6 and 8/8, saved preprocessing, loss, frozen-current
  centered EqProp semantics, and float64 fixed. Restore the same post-T state
  for the positive/negative branches and BPTT reference. Take no optimizer
  steps; verify checkpoint and in-memory parameter hashes remain unchanged.
- Perturb endpoint readout only. Use independent positive/negative draws;
  preserve the noiseless dynamics and inputs. Pre-generate shared standard
  normal draws across same-shaped scheme cases on one execution environment
  and scale them by sigma, retaining exact draw identities.
- Keep each training contract's injected beta: baseline/ours/legacy is
  100/30/3 for Conv1, 100/10/0.03 for Conv2, and 10/3/0.001 for Conv3 in the
  completed noise comparison. Show the extra Conv1 baseline beta-200 control
  separately. The earlier Conv3 baseline beta-100 clean run is not its
  beta-10 noise control.
- In the supporting CSV, retain gradient norms, relative vector error,
  near-zero fractions and undefined-cosine counts. A high cosine can coexist
  with a large magnitude error. Do not assign cosine one to zero vectors.

The proposed measurement asks whether the layers needed for learning lose
gradient direction as read noise increases. It must not compare gradients
from independently noise-trained checkpoints across sigma as its primary
noise-isolation experiment: their weights have also changed. Such checkpoints
can support a separate training-outcome analysis.

The existing [zero-bias gradient analyzer](../experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py)
already records `cosine`, `clean_eqprop_bptt_cosine` and
`noisy_clean_eqprop_cosine`, together with norm/error diagnostics. This
supports extending a read-only replay rather than new training. Existing
historical checks need a checkpoint/beta/cohort compatibility audit before
reuse; their presence does not establish coverage of this proposed table.

## Manuscript update already supported by collected evidence

Section 4.3 and its accuracy table still label baseline coverage as pending.
The [September 18 completion report](baseline_read_noise_results_20260916.md)
records all 22 new baseline trainings collected and validated, including
matching clean controls where required. Update that table alongside the
diagnostic addition, keeping best and final accuracy distinct and using the
correct beta-specific references.

Preserve the existing single-training-seed, beta-qualification and
environment limitations. A matched diagnostic replay can isolate immediate
readout distortion on fixed weights, but cannot retroactively make the
historical noisy training trajectories matched. Accuracy values remain
validation measurements; this proposal requires no official-test access.
