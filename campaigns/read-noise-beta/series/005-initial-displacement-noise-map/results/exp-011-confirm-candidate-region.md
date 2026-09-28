---
experiment: "exp-011"
evidence: "validated-local"
summary: "Conv1 and Conv2 selected ours-versus-legacy worst-layer advantages pass both cells in all five fresh draws; Conv3's tiny candidate is mixed (6/10 positive)."
verdicts: {"H-008": "supports"}
---
# Confirmation of selected initialization neighborhoods: exp011

All12 selected contexts,18 noisy scheme/cells and11,016 finite layer/batch/draw
rows are collected and scientifically validated; no failures, exclusions,
retries or replacement seeds. All three schemes are included. The selected
competitor is ours, and each draw computes W=min_layer(mean cosine over36 batches).

| Architecture | Selected cells (D/F,sigma) | Positive cell/draw contrasts | Ours-minus-legacy W margin range | Scoped outcome |
|---|---|---:|---:|---|
| Conv1 | (0.8,0.001), (0.8,0.002) | 10/10 | +0.309521 to +0.381923 | confirmed |
| Conv2 | (6,0.0001), (10,0.0001) | 10/10 | +0.130685 to +0.215791 | confirmed |
| Conv3 | (0.2,0.002), (0.2,0.005) | 6/10 | -0.00341023 to +0.00128070 | mixed; not confirmed |

Conv3 seeds96m and97m are negative at both cells and are retained. None of the
selected ours comparisons has positive mean differences in every parameter
layer, so the confirmed result is worst-layer improvement, not all-layer dominance.
The [full report](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/confirmation/analysis/report.md)
links the five-draw plot, every scheme/layer/draw comparison, physical controls,
coverage audit and identities. The
[confirmation figure](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/confirmation/analysis/confirmation_margins.png)
uses explicitly labelled per-panel y scales.

The exact screen initializers,576-example cohort, calibrated betas, clean finite-K
BPTT reference, native float64, zero frozen biases, physical20-output squared
loss, bounds[0,100], gains40/100/360 and T=K4/6/8 remain fixed. Fresh independent
noise seeds94m–98m are paired across schemes within each depth. No optimizer,
accuracy evaluation or official-test read occurred. Hash-valid reconstructed
clean trajectories and BPTT references reproduce the screen exactly; this bounded
reconstruction was explicitly recorded before selection because raw endpoint
caches were not persisted. All its compute is included in the confirmation cap.

Canonical configs and runner, source/calibration/checkpoint/cohort hashes, exact
batch/layer/noise coverage, paired draws, frozen-force/read-only/float64 guards,
same-host/runtime identity and actual output D/F matching pass. Authoritative
roots are study `confirmation/conv3` and `collected/akib/confirmation/{conv1,conv2}`;
Conv1/2 stayed on Akib RTX3080 and Conv3 on local RTX3090. The separate confirmation
source archive SHA256 is
`e540c0cf3dc457bd8614be134704d02303814b778a51e75492cd23f7e79816b1`;
runner SHA256 `2525967925b75c566537ea3e0fae67921220f93444964b4c31a01a6f50e01529`.
Confirmation used approximately0.159GPUh including smokes, below2GPUh. Matching,
screen and confirmation together used approximately0.97GPUh; both GPU lanes
finished before12:30Paris, before the20:00 deadline.

**H008 is supported for its existential initialization claim by Conv1 and Conv2.**
The claim requires at least one reproducible sampled neighborhood in at least
one architecture; those two complete pairs satisfy its five-draw rule. Conv3's
mixed pair remains inconclusive in its own scope and does not establish an
all-depth advantage. This is conditional evidence on one initializer per depth
and one cohort, not model-seed significance, a continuous-region proof, training
accuracy improvement or equal SNR under relative-output matching.

The authorized series is complete. No further GPU work, training or trained-state
replay is launched. Any later persistence study needs its own frozen checkpoint
selection and assignment.
