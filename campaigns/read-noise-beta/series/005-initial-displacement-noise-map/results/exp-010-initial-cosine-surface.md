---
experiment: "exp-010"
evidence: "validated-local"
summary: "All45 initialization contexts/77760 cosine rows validated; ours qualifies for one adjacent worst-layer advantage pair per depth, awaiting independent-noise confirmation."
verdicts: {"H-008": "inconclusive"}
---
# Initialization layerwise cosine screen: exp010

All45 canonical contexts are collected and scientifically validated:270
scheme/target/noise cells,810 layer cells and77,760 finite layer/batch/draw
cosines. All45 actual pooled output D/F matches pass1%; no residual-flagged,
failed or excluded cases. No optimizer, accuracy evaluation or official test.

H008 selected **ours** for all three adjacent pairs. Conv1 at D/F0.8,
sigma0.001/0.002 has W margins+0.322373/+0.372188 against legacy. Conv2 at
D/F6/10,sigma0.0001 has+0.144026/+0.204071. Conv3 at D/F0.2,
sigma0.002/0.005 has only+0.000107823/+0.000108791; its W values are
negative, so this is not good absolute alignment. W is the minimum of the
layer means, including readout. No competitor dominates legacy in every layer
at any nonzero-noise grid cell.

The [full report](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/report.md)
links all layer/paired/per-draw summaries, physical D/F and D/sigma controls,
raw beta lookup, gradient scale/error tables, clean curves and PNG/PDF heatmaps.
The [coverage audit](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/analysis/coverage.json)
checks canonical bundles, frozen config/source/checkpoint/calibration identities,
exact cohort and noise coverage, paired gradients/draws, read-only guards and
actual output matching. No best-case omission or unmeasured interpolation is used.

The contract is the identical576-example ordinary-MNIST validation cohort,
seed-0 initializers, centered frozen-current EqProp, native float64, physical20
outputs, zero frozen biases, weights[0,100], gains40/100/360 and accepted
T=K4/6/8. The reference is finite-K BPTT from the common post-T state.
Matched D/F targets0.2/0.8/2/6/10, absolute endpoint sigma0/1e-4/5e-4/1e-3/2e-3/5e-3
and three draws91m/92m/93m remain as declared. Conv1/2 used Akib RTX3080;
Conv3 used local RTX3090, with every scheme for a depth on the same host.
Source archive SHA256 is
`666df5a434daad68074097ee526809d618a52f58d2298bf2c85eda857055daf1`;
runner SHA256 `cacca7852b8949164c65d19761ec7541acc3fd8ec18f0e77facae72131d06e0a`.
Authoritative roots are study `screen/conv3` and `collected/akib/screen/{conv1,conv2}`.
Screen use including all smokes was approximately0.784GPUh, below4GPUh.

**H008 remains inconclusive at the screening stage.** The complete grid yields
three predeclared candidates; [exp011](../experiments/exp-011-confirm-candidate-region.md)
now tests those exact pairs with five fresh noise draws before any reproducibility
claim. The screen experiment itself is complete. This does not establish
all-layer dominance, training benefits, other model-seed behavior or a continuous
region. Relative-D matching under absolute additive noise does not equalize SNR.


Confirmation has since completed: the selected Conv1/2pairs pass all five fresh
noise draws; Conv3 is mixed. See [exp011 reviewed result](exp-011-confirm-candidate-region.md). The screening-stage
verdict above is preserved; the final H008 interpretation uses that confirmation.
