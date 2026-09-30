---
id: "exp-007"
title: "Increase only nudgedK in trained block2 at one fixed beta per scheme"
status: "complete"
hypotheses: ["H-005"]
---
# Fixed-beta nudged-K screen

Filip requests increasing nudgedK and then clarifies: use a single beta value
to see whether it matters. This supersedes the initial suggestion to recalibrate
beta at eachK. Test block2 atK=[6,12,24,48] on the same original final epoch50
BPTT checkpoints and256-image cohort from exp-006. No intermediate-epoch replay
is included in this screen; actual epoch10/30 final checkpoints exist in all
three trajectories and their hashes match continuation provenance.

Fix physical injectedB at the previously selected block2 peak: baseline
1.0249903844184778; ours2.1340009060890868; legacy0.6540032060609531. Their previous
RMS values were0.01993,0.03488,0.002483 respectively. No beta calibration or sweep.
Full free-stateT=[6,6,4], native BPTT K=[6,6,4], frozen output forces, matched-local
BPTT K=6, saved weights/BN, batch32, preprocessing, no noise/augmentation and
native endpoint convention remain fixed. Each nudged phase restarts from the
same captured state. Record actual displacement as an outcome of changingK.

Primary score: minimum pooled EqProp/native-BPTT cosine among block2's three
Conv weights, with per-layer norms/sparsity and per-minibatch spread. H-005
predicts an absolute gain>=0.01 for ours; <0.001 changes count as unchanged here.
Do not equate operational BPTT with a fully converged equilibrium reference.

## Execution and monitoring handoff

Root: `results/cifar-block2-nudged-k-e50-20260929-v1/`; configs
`configs/cifar/block2_nudged_k_e50_20260929/`. Each scheme has frozen source/inputs,
job spec, canonical run and CPU collector. Intermediate checkpoint inventory is
`analysis/intermediate_checkpoints.json`. K6 must reproduce exp-006's same-beta
cosines/norms/RMS; compare with recorded tolerance, not bitwise CUDA identity.
Validate12 pooled/96 batch rows per scheme,3 weights x4K x8 minibatches, epoch50,
unchanged checkpoint bytes and model/BN, identical native reference acrossK.

Ours/baseline on Loulou5090 sequentially; legacy on Trex5090. Expected2–4min/scheme,
600s hard allowance each,1800s combined including attempts, deadline October1
08:00 Paris. Shared queue owns admission, monitoring and collection; jobs
`cifar-block2-k-e50-{ours,legacy,baseline}-5090-20260929`. No smokes or training.
Stop/preserve failures on nonfinite results, identity/coverage mismatch or budget
exhaustion; no scientific retuning. All three queue jobs/collections are complete;
12 pooled/96 batch rows per scheme validate, K6 anchors reproduce exp-006, and
all source checkpoints/model/BN remain unchanged. Existing six numerical tests pass.
No failures;191.7s combined charged runtime. [Reviewed result](../results/exp-007-fixed-beta-nudged-k.md)
records the H-005 contradiction and verified intermediate checkpoint inventory.
