---
id: "H-004"
title: "Block2's gradient-alignment deficit persists after BPTT training"
---
# Trained block2 alignment

Initialization sweeps find lower maximum EqProp/native-BPTT cosine in block2,
especially ours/legacy, despite block1 having the same depth. A working explanation
is block-specific response conditioning under finiteT/K: insufficient useful response
at low displacement and nonlinear distortion at higher displacement leave a narrower
high-alignment range. Its cause has not been isolated.

Prediction: at epoch50, ours/legacy block2 still has a lower best worst-layer cosine
than both other blocks, with block2 below0.99 while blocks1/3 can exceed0.99.
Mixed scheme outcomes qualify this claim; disappearance of the gap argues against
a persistent deficit. Also test whether the shared R≈0.075 remains above0.95.
Use each original trained checkpoint with the same initial replay cohort/settings.
Local/native BPTT agreement helps distinguish reference mismatch from EqProp error;
this replay does not identify a unique mechanism or establish EqProp training quality.

Test: [exp-006](../series/001-beta-selection/experiments/exp-006-trained-e50-rms.md).
