---
experiment: "exp-017"
evidence: "validated-local"
summary: "Raising the beta ceiling removes all RMS misses but worsens ours epoch1 accuracy32.64→28.04% and CE1.788320→1.995131; no continuation."
verdicts: {"H-013": "contradicts"}
---
# Removing clipping did not improve learning

The exact retry completed one full epoch:1,407 updates,45,000 training examples
and5,000 validation examples. Canonical collection validated selected-beta
history, controller state, inputs and coverage. The pretraining CUDA failure
remains archived and charged; it produced no training evidence. Successful
training/evaluation took3,038.4seconds within the remaining5,396seconds.

| Matched arm | Accuracy | CE |
| --- | ---: | ---: |
| Native BPTT |37.06%|1.726060|
| Current-batch beta, ceiling10,000 |32.64%|1.788320|
| Current-batch beta, ceiling30,000 |28.04%|1.995131|

The higher ceiling removed all target-tolerance misses, versus77 block2 misses
in exp-015. Median RMS/target became[.9980,.9893,.9864]; maximum selected
block2 beta was18,082, below the new bound. Average centered-pair counts were
[1.380,1.151,1.138]. The first56 training losses exactly reproduced exp-015,
before its first ceiling miss, supporting the intended single-setting change.

Despite improved RMS tracking, accuracy fell4.60pp and CE rose11.56% relative
to exp-015. Both matched-BPTT continuation limits fail. This contradicts the
proposed learning benefit of removing this ceiling in the tested seed and
regime; it does not establish a universally preferable smaller ceiling. No
long extension. Accurate RMS tracking alone is insufficient to select a useful
training beta.

Exp-018's larger block3 RMS test remains active. If it also fails, inspect
actual selected training gradients during the early updates before choosing
another target. Fixed-cohort warm-start diagnostics do not answer that question.
Evidence: `results/cifar-beta-current-batch-high-cap-20260930-v1/`, with retry
provenance under `results/cifar-beta-current-batch-high-cap-retry-20260930-v1/`.
