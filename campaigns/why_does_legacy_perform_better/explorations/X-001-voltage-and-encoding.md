# X-001 — Voltage magnitude and target encoding

## Observation and interpretation

At clean Conv2 target 1/16, best validation accuracy is 98.42% ours versus
98.38% legacy; at target 1/4 it is 98.36% ours versus 98.40% legacy. Baseline
changes from 97.98% at 1/4 to 98.28% at 1/16. The intervention shows sensitivity
to encoding and motivates Filip's interpretation that encoding/voltage scale
largely explains the clean advantage. The ours/legacy changes are small and
single-seed, so the strength of that explanation remains to be measured.

With fixed targets, changing output scale changes the squared-error objective.
For a scalar illustration, L(s f, t) / s² = (f - t/s)² / 2. The corresponding
target in the unscaled coordinates changes; the scalar loss factor can also
affect optimization. This illustration is not a measured network identity.

The trained clean EqProp profiles show large changes in voltage over depth.
Baseline has the largest hidden RMS, whereas legacy has the largest readout
RMS among the three tested schemes. Independent training need not preserve
the baseline/legacy scaling relation seen at identical initial weights.

## Scope and confounds

The encoding intervention is clean BPTT; the voltage replay is clean centered
EqProp. Both use finite iteration budgets and ordinary-MNIST validation.
Neither uses official test accuracy. Separate scheme-specific learning rates
remain fixed, and the encoding test uses native MSE without compensating loss
normalization. Bounds [0,100] are held fixed. Thus there is no direct estimate
of the upper/lower-bound effect or of a finite positive w_max/w_min ratio.

## Discriminating controls

1. Check baseline/legacy output-score and conductance-gradient agreement at
   identical weights after the predicted output rescaling inside the clean BPTT
   loss. Begin with a paired saved-checkpoint computation.
2. Compare baseline at its own rates, baseline at legacy's rates, output-scaled
   baseline at legacy's rates, and legacy at its own rates, with identical
   initialization and minibatch order. Distinguish rate and output-scale effects.
3. Replicate target 1/16 and 1/4 across independent seeds; record best and final
   accuracy with a frozen decision rule.
4. Only then vary conductance bounds under an explicit matched encoding and
   optimizer contract to test the finite-bound part of the research question.

These are proposed controls, not authorized new runs. Original pilots and result
bundles stay at their historical paths, listed in the imported result notes.
