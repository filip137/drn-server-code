# X-002 — Separate beta, learned state, settling and noisy credit assignment

The [beta synthesis](../../../docs/beta_study.md) reports a strong T-dependent readout
cosine improvement at older p90-trained ours weights, but almost no T/K dependence at
initialization or current p99-trained checkpoints. The older weights remain sensitive
even at tiny clean replay beta, whereas the newer weights remain insensitive when
replay beta is raised. A universal cutoff in instantaneous beta is therefore a poor
explanation of the existing observations; the trajectory and resulting state matter.

This also resolves a misleading comparison: the higher-T p99 training repeat does not
share the large gradient improvement measured at the older p90 checkpoint. The matched
p99 replay sees no meaningful gain, consistent with nearly unchanged accuracy. Neither
study proves that a genuinely better gradient could never improve training.

At current p99 ours epoch 30, noisy layer medians are approximately -0.005645, 0.008845,
0.631672 and 0.999948 for Conv1, Conv2, Conv3 and readout. A nearly perfect readout cosine
can coexist with poor hidden-layer measurements. Cosine alone also suppresses magnitude
information; record norms, relative error and actual optimizer movement.

In this runner T advances the free phase and K controls both the nudged phase and the
BPTT unroll used for comparison. A K sweep that changes the reference cannot isolate
improvement in EqProp. New replay comparisons should include a fixed, declared BPTT
reference and its own adequacy checks, without calling a finite-unroll gradient exact.

Other confounds remain: trained weights differ between historical policies, read-noise
seeds can be coupled differently, and the historical replay environments are not all
the same. Frozen-weight replay distinguishes instantaneous response from training
history, but cannot prove the causal origin of a final accuracy gap. A beta/LR factorial
or adaptive schedule is a later study, not an implicit change to the first comparison.

See [H-003](../hypotheses/H-003-state-dependent-settling.md) and
[H-004](../hypotheses/H-004-layer-signal-disparity.md).
