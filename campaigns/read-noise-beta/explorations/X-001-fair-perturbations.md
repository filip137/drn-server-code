# X-001 — What should equal beta mean?

The numerical beta is not a scheme-independent measure of the physical perturbation.
The direct initialization replay already found that injected betas 88.7, 1.385 and
0.02173 give approximately the same unit output RMS in baseline, ours and legacy.
Source: [RMS note](../../../docs/conv3_unit_output_displacement_beta_20260922.md).

This makes output displacement a plausible comparison coordinate. It is not equivalent
to matching hidden-layer displacement, learning-signal SNR, finite-beta gradient bias,
nudging energy, or stability after conductances have changed. Moreover, pooled RMS can
hide a tail of examples with large excursions. Matching the positive-free RMS must not
be silently replaced by averaging batch RMS or by another centered statistic.

The immediate opportunity is not another unbounded beta sweep. Reuse the measured
candidate, verify the exact contract, and test both signs, layerwise behavior and
trained checkpoints. Then compare training under the candidate rule and the current
p99 policy with all non-beta factors fixed within each scheme.

A largest-stable-beta rule is an alternative, but stability depends on the horizon,
seed, checkpoint and optimizer. A search that gives one scheme more trials can create
an apparent fairness advantage. The selection budget must therefore be explicit.
A fixed initial RMS rule has a simpler interpretation, but may fail later; that failure
is useful evidence rather than a reason to adjust its target silently during training.

[H-001](../hypotheses/H-001-output-rms-match.md) captures the calibration observation;
[H-002](../hypotheses/H-002-training-transfer.md) asks whether its transfer to training
is beneficial. They are deliberately different claims.
