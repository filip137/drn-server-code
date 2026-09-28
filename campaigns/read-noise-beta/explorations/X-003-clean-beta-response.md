# X-003 — Does legacy displacement decay faster as beta decreases?

Filip requested this comparison on September 24, 2026, explicitly selecting
clean EqProp training and a beta sweep at fixed weights. The proposed trend is
a steeper decay with decreasing beta, rather than a decrease across epochs or
network depth. He also requested clean EqProp gradient cosine against BPTT at
the same beta values and excluded any additional T/K experiments.

Plot all three convolutional states and the readout at initialization and
epoch 30. A vertical offset in a log-log displacement plot is an amplitude
difference; a larger log-log slope indicates faster decay as beta decreases.
Earlier measurements report approximately linear scaling in some ranges, so
legacy's sharper decay is a hypothesis, not a plotting assumption.

Literal RMS(v_positive-v_free) can contain continued free-state relaxation.
Show it alongside RMS((v_positive-v_negative)/2), with zero-nudge drift saved
separately. Small signals may also reach float64 resolution; undefined slopes
and cosine must not be converted into zeros. BPTT means the existing K=8
unroll from the same post-T state, not an asserted exact equilibrium gradient.

Use one complete clean training family and one replay environment. The
three A100 seed-0 p90 runs share initialization and the epoch-30 horizon but
have different selected training betas and learning rates. The replay
isolates instantaneous beta response at those weights; it does not isolate
the cause of their training outcomes. Noisy-training results remain separate.

See [H-005](../hypotheses/H-005-clean-beta-response.md) and
[exp-006](../series/003-clean-beta-response/experiments/exp-006-rms-and-cosine-beta.md).
