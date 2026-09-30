---
id: "H-014"
title: "The larger trained-checkpoint block3 RMS candidate may improve fresh learning"
---
# Test the cross-checkpoint block3 target in training

Exp-008 recommended ours RMS[.02,.035,.005] across trained checkpoints;
the current training tests instead use epoch10's sampled peak[.02,.035,.001].
The larger block3 target retained strong diagnostic cosine across epochs10/30/50
and was bracketed by strong initialization measurements. It may improve learning
by avoiding an unnecessarily small displacement, but neither gradient alignment
nor this interpretation establishes a learning benefit.

[Exp-018](../series/001-beta-selection/experiments/exp-018-block3-rms-training.md)
tests only block3 target.001→.005 against exp-017, with the same current-batch
search and30,000 ceiling. This is prospective; exp-017's final validation is
still pending. Judge the full epoch against the unchanged matched-BPTT gate.
