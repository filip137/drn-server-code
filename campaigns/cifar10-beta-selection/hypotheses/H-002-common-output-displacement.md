---
id: "H-002"
title: "A common relative output displacement identifies high CIFAR gradient alignment"
---
# H-002 — A common displacement range across CIFAR blocks

Prospective, motivated by exp-001 and Filip's request for a focused sweep.
Claim: despite different betas and block depths, a common relative physical-output
RMS range gives pooled native-BPTT cosine >=0.95 for every Conv in each block.
Use R=RMS((s_plus-s_minus)/2)/RMS(s_free), with the existing frozen-force convention.

Support requires overlapping observed high-cosine bands across all three blocks;
report the corresponding B values, both BPTT references and batch variation.
Also test >=0.99, and distinguish a shared sufficient range from a common
upper cutoff or identical cosine-versus-R curves. Different cutoff locations
would limit R as a universal predictor even if a useful shared band exists.
No overlap on the focused tested range contradicts the bounded claim; unresolved
boundaries remain inconclusive. This is an initialization-only, noiseless test,
not a claim about trained checkpoints or accuracy.
