---
id: "H-003"
title: "The selected output displacement transfers across amplification schemes"
---
# H-003 — Does the common RMS target transfer across schemes?

Prospective: exp-003 selected relative output displacement R≈0.075 for ours.
Claim: calibrating baseline and legacy to the same R yields native-BPTT cosine
>=0.95 for every Conv weight, with identical initialization and native T/K.
Test all three blocks in each scheme. Report per-batch and matched-local BPTT
agreement separately, including any exceptions to the stronger per-batch rule.

Support requires the claim in both new schemes. Failure in either contradicts
transfer of the selected target in this tested regime; it need not rule out a
different common interval. Sweep R=0.02–0.25 to identify such overlap, also
reporting >=0.99 and absolute RMS. Calibration failure or dead references leave
the affected comparison unresolved. This is a noiseless initialization test,
not a ranking of training accuracy or a convergence-qualified equilibrium claim.
