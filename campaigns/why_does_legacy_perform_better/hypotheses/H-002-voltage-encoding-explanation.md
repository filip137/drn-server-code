---
id: "H-002"
title: "Voltage magnitude and target encoding explain much of legacy's clean advantage"
---
# H-002 — Voltage and encoding explanation

Filip's interpretation adopted September 29: much of legacy's advantage in
clean finite-bound experiments comes from the operating voltage scale relative
to target encoding, rather than an advantage independent of those choices.

Existing evidence combines a clean BPTT target-amplitude intervention with a
separate clean EqProp voltage replay. It motivates this explanation but does not
quantify a causal fraction or establish equivalence between those algorithms.
No claim that legacy always has the largest hidden voltage is made.

Support for the stronger explanation requires matched interventions that remove
most of a reproducible gap while controlling optimizer history, learning rates,
loss normalization, initialization and conductance bounds. Persistence of the
gap after those controls would weigh against it. Freeze a quantitative meaning
of "most" and the comparison contract before collecting new causal evidence.
The present evidence is inconclusive for this broader claim.
