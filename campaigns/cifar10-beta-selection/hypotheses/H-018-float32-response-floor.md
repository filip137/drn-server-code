---
id: "H-018"
title: "Float32 state precision limits small-beta CIFAR gradient agreement"
---
# Float32 state precision and small nudges

Prospective follow-up to exp-022 and Filip's September30 request. Earlier
MNIST Conv1/2/3 EqProp diagnostics used float64, while these CIFAR replays
use float32. At identical checkpoints, cohort, beta and finite T/K, promote
the CIFAR computation to float64 and test whether low-beta agreement recovers.

Support requires recovery of cosine/norm agreement at beta values where the
matched float32 computation fails, with comparable BPTT reference directions.
Float64 arithmetic on unchanged float32 endpoints separates the gradient
calculation from precision in state solving. Recovery only with full float64
supports a limitation upstream of gradient arithmetic; unchanged low-beta
failure weakens this explanation within the fixed finite-iteration contract.
Full promotion also affects free states, BN and forcing, so report their
changes and do not attribute every difference solely to endpoint storage.

Use pooled native cosine>=0.95 and norm ratio0.9–1.1 as the existing descriptive
criterion, retain minibatch spread, and report unbracketed lower boundaries.
High-beta errors can persist. No claim of improved training accuracy follows.
