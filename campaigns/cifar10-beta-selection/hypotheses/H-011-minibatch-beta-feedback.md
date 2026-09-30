---
id: "H-011"
title: "Following the training-minibatch response can improve fresh EqProp learning"
---
# Can beta follow the response within the first epoch?

Retrospective motivation: exp-012's eight fixed coefficients all miss the
matched-BPTT gate. Larger coefficients improve final gradient alignment but can
misalign initialization gradients; observed displacement changes greatly within
one epoch. The earlier epoch-wise controller cannot track that change.

Compare fixed beta with a lagged controller using the actual training minibatch's
centered displacement to set the next minibatch's beta. Both begin with exp-012's
best fixed coefficient. The hypothesis is improved learning, not merely maintained
RMS. [Exp-014](../series/001-beta-selection/experiments/exp-014-minibatch-beta-feedback.md)
fixes the update, targets, matched control and decision rule before execution.
