---
id: "H-013"
title: "An early beta ceiling may limit current-batch RMS selection"
---
# Does the observed ceiling explain part of the learning deficit?

Exp-015 reached its block2 beta ceiling on75 early updates and undershot the
requested RMS, while later updates usually reached it. Its full-epoch accuracy
32.64% missed the matched-BPTT gate despite improved cross-entropy. This motivates
one prospective follow-up: increase only the current-batch search ceiling from
10,000 to30,000. The new ceiling extrapolates from the worst measured response
deficit; it is not a guaranteed solution. Test both reduced clipping and full
validation performance in [exp-017](../series/001-beta-selection/experiments/exp-017-current-beta-ceiling.md).
Better target tracking alone cannot support a learning-benefit claim.
