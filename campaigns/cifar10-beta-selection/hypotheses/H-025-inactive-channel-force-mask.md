---
id: "H-025"
title: "Masking force on inactive output channels restores block3's float32 window"
---
# Remove the force that only inactive units receive

Prospective, following [X-002](../explorations/X-002-inactive-channel-force.md)
and [exp-029](../series/001-beta-selection/results/exp-029-block3-diode-switching.md).
Intervention: before nudging, zero the block-output force on logical channels
whose excitatory and inhibitory states are exactly zero for the whole minibatch
(`force_mask: "inactive_output_channels"`). Population: the 18 exp-025
checkpoints, all schemes, same cohort and T/K, float32 throughout, against the
unmasked block-local BPTT reference.

Proposed thresholds, frozen before measurement:

1. **Exactness:** for every block, layer and minibatch, the local reference
   computed with the masked force has cosine ≥1−1e-9 and norm ratio within
   1±1e-9 of the unmasked reference.
2. **Block3 recovery:** in every checkpoint whose block3 has inactive
   channels, the largest sampled target in the contiguous passing run above
   the float32 floor (every layer's pooled local cosine ≥0.99) is ≥1e-3 with
   the mask, whereas the matched unmasked float32 control stays below it.
3. **Common float32 target:** within each training group (EqProp0/1/5 and
   BPTT0/10/30/50, all schemes), at least one sampled block3 target passes every
   layer's pooled cosine ≥0.99, every minibatch ≥0.95 and pooled norm ratio
   0.9–1.1 at every checkpoint.

Support requires 1–3. Contradiction: criterion 2 fails in most applicable
checkpoints, so the mask does not move the limit. Mixed outcomes are partial and
remain checkpoint-specific. Blocks without inactive channels are reported but
cannot test the claim. Gradient agreement does not establish learning; see
[H-026](H-026-float32-masked-training.md).

[exp-030](../series/001-beta-selection/experiments/exp-030-float32-force-mask-replay.md)
owns execution and review.
