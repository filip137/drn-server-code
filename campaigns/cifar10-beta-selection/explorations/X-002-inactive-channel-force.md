# X-002 — Why block3 tolerates so much less output displacement

Filip asked why the usable block2/3 output RMS is so much smaller than block1's.
In the full64 local comparison ([exp-025](../series/001-beta-selection/results/exp-025-all-scheme-layer-rms.md))
blocks1/2 keep upper limits near 1e-2 at every checkpoint, whereas block3 falls
from about 0.12 at initialization to 3e-5 (EqProp epoch1) and 1e-4–1e-3
(epoch5). [exp-029](../series/001-beta-selection/results/exp-029-block3-diode-switching.md)
showed that freezing diode decisions restores block3 at ours epoch1, without
identifying which switches matter. Block2 is not anomalous in the same way:
its limits overlap block1's, and common-target minima are set by single cases.

## Proposed mechanism

After training, some block3 output channels are **inactive**: both diode halves
are off for every image and position in the minibatch. The bridge BatchNorm
then has zero batch variance and scales the backpropagated force by
γ/√ε = 316γ, versus γ/σ ≈ 0.3–2.5 on live channels. Max-pool ties send that
force to one unit per window. These units have no linear response and no BPTT
gradient (their clamp blocks it), so they are invisible to the output RMS,
which measures the live units. They are nevertheless pushed toward threshold
roughly a thousand times harder. A unit switches once β|f|/(2a) exceeds its
unclamped margin; switched units then dominate both displacement and gradient
error. The allowed RMS is therefore approximately
(dead-unit margin / free RMS) × (live force / dead force).

## Interactive evidence

Root: [results/cifar-inactive-channel-force-20261001-v1/](../../../results/cifar-inactive-channel-force-20261001-v1/README.md),
local RTX3090, PyTorch2.5.1 without cuDNN; unmasked rows reproduce stored
exp-022/024 values. Ours fixed-0.3× trajectory unless stated.

- Forward census of all 18 exp-025 checkpoints: zero-variance channels never
  occur in blocks1/2 or at initialization; block3 has 1–87 per batch in every
  trained EqProp checkpoint and 1–37 in BPTT epochs10–50. Block3's sampled
  full64 upper target times its BN gain spread (max/median γ/√(var+ε)) stays
  within 0.01–0.1 across all 15 trained checkpoints, over three decades.
- Epoch1, batch0: 4/512 dead channels carry 99.98% of Σf² (force RMS 2.01
  versus 0.002–0.003 on live units). Margins 0.14–2.4 V predict first switches
  at beta multipliers 8–28; agreement collapses between 10 and 30 while dead
  units' share of the centered output displacement rises from 27% to 98%.
- Zeroing the force on these channels leaves the local BPTT reference exactly
  unchanged and restores agreement. Pooled float32 over 256 images, worst
  layer: epoch1 ≥0.9995 pooled and ≥0.9989 every batch at output RMS
  6.4e-4–2.1e-3 (unmasked best: 0.988/0.921 at 2.2e-5); epoch5 0.999/0.997 at
  7.5e-4. Float64 batch0 at epoch1 moves the limit from ~2e-5 to ~1e-2.

## Alternatives and limits

Live units near threshold also exist (exp-029 found tens of entries within
1e-6), but masking only inactive channels restores agreement up to ~1e-2, so
they matter only at much larger displacements. Settling drift is present at
passing betas as well. The mask does not change the float32 small-response
floor; it moves the upper limit away from it. Evidence covers one trajectory
interactively plus a forward census; other schemes have not been replayed with
the mask, and no training has used it. Why channels die, whether near-dead
low-variance channels set the remaining epoch5 limit, and read-noise effects
remain open.

The force is computed digitally through BN and the readout, so the mask is a
digital operation on the injected current. Codified as
[H-025](../hypotheses/H-025-inactive-channel-force-mask.md) and
[H-026](../hypotheses/H-026-float32-masked-training.md).
