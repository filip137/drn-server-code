# Measured cohort-B memristor LoRA recovery

`measured_cohort_b_lora` is the physical low-rank accuracy-recovery protocol
for a one-hidden-layer MNIST DRN. It combines the
`passive_layerwise_low_rank` model adapter with backpropagation and measured
cohort-B projection.

The loaded cohort-A checkpoint supplies only the two dense base matrices and
hidden bias. On deployment, each base conductance is independently assigned a
virtual cohort-B device curve and written once to the globally nearest
measured conductance. Those base tensors then remain frozen.

Four trainable memristor arrays form two explicit rank-node branches:

```text
input  -- A1 -- rank1 -- B1 -- hidden
hidden -- A2 -- rank2 -- B2 -- output
```

Every element of `A1`, `B1`, `A2`, and `B2` starts at the final measured
RESET-sweep point (pulse index 4999 in the example).
There is no random or ideal floating-point LoRA initialization. The optimizer
keeps a digital target shadow for these four arrays only and projects every
write onto that element's assigned measured curve. Checkpoints retain the
realized frozen base, factor targets, pulse indices, assignments, and optimizer
state.

The base edges retain `voltage_amp=4` and `current_amp=0.25`. The internal
rank branches are passive unity-gain connections. Amplification depth is
derived from the logical DRN topology, so constructing an LR-selection model
before the production model cannot alter the circuit equations.

Run the nested example with an explicit cohort-A base checkpoint and device
file:

```bash
python -m ebl train \
  --config examples/small_drn/measured_cohort_b_lora_mnist.json \
  --base-weights /path/to/cohort-a/checkpoints/weights.pt \
  --device-data data/march_slope_x3_5k.hdf5 \
  --output-dir results/measured-cohort-b-lora
```

The initialization artifact separately records the frozen-base deployment and
proves a last-pulse fraction of one for every LoRA factor. The example uses
per-factor learning rates because the measured reset-state gradient RMS differs
substantially between the four physical arrays.

## Exploratory result

The corrected rank-4 MNIST run on 2026-08-15 did **not** produce meaningful
accuracy recovery:

| checkpoint | validation accuracy | held-out test accuracy |
| --- | ---: | ---: |
| deployed cohort-B base | 93.64% | not measured at initialization |
| fully reset LoRA initialization | 93.70% | not measured at initialization |
| epoch 1 (best validation accuracy) | 93.76% | 94.42% |
| epoch 20 (best validation cost) | 93.68% | 94.38% |

The corresponding selected full-network cohort-B fine-tuning reference reached
94.56% on the held-out test set. By epoch 20, 43.75--63.36% of the four LoRA
shadow arrays were at a conductance bound, and just 2.48--5.00% of their pulse
indices changed during the final epoch. This is consistent with a structural
limitation rather than an insufficient run length: two finite, nonnegative
conductance factors initialized at RESET do not form a zero-centered additive
correction, and the physical factors saturate while validation accuracy stays
flat. A signed or differential factor encoding is the appropriate next
experiment.

The run artifacts are under
`results/measured-cohort-b-lora-rank4-fully-reset-20260815-v1/`; the one-epoch
peak reproduction and both held-out evaluations use sibling result roots with
`peak` and `test` in their names.
