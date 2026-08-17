# Digital low-rank recovery of a programmed DRN

The `digital_low_rank` adapter is a hybrid analog-digital recovery experiment.
The DRN contains one frozen non-negative dense conductance matrix. AIHWKit's
Wan-2022 ReRAM model produces one persistent device realization of that
matrix. Two ideal signed FP32 matrices then add a residual only to the settled
class scores:

```text
input x ---> frozen programmed DRN ---> differential scores ---+
                                                                +--> logits
input x ---> ideal FP32 A ---> ideal FP32 B --------------------+
```

For rank `r`, the evaluated scores are

```text
scores = paired_scores(DRN_W_device(x)) + (alpha / r) * x @ A @ B
```

`A` has shape `logical_input_width x r`; `B` has shape
`r x num_classes`. `A` is initialized symmetrically and `B` is initialized to
zero, so enabling the adapter initially leaves every prediction unchanged.
Only `A` and `B` are optimized. The programmed base matrix is frozen.

## Physical conductance calibration

AIHWKit's Wan-2022 coefficients are expressed in micro-siemens. The adapter
configuration supplies both sides of a proportional calibration:

```text
G_target_us = W_drn / drn_conductance_at_g_max * g_max_us
```

The source checkpoint must not contain a base conductance larger than
`drn_conductance_at_g_max`. The inverse proportional map converts the sampled
device conductances back into the DRN's numerical units. No offset or
background-conductance subtraction is assumed.

The supported `t_inference_seconds` values are the measurement ages exposed by
the AIHWKit model: `1`, `86400`, and `172800` seconds. The realization is
sampled once using `programming_seed`, written into the model, and subsequently
preserved by named-weight and resume checkpoints. It is never resampled per
batch or epoch.

## Running

AIHWKit is optional and must be importable in the Python environment used for
the initial `--base-weights` run:

```bash
python -m ebl train \
  --config examples/small_drn/digital_lora_reram_wan2022_digits.json \
  --base-weights /path/to/clean-base-weights.pt \
  --output-dir /path/to/new-output-root
```

`--base-weights` means that the supplied base matrix is clean and must be
programmed once. `--weights` loads an already programmed full
base-plus-adapter checkpoint and does not program it again. `--resume`
restores the exact saved realization, adapter, optimizer, RNG, and carried
layer state.

The initialization metric records the AIHWKit version, device model, seed,
physical calibration, device age, conductance-error statistics, and held-out
accuracy before the first digital update.

## Validated combination

The registered `small_drn.v1` combination is:

| Model adapter | Modifier | Algorithm | Update backend |
| --- | --- | --- | --- |
| `digital_low_rank` | `none` | `digital` | `direct` |

The initial implementation requires a single dense DRN matrix and differential
outputs. Native AIHWKit pulsed updates are deliberately not used for the
digital factors.

For a DRN with one hidden layer and physical factor conductances, use
`passive_layerwise_low_rank`. It adds two explicit rank-node branches trained
by EP; see
[Passive layerwise low-rank recovery](passive_layerwise_low_rank_recovery.md).
