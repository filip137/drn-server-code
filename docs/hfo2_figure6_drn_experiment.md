# Figure-6 HfO2 DRN initialization, HWA, and recovery

This exploratory path applies the analyst-defined endpoint model from
[`hfo2_figure6_endpoint_regimes.py`](../experiments/mnist_relu_drn/hfo2_figure6_endpoint_regimes.py)
to the fixed `784-50-10` ReLU teacher and the existing four-cell, positive-
conductance DRN topology. It does not reinterpret the synthetic endpoint
states as microSiemens or raw measured device identities.

## Three distinct operations

For every physical cell, normalized progress `p` is embedded as

```text
G = full_tile_RESET + p * (full_tile_SET - full_tile_RESET).
```

1. **Direct initialization.** Each teacher layer is normalized by its absolute
   maximum. A positive signed weight uses progress `(u,0,0,u)` and a negative
   weight uses `(0,-u,-u,0)` in canonical `(++,+-,-+,--)` order. Thus inactive
   cells begin exactly at their sampled RESET endpoint and unit-magnitude
   cells begin exactly at their sampled SET endpoint. No programming error is
   simulated in this arm.
2. **Direct endpoint-noise HWA.** A clean signed logical master is optimized by
   digital Adam. Each minibatch is evaluated using a Figure-6 endpoint
   assignment, and the physical gradient is lifted through each cell's exact
   `SET-RESET` window. The pilot uses four valid marginal draws and eight joint
   RESET/SET identity rotations per draw, for 32 assignments. Rotations retain
   every endpoint value and RESET/SET pair exactly; they only change which
   logical edge receives the device. This arm injects endpoint population
   variation, not pulse-programming or inference-read noise.
3. **Persistent pulse recovery.** The selected HWA master is directly deployed
   on one held-out endpoint assignment. That exact normalized cell position is
   then the sole weight authority. Digital BPTT and Adam choose stochastic
   one-pulse requests; all weight mutations pass through the additive-noise
   soft-bounds update parameters of AIHWKit 1.1.0
   `ReRamArrayHfO2PresetDevice`. The synthetic endpoints replace only the
   bounds coordinate. Circuit inference always uses persistent full positive
   conductance, not the apparent write-noise observation.

The accurate recovery label is **hardware-in-loop pulse-mediated recovery**.
The gradients, Adam moments, and pulse selection are digital, so this is not
autonomous on-chip Adam.

## Pilot contract and result

The completed `pilot` profile used the same frozen teacher, solver, MNIST
split, 8,192-example training subset, fixed logit gain, and fixed evaluation
endpoint assignment for both pairing regimes. HWA ran for five epochs; recovery ran for
three epochs. Validation selected trained HWA and recovery epochs before their
test metrics were consumed within the completed run. The same target assignment
was visible during exploratory method development, so these are development
results rather than sealed confirmatory evidence.

| Endpoint pairing | Direct initialization | Selected HWA | Selected pulse recovery | Recovery pulses |
| --- | ---: | ---: | ---: | ---: |
| Independent | 95.85% | 91.70% (epoch 1) | 92.89% (epoch 1) | 8,067 |
| Rank matched | 94.83% | 93.18% (epoch 1) | 93.40% (epoch 1) | 8,283 |

The ideal global `G=2p` control reached `96.88%`. In this short pilot, direct
initialization was therefore better than endpoint-noise HWA. Recovery did
operate correctly from the exact deployed HWA state and recovered `+1.19`
percentage points for independent pairing and `+0.22` points for rank-matched
pairing, but it did not regain the direct-initialization accuracy. Later
recovery epochs degraded validation, consistent with the very coarse HfO2
nominal progress step (`0.2311` of the full interval per central pulse).

The no-SET-clipping assumption is consequential at network scale. The target
seed was valid immediately, but 21 candidate seeds were rejected while
constructing four HWA populations because at least one of 158,800 cells had a
non-positive or RESET-crossing Gaussian SET draw. Rejection occurred only at
the whole-population level and was shared by both pairing arms; no individual
cell was clipped, repaired, or redrawn.

The implementation is
[`hfo2_figure6_drn.py`](../experiments/mnist_relu_drn/hfo2_figure6_drn.py), and
the executable composition root is
[`hfo2_figure6_drn_experiment.py`](../experiments/mnist_relu_drn/hfo2_figure6_drn_experiment.py).
The subsequent four-arm program-and-verify comparison is specified separately
in
[`hfo2_figure6_program_verify_experiment.md`](hfo2_figure6_program_verify_experiment.md).
Run the smoke or pilot from the repository root with:

```bash
/home/filip/miniconda3/envs/py312/bin/python \
  -m experiments.mnist_relu_drn.hfo2_figure6_drn_experiment \
  --profile smoke
```

The September 3, 2026 pilot summary has SHA-256
`1e4d8b5c0dd0357ca7613ffdf6097a88d5c73559df43e17bc160489960e192c8`.
It is a synthetic/model-based control and supports no measured-hardware or
physical-power claim.
