# Active comparison: zero-order feedback, VF EqProp and Jacobian homeostasis

14 September 2026. The user's focus is mismatch between independently trained
forward and backward connections, as in vector-field EqProp and Jacobian
homeostasis. This is the active scope for subsequent comparisons.

Use the existing untied-weight model in `labs/directed_eqprop.py`. Each pair of
forward and backward matrices is independently trainable throughout learning.
The initial backward matrix mixes a forward transpose with an independent
Gaussian matrix; mixing angles 0/45/90 degrees span initially reciprocal,
partially mismatched and independent backward weights. Even at zero mixing,
the weights remain untied during training.

The earlier model with a fixed antisymmetric matrix and a few controller gains
is excluded. Mathematical symmetry diagnostics on the current directed matrices
do not mean that the current model has that old parameterization.

## Three primary methods

All three methods share the same initial network at each seed and mixing angle,
the same data, cost, trainable parameters, optimizer and activity-coordinate
convention. For the task-gradient terms, omitting the shared direct readout
gradient:

| Method | Update supplied to the parameter optimizer |
|---|---|
| VF EqProp | `g = -F_theta^T q`, with `q` measured from paired error nudges |
| VF EqProp + Jacobian homeostasis | The same `g`, plus the AD gradient of the homeostatic penalty |
| Zero-order feedback estimation | `g = -F_theta^T lambda_hat`, using four fresh random state-nudge pairs and the learned feedback baseline |

The VF control uses the activity-response convention shared with the
[homeostasis implementation](https://github.com/Laborieux-Axel/generalized-holo-ep/blob/30592f576bd4d4a20d3c13632f0792b0fa452781/models/dyn.py).
The general [VF learning rule](https://arxiv.org/html/1808.04873) applies to the
equivalent activity equilibrium force. The original membrane-state control is
retained in the full study as an additional convention check. Holding the
activity convention fixed isolates the effect of adding homeostasis.

Homeostasis penalizes functional asymmetry and changes the network parameters.
The [paper's estimator](https://arxiv.org/html/2309.02214v2) is implemented with
five Gaussian vectors per example, two forward JVPs per vector, and parameter
AD. Our MLP follows its public code's activation-bypassed Jacobian. Details and
deviations are in [directed_eqprop_mnist.md](directed_eqprop_mnist.md).

Zero order here means estimating the required **feedback/adjoint** from state
perturbations and scalar response measurements. It does not mean reconstructing
a full Jacobian or estimating every parameter derivative with SPSA. It retains
the known cost gradient and local parameter-force derivatives needed by the VF
synapse update. Its learned predictor is fitted from past physical measurements;
the current gradient receives a correction from four fresh probe pairs. No
homeostasis is included in this arm.

## Focused results from the completed MNIST study

The new focused report reuses 18 independently verified trajectories from the
completed 48-case study. No new training was needed. A further check confirmed
identical initial network parameters across the three arms and that both
directions changed during training. Selection is by requested method, not by
performance; both seeds and all three mixing angles are included.

Five-epoch test accuracy (%), mean ± population SD across two seeds:

| Method | Initial mixing 0° | Initial mixing 45° | Initial mixing 90° |
|---|---:|---:|---:|
| VF EqProp | 49.45 ± 16.09 | 36.11 ± 0.78 | 34.28 ± 0.25 |
| VF EqProp + homeostasis | 71.01 ± 3.58 | 36.25 ± 0.63 | 35.86 ± 0.04 |
| Zero-order feedback, learned baseline and four probes | 74.40 ± 5.84 | 74.59 ± 5.40 | 75.18 ± 4.78 |

Zero order uses nine equilibrations per example; the other two use three.
Consequently, epoch-matched comparisons give zero order three times the
equilibrium budget. At **exactly 495,000 training equilibrations**, stored
validation measurements allow the following comparison without interpolation:

| Method | Epoch | Initial mixing 0° | Initial mixing 45° | Initial mixing 90° |
|---|---:|---:|---:|---:|
| VF EqProp | 3 | 44.99 ± 11.95 | 30.08 ± 3.36 | 31.31 ± 1.39 |
| VF EqProp + homeostasis | 3 | 63.02 ± 0.86 | 34.36 ± 0.24 | 32.69 ± 1.53 |
| Zero-order feedback | 1 | 54.26 ± 8.82 | 54.41 ± 8.57 | 54.84 ± 8.58 |

Homeostasis additionally uses 1,650,000 digital JVPs and their parameter-gradient
work at that endpoint. Equal equilibrium counts do not mean equal wall time or
energy. The matched endpoint was selected by available measurement counts,
without looking for an accuracy maximum or evaluating intermediate test data.

Under independent backward initialization, zero-order feedback has higher
accuracy in both comparisons. Homeostasis gives much better single-example
feedback alignment: the final first-hidden-layer cosine is approximately
0.998, compared with 0.712 for VF and 0.208 for zero order. The latter remains
a noisy per-example estimate; training averages 128 examples and uses Adam.
These results do not identify the cause of its optimization advantage.

With initially reciprocal weights, homeostasis leads at the common equilibrium
budget. The conclusion depends on initial mismatch and the resource being held
fixed. The plain four-probe correction with a measured VF baseline, without a
learned predictor, remains an ablation in the full report (`ep_probe4`); at
90-degree mixing it achieved 69.58 ± 10.40% test accuracy with eleven
equilibrations per example.

This is a preliminary comparison: five epochs, two seeds, untuned shared Adam,
zero training read noise and a norm cap active on 96–99% of updates. The low
baseline accuracy and restrictive cap prevent a general superiority claim.

The report and matplotlib figures are under
`simulation_results/directed_eqprop_20260914/zero_order_vs_vf_homeostasis/`.
Generate them with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m labs.tools.compare_untied_feedback_methods \
  simulation_results/directed_eqprop_20260914/main \
  --output simulation_results/directed_eqprop_20260914/zero_order_vs_vf_homeostasis
```

For a fresh run restricted to these methods, use the existing runner with
`--methods ep_activity ep_homeo learned_probe4` and a fresh output directory.
The original full-study protocol and all previous results remain preserved.
