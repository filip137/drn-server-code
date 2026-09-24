# CIFAR L8 amplification and KCL audit

Date: 2026-09-23. Evidence class: implementation diagnostic, synthetic inputs.

The current CIFAR L8 correctly implements the corrected physical-KCL model
for separately voltage-clamped analog circuits. Its convolution blocks use
the same solver engine as the fully coupled MNIST networks. The complete
CIFAR network is a composition of circuits with digital boundaries, not one
electrically coupled equilibrium. No production configuration, checkpoint,
learning rate, or training implementation was changed during this audit.

## Solver path

| Component | CIFAR path | Fully coupled MNIST comparison |
|---|---|---|
| Convolution interactions | `model.resistive.interaction.ConvResistive` | `labs.custom_classes.ConvResistive`, a separate implementation checked numerically against the core class |
| Perfect-diode solver | `labs.custom_minimizer.CustomQuadraticMinimizer` | `TrackingQuadraticMinimizer`, which subclasses the same engine and delegates its updates |
| Coordinate update | `CustomQuadraticUpdater`, inheriting `QuadraticUpdater.pre_activate`: `-b/(2*a)`, then excitatory/inhibitory diode projection | Same update and diode projection |
| Sweep policy | Fixed asynchronous odd/even sweeps, T=K `[6,6,4]` | Same engine; MNIST iteration budgets and circuit geometry differ |
| Analog classifier | `DenseResistive` coefficients followed by exact `-b/(2*a)` | Same linear-coordinate solution; no iteration is necessary for a single free layer with clamped input |
| Digital boundaries | Differential decode, max-pool, affine BN, learned positive voltage gain, new voltage clamp | Absent between convolution layers of the fully coupled MNIST circuit |

The CIFAR free pass runs without a gradient graph; the tracked pass starts
from detached free states and backpropagates through the additional fixed
sweeps, digital bridges, and analog classifier. BN updates running statistics
once, using minibatch statistics in both passes. Electrical feedback does not
cross a digital voltage clamp; backpropagation does cross it.

## Independent physical check

Let A and B denote voltage and current amplification. At free layer l, the
physical net incoming resistive current is

```text
I_l = sum_in g*(alpha_previous*v_previous - v_l)
      + B*sum_out g*(v_next - A*v_l),
alpha_previous = 1 at the clamped input, A otherwise.
```

The last free layer has no outgoing term. The energy derivative must be
`dE/dv_l = -(B/A)^(l-1) * I_l`. Tests assemble these physical currents using
convolution and transposed convolution independently of the energy and its
coefficient methods. They verify both the derivative and coordinate solution
in float64 at absolute/relative tolerance `2e-12`, including diode
complementarity at clamped coordinates.

Coverage: two- and three-convolution blocks, all three production schemes
`(1,1)`, `(4,1)`, `(4,.25)`, plus `(2,3)` as a non-reciprocal control.
The corresponding six-sweep states and gradients with respect to block input
and every conductance tensor match the MNIST interaction/tracking-minimizer
path bit-for-bit in the small float32 circuits.

The classifier matches the standard quadratic updater exactly. An independent
sum of its branch currents gives normalized residual below `2e-6` in all
four settings.

## Consequences of the boundaries

Each circuit restarts naming at `Layer_0`, followed by its own free layers.
This preserves the input-clamp convention and the local amplification factors.
There is no electrical load from the next block on the preceding block's
output. The digital bridge computes the next circuit's clamped input.

The classifier is a separate `Layer_0 -> Layer_1` circuit. By the shared input
convention its only edge has `alpha=1` and `(B/A)^0=1`. Thus A and B have no
effect on this classifier at fixed input, conductances, and boundary gain.
Its learned conductances still define an analog equilibrium, rather than an
unconstrained digital linear layer:

```text
v_output[j] = sum_i g[i,j]*v_clamped[i] / sum_i g[i,j].
```

In the fully coupled MNIST model, the classifier is connected directly to a
free convolution layer. It therefore has an amplified incoming edge and
loads that layer through the current-reflection factor B. These are different
circuit topologies even though their local physical equations are shared.

For zero biases and ideal perfect diodes, set
`v_l = A^(l-1)*x_l` within a block. The corrected energy becomes a baseline
energy with effective edge weights `(A*B)^p * W_p`, where p starts at zero.
Legacy has `A*B=1`, so identical-weight baseline and legacy block outputs
differ only by a positive scale: 16 for a three-convolution block and 4 for a
two-convolution block. All three assembled CIFAR blocks reproduce this
identity exactly in the synthetic test. Setting BN epsilon to zero only in
the algebraic test makes the complete matched-weight training-mode networks
identical. Production retains BN's original epsilon and running statistics.
The identity does not claim equality of separately trained checkpoints.

This supports the earlier BN explanation for why the MNIST scheme ordering
need not transfer. It does not establish a causal explanation of the final
accuracy gap, nor a hardware/SPICE validation of the complete CIFAR circuit.

## Validation and provenance

```bash
KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/home/filip/miniconda3/envs/py312/bin/python -m pytest -q \
  labs/tests/test_cifar_l8_amplification.py \
  labs/tests/test_conv_resistive_amplification.py \
  labs/tests/test_dense_resistive_amplification.py
```

Result: **136 passed in 1.51 seconds**, local CPU, no dataset or optimizer
steps. This includes 21 new architecture checks and 115 existing interaction
checks. The new tests are in
[`test_cifar_l8_amplification.py`](../labs/tests/test_cifar_l8_amplification.py).

The nine runtime files below are byte-identical between the audited checkout
and each of the five production source snapshots: initial LR search
`source-v2`; epoch10-to30 `source` and `source-gpu-only`; proposed epoch30-to50
`source`; and baseline/legacy epoch30-to50 `source`.

```text
labs/cifar_l8_analog.py
labs/custom_minimizer.py
model/resistive/interaction.py
model/resistive/minimizer.py
model/resistive/layer.py
model/function/interaction.py
model/minimizer/minimizer.py
model/variable/layer.py
model/variable/parameter.py
```

All three existing epoch50 results also record passing final T/K audits.
Those are saved trained-model convergence/gradient comparisons, distinct
from this synthetic physical-equation audit. No new training was required.

Subsequent implementation: the [mechanism study](cifar_l8_mechanism_plan_20260923.md)
adds an opt-in output-normalization control to `labs/cifar_l8_analog.py`.
The byte-identity statement above describes the audited implementation before
that addition. Existing configurations still select the unchanged `none`
behavior; physical interactions and KCL solvers are unchanged. New tests cover
production-epsilon BN and multiple Adam steps. GPU equivalence and causal
training results are tracked separately in the mechanism study.
