# Finite conductance floors in DRNs

Last updated: 2026-07-30

## Main finding

The first CMO result mixed together two different effects:

1. every physical edge retained the measured `9 µS` minimum conductance;
2. the literal mapping collapsed every learned conductance below
   `9/88.199997 = 0.1020408` DRN units to the same value.

The second effect caused most of the observed accuracy loss. On the shared
MNIST HWA checkpoint, a noisy **affine** mapping retained the same physical
`9 µS` floor without deterministically merging the small physical targets and
obtained `96.12%` accuracy on the first write (`96.11%` in the stored second
realization), compared with `96.37%` before deployment. The literal hard-floor
mapping obtained `80.61%` on its first write and `80.34%` in its stored
realization.

This does not make the floor electrically harmless. Under the affine mapping,
hidden/output voltage RMS fell from `0.8761/0.2188` to
`0.1922/0.02469`, approximately `4.6x/8.9x` attenuation. Accuracy remained
high in this idealized perfect-diode model, but analog noise margin, diode
operating point, readout resolution, power, and settling still require
separate study.

## Where the floor enters the coordinate update

For the unit-amplification, one-hidden-layer model used here, the implemented
asynchronous coordinate update is

```text
              x^T W1[:,j] + y^T W2[j,:] + b[j]
h[j]  =  ------------------------------------------------
          sum_i W1[i,j] + sum_k W2[j,k]

                  h^T W2[:,k]
y[k]  =  -----------------------------
              sum_j W2[j,k]
```

The perfect-diode constraint is applied after the hidden pre-activation.
These expressions follow the repository's generic `-b/(2a)` minimizer and
the numerator and conductance-sum coefficients in
[`model/resistive/interaction.py`](../model/resistive/interaction.py).
They are also the nodewise KCL minimizers of the passive quadratic edge
energy used in the DRN formulation of
[Scellier et al.](https://arxiv.org/html/2402.11674).

Let an affine physical mapping produce

```text
W1' = f + s W1
W2' = f + s W2
f = G_min / G_max
s = 1 - f
```

for the current unit DRN reference. The hidden update becomes

```text
       s(x^T W1[:,j] + y^T W2[j,:])
       + f(sum_i x[i] + sum_k y[k]) + b[j]
h' = ----------------------------------------------------
       s D[j] + f(n_input + n_output)
```

where `D[j]` is the clean hidden-node conductance sum. The finite floor is
therefore a complete-bipartite graph Laplacian, not merely an additive output
bias. It changes:

- numerator current through the sum of adjacent voltages; and
- the conductance-sum denominator that normalizes every voltage update.

The MNIST input is encoded as `gain * [x, -x]`, so its sum is exactly zero.
That cancels the input-array floor in the hidden numerator, but it does not
cancel the `1568 f` contribution to the hidden denominator. Common floor
terms also partly cancel in the paired output score, but the denominator
loading remains.

This is why ordinary crossbar reference-current subtraction is incomplete for
a DRN. It can remove a common term from a one-way MVM numerator, as in the
multi-device signed-weight unit cells of
[IBM HERMES](https://arxiv.org/abs/2212.02872), but the same physical cells
still load the reciprocal DRN nodes. Exact cancellation must correct both the
numerator and the row/column conductance sums. Simply placing two positive
devices in parallel doubles the loading.

## Three mappings that must remain separate

For `drn_conductance_at_g_max = 1`, define `f = G_min/G_max`:

| Mapping | Physical target | Effective passive DRN value |
| --- | --- | --- |
| `literal_conductance` | `clip(G_max w, G_min, G_max)` | nominally `max(w, f)` |
| `affine_floor` | `G_min + (G_max-G_min)w` | nominally `f + (1-f)w` |
| `normalized_offset` | same target as `affine_floor` | `w`, after subtracting and rescaling the physical floor |

Both `literal_conductance` and `affine_floor` retain a finite physical
connection. Only the literal mapping merges all sub-floor learned targets.
The affine mapping preserves their ordering and uniformly scales pairwise
differences; it does not preserve ratios. `normalized_offset` is a
cancellation abstraction, not the passive conductance seen by the DRN.

## Controlled MNIST results

All rows use the same `[1568, 100, 20]` perfect-diode HWA checkpoint and the
canonical 10,000-example MNIST test split. The noisy CMO rows use device seed
`17`, the `9–88.199997 µS` window, the `0.2%` program-and-verify fit, and
one-second relaxation/read noise.

| Scenario | Accuracy | Hidden RMS | Output RMS |
| --- | ---: | ---: | ---: |
| Clean HWA | 96.37% | 0.8761 | 0.2188 |
| Deterministic literal hard floor | 81.83% | 0.1460 | 0.01644 |
| Noisy literal floor, stored realization | 80.34% | 0.1441 | 0.01604 |
| Deterministic affine retained floor | 95.94% | 0.1892 | 0.02461 |
| Noisy affine retained floor, stored realization | 96.11% | 0.1922 | 0.02469 |
| Exploratory selector mask, top `384/60` inputs per column | 94.93% | 1.7993 | 0.3537 |

In the literal case, `98.13%` of W1 and `88.10%` of W2 clean targets are
below the floor. The nominal floor supplies a median `97.82%` of the realized
hidden degree and `85.18%` of the realized output degree. In the affine case,
the floor still supplies a median `86.79%` and `68.22%` of those respective
degrees, yet the distinct learned values remain available to the computation.
The affine run's total `realized-clean` RMSE is about `0.101`, but that number
is dominated by the intentional affine offset. Relative to the ideal mapped
target, stochastic programming/relaxation/read RMSE is `0.00428` for W1 and
`0.00452` for W2. Endpoint noise followed by physical clamping still places
`18.32%` of stored W1 and `11.80%` of stored W2 values exactly at `G_min`;
affine calibration avoids deterministic target clipping, not stochastic
floor hits.

The deterministic on/off-ratio sweep makes the distinction explicit:

| `G_min/G_max` | Literal hard floor | Affine retained floor |
| ---: | ---: | ---: |
| 1% | 96.51% | 96.30% |
| 2% | 96.10% | 96.20% |
| 3% | 95.27% | 96.16% |
| 5% | 93.14% | 96.08% |
| 7.5% | 89.31% | 95.97% |
| 10.204% (CMO) | 81.83% | 95.94% |
| 15% | 60.84% | 95.87% |
| 20% | 49.39% | 95.83% |

For this checkpoint only, reducing the literal floor ratio to `2%` keeps the
deterministic loss within `0.27` percentage points. The affine mapping is far
less accuracy-sensitive over the tested range, while its voltage amplitude
continues to shrink as the floor grows.

## Literal-versus-affine KL divergence

There is no unique KL divergence for these mappings until the compared
distribution is specified. Two complementary measurements were therefore
made on the matched stored CMO realizations:

1. a per-example 10-class distribution obtained by applying softmax to the
   paired DRN output differences;
2. a smoothed histogram distribution over the realized conductances.

The paired voltage differences are not calibrated logits. Raw-softmax KL is
strongly affected by the mapping-dependent voltage scale, so a second
predictive measurement centers each example and divides by its class-score
RMS before softmax. This normalization preserves top-1 decisions but imposes
an explicit deterministic temperature.

| Predictive comparison, literal vs affine | Literal → affine KL | Affine → literal KL | Symmetric KL | JS divergence | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw-score softmax | `0.0000878` | `0.0000880` | `0.0000879` | `0.0000220` | 81.23% |
| Per-example RMS-normalized | `0.3311` | `0.2631` | `0.2971` | `0.06449` | 81.23% |

All divergences are means over 10,000 examples and use natural-log nats. The
near-zero raw value is misleading: both floor-heavy networks have small
output amplitudes and therefore produce softmax distributions close to
uniform. Scale-normalized KL reveals their different class-score shapes.
For comparison, clean versus affine has only `0.00651` mean symmetric
scale-normalized KL and `98.95%` top-1 agreement, whereas clean versus literal
has `0.31285` nats and `81.18%` agreement.

For the stored conductance tensors, a 128-bin Jeffreys-smoothed histogram over
`[0,1]` gives:

| Conductance distribution | Literal → affine KL | Affine → literal KL | Symmetric KL | JS divergence |
| --- | ---: | ---: | ---: | ---: |
| W1 | `0.5371` | `1.1006` | `0.8188` | `0.1560` |
| W2 | `0.6976` | `0.9716` | `0.8346` | `0.18275` |
| W1 and W2 combined | `0.5383` | `1.0932` | `0.8157` | `0.15614` |

Histogram KL depends on binning. Across 64, 128, and 256 bins, combined
symmetric KL ranges from `0.7219` to `0.8670` nats, while JS ranges from
`0.1468` to `0.1662`, so the qualitative separation is stable.

## Why the affine floor causes little decision damage

The finite floor is not harmless. It strongly loads the network and attenuates
its voltages. Its unexpectedly small effect on MNIST accuracy follows from a
specific combination of an information-preserving map, common-mode rejection,
positive homogeneity, and an ideal readout.

### The affine map preserves the trained parameters

For `f = 9/88.199997 = 0.1020408` and `s = 1-f`, each crossbar is

```text
W' = f J + s W,
```

where `J` is the all-ones matrix. For every `f < 1`, this map is one-to-one:

```text
W = (W' - f J) / s.
```

It reduces every pairwise conductance difference by the common factor
`s = 0.8979592` but does not merge distinct values. The HWA checkpoint's
largest W1/W2 conductances are only `0.7241/0.7663`, so the deployment ceiling
of `1` clips no weights. This is qualitatively different from literal
hard-floor programming, which maps almost all small weights to one value.

### The input floor cancels and the output floor is common mode

The physical input is `50 [x,-x]`, whose paired residual is exactly zero.
Consequently,

```text
[x,-x]^T (f J) = 0,
```

so the uniform W1 floor contributes no mathematical input current to any
hidden node. Float32 reduction leaves only a negligible numerical residual.

At the output, the floor adds the same numerator term to all 20 output nodes.
The class readout subtracts each output pair. For class `c`, the floor leakage
is

```text
f sum(h) [1 / d'_(2c) - 1 / d'_(2c+1)].
```

It vanishes when the two degrees match. The affine floor makes output degrees
more uniform: their coefficient of variation falls from `0.1030` to `0.0326`,
and mean relative within-pair mismatch falls from `11.02%` to `3.52%`.
Measured at the last update, floor leakage has RMS `0.000557`, only `2.60%`
of the total differential-score RMS `0.021465`.

### Degree loading mostly changes positive gains

The floor remains in the KCL denominators and supplies most of each node's
degree. Hidden-degree coefficient of variation falls from `0.5044` to
`0.0702`; output-degree coefficient of variation falls from `0.1030` to
`0.0326`. The perfect-diode projection is positively homogeneous, so positive
denominator changes alter amplitudes without changing activation signs unless
a preactivation crosses zero.

This checkpoint is especially stable:

- deterministic clean-versus-affine gate agreement is exactly `100%` after
  the first update and `99.997%` after the fourth;
- final active-set Jaccard similarity is `0.99944`, corresponding to only
  `0.003` gate flips per example out of 100;
- the first affine update already gives the final predictions, so recurrent
  relaxation is not rescuing the mapped network;
- the learned input term supplies `99.999%` of final hidden-preactivation RMS;
- all 108 deterministic prediction changes lie in the lowest clean-margin
  quintile.

The controlled corrections agree with this mechanism:

| Deterministic scenario | Accuracy | Symmetric normalized KL |
| --- | ---: | ---: |
| Clean HWA | `96.37%` | `0` |
| Full affine floor | `95.94%` | `0.00715` |
| Affine W1 only | `95.95%` | `0.00637` |
| Affine W2 only | `96.27%` | `0.00029` |
| Remove floor from numerators only | `95.92%` | `0.00721` |
| Remove floor from denominators only | `96.34%` | `0.00141` |
| Remove both floor terms with matched bias scaling | `96.37%` | approximately `0` |

Removing numerator floor current alone does essentially nothing because the
encoding already rejects it. Correcting denominator loading removes most of
the remaining functional drift. Most residual distortion arises in W1, where
the floor changes hidden-node degree normalization.

### Ideal accuracy hides a large loss of electrical margin

The clean-to-affine decision geometry is preserved mainly up to a positive
scale:

- centered paired-score RMS falls from `0.2985` to `0.01969`, a `15.2x`
  attenuation;
- median absolute top-two margin falls from `0.9419` to `0.05585`, a `16.9x`
  attenuation;
- nevertheless, centered-score cosine averages `0.9922` and top-1 agreement
  is `98.92%`.

Exact floating-point `argmax` is insensitive to positive score scaling. In a
wide deterministic sweep, accuracy is still `95.63%` at a floor ratio of
`90%`, even though centered-score RMS is only `1.96e-5` of clean. This is a
vanishing-margin result, not evidence for a usable `90%`-floor circuit.
At floor ratios above roughly `98%`, float32 cancellation also makes
normalized KL and cosine unreliable.

Synthetic post-inference controls expose the hidden sensitivity:

| Absolute output-node perturbation | Clean HWA | Affine deterministic | Affine stored |
| --- | ---: | ---: | ---: |
| Gaussian noise `sigma=0.003` | `96.36%` | `95.74%` | `95.97%` |
| Gaussian noise `sigma=0.01` | `96.35%` | `91.32%` | `92.12%` |
| Gaussian noise `sigma=0.03` | `96.29%` | `48.52%` | `50.23%` |
| Quantization step `0.03` | `96.35%` | `92.31%` | `93.14%` |

The Gaussian rows average 32 matched random draws. The noise amplitudes and
quantization steps are in model output-voltage units and are not calibrated
CMO, ADC, or circuit models; they are mechanism controls. They demonstrate
that the retained classification ordering is far less robust after the floor
attenuates the differential voltage.

The current CMO “read noise” is sampled once into a fixed conductance tensor.
Inference then uses deterministic float32 coordinate updates and an exact
argmax. It does not include fresh output/current noise, ADC quantization,
amplifier offset, IR drop, finite voltage resolution, or a nonzero diode
threshold. These missing effects are precisely those expected to expose the
lost voltage margin.

### Interpretation for expressivity

The affine interval `[f,1]` has the same continuous parameter dimension as
`[0,1]`, and the affine map preserves every weight distinction for `f<1`.
Therefore a finite lower bound does not automatically remove the particular
function learned by this overparameterized MNIST network. The structured
uniform offset also lies largely in common-mode directions rejected by this
specific input and output coding.

This does not establish equality of the global loose-range and bounded
hypothesis classes. Conductance-ratio constraints can still exclude functions,
and usable hardware expressivity depends on resolving the resulting currents
above noise and converter thresholds. The protection is expected to weaken
with unbalanced input rails, device-dependent floors, unmatched output pairs,
finite-threshold or saturating nonlinearities, deeper networks, and dynamic
readout noise. For example, adding the same `0.01` logical common offset to
every positive and negative input rail lowers deterministic affine accuracy
to `89.89%` while clean remains `96.14%`, confirming dependence on the
zero-sum encoding. This offset experiment is synthetic, not a calibrated rail
mismatch.

## Cancellation and mitigation options

| Method | Passive? | What it fixes | Current evidence and caveat |
| --- | --- | --- | --- |
| Affine physical programming | yes | Preserves sub-floor distinctions while retaining `G_min` | Best immediate result: `96.12%` first-write accuracy. It does not remove denominator loading or voltage attenuation. |
| Lower `G_min/G_max` | yes, device-level | Reduces clipping and graph loading | Literal mapping stayed near baseline at `1–2%` in the single-checkpoint sweep. |
| Selector-backed sparsity | yes if an access device can disconnect an edge | Removes the floor and its degree contribution on masked edges | A post-hoc, noiseless top-`384/60` oracle reached `94.93%`; the mask/counts were explored on the same test set. |
| Floor-aware training | yes at inference | Lets weights, biases, and node degrees adapt to the physical affine map | Not yet run. It should optimize the mapped conductances directly rather than clamp a clean checkpoint after training. |
| Per-node denominator normalization | active | Compensates predictable row/column-sum loading | The ideal oracle reached `96.34%`, but it also scaled the hidden bias by `s`; a circuit needs calibrated gain/current scaling and still does not remove common numerator currents. |
| Full KCL floor cancellation | active | Removes both numerator and denominator floor terms | The deterministic oracle recovered exactly `96.37%`; it also scaled bias by `s`, so this is an equation-level bound, not a demonstrated circuit. |
| Differential/dual-rail constant-sum topology | passive cells plus additional routing/readout | Moves much of the floor into common mode and makes loading more uniform | Promising redesign, but a conventional differential MVM alone still leaves DRN node loading and is not an exact drop-in replacement. |

### Exact active correction

For a node voltage `v` connected to `d` neighbors `u_i`, a uniform floor
contributes

```text
I_floor(v) = f * (d v - sum_i u_i).
```

An exact active implementation must inject `-I_floor` at every dynamic
endpoint. Because the uniform floor matrix is rank one, the circuit does not
need one independent cancellation coefficient per edge: it can measure one
layer-voltage sum, broadcast it, and combine it with a calibrated `d f v`
term at each node. The hidden-to-output boundary must be corrected at both
ends to retain reciprocal dynamics.

After removing the floor from `f + sW`, the desired edges are `sW`. The bias
current and any finite nonlinear-device current must retain the same relative
scale. The deterministic oracle therefore scales the hidden bias by `s`;
with a non-ideal diode, its current law would need corresponding treatment.
Negative transconductance also makes stability and energy convexity a circuit
question, so this proposal needs SPICE-level validation.

A nominal reference cancels only the uniform component. Device-to-device
floor mismatch, programming error, relaxation, and read noise leave a
non-rank-one residual. Reference cells or a calibrated per-cell subtractive
array can reduce that residual, at added area, read noise, and stability cost.

### Why post-hoc subtraction could not rescue the literal checkpoint

Subtracting fractions of the nominal floor from the already programmed
literal checkpoint improved stored accuracy only from `80.34%` to a
denominator-only peak of `82.88%` at 90% subtraction; simultaneous numerator
and denominator subtraction peaked at `82.51%`. Numerator-only subtraction
remained near `80.1%`. All replay points stayed finite, but voltage RMS grew
and accuracy collapsed as the corrected conductance sums became small near
complete subtraction. This is not evidence against exact cancellation: the
literal write had already replaced most distinct targets with the same floor
value, so no later reference path can reconstruct the erased weights.

Changing the input gain from `10` to `800` or the inference iteration count
from `1` to `32` also left literal accuracy at approximately `80.34–80.36%`.
For the present perfect-diode equations, global gain does not undo a
conductance-range mismatch.

## Recommended sequence

1. Use `affine_floor` as the primary floor-retaining deployment mapping when
   the program-and-verify device can target its full analog window. Keep
   `literal_conductance` as an explicit absolute-scale/clipping stress test
   and `normalized_offset` as an explicit cancellation abstraction.
2. Train several bases from initialization through the affine physical
   mapping and endpoint noise. Track accuracy, hidden/output voltage
   distributions, conductance sums, residual-current tolerance, and estimated
   power rather than accuracy alone.
3. Repeat the floor-ratio sweep across independently trained bases with
   validation-selected settings and an untouched final test split.
4. Train structured masks or bounded fan-in on validation data, then realize
   masked edges as true open circuits rather than devices held at `G_min`.
5. Prototype full uniform-offset KCL cancellation with gain error, floor
   mismatch, read noise, finite output resistance, and stability checks.
   Each all-ones crossbar offset is rank one, but the complete KCL/Laplacian
   perturbation also contains diagonal degree loading and is not rank one.
   Compare it with denominator-only calibration and a dual-rail
   constant-sum topology.

## Reproduction and artifacts

The validated analysis utility is
[`labs/tools/analyze_mnist_conductance_floor.py`](../labs/tools/analyze_mnist_conductance_floor.py).
The main machine-readable result is the local ignored artifact
`results/mnist-cmo-floor-mitigation/analysis/replay-v5/summary.json`; the
floor-ratio and cancellation plots are in the same directory. The official
affine probe is the local ignored run
`results/mnist-cmo-floor-mitigation/affine_floor_probe_v2/20260730T095125.044508Z-ee47b30f-7664fae3/`.
The predictive and conductance divergence result is the local ignored
artifact
`results/mnist-cmo-floor-mitigation/analysis/mapping-kl-v1/summary.json`.
The gate, component, wide-floor, imbalance, and synthetic-readout controls
are in the local ignored artifact
`results/mnist-cmo-floor-mitigation/analysis/affine-floor-mechanism-v3/summary.json`;
its plots are in the same directory. The corresponding utility is
[`labs/tools/analyze_mnist_affine_floor_mechanism.py`](../labs/tools/analyze_mnist_affine_floor_mechanism.py).

```bash
python labs/tools/analyze_mnist_conductance_floor.py \
  --clean-weights <clean-hwa-weights.pt> \
  --literal-weights <literal-cmo-weights.pt> \
  --affine-weights <affine-cmo-weights.pt> \
  --output-dir results/mnist-cmo-floor-mitigation/analysis/<new-run>
```

```bash
python labs/tools/analyze_mnist_affine_floor_mechanism.py \
  --clean-weights <clean-hwa-weights.pt> \
  --affine-weights <affine-cmo-weights.pt> \
  --output-dir results/mnist-cmo-floor-mitigation/analysis/<new-run>
```

The study is exploratory: it uses one trained base/device seed, reports on
the canonical MNIST test split, and models active correction algebraically
rather than with a transistor-level circuit. The CMO endpoint equations come
from AIHWKit's model of the measured device data; the physical window and
program-and-verify context are described by
[Falcone et al.](https://arxiv.org/abs/2502.04524).
