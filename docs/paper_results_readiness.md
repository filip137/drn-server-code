# Paper Results Readiness

Updated: 2026-08-17

Status: working research synthesis. This note tracks progress toward the
paper's intended figures; it does not supersede the active scientific
protocols. Individual analyzed studies remain authoritative in
[`experimental_manifest.md`](experimental_manifest.md), and
`current_state.md` remains human-authored. On 2026-08-17 Filip selected
ordinary MNIST, rather than deterministic medium-affine MNIST, as the active
paper dataset.

## Executive Assessment

The results now support a coherent paper claim that is stronger than “our
amplification always performs best”:

> Energy-compatible bidirectional amplification is a design degree of freedom
> whose preferred setting depends on the learning rule and hardware
> constraints. Under the tested contracts, legacy amplification is favored by
> BPTT when conductance range is tight, but its smaller EqProp phase signal
> makes it more vulnerable to fixed absolute read noise. Ours retains much of
> the clean amplification benefit while preserving a substantially larger
> EqProp noise margin than legacy.

This claim directly connects the theory—multiple voltage/current
amplification choices preserve the energy formulation—to an experimental
message: energy equivalence does not imply equal trainability under finite
dynamic range, finite nudging, and noisy reads.

The mechanism and hyperparameter evidence is now strong, and the dataset
switch makes several existing checkpoint families potentially reusable. The
main figures are not yet paper-ready because the final EqProp beta, inclusion
sets, and Conv3 residual policy are not frozen, no existing family has passed
the paper-reuse audit, and the official MNIST test split has not been evaluated
under the new sealed contract.

## Readiness of the Three Intended Result Blocks

| Result block | Evidence in hand | What it supports now | Remaining paper gate | Status |
|---|---|---|---|---|
| Ideal BPTT versus EqProp, Conv1/2/3 | All nine exact-zero-bias Adam EqProp cases complete `10/30/30` epochs at the one-decade beta. The direct gradient gate passes `213/216` rows at one decade and `216/216` at two decades. | Clean EqProp is viable at all three depths. Beta selection is a phase-signal versus local-gradient-fidelity tradeoff, not an accuracy maximization problem. | Freeze beta; resolve or explicitly accept the trained Conv3-baseline `T=8` residual caveat; audit the existing zero-bias BPTT/EqProp checkpoints for exact matching; retrain only ineligible cells; then perform one sealed official-test evaluation. | **Selection-ready; checkpoint audit pending** |
| EqProp with read noise, Conv1/2/3 | Conv2 sigma-`5e-4` training repeats the same qualitative scheme ordering with Adam and SGD. Conv3 endpoint-gradient replay gives ours a `100x` initialization and `1000x` trained-checkpoint usable-noise advantage over legacy. A matched Conv1/Conv3 sigma-`5e-4` Adam study is running. | Legacy's smaller upstream phase signal creates selective read-noise sensitivity; ours is the highest-accuracy noisy Conv2 scheme in both tested optimizer/LR contracts. | Complete and review Conv1/Conv3 noisy training at the frozen beta; seal the noise levels and inclusion set; audit eligible checkpoints; evaluate the official test once per retained condition. | **Mechanism strong; all-depth training nearly covered** |
| Constrained dynamic range, Conv1/2/3 | A 27-case Adam study shares one fixed initializer and changes only `wmax` over `1e-4,5e-4,1e-3`. A separate Kaiming raw-LR transfer preserves the legacy gaps. | A tight upper conductance bound disproportionately throttles baseline and ours, explaining much but not all of legacy's advantage. The initializer family alone does not explain the ordering at transferred rates. | Decide whether the fixed-initialization `wmax` sweep itself is the paper mechanism figure or whether a canonical bounded-initializer/rho handoff is required; audit or rerun the chosen MNIST cells and seal official-test evaluation. | **Mechanism strong; final contract pending** |

## Evidence Already Established

### 1. Clean EqProp training is stable at one decade below the boundary

The zero-bias, shared-`T/K` Adam qualification uses Conv1/2/3
`T/K=4/4,6/6,8/8`, injected beta baseline/ours/legacy
`100/30/3`, `100/10/.03`, and `100/3/.001`, and exact inherited BPTT weight
learning rates. All `9/9` runs complete their full budget with finite metrics.

| Architecture | Baseline best/final | Ours best/final | Legacy best/final |
|---|---:|---:|---:|
| Conv1 | `96.18/96.04%` | `96.42/96.34%` | `96.44/96.42%` |
| Conv2 | `97.22/97.22%` | `98.10/98.02%` | `98.44/98.42%` |
| Conv3 | `97.76/97.68%` | `98.72/98.64%` | `99.08/99.02%` |

These validation measurements are not paper-facing accuracy and are not yet a
BPTT--EqProp comparison. Their role is to establish that the proposed EqProp
contract trains. Under the new dataset decision, the checkpoints may become
paper evidence without retraining only if they pass the exact reuse/matching
audit and are then evaluated once on the previously untouched official test
split.

### 2. Beta exposes a real fidelity-versus-signal tradeoff

At reconstructed initialization and best checkpoints, over all architectures,
schemes, four fixed minibatches, and all weight layers:

| Injected-beta tier | Direct-gradient passes | Worst cosine | Maximum symmetric norm delta | Interpretation |
|---|---:|---:|---:|---|
| Original boundary | `159/216` | `.801608` | `.533422` | Ineligible; outside the local-gradient regime in many contexts |
| One decade lower | `213/216` | `.939168` | `.278178` | Training-stable, but three Conv3-initialization C0 rows fail |
| Two decades lower | `216/216` | `.998877` | `.039421` | Fully direct-gradient-qualified; long matched stability still pending |

The remaining beta decision should therefore not be framed as “small beta is
standard” or “10% output motion is safe.” Neither is the empirical rule used
here. The supported choice is between two observed operating points with
different benefits:

- one decade has demonstrated all-depth long training and a tenfold larger
  phase signal, improving robustness to fixed absolute read noise;
- two decades satisfies every direct-gradient comparison and has the larger
  finite-nudge safety margin, but still needs the matching `10/30/30`-epoch
  stability qualification.

Beta is a force coefficient, not a displacement. In the local regime the
response is governed schematically by
`delta_s = -beta H_E^{-1} grad_s C`, so changing weights, conductances, or
amplification changes susceptibility and the allowed beta. A common relative
margin below each scheme's gradient boundary does not equalize displacement:
at the one-decade tier the measured output displacement spans `4.89x` across
schemes in Conv1 and `6.50x` in Conv3.

For the paper, report base beta, actual injected beta, the gradient-boundary
margin, and measured absolute and free-state-relative displacement at every
layer. This makes the selection defensible without inventing an unsupported
universal displacement threshold.

### 3. Conv2 read-noise sensitivity repeats across optimizer contracts

At endpoint read-noise sigma `5e-4`, using the same one-decade scheme-specific
betas:

| Scheme | Adam clean -> noisy final | Adam change | SGD clean -> noisy final | SGD change |
|---|---:|---:|---:|---:|
| Baseline | `96.88 -> 96.78%` | `-.10 pp` | `96.52 -> 96.34%` | `-.18 pp` |
| Ours | `97.92 -> 97.44%` | `-.48 pp` | `97.80 -> 97.46%` | `-.34 pp` |
| Legacy | `98.40 -> 96.90%` | `-1.50 pp` | `97.52 -> 96.72%` | `-.80 pp` |

All runs remain finite. Ours has the highest noisy endpoint accuracy and legacy
the largest clean-relative loss under both contracts. The exact loss magnitude
is optimizer/LR-contract dependent; this is not a pure optimizer ablation
because each optimizer uses its own historical LR vector.

The Conv3 read-only mechanism study is stronger in scale but narrower in
scope. At a predeclared gradient-fidelity gate, ours tolerates `100x` more
endpoint voltage noise than legacy at initialization and `1000x` more at the
trained checkpoint. This diagnoses layerwise signal transport; it is not a
training-accuracy result or a calibrated ADC/ENOB claim.

### 4. Tight weight range changes the preferred amplification

The all-depth fixed-initialization Adam sweep changes only the upper projection
limit. Best-validation accuracy is:

| Architecture | Scheme | `wmax=1e-4` | `wmax=5e-4` | `wmax=1e-3` | Gain, tight to wide |
|---|---|---:|---:|---:|---:|
| Conv1 | Baseline | `89.68%` | `94.44%` | `94.80%` | `+5.12 pp` |
| Conv1 | Ours | `92.92%` | `95.00%` | `95.02%` | `+2.10 pp` |
| Conv1 | Legacy | `95.78%` | `96.68%` | `96.68%` | `+.90 pp` |
| Conv2 | Baseline | `85.60%` | `92.44%` | `93.52%` | `+7.92 pp` |
| Conv2 | Ours | `90.30%` | `96.56%` | `97.18%` | `+6.88 pp` |
| Conv2 | Legacy | `96.84%` | `98.16%` | `98.20%` | `+1.36 pp` |
| Conv3 | Baseline | `77.66%` | `87.44%` | `89.60%` | `+11.94 pp` |
| Conv3 | Ours | `84.54%` | `94.74%` | `95.80%` | `+11.26 pp` |
| Conv3 | Legacy | `93.68%` | `97.68%` | `97.86%` | `+4.18 pp` |

Legacy is best in all nine architecture-by-ceiling cells, but its advantage
over ours contracts from `2.86 -> 1.66 pp`, `6.54 -> 1.02 pp`, and
`9.14 -> 2.06 pp` in Conv1/2/3 as the ceiling widens. This is the clearest
evidence that amplification ranking depends on the hardware constraint rather
than reflecting a universal algorithmic ordering.

The Kaiming raw-LR transfer changes final accuracy by only `-.42` to
`-.16 pp` relative to matched bounded-uniform source cells, and none of the
legacy margins shrinks. Initializer family alone therefore does not explain
the bounded legacy advantage at those transferred rates. Because the raw
rates were deliberately not reselected, this is a diagnostic rather than the
canonical bounded handoff.

## Minimum Path to Paper-Ready Results

1. Run the exact two-decade zero-bias/shared-`T/K` Adam stability counterpart
   to the completed one-decade study, then freeze beta under the predeclared
   fidelity/stability/noise rule without consulting the official test split.
2. Resolve the trained Conv3-baseline `T=8` free-state residual caveat by
   either explicitly accepting the shared truncated dynamics or requalifying
   a new BPTT/EqProp-common operating point.
3. Complete and review the already prepared Conv1/Conv3 one-decade
   sigma-`5e-4` Adam training study. If two decades is ultimately selected,
   rerun the corresponding noisy comparison at that frozen tier rather than
   mixing beta contracts.
4. Audit the existing ordinary-MNIST BPTT/EqProp checkpoint families for
   exact-zero biases, hash-identical initialization/order, identical weight LR
   vectors, frozen `T/K`, complete coverage, and zero prior official-test
   reads. Retrain only cells that fail this gate.
5. Decide whether the fixed-initialization `wmax` sweep is the final dynamic-
   range figure or whether the global bounded-initializer/rho handoff remains
   required. Freeze that choice before any test evaluation.
6. Freeze the primary comparisons, then add seeds to those comparisons only.
   Do not spend the first multi-seed budget repeating every mechanism
   diagnostic.
7. Seal the exact checkpoint identities and inclusion set, then read the
   official 10,000-example MNIST test split exactly once per retained paper
   run.

The unresolved scientific scope decision is whether the constrained-range
figure is BPTT-only, matching the original Conv scope, or a second full
BPTT/EqProp factorial. That decision should be made explicitly because the
latter materially enlarges the paper grid.

## Suggested Main-Figure Organization

1. **Clean algorithm comparison:** official-test MNIST Conv1/2/3 accuracy for
   baseline, ours, and legacy under matched BPTT and EqProp contracts.
2. **Read-noise robustness:** clean-relative EqProp degradation versus noise,
   with Conv1/2/3 training as the main panel and layerwise gradient/noise
   thresholds as mechanism support.
3. **Dynamic-range dependence:** accuracy versus `wmax/wmin` or upper ceiling
   across Conv1/2/3, emphasizing that the legacy-minus-ours gap contracts as
   the constraint relaxes.
4. **Tradeoff summary:** a compact synthesis showing that no amplification is
   uniformly optimal across clean BPTT, noisy EqProp, and constrained weights.

CIFAR-10 would be a useful scaling validation after these contracts are
frozen, particularly if it repeats the constraint-dependent ordering. It
should be a capstone rather than a prerequisite for the core paper. ImageNet
would be substantially more impressive but is not necessary to make the
present mechanism story coherent and carries a much larger contract-selection
and compute burden.

## Claim Boundaries to Preserve

- Ordinary-MNIST validation accuracy is selection/mechanism evidence. Only the
  once-read official MNIST test metric from a sealed eligible checkpoint is
  paper-facing accuracy.
- Every current headline result is seed 0; uncertainty is not yet estimated.
- The historical medium-affine BPTT grid used learned, unscaled biases. It is
  optional robustness provenance, not the active paper grid or the matched
  control for new exact-zero-bias EqProp rows.
- The read-noise sigma is in simulator voltage units and models independent
  endpoint-read noise; it is not yet a calibrated hardware voltage or ENOB.
- Adam and SGD comparisons also change their accepted parameter-wise LR
  vectors and must not be described as isolated optimizer effects.
- The fixed-`wmax` causal contrast is within a scheme at fixed initializer and
  LR vector. Cross-scheme levels still use scheme-specific LR vectors.

## Primary Evidence

- [One-decade zero-bias EqProp training report](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1/analysis/report.md)
- [Zero-bias EqProp--BPTT gradient qualification](../results/perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-bptt-gradient-gate-tk4-6-8-seed0-20260816-v1/review.md)
- [EqProp beta decision record](beta_study.md)
- [Conv2 Adam/SGD read-noise comparison](../results/perfectdiode-conv2-centered-float64-eqprop-read-noise-one-decade-sigma-5em4-sgd-10ep-seed0-20260816-v1/analysis/report.md)
- [Conv3 layerwise endpoint-read-noise analysis](../results/perfectdiode-conv3-centered-float64-voltage-read-noise-tk8-tk12-20260812-v1/analysis/report.md)
- [All-depth fixed-initialization dynamic-range report](../results/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/analysis/report.md)
- [Bounded-Kaiming raw-LR transfer report](../results/perfectdiode-conv12-bounded-kaiming-adam-lr-transfer-3ep-seed0-20260813-v1/analysis/report.md)
