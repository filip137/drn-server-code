# Conv Paper Experiment Definition

Updated: 2026-08-17

Status: active comparison contract. On 2026-08-17 Filip changed the active
paper dataset from deterministic medium-affine MNIST to ordinary MNIST. The
architecture, amplification, matching, and official-test safeguards remain
frozen; bounded handoffs and later multi-seed scope remain pending in the
[protocol index](conv_paper_hyperparameter_protocol.md). A direct seed-0
BPTT--EqProp extension is defined by the
[matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md).

## Question

The paper compares bidirectional amplification in perfect-diode dissipative
resistive Conv networks under a wide conductance range and a narrow physical
conductance contract.

Training uses backpropagation through time (BPTT) through the explicit,
fixed-`K` unrolled equilibrium iterations. It does not use EqProp, nudging
updates, or a feed-forward surrogate.

That statement defines the original amplification comparison. The separate
one-seed training-algorithm extension compares BPTT with centered EqProp under
an exact matched-pair rule. In that extension, both algorithms use the same
named learning-rate vector and `T/K`, and every bias is initialized and frozen
at exact zero. An EqProp run outside that matching rule is not part of the
paper-ready algorithm comparison.

## Paper Dataset

Paper runs use ordinary **MNIST**, without affine augmentation or pixel
permutation. Apply the frozen DRN preprocessing:

```text
x_normalized = 0.3 * (x - 0.1307) / 0.3081
x_signed = concat(x_normalized, -x_normalized)
```

The model's frozen `input_gain` is applied after this preprocessing. The
official 60,000-example training split is deterministically divided into
55,000 training and 5,000 validation examples. The official 10,000-example
test split remains untouched throughout `T/K`, rho, learning-rate,
initializer, beta, noise-level, epoch, and checkpoint selection.

Ordinary-MNIST validation metrics remain selection or mechanism evidence and
are not paper-facing accuracy. After the complete scientific contract and
maximum-validation checkpoint rule are frozen, each eligible paper run reads
the official MNIST test split exactly once from that selected checkpoint. The
resulting test metric is the paper-facing accuracy.

The earlier deterministic medium-affine dataset—rotation in
`[-25 deg,+25 deg]`, translation up to `20%`, scale `[0.8,1.2]`, affine seed
`1729`—is retained as historical or optional robustness evidence. It is no
longer required for the active paper grid.

## Architecture

All convolutional layers use kernel size `3`, padding `1`, and no pooling.

| Architecture | Hidden channels | Strides | Hidden shapes | Output |
|---|---|---|---|---|
| Conv1 | `[64]` | `[2]` | `[64,14,14]` | `[20]` |
| Conv2 | `[64,128]` | `[2,2]` | `[64,14,14]`, `[128,7,7]` | `[20]` |
| Conv3 | `[64,128,256]` | `[2,2,1]` | `[64,14,14]`, `[128,7,7]`, `[256,7,7]` | `[20]` |

The classifier uses 20 paired outputs for 10 classes and paired squared-error
loss. Output-10 checkpoints are a different protocol.

## Nonlinearity And Amplification

The active nonlinearity is `perfect_diode` with explicit diode parameter
dictionaries and the protocol's clamp epsilon.

| Label | Run name | Voltage amp | Current amp |
|---|---|---:|---:|
| baseline | `mnist_bp_amp_v1_c1` | `1` | `1` |
| proposed/ours | `mnist_bp_amp_v4_c1` | `4` | `1` |
| legacy | `mnist_bp_amp_v4_c0p25` | `4` | `0.25` |

The old five-setting grid and hard sigmoid are outside the active paper
contract.

## Optimizers

Every architecture and amplification scheme is run with:

- plain SGD with momentum `0` and weight decay `0`; and
- Adam with the exact optimizer settings frozen by the rho protocol.

Rho targets, proposal units, and raw learning-rate vectors are selected
independently for every architecture x scheme x optimizer surface.

## Weight Contracts

Each paper surface has two result conditions:

1. **Wide-range reference:** conductance weights are projected to `[0,100]`
   and use the reference initialization contract.
2. **Bounded hardware contract:** `ConvWeight_*` and `DenseWeight_*` are
   projected to `[1e-5,1e-4]`. Initialization is selected globally between:
   - `bounded_uniform`, sampled from `[1e-5,1e-4)`; and
   - `bounded_kaiming_uniform`, the midpoint-centered, fan-in-scaled bounded
     Kaiming initializer with gain `1`.

Biases are not conductance weights and are not projected to this interval.
The bounded initializer and bounded raw LR vectors are selected on ordinary
MNIST under the
[bounded-weight protocol](perfectdiode_bounded_weight_protocol.md).

For the matched BPTT--EqProp extension only, the bias tensors remain present
but their initialization and learning rates are exactly zero under the
[one-seed matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md).
This explicit downstream rule supersedes the historical learned-bias handoff
entries for those paired rows.

If uniform initialization wins, the bounded condition differs from the
wide-range reference in both range and initialization. Report it as one
hardware-constrained contract, not as a bounds-only causal ablation.

## Comparison Contract

Before a fresh paper launch—or before an existing checkpoint is sealed for
paper evaluation—every row must freeze and record:

- the exact dataset realization and preprocessing;
- architecture, convolution pipeline, output encoding, and loss;
- nonlinearity and explicit diode parameters;
- amplification values;
- weight contract and initialization;
- operational `T/K`;
- optimizer and complete parameter-specific LR vector;
- training algorithm and, for a BPTT--EqProp pair, proof of exact LR and
  `T/K` equality plus exact-zero biases;
- model and loader seeds;
- epoch budget; and
- checkpoint and inclusion rules.

The final ordinary-MNIST paper config consumes the corresponding selection
handoff unchanged. It must not recalibrate rho or rewrite raw learning rates.
Changing architecture, scheme, optimizer, `T/K`, weight contract, or
initializer invalidates the dependent LR handoff. Changing EqProp beta
invalidates the dependent EqProp qualification and paper eligibility until it
is requalified.

## First Ordinary-MNIST Paper Execution

The first `[0,100]` batch uses model seed `0` and loader seed `0`. Conv1 trains
for 10 epochs; Conv2 and Conv3 train for 30.

The 60,000-example training split is deterministically partitioned into 55,000
training and 5,000 validation examples. Checkpoint selection uses maximum
validation accuracy. After training completes and the full paper contract is
frozen, the official 10,000-example MNIST test split is evaluated exactly once
from that selected checkpoint.

All 18 predeclared architecture x scheme x optimizer rows are included if they
complete the exact budget with finite metrics and verified artifacts.
Operational failures may be retried only with the unchanged scientific config;
outcomes are never an exclusion criterion.

The direct BPTT--EqProp extension contains two predeclared algorithm members
for every such row, or 36 rows per weight contract. It uses the same
`10/30/30` epoch budgets and inclusion rule, with the additional fail-closed
pair receipt required by its matching protocol.

Filip authorized consuming the Conv3 LR vectors selected under
`weight_max=null` in these downstream `[0,100]` configs. The source-contract
mismatch remains a declared limitation.

## Existing Ordinary-MNIST Reuse Gate

Changing the paper dataset does not automatically promote every existing
ordinary-MNIST experiment. An existing run may be used without retraining only
after a fail-closed audit establishes all of the following:

- its resolved configuration exactly matches the subsequently frozen paper
  contract, including bias policy, `T/K`, optimizer vector, beta/noise fields,
  weight bounds, epoch budget, seeds, and checkpoint rule;
- the complete comparison surface was predeclared and every eligible arm is
  retained, so the result was not selected because it looked favorable;
- the canonical bundle and required checkpoints validate, and any matched
  BPTT--EqProp pair passes the initialization/order/LR equality receipt;
- `official_test_read=false` for every selection and training artifact; and
- the official test evaluation is performed once only after the dataset,
  contract, inclusion set, and checkpoint identities are sealed.

If any condition fails, rerun that paper surface from fresh matched
initialization. Passing this gate permits checkpoint reuse and one sealed test
evaluation; it does not retroactively turn validation accuracy into a paper
metric or erase the study's selection provenance.

## Historical Medium-Affine Batch

The completed seed-0 medium-affine wide-range batch remains valid historical
evidence under its original learned-bias contract. It is not the active paper
grid and is not the matched control for exact-zero-bias BPTT--EqProp results.
