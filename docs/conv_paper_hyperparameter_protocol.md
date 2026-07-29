# Conv Perfect-Diode Hyperparameter Protocol Index

Updated: 2026-07-29

Status: active index. The paper comparison is frozen; wide-range and bounded
learning-rate handoffs are not yet complete for all Conv1/2/3 surfaces.

## Active Scope

The active work is perfect-diode BPTT for:

- Conv1, Conv2, and Conv3;
- strides `[2]`, `[2,2]`, and `[2,2,1]`;
- kernel `3`, padding `1`, and no pooling;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- plain SGD and Adam; and
- wide-range reference and bounded hardware weight contracts.

The [experiment definition](conv_paper_experiment_definition.md) is the
paper-facing comparison authority.

## Dataset Handoff

The workflow deliberately uses two datasets:

1. Ordinary MNIST with the deterministic 55,000/5,000 training/validation
   split selects operational `T/K`, rho targets, parameter-specific raw
   learning rates, and the bounded initializer. The official test split is not
   read during selection.
2. Deterministic medium-affine MNIST consumes the frozen handoffs for paper
   training.

Ordinary-MNIST metrics are optimization diagnostics. They do not enter paper
accuracy tables. A medium-affine paper run must use the matching LR vector
unchanged; changing a scientific surface requires a new ordinary-MNIST
handoff.

## Protocol Order

1. Freeze the paper comparison axes.
2. Resolve perfect-diode input gain and operational `T/K`.
3. Select the wide-range LR vector independently for every architecture x
   scheme x optimizer surface.
4. Repeat rho selection independently under both bounded initializers.
5. Select one globally eligible bounded initializer with the declared paired
   2% rule.
6. Freeze medium-affine seeds, epoch budgets, checkpoint rules, and inclusion
   rules.
7. Run the wide-range and bounded paper grids from fresh initialization.

Changing an earlier stage invalidates its dependent later stages.

## Active Protocols

| Stage | Authority | Status |
|---|---|---|
| Paper comparison | [`conv_paper_experiment_definition.md`](conv_paper_experiment_definition.md) | frozen |
| Conv1/Conv2 ordinary-MNIST rho | [`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) | nine interim unbounded vectors authorized; three rows missing; terminal evidence incomplete |
| Conv3 ordinary-MNIST `T/K` and rho | [`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md) | protocol authorized; terminal handoffs incomplete |
| Bounded initializer and rho | [`perfectdiode_bounded_weight_protocol.md`](perfectdiode_bounded_weight_protocol.md) | all-depth selector specified; execution pending |
| Execution and targets | [`experiment_workflow.md`](experiment_workflow.md) | active |
| Result reporting | [`experiment_reporting.md`](experiment_reporting.md) | active |
| Dashboard and nodes | [`current_state.md`](current_state.md) | active |

The immutable Conv1/Conv2 bounded-initialization JSON predates the all-depth
global selector. It remains provenance and must not be edited in place.

## Handoff Contents

Every successful ordinary-MNIST LR handoff records:

- architecture, scheme, optimizer, weight contract, and initializer;
- input gain and operational `T/K`;
- optimizer-specific rho units and selected targets;
- ordered raw LR vector for every scientific parameter;
- initialization, split, cohort, minibatch, config, code, and artifact hashes;
- candidate metrics and safety diagnostics; and
- `official_test_read=false`.

The matching medium-affine config imports the complete handoff. It does not
transfer a rho target or raw LR across schemes, optimizers, architectures,
weight contracts, or initializers.

## Current Gates

Complete paper-grid execution remains incomplete because:

- only nine Conv1/Conv2 wide-range surfaces have user-directed interim
  vectors; Conv1 legacy Adam and Conv2 ours SGD/Adam remain unresolved;
- all Conv1/Conv2 surfaces still need terminal selection and confirmation;
- Conv3 `T/K` and wide-range rho handoffs are not terminal;
- the two bounded initializer families have not completed all 18 surfaces;
- no global bounded initializer has been selected;
- Conv3 epoch budget, final model seeds, checkpoint rules, and inclusion rules
  remain pending.

Conv1's current paper budget is 10 epochs and Conv2's is 30 epochs. Do not
invent the unresolved Conv3 or final multi-seed choices.

## Historical Material

Hard-sigmoid, Hopfield/EqProp, older optimizer diagnostics, and legacy sweep
instructions are preserved under
[`archive/out_of_scope_20260729/`](archive/out_of_scope_20260729/). Existing
older archives and snapshots remain provenance. If historical material
conflicts with this active index, this index and its linked active protocols
win.
