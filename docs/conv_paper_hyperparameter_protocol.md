# Conv Perfect-Diode Hyperparameter Protocol Index

Updated: 2026-08-17

Status: active index. The original BPTT comparison axes are frozen, and the
active paper dataset is ordinary MNIST. Wide-range and bounded learning-rate
handoffs are not yet complete for all
Conv1/2/3 surfaces. The seed-0 matched BPTT--EqProp extension is governed by
the exact-zero-bias matching protocol linked below.

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

The first direct BPTT--EqProp comparison additionally follows the
[one-seed matching protocol](conv_paper_one_seed_bptt_eqprop_protocol.md).
For those paired rows, it freezes every bias at zero and requires exact
learning-rate and `T/K` equality between algorithms.

## Dataset And Evaluation Handoff

Ordinary MNIST is used for both selection and the active paper runs. The
deterministic 55,000/5,000 training/validation split selects operational
`T/K`, rho targets, parameter-specific raw learning rates, bounded
initializer, and EqProp beta. The official 10,000-example test split is not
read during any of those choices.

Validation metrics remain optimization diagnostics and do not enter the paper
accuracy tables. Once the complete contract, inclusion set, and selected
checkpoint are frozen, the official MNIST test split is evaluated exactly once
per eligible paper run. Deterministic medium-affine MNIST is historical or
optional robustness evidence rather than a required handoff target.

## Protocol Order

1. Freeze the paper comparison axes.
2. Resolve perfect-diode input gain and operational `T/K`.
3. Select the wide-range LR vector independently for every architecture x
   scheme x optimizer surface.
4. Repeat rho selection independently under both bounded initializers.
5. Select one globally eligible bounded initializer with the declared paired
   2% rule.
6. Freeze ordinary-MNIST paper seeds, epoch budgets, checkpoint rules,
   inclusion rules, and official-test evaluation receipts.
7. Run the wide-range and bounded paper grids from fresh initialization, or
   audit already completed checkpoints through the explicit reuse gate.

Changing an earlier stage invalidates its dependent later stages.

The matched BPTT--EqProp extension inserts two fail-closed steps before its
paper grid: derive the shared zero-bias vectors from the accepted weight
handoffs, then qualify EqProp beta on the training/validation partition at the
architecture's shared `T/K`. Only after both steps are frozen may paired BPTT
and EqProp checkpoints be sealed for official-test evaluation.

## Active Protocols

| Stage | Authority | Status |
|---|---|---|
| Paper comparison | [`conv_paper_experiment_definition.md`](conv_paper_experiment_definition.md) | axes frozen; paper dataset switched to ordinary MNIST on 2026-08-17 |
| Seed-0 BPTT--EqProp matching | [`conv_paper_one_seed_bptt_eqprop_protocol.md`](conv_paper_one_seed_bptt_eqprop_protocol.md) | matching rule frozen; one-decade Adam training stability and two-decade Adam gradient fidelity are established, while the two-decade stability comparison, final beta freeze, and Conv3 residual decision remain pending |
| Conv1/Conv2 ordinary-MNIST rho | [`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) | all 12 current unbounded vectors fixed; Conv1 review and Conv2 confirmation/terminal evidence incomplete |
| Conv3 ordinary-MNIST `T/K` and rho | [`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md) | fixed `T=K=8` unbounded six-surface rho study and LR handoff complete; downstream `[0,100]` use explicitly authorized with source mismatch retained; confirmation pending |
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

The final paper config imports the complete handoff. It does not transfer a
rho target or raw LR across schemes, optimizers, architectures, weight
contracts, or initializers.

## Current Gates

Complete paper-grid execution remains incomplete because:

- the matched BPTT--EqProp contract has exact-zero-bias LR vectors and shared
  `T/K`; one-decade Adam training is stable in all nine cases and two decades
  passes all `216/216` direct-gradient comparisons, but the matched
  two-decade long-stability branch and final beta decision are still open;
- trained Conv3 baseline retains the shared-`T=8` free-state residual caveat,
  which must be explicitly accepted or resolved by requalifying a common
  BPTT/EqProp operating point before paired paper runs;
- existing ordinary-MNIST zero-bias checkpoints have not yet completed the
  fail-closed paper-reuse and matched-pair audit, and no official-test
  evaluation is authorized before beta and the inclusion set are frozen;
- all 12 Conv1/Conv2 wide-range surfaces have current user-directed vectors,
  but Conv1 confirmation review and terminal selector publication are pending;
- Conv2 has three-epoch evidence only and no long confirmation;
- Conv3's fixed-`T=K=8`, `weight_max=null` rho search is complete; Filip
  explicitly authorized consuming its six selected LR vectors under the
  `[0,100]` paper contract, while the source-contract mismatch remains recorded
  as a limitation;
- the two bounded initializer families have not completed all 18 surfaces;
- no global bounded initializer has been selected.

The first ordinary-MNIST paper batch is frozen at model/loader seed `0`: Conv1
uses 10 epochs and Conv2/Conv3 use 30. It selects the best checkpoint by the
held-out 5,000-example validation accuracy, evaluates the official MNIST test
split once from that checkpoint after all contracts are sealed, and includes
every predeclared arm that meets the exact completion and artifact-integrity
criteria. Existing checkpoints may be reused only through the
[paper experiment reuse gate](conv_paper_experiment_definition.md#existing-ordinary-mnist-reuse-gate).
Later multi-seed scope remains a separate decision.

For the matched BPTT--EqProp extension, the current weight learning rates are
copied unchanged, all bias rates are set to zero, and both algorithms use
Conv1 `T/K=4/4`, Conv2 `6/6`, and Conv3 `8/8`. Existing trained-bias EqProp
runs and Conv1/Conv2 EqProp runs at `8/8` are not paper-ready under that rule.

## Historical Material

Hard-sigmoid, Hopfield/EqProp, older optimizer diagnostics, and legacy sweep
instructions are preserved under
[`archive/out_of_scope_20260729/`](archive/out_of_scope_20260729/). Existing
older archives and snapshots remain provenance. If historical material
conflicts with this active index, this index and its linked active protocols
win.
