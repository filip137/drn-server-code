# Apparent-state Adam recovery from P&V

This exploratory experiment compares recovery from exact saved
program-and-verify (P&V) states for two IBM-OM-based device models:

- the older Winsorized normalized model with `G=clamp(a_apparent+1, 0, 2)`;
- the Figure-6-informed independent endpoint model with
  `G=G_RESET+clamp(p_apparent, 0, 1)*(G_SET-G_RESET)`.

It does not rerun programming. Each arm loads the saved recovery P0, including
both its persistent state and its held post-write apparent observation. For the
Figure-6 checkpoints this is `recovery.initial_plant_state`, never the selected
state produced by the earlier recovery implementation.

## Frozen arm matrix

The main comparison is `2 device models x 2 P&V targets x 2 corruption
states`, for eight arms total. The targets are direct signed-weight
initialization and off-chip HWA, both after P&V. The corruption states are the
counterfactual repaired population and the literal published-corrupt
population.

Every gradient-producing forward, training metric, validation pass, selection
pass, and final headline evaluation uses the current held apparent state. A
persistent-state forward is evaluated only as a named secondary diagnostic and
cannot affect Adam, pulse selection, or conclusions. Adam has no authoritative
weight shadow: its command is divided by the nominal OM progress step, capped
to a Bernoulli probability, and realized as at most one stochastic SET or RESET
pulse per eligible cell and minibatch. There are no verify reads during
recovery. Published corrupt singleton cells may be addressed, but their
persistent state must remain bitwise immobile.

The old P&V checkpoints reserve 257 per-cell random draws, exactly enough for
their RESET observation and cap-128 programming. Recovery extends only this
deterministic buffer capacity to 385 draws per cell so that 64 two-draw recovery
pulses are representable. The per-cell seeds and saved draw indices are
unchanged, the original stream prefix is bitwise identical, and the extension
consumes no draw or changes persistent/apparent P0 state.

## Adam selection and main run

One common progress-coordinate Adam learning rate is selected from
`{3e-6, 1e-5, 3e-5, 1e-4, 3e-4}`. Each rate gets 512 minibatches on the two
repaired/direct anchors, starting from the exact same P0 and data-order/RNG
state. Selection maximizes macro apparent validation accuracy across the two
models, followed by lower macro apparent KL, fewer applied pulses, and the
lower rate. Test data remain unopened during this screen.

After the rate receipt is frozen, every main arm processes exactly one shuffled
55,000-example epoch (3,438 minibatches at batch size 16), followed by full
validation and test evaluation of apparent primary and persistent secondary
states. Checkpoints preserve the exact plant, Adam moments, pulse-selection
RNG, data-loader RNG endpoints, source hashes, projection counts, saturation,
per-layer pulse counts, probability clipping, pulse caps, and corrupt-cell
attempts.

The runner is
`experiments.mnist_relu_drn.ibm_om_apparent_pv_adam_recovery_experiment`.
Use its `screen` subcommand first, then pass the generated
`common_learning_rate_receipt.json` to each `arm` subcommand. The
`--canary-batches` path is explicitly reduced and cannot consume a frozen
receipt or open test data.

These are hybrid, model-based exploratory results, not measurements of an
autonomous hardware Adam implementation or physical-chip power.

## Executed 2026-09-04 result

The frozen screen selected `1e-5`. The eight main arms then completed one
55,000-example epoch each on `nom-cool-2` and `integnano-akib`. Test accuracy
below is the primary held-apparent state before and after recovery; the last
column is the final persistent-state diagnostic.

| Endpoint model | P&V target | Population | Apparent before | Apparent after | Persistent after |
|---|---|---:|---:|---:|---:|
| old `[0,2]` | direct | repaired | 94.80% | 94.71% | 84.35% |
| old `[0,2]` | direct | corrupt | 39.40% | 84.43% | 78.66% |
| old `[0,2]` | HWA | repaired | 84.33% | 92.87% | 90.16% |
| old `[0,2]` | HWA | corrupt | 60.45% | 88.73% | 88.43% |
| Figure-6 OM | direct | repaired | 95.64% | 96.11% | 94.57% |
| Figure-6 OM | direct | corrupt | 48.28% | 82.48% | 78.98% |
| Figure-6 OM | HWA | repaired | 89.97% | 94.19% | 92.13% |
| Figure-6 OM | HWA | corrupt | 49.55% | 82.97% | 80.16% |

Seven of eight apparent-state accuracies improved; the exception moved only
`-0.09` percentage points from an already-strong repaired/direct old-OM P0.
All eight final apparent accuracies exceed their persistent diagnostics. Across
the matrix, 161,303 recovery pulses were applied, with zero verify reads, zero
probability-clipped cell events, zero cells reaching the recovery pulse cap,
and verified persistent immobility of every published-corrupt singleton.

The strict aggregate is stored under
`simulation_results/ibm_om_apparent_pv_adam_recovery/aggregate/main_8arm_20260904_v2`.

## Three-epoch higher-rate corrupt Figure-6 follow-up

The two weak Figure-6 corrupt arms were rerun from their same exact saved P&V
P0 states with one combined intervention: three complete shuffled training
epochs, Adam learning rate `3e-5`, and a cumulative recovery cap of 192 pulses
per physical cell. The rate is the next higher member of the already executed
screen grid and matches the recent successful old-OM pulse-Adam control. The
cap is three times the one-epoch budget, so extending training does not silently
tighten the lifetime budget. All other state, data, pulse-model, seed,
apparent-forward, and no-verify-read contracts are unchanged.

| P&V target | Apparent test P0 | Prior 1 ep, `1e-5` | New 3 ep, `3e-5` | Persistent after 3 ep |
|---|---:|---:|---:|---:|
| direct | 48.28% | 82.48% | **89.08%** | 80.55% |
| HWA | 49.55% | 82.97% | **90.83%** | 89.53% |

The apparent validation trajectories after epochs 1, 2, and 3 were
`87.12%, 88.92%, 87.74%` for direct and
`87.02%, 89.24%, 89.94%` for HWA. The final epoch is reported without
test-set checkpoint selection. Relative to the earlier one-epoch runs, final
apparent test accuracy gained `+6.60` points for direct and `+7.86` points for
HWA.

Direct and HWA applied 172,836 and 165,354 recovery pulses, respectively.
Their maximum observed cumulative counts were only 9 and 8 pulses per cell;
there were zero cap hits and zero probability-clipped events. Consequently,
raising the configured cap from 64 to 192 had no numerical effect in these
executions: the same pulse sequence never approached even the old limit.
There were again zero recovery verify reads, no authoritative digital weight
shadow, and bitwise-verified persistent immobility of every published-corrupt
singleton.

This is an exploratory combined intervention, so the additional gain cannot be
uniquely divided between the higher rate and the extra epochs. The inactive
cap can be ruled out as its cause. The direct validation peak at epoch 2 also
suggests that extending beyond two epochs is not uniformly beneficial, whereas
the HWA arm was still improving at epoch 3.

The local direct result is under
`simulation_results/ibm_om_apparent_pv_adam_recovery/extensions/figure6_corrupt__3ep__lr_3em05__cap_192/figure6_independent__direct__corrupt/local_extension_3ep_lr3e5_cap192_20260904`.
The byte-verified Akib HWA copy is under
`simulation_results/ibm_om_apparent_pv_adam_recovery/akib_import/extensions/figure6_corrupt__3ep__lr_3em05__cap_192/figure6_independent__hwa__corrupt/akib_extension_3ep_lr3e5_cap192_20260904`.

### Partial interpretation and next decision

Adam can partially recover the results. The network evaluated here is much
smaller than the network reported in IBM's work. The next decision is whether
we want to use a bigger network or continue with this one.

This interpretation remains exploratory and model-based: it describes the
recovery observed in these matched simulated OM-pulse arms, not measured
autonomous on-chip Adam behavior.
