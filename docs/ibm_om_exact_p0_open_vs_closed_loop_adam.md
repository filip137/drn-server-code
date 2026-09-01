# Exact-P0 open-loop versus incremental P&V Adam

## Executive result

Starting from the exact same persistent target-array state, incremental
apparent-feedback program-and-verify (P&V) reached **93.34%** MNIST test
accuracy, compared with **92.78%** for the existing stochastic open-loop pulse
writer. The P&V policy therefore gained **0.56 percentage points**, or 56
additional correct examples out of 10,000.

The gain was not free. P&V issued **303,231** pulses instead of **170,931**
(**1.774x**), caused **247,243** rather than **141,515** effective state-change
events (**1.747x**), and added **10,314** vectorized verify calls. This result
supports P&V as a small final-accuracy refinement under this simulator, not as
the main explanation for recovery: endpoint-specific Adam already recovered
the network from 44.45% to 92.78% with open-loop pulses.

The comparison is exploratory and mechanism-oriented. It uses one previously
inspected target array identity and one exact deployed endpoint. The closed
arm is incremental one-pulse target tracking, not full convergence to every
new Adam target after each minibatch.

## Question and frozen branch point

The experiment asked:

> If target-array retraining starts from exactly the same deployed physical
> state, does apparent-feedback P&V realize the digital Adam updates better
> than stochastic open-loop pulses?

Both arms branched from target assignment 94004, P&V endpoint seed 94301:

- P0 checkpoint SHA-256:
  `299d4800f89f379bd85cb25b824c89ef9fe92313fd82d70e27e7ee3f320bbb04`
- P0 validation accuracy: 44.70%
- P0 test accuracy: 44.45%
- active published-corrupt cells: 21,345 of 158,800
- source HWA winner SHA-256:
  `d976b28789b1799cc6c7b1c0d66aadf4f8d635a5dce56ede5d6f1d76c1e41b72`

The implementation was committed before launch as `2831bbb2` and run through
the public `python -m ebl train` entry point. The manifest nevertheless records
`source.dirty=true`, with dirty-tree hash
`c7debc34bb5139806c7a9d770eeeedb303f3ae551d6e0916fcdf44e29c99e7a1`,
because unrelated pre-existing documentation edits remained in the worktree.
The run is:

`results/mnist-ibm-om-exact-p0-open-vs-closed-loop-adam-exploratory-20260901-v1/20260901T080531.583567Z-12c11951-44a7d43b`

It completed successfully in 369.924 seconds. All seven registered artifacts
matched their declared sizes and SHA-256 hashes.

## Matched protocol

Both arms used:

- the same exact persistent and apparent P0 tensors;
- the same per-cell pulse-plant RNG continuation;
- the same MNIST minibatch order and generator state;
- the same frozen ReLU teacher and output-KL objective;
- the same BPTT physical-conductance gradients for both DRN layers;
- the same digital Adam equations, learning rate `3e-5`, betas `(0.9, 0.999)`,
  epsilon `1e-8`, and three epochs;
- at most one pulse per cell per minibatch and 64 additional recovery pulses
  per cell;
- persistent full conductance `G = a + 1 = 2x` as the only forward authority;
- no negative conductance, inference-read-noise, retention, or drift model.

The open arm converted each small Adam displacement into an independent
Bernoulli SET/RESET command using the nominal pulse spacing. It made no verify
calls during training.

The closed arm kept a controller-owned desired-target accumulator. It was
initialized from the P0 apparent observation and projected to the public
nonnegative command domain `x in [0,1]`. Every minibatch accumulated the same
Adam displacement, projected the updated target to `[0,1]`, compared it with
cached apparent state, and issued at most one feedback-directed pulse wherever
the error exceeded the P&V tolerance `0.023725`. A post-pulse verify refreshed
the apparent cache. The controller never received persistent state, cell
bounds, support labels, or the corrupt mask, and it never clipped targets to
individual device support.

The target accumulator and global projection are intrinsic co-interventions
with feedback. The result therefore compares complete writer policies; the
0.56-point difference cannot be attributed to verify feedback alone.

## Accuracy and KL

| Arm | Epoch | Train accuracy | Validation accuracy | Validation KL | Issued pulses, cumulative |
|---|---:|---:|---:|---:|---:|
| Open-loop | 1 | 88.07% | 90.60% | 0.23618 | 56,068 |
| Open-loop | 2 | 92.12% | 91.60% | 0.19993 | 113,156 |
| Open-loop | 3 | 92.87% | 92.20% | 0.18600 | 170,931 |
| Incremental P&V | 1 | 74.08% | 88.24% | 0.36554 | 157,234 |
| Incremental P&V | 2 | 90.76% | 91.54% | 0.24130 | 240,003 |
| Incremental P&V | 3 | 93.18% | 92.70% | 0.18168 | 303,231 |

Both arms selected epoch 3 using validation accuracy, then KL, then the earlier
epoch. Both selections and selected-checkpoint validation replays completed
before terminal test traversal.

| Terminal readout | P0 | Open-loop | Incremental P&V | P&V minus open |
|---|---:|---:|---:|---:|
| Test accuracy | 44.45% | 92.78% | **93.34%** | **+0.56 pp** |
| Correct / 10,000 | 4,445 | 9,278 | **9,334** | **+56** |
| Test KL | 3.20612 | 0.16868 | **0.14869** | **-0.01999** |
| Teacher agreement | -- | 93.50% | **94.03%** | **+0.53 pp** |

The open arm reproduced the prior validation predictions, KL values, attempted
and effective pulse counts, data-order hashes, selected epoch, and terminal
test prediction exactly. This is an exact replay rather than a comparison to a
loosely similar historical run.

## Programming and verify cost

| Cost or controller diagnostic | Open-loop | Incremental P&V | Ratio / difference |
|---|---:|---:|---:|
| Issued pulses | 170,931 | 303,231 | 1.774x |
| Effective state-change events | 141,515 | 247,243 | 1.747x |
| Vectorized full-port verify calls | 0 | 10,314 | +10,314 |
| Conceptual post-pulse cell observations | 0 | 303,231 | +303,231 |
| Commands targeting corrupt cells | 23,083 | 53,241 | +30,158 |
| SET pulses | 85,739 | 77,814 | -7,925 |
| RESET pulses | 85,192 | 225,417 | +140,225 |
| Direction reversals | not tracked for open | 87,371 | -- |
| Maximum recovery pulses per cell | 10 | 64 | +54 |

The vectorized simulator returns a full apparent array per API verify call,
although only pulsed coordinates are conceptually consumed. If an
implementation required reading every cell on every call, 10,314 calls would
correspond to 1,637,863,200 cell-value observations. If it supported selective
sensing, the relevant count would instead be the 303,231 pulsed-cell
observations. The present model does not justify an energy or latency claim.

P&V required 132,300 extra pulses for 56 additional correct test examples,
about 2,363 extra pulses per additional correct example. This is only a
descriptive ratio for this endpoint, not a general efficiency law.

## What the controller did

### P&V concentrated updates but repeatedly corrected them

At the selected endpoints:

- open-loop training left 86,534 cells different from P0;
- incremental P&V left only 27,098 cells different from P0;
- open-loop net raw-`a` displacement from P0 had RMS 0.20337;
- incremental P&V net displacement had RMS 0.05622.

Thus, P&V reached slightly better task accuracy with a substantially smaller
net physical change, but it spent more pulse events repeatedly moving a much
smaller subset. This is consistent with targeted accumulation of small Adam
commands plus feedback correction, while stochastic coincidence distributes
many one-off changes more broadly.

The pulse-count tensors tell the same story: closed loop touched 32,390 cells,
whereas open loop touched 104,562. Among commands sent to healthy cells, 98.90%
of closed-loop commands changed persistent state versus 95.72% open-loop. The
closed writer was therefore more directionally effective on programmable
cells, even though its overall issued-to-effective fraction was slightly worse
because it sent disproportionately many commands to immutable faults.

The closed controller issued 225,417 RESET versus 77,814 SET pulses and
recorded 87,371 cross-minibatch direction reversals. Those counts show that the
extra cost was not a monotonic program-to-target trajectory; it included
correction and/or changing target direction. The artifacts do not separately
identify overshoot as the cause of a reversal.

### Permanent faults consumed feedback effort

The controller was deliberately blind to the 21,345 corrupt devices. Of the
303,231 closed-loop pulses, 53,241 targeted corrupt cells and could not move
their persistent state. At the selected epoch:

- 345 cells had reached the 64-pulse cap;
- 278 of those 345 were corrupt;
- 350 cells still had target debt above tolerance;
- 276 of those 350 were corrupt;
- zero corrupt persistent cells moved.

Feedback therefore made permanent non-reachability visible as repeated target
debt, but it did not provide remapping or fault localization. A defect-aware
mapper or learner could avoid some of this cost, but that would be a different
intervention because the present learner was not allowed to see the mask.

### Apparent acceptance still did not imply persistent accuracy

For the closed arm's selected digital targets:

- apparent values were within the verify tolerance for 99.780% of cells;
- persistent values were within tolerance for only 34.890%;
- persistent target MAE was 0.04628 raw `x`;
- persistent target RMSE was 0.06009 raw `x`.

This reproduces the earlier controller-observation diagnosis. The model's held
apparent post-write value can make a cell look accepted even when its
authoritative persistent state is far from the desired target. The network can
still perform well because Adam learns a task-effective physical state; high
classification accuracy must not be interpreted as faithful conductance
realization.

### Projection was material but separate from corrupt-cell debt

The initial apparent target projected 9,941 healthy cells into `[0,1]`, although
only five initial corrections exceeded the verify tolerance. During training,
projection touched 12,964 unique healthy cells across 15,913,449 cell-events.
The accumulated lost absolute target displacement was about 146.42 raw-`x`
across both layers. No corrupt cell was part of the update-target projection
set, so the repeated corrupt-cell debt and the global command-domain projection
are distinct mechanisms.

## Interpretation

The answer to the narrow question is **yes, but only slightly**: in this exact
matched endpoint, incremental P&V improves test accuracy by 0.56 points and
reduces test KL. It is not more pulse-efficient and it introduces a large
verify burden.

The dominant recovery mechanism remains endpoint-specific gradient-based
adaptation. Open-loop Adam already closes 48.33 of the 55.55 percentage-point
gap between P0 and perfect accuracy, while P&V closes 48.89 points. P&V changes
the last 0.56 points; it does not explain the roughly 48-point recovery shared
by both arms.

The learning curves also do not establish convergence. Both arms selected the
third and final epoch, and closed validation improved by 1.16 points from epoch
2 to epoch 3. Longer training might change the ranking, but closed P&V already
has cap-bound cells and accumulating unresolved debt. Extending the horizon
would need a new predeclared pulse-budget decision rather than silently
resetting per-cell caps.

## Claim boundary

This run supports only the following statement:

> On one previously inspected simulated IBM-OM target endpoint, a projected
> digital-target incremental one-pulse P&V Adam policy reached 93.34% test
> accuracy versus 92.78% for a matched stochastic open-loop pulse policy, at
> 1.77 times the pulse count and with 10,314 vectorized verify calls.

It does not establish:

- superiority across independent array identities or write endpoints;
- full-convergence P&V after every minibatch;
- verify feedback as the sole causal intervention;
- autonomous or fully on-chip learning;
- faithful reconstruction of the requested conductances;
- measured-device energy, latency, retention, drift, or endurance behavior.

The plant is a normalized, model-based AIHWKit 1.1.0 IBM optimized-material
preset with an analyst Winsorization intervention and published-rate simulated
corrupt devices. Digital teacher logits, BPTT gradients, Adam moments, and the
closed target accumulator remain digital/off-array.

## Recommended next decision

The current result does not justify replacing stochastic pulse training with
P&V solely for accuracy. A practical decision requires a declared cost model
for selective versus full-array verify and replication across independent
target identities. The smallest useful follow-up is:

1. freeze the same two writer policies and three-epoch budget;
2. run them on at least three new independent target identities, one
   predeclared endpoint each;
3. report paired accuracy, pulses, vector verify calls, corrupt-targeted cost,
   cap debt, and persistent target error;
4. only then decide whether the approximately half-point gain is systematic.

If the goal is instead an accuracy ceiling, test a separately labelled
full-convergence or multi-pulse P&V controller. It must use persistent
per-cell remaining budgets and must not reset a 64-pulse allowance on every
minibatch.
