# IBM-OM 784-256-10 standard-crossbar versus DRN: production results and retuned recovery diagnostic

- **Report date:** 2026-09-05
- **Task:** MNIST, bias-free logical `784-256-10`
- **Primary inference state:** current held post-write apparent state
- **Secondary diagnostic:** hidden persistent state on the same examples
- **Device evidence:** model-based IBM optimized-material ReRAM simulations,
  not measured-array experiments
- **Scientific status:** the crossbar production and exact-P0 diagnostic are
  artifact-verified; the DRN campaign is complete
  `exploratory_noncanonical` evidence. The exact-P0 diagnostic is
  `ready_for_review` and has not yet received a final human outcome label

## Executive summary

The completed DRN endpoints are numerically higher than the endpoints from
the original standard-crossbar production protocol. That raw difference is
descriptive, not a causal architecture comparison. The two campaigns use
different teacher checkpoints, physical representations, device-forward
models, training objectives, learning-rate screens, checkpoint rules, and
replication structures.

The evidence does not support a broken Adam update as the main explanation
for the largest apparent discrepancy. It localizes the observed collapse to
the tested schedule and selection policy. The original crossbar study
selected one `3e-3` learning rate jointly for HWA
recovery and training from scratch, forced the tenth epoch, and did not permit
the unchanged programmed state to win. From the already accurate HWA state,
that protocol touched about 95% of cells in its first epoch, applied roughly
2.3 million pulses immediately, and reduced apparent accuracy from about 97%
to about 87%. It eventually applied 11--12 million pulses and ended near 93%.

The exact-P0 follow-up screened much smaller rates, separated supervised
cross-entropy from teacher KL, made epoch zero eligible, and selected by held-
apparent validation accuracy. It found:

- healthy HWA P0: `97.12% -> 97.38%` with CE at `1e-5`, epoch 1;
- corrupted HWA P0: `96.26% -> 96.82%` with CE at `3e-5`, epoch 1;
- healthy HWA P0: `97.12% -> 97.30%` with teacher KL at `1e-5`;
- corrupted HWA P0: `96.26% -> 96.66%` with teacher KL at `3e-5`.

The CE arms produced the highest top-1 validation accuracy. The teacher-KL
arms reduced teacher-distribution KL more consistently. None of the selected
runs hit the 640-pulse cap; the four state/objective-winning checkpoints used
only 8,554--30,569 pulses in total.

The corrupted crossbar recovery gain, `+0.56` percentage points on the
development validation set, is of the same order as the DRN campaign's
`+0.437`-point aggregate test gain. This demonstrates improvement on this
exact development P0 under conservative PulseAdam. It does not establish
recovery on unseen assignments or test data, nor residual crossbar--DRN
parity, because the crossbar result is a selected one-assignment validation
result and the DRN result is a three-array test aggregate.

## Evidence sets

This report keeps three evidence sets separate.

1. **Standard-crossbar production:** four device assignments and four P&V
   writes per assignment. The primary summary first averages writes within an
   assignment and then summarizes the four assignment means. It contains 160
   completed stage runs and eight headline outcomes with 16 realizations each.
2. **DRN exploratory campaign:** three assignments and three writes per
   assignment, summarized as the mean and range of three array means. It
   contains 72 final records. Its evidence tier is explicitly
   `exploratory_noncanonical`.
3. **Exact-P0 crossbar canary:** one exact healthy and one exact corrupted P0
   from the prior tuning assignment/write, evaluated on all 5,000 validation
   examples. It contains 24 nonzero-LR Adam arms, four LR-zero recovery
   controls, and two no-write fresh-apparent controls. It did not open the
   official test set.
4. **Contextual native-OM three-layer feasibility run:** one
   `784-256-128-10` scratch run used only to test whether the same PulseAdam
   mechanics can learn at all. It is neither part of the matched production
   comparison nor replication evidence for the `784-256-10` result.

Ranges in this report are full ranges across assignment/array means. They are
not confidence intervals. Pooled programming writes are secondary because
writes on the same assignment are not independent arrays.

## Comparison contract and known mismatches

| Field | Standard crossbar | DRN |
|---|---|---|
| Logical network | bias-free `784-256-10`, digital ReLU | logical `784-256-10`, perfect-diode equilibrium network |
| Stored/physical layout | `(392,256)`, `(392,256)`, `(256,10)` | rail-encoded `1568-512-20` |
| Logical weights | 203,264 | 203,264 |
| Programmable states | 203,264 active OM states, plus fitted fixed-reference parameters | 813,056 active conductance states, four per logical weight |
| Signed representation | native effective `q=a-r` | four-cell dual rail |
| Forward physics | standard MVM, digital ReLU, standard MVM; no passive conductance-sum denominator | perfect-diode solver with conductance-sum loading |
| Device realization | native sampled OM effective-state plant | analyst-defined Figure-6 endpoints with OM pulse dynamics |
| Source teacher | SHA-256 `7c1f826c...a4565`, test `97.45%` | SHA-256 `a22be1ee...ba1d4`, test `97.92%` |
| Production replication | 4 assignments x 4 writes | 3 assignments x 3 writes |
| Original on-chip objective | supervised CE | teacher KL |
| Original HWA recovery LR | `3e-3`, jointly chosen with scratch | `1e-4`, selected separately from scratch |
| Checkpoint rule | fixed epoch 10 | best apparent validation epoch; all HWA runs chose epoch 1 |
| Update implementation | apparent-forward gradient, persistent pulse write, digital Adam moments | apparent-forward gradient, persistent pulse write, digital Adam moments |

The reference values in the standard-crossbar model are parameters of the
AIHWKit `SoftBoundsReferenceDevice` fit. This report does not reinterpret one
active state plus one abstract reference value as a known fabricated-device
count. Likewise, the DRN's larger stored tensors are rail encoding, not a
larger logical neural network.

Both recovery implementations are digitally assisted, open-loop pulse-
mediated Adam. Gradients and Adam moments are digital. The word “on-chip” in
this report denotes the chronological same-array update of the named deployed
state; it does not claim an autonomous local-learning circuit.

## Crossbar production and DRN exploratory test results

The table reports held-apparent test accuracy and teacher KL. KL is measured
against each campaign's own teacher, so it is meaningful within an
architecture but is not a matched cross-architecture distance.

| Stage | DRN accuracy [array range] | DRN KL | Crossbar accuracy [assignment range] | Crossbar KL | Crossbar minus DRN |
|---|---:|---:|---:|---:|---:|
| Direct P&V | 97.513% [97.353, 97.643] | 0.0328 | 97.039% [96.965, 97.140] | 0.0099 | -0.475 pp |
| Direct P&V + corruption | 96.279% [96.107, 96.560] | 0.0882 | 94.896% [93.698, 95.585] | 0.0805 | -1.383 pp |
| HWA + P&V | 97.716% [97.517, 97.817] | 0.0303 | 97.124% [97.063, 97.228] | 0.0253 | -0.591 pp |
| HWA + P&V + corruption | 97.046% [96.823, 97.353] | 0.0587 | 96.076% [95.788, 96.208] | 0.0568 | -0.969 pp |
| HWA + P&V + Adam | 97.719% [97.700, 97.733] | 0.0278 | 92.937% [92.728, 93.028] | 0.1899 | -4.782 pp |
| HWA + P&V + corruption + Adam | 97.482% [97.457, 97.507] | 0.0373 | 92.604% [92.310, 92.800] | 0.1950 | -4.878 pp |
| Scratch + Adam, clean | 95.533% [95.500, 95.583] | 0.1183 | 85.924% [85.225, 86.583] | 0.4568 | -9.609 pp |
| Scratch + Adam, corrupted | 95.599% [95.503, 95.660] | 0.1115 | 85.015% [84.310, 86.033] | 0.4798 | -10.584 pp |

### Persistent-state diagnostic

Persistent-state accuracy is secondary and never selected a checkpoint or
drove an optimizer update.

| Stage | DRN persistent | Crossbar persistent |
|---|---:|---:|
| Direct P&V | 97.308% | 90.835% |
| Direct P&V + corruption | 95.932% | 85.425% |
| HWA + P&V | 97.618% | 93.718% |
| HWA + P&V + corruption | 96.880% | 91.258% |
| HWA + P&V + Adam | 97.397% | 90.065% |
| HWA + P&V + corruption + Adam | 97.173% | 90.321% |
| Scratch + Adam, clean | 95.436% | 80.447% |
| Scratch + Adam, corrupted | 95.484% | 79.634% |

### Within-architecture effects

The clean direct deployment loss is nearly identical in the two campaigns:

- DRN: `97.92% -> 97.513%`, approximately `-0.407` points;
- crossbar: `97.45% -> 97.039%`, approximately `-0.411` points.

Consequently, the `0.475`-point clean direct endpoint gap almost exactly
matches the `0.47`-point difference between the two source teachers. It must
not be attributed to architecture.

Original HWA recovery changed the two architectures very differently:

| Recovery contrast | Accuracy change | KL change |
|---|---:|---:|
| DRN HWA clean | +0.003 pp | -0.00249 |
| DRN HWA corrupted | +0.437 pp | -0.02140 |
| Crossbar HWA clean, original schedule | -4.188 pp | +0.16468 |
| Crossbar HWA corrupted, original schedule | -3.473 pp | +0.13822 |

This table identifies a recovery-protocol discrepancy. On its own, it does
not identify an architecture limitation.

## Why the original crossbar Adam endpoint failed

### 1. One schedule was forced across two different tasks

The original tuning score averaged final CE over healthy and corrupted HWA
recovery and healthy and corrupted scratch learning. Scratch starts needed
large updates. HWA P0 was already near 97% and needed either no update or a
very conservative update. A single winner could not represent both regimes.

The grid included only `3e-4`, `1e-3`, and `3e-3`, paired with cap 128 or no
cap. It omitted `0`, `3e-6`, `1e-5`, `3e-5`, and `1e-4`. The selected winner
was `3e-3` with cap 128 because it minimized mean fixed-final CE across the
four starts. Epoch zero was not eligible.

### 2. Adam normalization made the first writes dense

At the first Adam step, bias-corrected `m/sqrt(v)` is approximately the sign
of a nonzero gradient. With nominal OM `dw_min_q=0.0949`, the old `3e-3`
rate corresponds to a nominal Bernoulli pulse probability near `3.16%` per
nonzero normalized step. Small gradient magnitude does not by itself make
that first normalized command small.

The old exact-P0 tuning trajectories therefore refreshed almost the whole
array in one epoch:

| Old exact-P0 run | Epoch-1 cells touched | Epoch-1 pulses | Epoch-1 apparent accuracy | Epoch-10 pulses | Epoch-10 apparent accuracy | Cap blocks |
|---|---:|---:|---:|---:|---:|---:|
| Healthy | 94.784% | 2.30M | 86.92% | 11.125M | 92.60% | 2.066M |
| Corrupted | 95.632% | 2.38M | 86.04% | 12.161M | 92.72% | 2.438M |

The cap did not cause the initial collapse. It began binding only after the
damage was already visible and was protective: removing it produced an even
worse endpoint. The blocked commands are evidence of update overdose, not an
argument to raise the cap.

### 3. The P&V-conditioned apparent state was fragile

P&V accepts a cell using a noisy post-write apparent verification value. The
accepted P0 is therefore conditioned on a favorable held apparent draw. A
subsequent pulse updates hidden persistent state and redraws the apparent
state of that touched cell. Dense writing can destroy the favorable P&V
snapshot even when the persistent state changes little.

On the exact HWA tuning P0, apparent/persistent accuracy was `97.12/92.10%`
when healthy and `96.26/90.38%` after corruption. At the smallest old rate,
`3e-4`, the first healthy epoch changed persistent accuracy by only about
`-0.12` points while apparent accuracy fell by `-3.50` points. That is
consistent with apparent-state refresh dominating the early accuracy loss.
At `3e-3`, persistent accuracy also collapsed, showing pulse overdose on top
of the same apparent-state sensitivity.

### 4. Fixed-final selection hid the correct answer

Every original crossbar HWA production realization was worse than its P0 at
every trained epoch. Post hoc, an accuracy-based rule with P0 eligible would
have selected P0 in all 16 healthy and all 16 corrupted realizations. The
predeclared fixed-final endpoints remain the valid result for the original
protocol.

The DRN headline also depends on checkpoint policy. All 18 DRN HWA recovery
runs selected epoch 1. Their mean epoch-10 validation accuracies fell to about
`96.13%` healthy and `95.79%` corrupted. Thus the DRN's `97.72/97.48%`
headlines largely preserve a high P0; they are not sustained tenth-epoch
endpoints.

## Exact-P0 low-rate recovery canary

The canary reused the exact teacher, HWA master, healthy P0, and corrupted P0
from the original crossbar tuning assignment. It changed only the declared
diagnostic recovery protocol:

- objectives: supervised CE and teacher KL;
- rates: `0`, `3e-6`, `1e-5`, `3e-5`, `1e-4`, `2e-4`, `3e-4`;
- three complete epochs of the same 55,000-example ordered stream;
- cap 640;
- selection: highest held-apparent validation accuracy, then lower value of
  the declared objective, then earlier epoch, with epoch zero eligible;
- held-apparent state for every gradient, metric, validation, and checkpoint;
- persistent state only as a same-example secondary diagnostic.

### Complete selection surface

Each cell reports selected held-apparent validation accuracy and epoch. `P0`
means the unchanged epoch-zero state won.

| LR | Healthy CE | Healthy KL | Corrupted CE | Corrupted KL |
|---:|---:|---:|---:|---:|
| 0 | 97.12% P0 | 97.12% P0 | 96.26% P0 | 96.26% P0 |
| `3e-6` | 97.20% e2 | 97.16% e3 | 96.66% e2 | 96.56% e3 |
| `1e-5` | **97.38% e1** | **97.30% e1** | **96.82% e3** | 96.52% e1 |
| `3e-5` | 97.12% P0 | 97.12% P0 | **96.82% e1** | **96.66% e1** |
| `1e-4` | 97.12% P0 | 97.12% P0 | 96.26% P0 | 96.26% P0 |
| `2e-4` | 97.12% P0 | 97.12% P0 | 96.26% P0 | 96.26% P0 |
| `3e-4` | 97.12% P0 | 97.12% P0 | 96.26% P0 | 96.26% P0 |

Only 10 of the 28 recovery-grid arms selected a nonzero-update checkpoint;
four of those 28 arms were explicit LR-zero controls. Every arm at or above
`1e-4`, including the explicit `2e-4` native-`q` coordinate translation of
the DRN HWA rate, selected P0. Raw learning rates are not portable across the
two device coordinates or physical topologies.

### Best checkpoints

| Start and objective | Selected LR/epoch | Apparent accuracy | Persistent accuracy | Apparent CE | Apparent KL | Pulses / changed cells | Max pulses/cell | Blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Healthy, CE | `1e-5`, e1 | 97.12 -> **97.38%** | 92.10 -> 92.58% | 0.11665 -> 0.10760 | 0.02724 -> 0.03053 | 8,554 / 8,358 | 3 | 0 |
| Healthy, teacher KL | `1e-5`, e1 | 97.12 -> **97.30%** | 92.10 -> 91.60% | 0.11665 -> 0.11322 | 0.02724 -> **0.02533** | 9,207 / 8,989 | 3 | 0 |
| Corrupted, CE | `3e-5`, e1 | 96.26 -> **96.82%** | 90.38 -> 91.58% | 0.14321 -> 0.11583 | 0.06030 -> 0.05394 | 30,023 / 27,690 | 4 | 0 |
| Corrupted, teacher KL | `3e-5`, e1 | 96.26 -> **96.66%** | 90.38 -> 91.72% | 0.14321 -> 0.12328 | 0.06030 -> **0.04207** | 30,569 / 28,187 | 4 | 0 |

The old exact-P0 tuning trajectories used about `1,300.6x` and `405.1x` more
pulses, respectively, than the new best CE checkpoints. The retuned result is
therefore a sparse early correction, not a slower route to the old high-pulse
endpoint. These ratios are not pulse-count summaries of the four-assignment
production aggregate.

### CE versus teacher KL

CE gave the highest selected top-1 accuracy for both starts. Teacher KL gave
the larger reduction in distributional divergence:

- healthy CE: apparent KL changed by `+0.003293`;
- healthy teacher KL: apparent KL changed by `-0.001909`;
- corrupted CE: apparent KL changed by `-0.006356`;
- corrupted teacher KL: apparent KL changed by `-0.018228`.

Teacher KL is therefore useful when matching the teacher distribution is the
objective, but it is not the missing explanation for the original accuracy
failure. Learning rate, pulse exposure, and checkpoint eligibility dominate
that failure.

## No-write fresh-apparent counterfactual

The two fresh-apparent controls kept hidden persistent state and the original
held apparent state unchanged. They drew four matched full-array post-write
noise vectors and evaluated those counterfactual apparent states without
commanding a write.

| Start | Held P0 apparent accuracy | Persistent accuracy | Fresh apparent mean [range] | Fresh mean KL | Writes |
|---|---:|---:|---:|---:|---:|
| Healthy | 97.12% | 92.10% | 88.265% [86.98, 89.58] | 0.36690 | 0 |
| Corrupted | 96.26% | 90.38% | 82.765% [81.10, 84.38] | 0.49972 | 0 |

This is not inference-read noise, retention, aging, or a prediction of a
second physical read. It is a deliberately labelled counterfactual redraw of
the modeled post-write apparent term. It shows that the accepted P&V snapshot
is highly sensitive to refreshing that term and provides a mechanism
consistent with why dense first-epoch writing is especially destructive.

## DRN comparison after correcting the crossbar schedule

| Exact HWA condition | DRN exploratory test aggregate | Retuned crossbar validation |
|---|---:|---:|
| Healthy P&V before Adam | 97.716% | 97.12% |
| Healthy selected after Adam | 97.719% | 97.38% |
| Within-architecture gain | +0.003 pp | +0.26 pp |
| Corrupted P&V before Adam | 97.046% | 96.26% |
| Corrupted selected after Adam | 97.482% | 96.82% |
| Within-architecture gain | +0.437 pp | +0.56 pp |

The comparison supports two narrow statements:

1. in this exact-P0 canary, conservative PulseAdam improved the selected
   native-OM standard-crossbar validation metric; and
2. the original approximately 4.8-point DRN--crossbar recovery endpoint gap
   between the two completed protocols was dominated by crossbar protocol
   confounding.

It does not support an architecture tie or crossbar superiority. The residual
values mix DRN test evidence across three arrays with a crossbar validation
winner from one exact assignment/write. Most of the absolute separation was
already present before recovery, and the different teachers contribute about
0.47 points before any device comparison.

## Why the two-hidden-layer native-OM run is not contradictory

The earlier `784-256-128-10` native-OM feasibility run trained from scratch.
It used sigmoid hidden units, biases, 235,146 active `q` states, LR `1e-3`,
batch size 64, five epochs, and 4,690 optimizer steps. It applied 1,659,981
pulses, or about 7.06 per state; its observed maximum was 30 and no cap bound.
Apparent accuracy rose from `9.58%` to `94.67%` (`94.98%` at epoch 4), with a
`92.72%` persistent diagnostic.

That result shows the same pulse-Adam sign and update mechanics can learn. It
does not validate the old HWA recovery schedule. A random scratch state has
no favorable 97% P&V-conditioned snapshot to preserve, so broad writing is
useful. The old crossbar HWA protocol instead used 34,380 steps and about
55--60 pulses per state, while starting close to an optimum.

The scratch rows in the production table are likewise endpoints of two
different completed protocols, not evidence that the DRN architecture is
intrinsically easier to train. The crossbar scratch arm inherited the pooled
LR/fixed-final selection problem.

## Scientific interpretation

The evidence supports the following diagnosis:

- there is no observed sign, state-routing, or Adam-equation failure;
- the old HWA endpoint is a negative result for dense, open-loop Bernoulli
  PulseAdam with `3e-3`, not for Adam in general;
- this high-quality verification-conditioned apparent P0 benefited from
  sparse, conservative correction and a no-update option;
- the cap was protective and should not be raised to cure the old result;
- CE gave the highest selected validation accuracy in this canary, while
  teacher KL better preserved the teacher distribution;
- HWA plus P&V is already strong, so the data do not establish a universal
  need for on-chip recovery; and
- the DRN's higher absolute endpoint cannot be assigned to architecture until
  the teacher, plant, objective, checkpoint rule, and evaluation cohort are
  matched.

The exact-P0 study still requires a human outcome classification. Its evidence
shows low-rate recoverability on the development P0, while teacher KL was not
the unique accuracy solution and the selected gains remain one-realization
development results.

## Limitations and claim boundary

- The crossbar canary selected LR and epoch on the same one-assignment,
  5,000-example validation evidence reported here. Its `+0.26/+0.56` gains
  are development estimates subject to winner's optimism.
- The canary did not evaluate official test data. A DRN-test minus
  crossbar-validation residual is not a valid generalization estimate.
- The two campaigns use different exact teachers. They are not two physical
  implementations of one frozen source tensor.
- The DRN uses a Figure-6 endpoint-marginal plus OM-pulse hybrid rather than
  the native `q=a-r` plant used by the crossbar.
- The topology changes physical state count and loading. The DRN has four
  active conductance cells per logical weight and a passive conductance-sum
  denominator; the standard crossbar does not.
- The post-P&V fault states differ: the crossbar replays each published
  companion's collapsed effective state `q=a_corrupt-r_corrupt`, whereas the
  DRN forces persistent conductance to RESET. The common 13.48% mask
  probability does not make these the same physical failure model; topology
  also changes the impact per logical weight. See the
  [corrupt-device audit](../../reviews/ibm_om_2026-09-04_05/corruption_review.md).
- All results are simulations of fitted or normalized device models. They are
  not measured-device evidence and do not support fabricated-array yield,
  retention, endurance, energy, physical power, IR-drop, ADC/DAC, or
  autonomous on-chip-learning claims.
- The held apparent state is the declared primary forward state. Persistent
  results diagnose the hidden substrate and must not replace the headline.
- The fresh-apparent intervention is a counterfactual post-write-noise redraw,
  not an inference-read-noise experiment.

## Candidate confirmatory study design

If the human closeout selects a confirmatory comparison as the next study, it
should be predeclared with:

1. one exact teacher checkpoint and hash for both architectures;
2. the same MNIST train/validation/test split, preprocessing, minibatch order,
   objective, and selection hierarchy;
3. explicit mapping receipts and architecture-specific physical counts;
4. the same device-law/endpoint source where possible, or a predeclared
   architecture-by-plant factorial where it is not;
5. multiple untouched assignments and multiple writes, averaging writes
   within assignment before reporting assignment uncertainty;
6. development assignments dedicated to LR/epoch selection, followed by a
   frozen schedule before target arrays and official test are opened;
7. separate HWA and scratch rate selection, with epoch zero eligible for HWA;
8. coordinate-aware rate grids rather than equating raw `p` and `q` LRs;
9. one named exact P0 per recovery fork, with no remapping or reprogramming;
10. apparent-primary and persistent-secondary accuracy, CE, KL, agreement,
    pulse, touched-cell, and cap telemetry normalized per programmable state;
11. frozen-master fresh-array deployment through every target array's own
    bounds/codebook, kept separate from same-array recovery and recovered-
    state transfer; and
12. official test evaluation only after every schedule and checkpoint rule is
    frozen.

For a confirmatory standard-crossbar target-array study, freeze CE `1e-5`,
epoch 1 for healthy recovery and CE `3e-5`, epoch 1 for corrupted recovery
now, and retain the unchanged P0 as a paired control. Do not reselect LR or
epoch on the target assignments or official test. If broader development
confirmation is desired first, declare it as a separate selection study and
freeze its winner before opening a second untouched target study.

## Reproducibility and artifact ledger

### Standard-crossbar production

- Teacher study:
  [`mnist-ibm-om-crossbar-784-256-teacher-20260904-v1.json`](../studies/mnist-ibm-om-crossbar-784-256-teacher-20260904-v1.json)
- Tuning study:
  [`mnist-ibm-om-crossbar-784-256-hwa-adam-tuning-20260904-v1.json`](../studies/mnist-ibm-om-crossbar-784-256-hwa-adam-tuning-20260904-v1.json)
- Production study:
  [`mnist-ibm-om-crossbar-784-256-production-20260904-v1.json`](../studies/mnist-ibm-om-crossbar-784-256-production-20260904-v1.json)
- Scientific execution commit:
  `1f4ef1b8d05b2b0613bf4e8bec4cce799c034cea`
- Teacher checkpoint SHA-256:
  `7c1f826c6da5e0a8b18024b49305a9c61230a93db28253a2e8ad427d788a4565`
- HWA master SHA-256:
  `b7daa18f5b148114cebc0c6b212099c236b0cebdd588063e72d9e9c127dc6233`
- Production assignment/pooled summary SHA-256:
  `2f3b3f9ef84e7320231b2ca1ed61f4b4feec54cea9b3b43163f05f1575956116`
- Production workflow summary SHA-256:
  `c85769fb00794e440b0a982ee247f9aec3160a945523adaf892209ee5ed4d6e9`
- Canonical Akib production study root:
  `/home/filiposana/staged/ibm_om_crossbar_784_256_1f4ef1b8/results/mnist-ibm-om-crossbar-784-256-production-20260904-v1`
- Canonical production files:
  `analysis/production_assignment_and_pooled_summary.json` and
  `analysis/summary.json`

### Exact-P0 canary

- Study ID:
  `mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1`
- Tracked plan at the scientific execution commit:
  `studies/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1.json`
- Scientific execution commit:
  `18dffb935516238eb69ab120eda7360a9f6c986b`
- Artifact/provenance analysis commit:
  `207a8f6b25105b6ceaee85f00bf5f5826b9821b2`
- Generic artifact-verified summary SHA-256:
  `3bb07a800e95e156337df779f7ef0727928acded7cff43feafd9bff48655474b`
- Scientific summary SHA-256:
  `1ed37d4a41dca2ca97731c8b47ee2340e9ed551699feca996122dfbc3078af3c`
- Canonical summary files:
  `analysis/summary.json` and `analysis/hwa_recovery_canary_summary.json`
- Federated collection receipt:
  `analysis/federated_collections/cfbfe28ea602c25876b6e9c46675f6cd5fd0b289493110a97143ecb7beb1b9f9-d786962be233e186.json`,
  SHA-256
  `e89cef14491d3b4dba84ba711efe23462a8de3083afe3203aede9be1bc9b15df`
- Coverage:
  30/30 valid arms, 142 registered artifacts, 5,355,012,134 artifact bytes,
  zero failed/invalid/running/missing/unknown coverage
- Compute:
  local RTX 3090 and Akib RTX 3080, PyTorch `2.5.1+cu121`; exact LR-zero
  cross-host state/prediction parity passed and the largest numeric metric
  difference was `6.5893e-8`
- Canonical Akib study root:
  `/home/filiposana/staged/ibm_om_crossbar_hwa_canary_18dffb93/results/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1`

### DRN exploratory campaign

- Tracked evidence receipt:
  [`drn_figure6_om_784_256_10_postpv_fault_20260905.json`](../studies/references/drn_figure6_om_784_256_10_postpv_fault_20260905.json)
- Campaign root:
  `/home/filip/server_code/.codex/worktrees/ibm-om-cell-aware-quantized/simulation_results/exploratory_noncanonical_figure6_om_784_256_10_postpv_fault`
- Source commit:
  `2b283365063931e9fdd41fbd30d2672e0e600f04`
- Teacher checkpoint SHA-256:
  `a22be1ee0c3f4392dde455a5d87903e88e089d3f281c29a9cbe0aa3f378ba1d4`
- Scientific summary SHA-256:
  `6ecbbb845c14652bf9692799bbc29fe3b1634b14509c0a3ab0731e2f34d3d89e`
- Raw-record artifact SHA-256:
  `ec987d388eaba120fd1f0faa6ff1bd9f9164e51dee0602ebc2017a342a12c605`
- Coverage:
  72/72 final records, three assignments by three writes
- Evidence class:
  `exploratory_noncanonical`, model-based Figure-6 endpoint plus IBM-OM pulse
  hybrid

### Contextual native-OM three-layer run

- Study:
  [`mnist-ibm-om-3fc-adam-short-cuda-20260904-v1.json`](../studies/mnist-ibm-om-3fc-adam-short-cuda-20260904-v1.json)
- Study summary:
  `results/mnist-ibm-om-3fc-adam-short-cuda-20260904-v1/analysis/summary.json`,
  SHA-256
  `89d85cec078ea8de8878281de8f724466a7eaba496994c3cea6d41d523b8fe62`
- Numerical summary artifact SHA-256:
  `d154ad03f6c1861c4346cbc8976628f679dbd2eb437353e401a8debe7dbef8a0`
- Recorded source identity:
  commit `4866202123552e5842bea2db838c29ddf7836e7c`, dirty-state hash
  `0fd0209dd4904adc038522ec7eed1809d349558d2a57aa6e45bae9bcd3f813aa`
- Scope:
  one-seed CUDA feasibility evidence, not a production architecture
  comparison

## Closeout status

The numerical report is complete. The original run artifacts were not
rewritten during federated collection or report preparation. The exact-P0
study is artifact-verified and ready for human scientific review. Per the EBL
study lifecycle, do not create or modify the exact-P0 study's
`analysis/review.json`, `analysis/final.json`, or experimental-manifest entry
until the user confirms its outcome interpretation and the next experiment.
