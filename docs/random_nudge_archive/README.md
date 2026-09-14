# EqProp probing and asymmetry: complete experiment archive

Compiled 14 September 2026 from the experiments performed on
`codex/hopfield-random-nudge-adjoint`, worktree
`/home/filip/server_code_random_nudge`, from 11–14 September. The branch starts
at `9083aa22`; this compilation covers the 25 experiment/analysis commits through
`240b6637`. This is an archive of exploratory evidence, including cancelled
grids and negative results, not a declaration that every proposed study finished.

**The evidence supports continuing learned equilibrium probing.** It trained
successfully on the teacher task, digits, and the constrained untied MNIST
model. Homeostasis improved feedback alignment. The recent Fashion-MNIST
diagnostics expose variance and settling problems under a different protocol;
they do not erase the successful training evidence. Individual feedback cosine,
minibatch gradient cosine, optimizer-step direction, and accuracy are different
measurements and must remain separate.

The current scientific target is independently trainable forward/backward
connections. The earlier fixed-skew and four-known-loop models remain useful
historical controls, with their additional assumptions stated below.

## Reading and preservation

- This document consolidates the questions, coverage, headline results, costs,
  limitations and status of **20 experiment groups**.
- [evidence.json](evidence.json) preserves 79 source snapshots/indexes, 61
  historical launcher records, and the 25-commit history. Small reports and JSON
  receipts are embedded with their original paths and SHA256 hashes. CSV indexes
  retain hashes, columns and row counts. This makes the reported evidence
  inspectable from Git without storing the datasets or model checkpoints.
- Dedicated notes linked below contain the full methods and reproduction
  commands. Raw measurements, plots and checkpoints remain under the linked,
  ignored `simulation_results/` directories in this worktree.
- Compilation checks source presence, parsing, hashes and saved coverage
  receipts. It does **not** rerun training or independently replay checkpoints
  again. Earlier replay checks are identified as historical verification.
- Counts overlap: later reports reuse controls and checkpoints. Do not add these
  group counts to claim a total number of independent experiments or seeds.

## Inventory

| ID | Experiment | Coverage and terminal state |
|---|---|---|
| E01 | Initial Hopfield identities, finite nudges, teacher learning | Complete: 72 ideal cases, 84 physical cases, 36 learning trajectories |
| E02 | Response designs, MC, LS, ridge, Kaczmarz, projection | Complete: 12 networks, 1,680 result rows |
| E03 | Recurrent digits and orthogonal-MC follow-up | Complete: 27 primary + 3 follow-up trajectories; separate 4-run LR screen |
| E04 | Statistical and frozen-network scaling | Complete: 57 cases; 45 statistical and 168 gradient rows |
| E05 | Physical width scaling and read-noise control | Complete: 27 new primary + 3 noise-control trajectories; 6 reused controls |
| E06 | Local/learned baselines and optimizer selection | Complete: 30 screening + 27 new full trajectories; 9 reused controls; 66 verified checkpoints |
| E07 | Saved predictor and vector-response identity | Complete diagnostic: 3 checkpoints, 96 validation examples each; separate linear identity |
| E08 | Dense circulation calibration | Complete: 9 calibrations, 45 training trajectories, 3 timestep-refinement calibrations |
| E09 | Four-loop circulation calibration | Complete: 6 structured + 6 same-network dense calibrations, 24 training trajectories |
| E10 | Swapped DC-response calibration | Complete: 12 calibrations across two sizes and two budgets |
| E11 | Training with frozen DC-calibrated controllers | Complete: 6 new training trajectories; 24 reused matched controls |
| E12 | Structured-controller full MNIST | Cancelled: 30 planned, 0 final trajectories, 8 partial; 12 calibrations completed |
| E13 | Untied-weight MNIST: eight learning rules | Complete: 48 trajectories, 8 methods × 3 initial angles × 2 seeds |
| E14 | Untied-MNIST probe amplitude/noise/budget audit | Complete diagnostic: 4 checkpoints, 72 rows, 8 examples, 32 probe trials |
| E15 | Shortened Fashion-MNIST reproduction | Cancelled: 20 planned, 7 complete, 4 partial, 9 unstarted; 4-method smoke completed |
| E16 | Public-code audit and JAX/Torch parity | Complete diagnostic: small and trained-network fixtures |
| E17 | Fashion-MNIST settling audit | Complete diagnostic: 4 checkpoints, 500 training images, 3 settling policies |
| E18 | Fashion-MNIST projection error and MC variance | Complete diagnostic: 1 checkpoint, 16 equilibrated images; 128 ideal variance trials |
| E19 | Frozen Fashion-MNIST cosine comparison | Complete diagnostic: 7 checkpoints, 27 common equilibrated examples, 8 probe trials |
| E20 | Measured local homeostasis identity | Complete algebra check: 8 states, 8 orthogonal probes, 3 amplitudes |

## Learning rules and measurement conventions

The common convention is `F(s,theta)-b=0`, with `J=partial_s F`, `R=J^-1`,
cost derivative `c`, ordinary feedback `q=Rc`, and true adjoint `lambda=R.T c`.
Measured-adjoint training uses `g_theta=-F_theta.T lambda_hat`, plus any direct
cost dependence on parameters. It still needs the known local parameter-force
derivatives. No full measured Jacobian is reconstructed for those learning arms.

Most later probes have unit norm: entries are `+/-1/sqrt(n)`, and MC therefore
uses the factor `n/m`. E01 instead uses unnormalized signs and the corresponding
`1/m` factor. Paired probes take two perturbed equilibrations. Counts below
include a free phase when stated, but do not equate an equilibrium, digital
matrix solve, current read, or continuous noise record with the same time/energy.

| Family | What supplies the task feedback | What changes additionally |
|---|---|---|
| Ordinary contrastive EP | Positive/negative cost-nudged states and energy-partial differences | Symmetric task parameters in the early fixed-skew model |
| VF/activity EqProp | Measured error response in the local directed-synapse rule | Independent forward/backward task weights in the later model |
| MC / fitted-adjoint methods | Random state perturbations plus a baseline | Feedback estimated per example; learned variants also maintain a predictor |
| Jacobian homeostasis | VF task feedback | Add the AD parameter gradient of a symmetry penalty |
| Known/calibrated AsymEP | Cost-nudged states with anchored corrective dynamics | Known or measured auxiliary controller; actual contrastive EP task update |
| Exact adjoint | Digital transpose solve or transpose iterations | Oracle/control work; not counted as physically measured feedback |

For the later sigmoid MLP, raw voltages `u` and activities `r=phi(u)` have
different Jacobians. The matched VF/homeostasis comparison uses activity-force
coordinates. The inspected author code uses `J_code=W-I` inside the regularizer;
its expected normalized loss is `||W-W.T||_F^2/(2n)`. The activity-force Jacobian
`W-D^-1` has the same antisymmetric part. The raw-voltage matrix `W D-I` should
not be substituted in its symmetry metric.

## E01 — Initial Hopfield measurements and teacher task

The soft-spin force is `-s-gamma*s^3+(S+alpha*A)s+drive`, with symmetric S,
fixed antisymmetric A, and a negative-definite symmetric force Jacobian.
The initial task trains only input-to-hidden weights in an eight-state teacher–
student network; it does not train every recurrent connection.

The ideal sweep covered n=8/16/32, linear/nonlinear dynamics, four asymmetries,
three seeds, five budgets and 128 trials. The finite-nudge sweep compared forward
and paired nudges, seven amplitudes and four read-noise levels. All 36 final
learning checkpoints reproduced their saved held-out losses; force residuals
were at most 1e-12. The maximum projection-identity error was 2.22e-15 and the
mean empirical/theoretical MC-MSE ratio was 1.0148.

At asymmetry 1.5, after 80 updates over batches of 64 examples:

| Feedback | Final held-out loss, mean ± SD | Equilibrations/example/update |
|---|---:|---:|
| Exact adjoint | 0.00044048 ± 0.00018730 | 1 + digital solve |
| Ordinary frozen-error EqProp | 1.039716 ± 0.284423 | 3 |
| MC, 1 probe | 0.00045655 ± 0.00021694 | 5 |
| MC, 8 probes | 0.00044989 ± 0.00019445 | 19 |
| MC, 32 probes | 0.00043083 ± 0.00018359 | 67 |
| Complete orthogonal 8-probe design | 0.00044048 ± 0.00018731 | 19 |

All corrected methods learned in this small task, including one probe per
example. The batch contains independent example-specific probes. Centered
measurements showed quadratic finite-nudge bias followed by a noise/tolerance
floor; arbitrarily reducing the nudge did not improve accuracy.

[Methods and all sweeps](../random_nudge_hopfield.md) ·
[raw report](../../simulation_results/random_nudge_full/report.md)

## E02 — Comparing response measurement designs

Twelve nonlinear networks (n=16/32, asymmetry 0/1.5, three seeds) compared five
budgets, two amplitudes, two read-noise levels and seven design/estimator pairs.
At n=32, asymmetry 1.5, amplitude .03 and read noise 1e-4, the baseline relative
adjoint error was .933:

| Estimator/design | 8 probe pairs | 32 probe pairs | 64 probe pairs |
|---|---:|---:|---:|
| Random-sign MC | 1.318 | .850 | .615 |
| Random-sign Kaczmarz | .877 | .589 | .339 |
| Random-sign LS | .872 | .195 | .0174 |
| Random-sign ridge | .872 | .403 | .202 |
| Coordinate projection / block LS | .743 | .0185 | .0132 |
| Hadamard projection / block LS | .832 | .0185 | .0130 |
| Random orthogonal projection / block LS | .784 | .0185 | .0129 |

Square random-sign designs had mean condition number 142.5, versus one for
orthogonal bases. At full rank, Hadamard response predictions on unseen probes
had relative error .0168 versus .218 for random-sign LS. Smaller partial-probe
MSE did not guarantee better learning, as E03 subsequently showed. This sweep
performed no parameter training.

[Methods, noise accounting and reproduction](../adjoint_measurement_methods.md) ·
[completion](../../simulation_results/adjoint_measurement_comparison/completion.json)

## E03 — Recurrent ten-class digits training

The task became sklearn digits, with 1,149/288/360 train/validation/test examples,
32 states, and trainable symmetric recurrence, input weights and biases. Fixed
skew remained; a spectral cap enforced stability. A separate four-setting
validation screen selected common LR .3. All 30 final checkpoints were replayed.

| Learning rule | Test accuracy, mean ± SD, 3 seeds | Equilibrations/example |
|---|---:|---:|
| Exact adjoint | 95.65 ± .13% | 1 + solve |
| Ordinary contrastive EP | 18.43 ± 11.06% | 3 |
| Known-skew AsymEP | 95.65 ± .13% | 3 + corrective circuitry |
| IID-sign MC8 | 95.09 ± .57% | 19 |
| Kaczmarz8 | 42.87 ± 24.92% | 19 |
| Minimum-change LS8 | 45.74 ± 28.67% | 19 |
| Fixed ridge8 | 44.26 ± 27.99% | 19 |
| Orthogonal projection8 | 46.76 ± 24.45% | 19 |
| Complete orthogonal32 | 95.65 ± .13% | 67 |
| Unbiased orthogonal MC8, separate follow-up | 95.46 ± .35% | 19 |

The partial projection estimate retained unmeasured baseline bias. Multiplying
the randomized-subspace correction by n/m changed its learning result markedly
at the same probe count. This is early direct evidence that reducing individual
adjoint MSE and improving training are different objectives. Methods shared a
common optimizer rather than receiving individual tuning.

[Full experiment](../adjoint_measurement_methods.md) ·
[merged report and verification](../../simulation_results/digits_combined/report.md)

## E04–E05 — Scaling and read-noise control

Synthetic projections tested n=32 through 4,096; the 57-case diagnostic also
tested initialized nonlinear networks through n=512 and batch prefixes 1/8/32/96.
The ideal law `E||lambda_hat-lambda||^2=(n-1)||lambda-q||^2/m` matched the samples.
At n=4,096, eight-probe measured MSE was 493.426 times discrepancy squared,
versus predicted 511.875. This was oracle projection data, not physical training.

For actual initialized networks and a 96-example batch, eight-probe gradient
cosine fell from .581 at n=32 to .262 at n=512. Proportional probe budgets
improved direction under this protocol. The physical 15-epoch digits comparison
then gave:

| States | Exact adjoint | MC8 | MC with n/4 probes | MC8 / proportional phases |
|---|---:|---:|---:|---:|
| 32 | 95.65% | 95.09% | 95.09%, m=8 | 19 / 19 |
| 64 | 95.74% | 94.91% | 95.37%, m=16 | 19 / 35 |
| 128 | 95.65% | 93.06% | 95.46%, m=32 | 19 / 67 |
| 256 | 95.56% | 89.72% | 95.56%, m=64 | 19 / 131 |

Means cover three seeds; the 32-state controls are reused. All 36 selected
checkpoints, including three new noise controls, were replayed. At n=256,
effectively removing read noise while preserving RNG consumption changed MC8
accuracy only from 89.72% to 89.91%. It did not explain the main gap. These are
fixed-optimizer scaling results, not a theorem that useful training always
requires a number of probes proportional to width.

[Statistical data, all SDs and costs](../random_nudge_scaling.md) ·
[physical report](../../simulation_results/mc_scaling_combined/report.md)

## E06 — Learned baselines and optimizer improvements

At n=256, local-slope and learned feedback baselines were compared with four/eight
fresh probes. Each method received a six-setting validation screen over learning
rate and gradient EMA. Training then ran 15 epochs over three seeds. The predictor
was fitted from previous measured equations; its fit is distinct from parameter
gradient momentum. All 66 unique screened/reused/final checkpoints were verified.

| Method | Original optimizer accuracy | Selected optimizer accuracy | Phases/example |
|---|---:|---:|---:|
| Ordinary baseline + MC8 | 89.72% | 94.54% | 19 |
| Local slope + MC4 | 92.59% | 95.09% | 9 |
| Local slope + MC8 | 92.69% | 95.19% | 17 |
| Learned baseline + MC4 | 95.19% | **96.30%** | 9 |
| Learned baseline + MC8 | 95.46% | 95.93% | 17 |

The selected learned-four configuration used LR .3 and gradient EMA .9. Its
training phase count was 155,115 per seed versus 327,465 for tuned ordinary MC8.
The complete screen itself cost 3,915,792 training equilibrations and must not
be treated as free. The exact-adjoint control used a different optimizer, so
these values do not establish superiority over exact gradients.

The learned map's held-out adjoint relative error was .152, versus .662 for
ordinary EqProp, at the same saved models. The stability projection remained
active on every learned-four update. These are positive results within the
fixed-skew toy, with no demonstrated dimension-independent scaling.

[All settings, costs and verification](../random_nudge_improvements.md) ·
[report](../../simulation_results/nudge_improvements_20260913/combined/report.md)

## E07 — Predictor-only audit and the DFA connection

Three saved learned-four models were replayed on 96 validation examples each,
with no fitting or network updates. For the raw whole-minibatch parameter gradient:

| Feedback | Mean cosine | Relative gradient error |
|---|---:|---:|
| Local slope only | .855 | .530 |
| Learned predictor alone | **.985** | .175 |
| Predictor + one fresh MC probe | .384 | 2.603 |
| Predictor + two fresh MC probes | .475 | 1.911 |
| Predictor + four fresh MC probes | .604 | 1.338 |

Each predictor had already received 68,940 scalar probe equations during training.
The fresh MC audit used ideal linear projections with modeled read noise, excluding
finite-nudge bias. No predictor-only training trajectory was run. A separate
16-state/3-output linear check recovered a full feedback map from 16 orthogonal
probes to norm error 5.15e-16 and verified the vector regression update.

The proposed extension retains all output responses per probe and learns a
reusable feedback map, rather than immediately reducing them to one scalar.
This remained a proposal for the nonlinear Fashion-MNIST model.

[Derivation, literature and audit](../direct_feedback_alignment_connection.md)

## E08 — Dense stochastic circulation calibration

An auxiliary controller was learned from time-ordered fluctuations of a physical
SDE model. Calibration applies `C(s-s0)`; cost nudging applies `2C(s-s0)`.
At the target C=-K, the learning-phase Jacobian is transposed while the original
free inference dynamics remain. This used an actual stochastic dynamical model,
not noise added to deterministic solver iterations.

| States | Mean controller relative error | Calibrated training accuracy | Coefficients |
|---|---:|---:|---:|
| 32 | .1017 | 95.28% | 496 |
| 64 | .1571 | 95.37% | 2,016 |
| 256 | .3341 | 95.19% | 32,640 |

Nine calibrations and all 45 final training checkpoints were verified. The n=64
half-timestep control changed mean controller error to .1545. Calibration required
13,120 total normalized recording-time units per model, including 32 records
and burn-in; n=256 charged 167,944,192 scalar voltage reads. Calibration quality
worsened with width even though task learning remained successful here.

The comparison includes ordinary EP, known-skew AsymEP, digital adjoint and learned
MC4 at common LR .1/EMA .9. That differs from E06's selected LR .3. Fixed skew
made reuse across changing symmetric task weights unusually favorable.

[Theory](../circulation_feedback_theory.md) ·
[all controls, transfer audits and costs](../../simulation_results/circulation_feedback_20260913/main_verified/report.md)

## E09–E11 — Structured noise calibration, swapped DC probes, and training

These experiments assumed four unknown loop gains in an independently specified
physical wiring basis. They did not discover an arbitrary dense asymmetry from
four measurements. Six structured noise calibrations, six same-network dense
controls, 12 DC calibrations, 24 primary training trajectories and six additional
DC-controller trajectories were completed and checked.

| Method | n=64 accuracy | n=256 accuracy | Training phases/example |
|---|---:|---:|---:|
| Ordinary contrastive EP | 12.50% | 5.19% | 3 |
| Known-skew AsymEP | 95.19% | 95.19% | 3 |
| Structured stochastic calibration | 95.19% | 95.28% | 3 |
| Learned baseline + MC4 | 95.09% | 95.09% | 9 |
| Ten-update DC calibration | 95.28% | 95.19% | 3 |

For the structured stochastic method, controller error was .0404/.0597 at
n=64/256. Same-K dense calibration at the identical short budget gave
1.1592/5.5093: an important negative result, not omitted or substituted into
the structured training rows. The structured process required eight projected
readout channels, 352,064 scalar projection reads and 880 recording-time units
per model; noise still excited all n states.

The DC method exchanges injection and measurement directions on each loop:

| States | Calibration updates | Perturbed equilibrations | Mean gain error | Minimum gradient cosine |
|---|---:|---:|---:|---:|
| 64 | 10 | 160 | .013751 | .99995568 |
| 64 | 60 | 960 | .000287 | .99999995 |
| 256 | 10 | 160 | .012792 | .99994784 |
| 256 | 60 | 960 | .000281 | .99999996 |

Each DC case also needs one free anchor and eight anchor projection reads. The
ten-update controller was then frozen for 15 training epochs: 161 calibration
equilibrations plus 51,705 training equilibrations per trajectory, versus 155,115
training equilibrations for learned MC4. No fresh random adjoint probes were
needed after calibration, but positive/negative cost nudges remained.

Doubling controller gain during task nudging was essential in the gradient
audit. Merely applying the cancellation controller once produced much poorer
alignment. The fixed wiring and constant unknown gains are strong assumptions;
these results are outside the current unrestricted untied-weight target.

[Complete experimental notes](../circulation_feedback_experiments.md) ·
[DC calibration theory](../response_loop_calibration.md) ·
[DC training comparison](../../simulation_results/circulation_feedback_20260913/response_training_verified/report.md)

## E12 — Cancelled structured-controller MNIST extension

The earlier structured model was expanded to all 784 MNIST input pixels, with
55,000/5,000/10,000 train/validation/test examples, n=64/256, three seeds, five
methods and 15 planned epochs. The user redirected the work toward untied
VF/homeostasis models before any final training trajectory completed.

All 12 calibrations completed. Eight n=256 training trajectories retain partial
checkpoints: seed 0 ordinary EP/known-skew/learned-MC4 reached epochs 7/8/4;
seed 1 reached 10/9/5; seed 2 ordinary EP/learned-MC4 reached 7/5. The other 22
planned trajectories never started. These are not final accuracy evidence.
The cancellation receipt takes precedence over preserved running heartbeats.

[Protocol and partial observations](../mnist_feedback_experiments.md) ·
[exact cancellation record](../../simulation_results/mnist_eqprop_20260914/main/cancellation.json)

## E13 — Complete untied-weight MNIST comparison

This replaced fixed skew with independently trainable forward/backward weights.
It used two 64-unit hidden layers and ten dynamical outputs (138 states), three
initial mixing angles, two seeds and five epochs. Activity coordinates matched
the inspected homeostasis code. All 48 final checkpoints were independently
replayed over validation and test data.

| Method | Initial 0° | Initial 45° | Initial 90° | Phases/example |
|---|---:|---:|---:|---:|
| Raw membrane VF | 20.55% | 20.56% | 21.11% | 3 |
| Activity VF | 49.45% | 36.11% | 34.28% | 3 |
| Activity VF + homeostasis | 71.01% | 36.25% | 35.86% | 3 |
| VF + four probes | 74.49% | 69.36% | 69.58% | 11 |
| VF + four probes + homeostasis | 75.19% | 70.76% | 70.28% | 11 |
| Learned baseline + four probes | 74.40% | 74.59% | 75.18% | 9 |
| Learned baseline + four probes + homeostasis | 75.30% | 75.79% | 75.92% | 9 |
| Digital adjoint | 73.96% | 69.53% | 68.42% | 1 + digital work |

Values are mean final test accuracies; the dedicated report retains every SD.
Homeostasis adds five Gaussian vectors, ten JVPs and parameter AD per example.
The common optimizer was Adam; the recurrent norm cap was active on 96.2–99.1%
of updates. This is a restrictive, short mechanism comparison.

At initial angle 90°, homeostasis reduced absolute asymmetry .524→.231 and
improved first-hidden cosine .712→.998. The learned probe arm's per-example
cosine was about .208, yet it trained better. At exactly 495,000 training
equilibrations, validation accuracy was 54.84% for learned probing, 31.31% for
VF and 32.69% for homeostasis. At 0°, homeostasis instead led that matched-budget
comparison. The outcome depends on initial asymmetry and the resource matched.

[All eight methods and implementation](../directed_eqprop_mnist.md) ·
[focused comparison and matched costs](../zero_order_vs_vf_homeostasis.md)

## E14 — Untied-MNIST probe-budget audit

Four final seed-zero, 90° checkpoints were held fixed. Eight training examples
and 32 probe trials tested budgets 1/4/16, amplitudes .1/.01/.001, and read noise
0/1e-5, producing 72 rows. No predictor or parameter updates occurred.

At four probes, MC correction MSE was 29.5–35.3 times baseline MSE, compared with
the ideal factor `(138-1)/4=34.25`. Activity-response finite-nudge error at .01
was only .025–.274%. This separated sampling variance from nudge bias while
remaining compatible with the successful batched training in E13. Noise affected
fresh responses only; baseline and free states were held noiseless.

[Full recorded audit](../../simulation_results/directed_eqprop_20260914/main/probe_budget_diagnostic.json)

## E15 — Shortened Fashion-MNIST reproduction and cancellation

The selected target became the adjacent-layer Figure 4a–d model from the
homeostasis paper: 784 inputs, 256/256 hidden units, ten dynamical outputs and
a trainable 10×10 readout. It used unit-scaled pixels, shifted sigmoid, zero
bias initialization, independent directional fan-in initialization, no recurrent
norm cap, Adam at 1e-4, batches of 50, 150 synchronous free steps and 20 response
steps. The user shortened 50 epochs to ten, then requested cosine analysis.

The declared four-method/five-seed grid was cancelled: seven trajectories
completed ten epochs, four retain partial checkpoints, and nine never started.
The four-method, full-width operational smoke was independently replayed.

| Method | Seed 0 logged final test accuracy | Seed 1 logged final test accuracy |
|---|---:|---:|
| VF exact forward response (`vf_ad`) | 82.38% | 82.71% |
| VF + AD homeostasis | 79.41% | 81.23% |
| Paired real VF nudges | 83.17% | 83.56% |
| Learned baseline + four probes | 70.99% | Incomplete |

These are saved 150-step prediction metrics from individual completed runs,
not a completed five-seed benchmark or a fresh independent full-test replay.
The main-grid independent accuracy report was not completed. Some inputs were
not equilibrated at 150 steps; E17 documents that limitation. High homeostasis
feedback cosine did not imply higher recorded accuracy here either.

Partial trajectories: seed 1 learned-four has five completed epochs; seed 2 VF
has four, homeostasis two, and paired VF two. The unstarted cases are seed 2
learned-four and all seed 3/4 cases. No partial value is substituted for an
epoch-ten endpoint.

[Full protocol and departures from the paper](../fmnist_homeostasis_replication.md) ·
[cancellation](../../simulation_results/fmnist_homeostasis_20260914/main/cancellation.json)

## E16 — Author-code parity and coordinate audit

The public source was pinned at `30592f576bd4d4a20d3c13632f0792b0fa452781`.
Its MLP force, finite-step response and homeostatic routines were compared with
the reimplementation under JAX/Flax on a small fixture and three difficult
examples from an actual trained 522-state checkpoint. Free states, forces,
responses, all task-gradient blocks, homeostatic loss and its gradient matched
to double-precision tolerance. Maximum trained-fixture state discrepancy was
4.0e-15 and response discrepancy 1.1e-15.

Unused stale imports were omitted and common Gaussian draws supplied for the
comparison. This establishes equation parity on the tested fixtures; it does
not establish identical random training trajectories or every revised paper
experiment. Exact differentiation of 20 response steps removes finite-nudge
bias, not finite-settling error.

[Public implementation analysis](../jacobian_homeostasis_connection.md) ·
[small parity](../../simulation_results/fmnist_homeostasis_20260914/parity_toy.json) ·
[trained parity](../../simulation_results/fmnist_homeostasis_20260914/parity_trained.json)

## E17 — Settling and period-two behavior

Read-only float64 checks used the first 500 training images on the four completed
seed-zero checkpoints and force-L2 threshold 1e-7:

| Checkpoint | Not settled after 150 synchronous steps | After 1,500 synchronous | After 1,500 damped, rate .5 |
|---|---:|---:|---:|
| VF | 9.8% | .4% | 0% |
| Homeostasis | 8.4% | 5.8% | .2% |
| Paired VF | 7.8% | .8% | 0% |
| Learned-four | 20.6% | 4.2% | .4% |

Persistent period-two fractions were .4%, 5.8%, .8% and 4.0%, respectively.
Damping preserves force zeros but can change the selected basin and was not
used during training. An even-horizon state comparison can miss a two-cycle.

[Full settling audit](../../simulation_results/fmnist_homeostasis_20260914/settling_seed0.json)

## E18 — Projection measurement error and ideal variance

The learned-four seed-zero checkpoint was frozen. Sixteen equilibrated examples
were selected from the first 64 training images; 12 of the 64 failed the free
equilibrium check. Thirty-two measured directions tested three amplitudes:

| Amplitude | Projection error against exact 20-step response | Against equilibrium response |
|---|---:|---:|
| .1 | .00335045 | .0293298 |
| .01 | .00003324 | .0318428 |
| .001 | .000000330 | .0318684 |

The predictor's full-state relative error against the equilibrium adjoint was
.9562 on this cohort; the finite-time adjoint error was .03463. In 128 trials
using **oracle projections**, MC MSE/baseline-MSE was 128.59, 32.59 and 8.208
at 4/16/64 probes, versus 130.25, 32.5625 and 8.140625 predicted. The measurement
sweep and oracle variance sweep are different controls, both noiseless.

[Recorded diagnostic](../../simulation_results/fmnist_homeostasis_20260914/probe_error_seed0.json)

## E19 — Frozen Fashion-MNIST feedback cosines

Seven saved checkpoints were compared: initialization, VF epoch ten and
homeostasis epoch ten for seeds 0/1, plus learned-four epoch ten for seed 0.
Of 40 stratified test candidates, 27 met the same free-residual threshold at
every checkpoint. This shared intersection was used throughout. Probe budgets
were 1/4/16, with eight trials, amplitude .01 and 20 response steps.

| Weights | VF first-hidden cosine | Four-probe MC | Four-probe projection |
|---|---:|---:|---:|
| Initialization | -.043 | .010 | -.041 |
| VF, epoch ten | .354 | .027 | .353 |
| Homeostasis, epoch ten | .965 | .169 | .964 |

At the separate learned-four checkpoint, predictor-only first-hidden cosine was
.454, predictor+MC4 .030, and predictor+projection4 .453. Mean input-weight
gradient cosine from VF feedback was .412 on VF-trained checkpoints and .951
on homeostasis-trained checkpoints. Homeostasis changed the model being
evaluated; these are not corrections of identical underlying weights.

The parameter-gradient audit used batches of 10/10/7, whereas training batches
were 50. E07's .985 was a full-minibatch parameter-gradient statistic over 96
examples. It must not be numerically compared with the per-example hidden
cosines here as a measure of transfer loss. The finite-time adjoint control also
showed that high cosine can hide magnitude error.

No training, predictor fitting or optimizer updates occurred. Checkpoint hashes
and parameter equality were checked before/after replay. Runtime for estimator
replay was 13.83 seconds after cohort screening. Results apply to the screened
cohort and do not alone rank training algorithms.

[Detailed comparison](../fmnist_feedback_cosines.md) ·
[plot](../../simulation_results/fmnist_homeostasis_20260914/cosines/cosine_comparison.png)

## E20 — Clamped-force homeostasis identity

A new mathematical adaptation was checked on an eight-state untied PaperNet.
Paired clamped activities supplied finite force differences. An antisymmetric
outer-product update averaged over eight orthogonal unit-norm sign directions
matched the mean paper-code homeostatic gradient in every recurrent parameter
block within 6.6e-13, across three amplitudes.

Each amplitude required 16 clamped configurations and 128 scalar current reads.
This is an algebra check at interior activities, not a Fashion-MNIST calibration
or a few-probe training result. It assumes stronger access than equilibrium
nudging: force/current measurement under state clamping. Substituting measured
equilibrium responses Rz for force responses Jz would change the algorithm.

[Check implementation](../../labs/tools/check_measured_homeostasis_identity.py) ·
[result](../../simulation_results/on_chip_direction_20260914/clamped_homeostasis_identity.json)

## Gates, provenance and pending work

Operational smoke runs and validation screens are indexed in `evidence.json`
with their original receipts. They are not extra independent scientific seeds.
Historical focused/full test gates progressed through 14, 65, 74, 84, 147 and
156 passing tests as the implementation expanded; these counts refer to their
recorded source versions and must not be added together. The final Fashion-MNIST
focused suite had ten passing tests. The broader suite excluded the pre-existing
`test_single_conv.py` import failure (`custom_classes` unavailable).

All experiment execution described here was local CPU simulation. Successful
physical-style access restrictions in code are not evidence of an on-chip
implementation. Device programming, analogue measurement circuitry, energy and
timing were not experimentally established.

The following remain proposed rather than completed experiments:

- A matched read-only comparison of earlier/current checkpoints at actual
  training batch sizes, separating individual feedback, averaged gradients and
  optimizer effects.
- Nonlinear untied-network training using the predictor alone, vector-output
  predictor fitting, periodic refresh, and fully charged calibration warm-up.
- Low-probe physical homeostasis calibration, beyond E20's full-basis identity.
- Generic gain-metric calibration, lock-in response measurements, and adjoint
  or full-Jacobian identification from fluctuation correlations. Circulation
  controller learning in E08/E09 is a different completed experiment.
- Hardware noise/programming studies, drift or state-dependent mismatch, and
  unrestricted large-network correction with demonstrated low measurement cost.

The agreed direction is to investigate on-chip asymmetry correction using
zero-order perturbations and to retain the positive learned-probing evidence.
The current proposed next step is the matched minibatch audit before promoting
a new hardware access requirement or rejecting a method on per-example cosine.
[Reconciled interpretation and candidate algorithms](../on_chip_asymmetry_direction.md)

Rebuild or verify the evidence snapshot without running experiments:

```bash
python3 -m labs.tools.compile_random_nudge_archive
python3 -m labs.tools.compile_random_nudge_archive --check
```

The builder captures only declared evidence paths and matching historical
launcher roots. Its source cutoff remains pinned so future unrelated experiments
do not silently become part of this archive's scientific claims.
