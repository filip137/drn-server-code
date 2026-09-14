# Scaling the physical-feedback comparison to MNIST

**Superseded experiment.** The fixed-four-gain full-MNIST grid was cancelled on
14 September 2026 at the user's request, before any final training trajectory
completed. Partial outputs remain under `simulation_results/mnist_eqprop_20260914/main`
with `cancellation.json`. The replacement uses untied directed weights and
trainable Jacobian homeostasis; see
[directed_eqprop_mnist.md](directed_eqprop_mnist.md). The notes below describe
the earlier protocol and interim observations, not completed MNIST evidence.

14 September 2026. Exploratory extension of the four-gain Hopfield comparison.
Use the existing `codex/hopfield-random-nudge-adjoint` branch and worktree.

The physical model is unchanged: F=-s-.25*s^3+(S+K)s+B*x+b. S is symmetric,
zero diagonal, projected to operator norm <= .7. The fixed K has four unknown
gains in a known independently generated skew-loop basis, normalized to norm
one. This prior does not identify general dense or state-dependent asymmetry.
The input dimension grows from 64 to all 784 original MNIST pixels. State
counts remain 64/256, including ten output states (54/246 hidden states).

The official 60,000 training images are split into 55,000 training and 5,000
validation examples, stratified with seed 20260914. The official 10,000 test
images remain separate. Per-pixel centering/scaling uses training images only,
after division by 255, with standard-deviation floor .1. No PCA, resizing,
augmentation, convolution or test-based selection. IDX content hashes and
exact split indices are saved once per run; checkpoints do not duplicate data.

Main declared coverage: five methods x two sizes x three seeds (0,1,2), hence
30 training trajectories and 12 calibrations. Methods: ordinary contrastive
EqProp, known-skew AsymEP reference, DC-calibrated AsymEP, noise-calibrated
AsymEP, and learned baseline plus MC4. The first four use actual centered
contrastive EqProp. MC4 uses its estimated adjoint and local parameter-force
derivatives, as before. Its learned feedback predictor and gradient momentum
are distinct states; prediction precedes current probes and the predictor is
updated only after the current gradient has been formed.

Common training settings: 15 epochs, batch size 250, LR .1, bias-corrected
gradient EMA .9, nudge amplitude .01, independent nudged-state read noise 1e-5.
Batch size changes from the small-digits study's 96; no method-specific tuning
or equal-optimizer-step claim is made. This is a controlled comparison, not a
best-achievable MNIST benchmark. Test evaluation is at the fixed final epoch.
Validation is measured each epoch. A fixed 16-training-example diagnostic
cohort is replayed at epochs 0,5,10,15, with copied predictor state; oracle
Jacobians/adjoints only audit those estimates and never update the learner.
Those diagnostic measurements and evaluation equilibria are charged separately.

DC calibration: 10 sweeps, rate .2, amplitude .01, scalar read noise 1e-5,
four loops, 160 perturbed equilibria plus one anchor, 168 scalar projection
reads including the anchor. Noise calibration: stochastic Heun, dt .02,
100 adaptive time units +10 burn-in, eight separately charged records,
temperature .05, rate .1, update every 10 steps, average after time 30,
no added projection-read noise. Both operate on the first training input at
initialization and freeze C. Calibration applies F+C(s-s0); task nudging applies
F+2C(s-s0)-beta*c. K remains fixed throughout training. Neither calibration
receives K, its gains, a Jacobian or an adjoint. Error comparisons use K afterward.

Numerical implementation: the new PyTorch backend uses float64 throughout,
the existing local implicit leak/cubic relaxation and force tolerance 1e-9.
It batches positive/negative phases and checks force residual every ten solver
iterations. This can settle beyond the first passing iterate, so bitwise
identity is not expected. Spectral projection, EMA and sequential predictor
updates retain the NumPy implementations. Optional CUDA graph support is not
GPU-validated in this run; all main evidence is local CPU execution.

Monitoring contract: local processes, single BLAS/Torch thread per worker,
at most eight workers. Fresh results under
`simulation_results/mnist_eqprop_20260914`. Run/config/dataset manifests live at
the root, each case has calibrations and each trajectory has status.json,
run.log, metrics.csv, audits.csv, latest.npz and terminal final.npz/result.json.
Worker heartbeats report epoch/batch, CPU time and counters every 20 seconds;
the parent aggregates terminal results/failures. Verify launch and artifact
progress twice, then inspect all active workers at least every 30 minutes.
Initial CPU benchmarks estimate roughly an hour or more for the complete
parallel comparison; actual runtime depends on contention and settling.
Only operational failures may be retried automatically with the same science;
poor accuracy remains a result. Preserve failed directories. Completion needs
30 terminal results, coverage/count checks, finite values and checkpoint replay.

Access note: Akib's GPU was idle, but automatic approval review rejected source
transfer because authorization for the remote destination could not be verified.
No source was uploaded. The experiment proceeds locally without that transfer.

Gates before the main launch: (1) CPU scientific regression suite, (2) measured
response/gradient and three-update parity against NumPy at 64/256 states,
(3) a one-epoch 250-image/100-validation operational smoke through all five
methods at 16 states, with test evaluation disabled. Benchmark artifacts are
`cpu_gate` and `cpu_full_batch_gate`. Maximum gradient relative difference in
the full-batch gate's small matched parity cohorts was 1.43e-6; all measured
force residuals were <=1e-9. The main 784-feature task is not a reduced-data run.

Pre-launch gate result: 125 tests passed with the unrelated existing
`labs/tests/test_single_conv.py` excluded because its `custom_classes` import
cannot be resolved. The one-epoch smoke completed all five methods. Independent
NumPy replay of every smoke checkpoint reproduced all validation accuracies
and losses within 1e-7; no test evaluation ran. Its measurement counts matched
three or nine equilibria per training image.

Main launch command (single parent, eight worker processes):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
/home/filip/miniconda3/envs/py312/bin/python -m labs.tools.train_mnist_feedback \
  --raw-dir /home/filip/python2cadence/data/MNIST/raw \
  --output simulation_results/mnist_eqprop_20260914/main
```

Largest probe-heavy jobs are submitted first to reduce the parallel makespan.
Each case uses independent deterministic RNGs, so scheduling does not change
its model initialization, shuffle, probe directions or measurement noise.

Launch receipt: source commit `3a5707f5`, local unified-exec session `1412`.
The parent process and eight intended workers produced manifests and initial
validation metrics; a second check confirmed advancing batch counters, fresh
worker heartbeats and no tracebacks. All 12 calibrations completed. DC controller
relative errors were 0.0084--0.0136; noise-controller errors were 0.0317--0.0703.
These references are post-calibration diagnostics, not stopping criteria.

Accounting at the declared full budget: 825,000 training-image presentations
per trajectory. Each EqProp variant therefore charges 2,475,000 training
equilibria; MC4 charges 7,425,000 (825,000 free equilibria and 3,300,000 paired
probes). DC adds 80 calibration probe pairs and one anchor before training.
All controlled error-nudge phases remain charged; zero additional random
adjoint probes does not mean zero nudged training phases. Validation/test and
fixed-cohort audits are additional costs in each terminal result.

Relative to the small-digits experiments, the input-weight parameter count
increases with 784 pixels, while the four unknown physical gains remain fixed.
There are 44,416 trainable scalar degrees of freedom at 64 states and 225,760
at 256 states, counting the independent off-diagonal symmetric couplings.
The calibrated unknown has four degrees of freedom in either case. MC4 still
learns an n-by-10 feedback predictor from measurements. Increasing the dataset
provides more observations of this shared predictor; the larger minibatch also
averages more probe noise. Neither effect removes the generic n/m variance
of a single unstructured random-projection estimate.

Reported wall-clock times include contention among the local worker processes
and are operational measurements, not a controlled hardware speed comparison.
The number of equilibria and scalar calibration reads is the explicit physical
measurement comparison. All eight initial workers have advancing CPU time and
batch counters; none is waiting on a GPU or another remote machine.

Interim scientific observation, at the predeclared epoch-five audit: ordinary
EqProp's overall gradient cosines were .945/.982/.983 across the three 256-state
seeds, but its input-weight cosines were -.663/.223/-.457. Its recurrent and
bias gradients dominate the aggregate measure. Therefore the final report
must retain blockwise audits and training accuracy; a large overall cosine is
not by itself evidence of successful feature learning. These observations do
not change any run settings or select the final epoch.
