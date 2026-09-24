# Refined beta boundaries and paper protocol assessment

User requests finer beta selection and a defensible protocol supported by the
simulations. This is exploratory calibration and analysis, not new training.
Conv2 and Conv3, baseline/ours/legacy, seed0, zero noise, T=K6/8, the unchanged
float64 centered frozen-current runner, and the saved initializer/BPTT-best
checkpoints remain fixed. Reuse the September18 cohort: 36 batches of16 from
the 5,000-example validation partition of MNIST's official training split.
It is selection evidence, not independent confirmation or official-test data.

Reuse all existing measurements. Measure geometric grids with adjacent ratio
at most1.10 in these intervals (endpoints already measured): Conv2 baseline
[500,750], ours [30,50] and [50,75]; Conv3 baseline [100,200] and [300,500],
ours [.9,3] and [3,6]. For legacy, extend the previous maximum30/1 by factors
1.5^j for j=1..6, up to341.71875/11.390625. Then subdivide the interval between
the largest observed passing beta and its next measured failing point for
each strict per-matrix threshold .90/.95 to adjacent ratios <=1.05. Evaluate
all added points, preserve nonmonotonicity and report sampled intervals rather
than asserting an exact mathematical maximum. Unclosed legacy boundaries at
the declared cap remain open. No norm gate is added; norm and global-gradient
statistics remain reported. No claims of exact equilibrium are added at T8.

Targets: Conv2 on local RTX3090, Conv3 on Trex RTX5090. One sequential worker
per architecture, running concurrently across hosts; previous two-worker replay
benchmark gave only1.068x throughput. Resource inventory checked before launch;
do not disturb unrelated jobs. Synchronous local same-runner smoke for each
architecture, then target smoke before production. Reproduce overlap first-batch
gradient measurements. Budget <=2hours per target including smoke, <=4GPUhours
total; expected1–2hours, revised from measured throughput. Maximum150 new beta
cases combined is a defensive cap; typical coverage is below100. Run-specific
canonical bundles, logs, inputs and aggregate progress live under
`results/eqprop-beta-refinement-20260919-v1/`. Remote root has the same basename
under `/home/filip/server_code/results/`. Monitor actual progress and collect
and validate remote results before conclusions. Scientific nonfinite values
are failed beta candidates; operational errors require diagnosis.

After completion, assess cosine, scale-sensitive gradient discrepancy and
optimizer/update relevance against the existing ten-epoch outcomes, with
explicit limits on retrospective prediction. Review read-noise evidence and
distinguish fixed-clean-beta robustness from noise-aware hyperparameter tuning.
Recommend an auditable, equal-budget selection and confirmation procedure;
do not retrofit a supposedly prospective rule to force existing betas or
claim full-horizon/noisy qualification for newly calibrated values. Any
additional training needed to establish the proposed paper rule is listed
as outstanding, not silently launched by this calibration study.

## Placement and staging amendments

Trex and then Riri acquired unrelated GPU clients before production admission.
No Conv3 production replay ran on either remote host. Their smokes and the
Trex admission-exit receipt are preserved. Final production placement is the
local RTX3090 for both architectures: one sequential worker per architecture,
concurrent until Conv2 finishes. Conv3 retains a three-hour cap after same-path
local smoke; the entire physical GPU budget remains four hours. Expected wall
time is now2–3hours. Local original Conv3 replay and final-placement smoke
agree within2.3e-15 in cosine/norm diagnostics.

An isolated-snapshot Git metadata error stopped the original Trex smoke and
first local production attempt before measurements. The attempts are preserved
under operational-abort-01, snapshot metadata was repaired without modifying
scientific source/configs, and production-environment smokes were repeated.
Trex's cosine agreement was2.1e-12; a1.66e-10 norm-diagnostic difference exceeded
an initial generic1e-10 assertion and was explicitly reviewed as negligible
within1e-9. The launch commands had already been issued when that assertion
failed; Trex's occupancy guard independently prevented production. Final
production uses the original local calibration hardware, with the tighter
regression passing. No scientific measurement or training outcome was excluded
on the basis of these transport checks.
