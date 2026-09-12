# Measuring and correcting EqProp feedback

Exploratory research note, 11 September 2026. The completed measurement sweep supports using well-conditioned probe designs and choosing a nudge amplitude above the read-noise floor. It does not establish that better individual adjoint estimates necessarily improve training. A separate recurrent digits experiment tests that question with trainable recurrent weights and actual state-dependent cost nudging.

The [12 September scaling follow-up](random_nudge_scaling.md) checks fixed and
increasing Monte Carlo probe budgets as the state dimension grows.

## What is measured

For a free equilibrium satisfying \(F(s^0,\theta)=0\), use the convention \(F(s(b),\theta)-b=0\). With \(J=\partial_sF\), \(R=J^{-1}\), and \(c=\nabla_sC(s^0)\), ordinary small-error nudging gives \(q=Rc\), while the parameter gradient requires

\[
\lambda=R^\mathsf Tc,
\qquad \nabla_\theta C=-(\partial_\theta F)^\mathsf T\lambda.
\]

A centered constant-current probe measures

\[
r_u=\frac{s(hu)-s(-hu)}{2h}=Ru+O(h^2),
\qquad y_u=c^\mathsf Tr_u\approx u^\mathsf T\lambda.
\]

Thus each experiment supplies one scalar equation for the adjoint. The implemented estimators consume measured responses and a measured baseline; they do not estimate or evaluate a Jacobian. Analytic Jacobians appear in evaluation controls and the explicitly named exact-adjoint training baseline. The inverse-versus-transposed-inverse distinction and correction through augmented learning dynamics are developed by [Scurria et al., 2026](https://arxiv.org/html/2602.03670v2).

The smooth central-difference interpretation requires remaining on one differentiable equilibrium branch. Hardware would measure \(P J^{-1}B_{\rm inj}\), so injection/readout gains and inaccessible nodes must be accounted for. Only output injection does not generally identify the complete adjoint. If all required nodes can be injected and the cost depends only on outputs, the scalar \(c^\mathsf TRu\) needs only output readout. These simulations assume injection and voltage readout at every state.

## Implemented estimators and their limits

[adjoint_estimators.py](../labs/adjoint_estimators.py) implements unit-L2 random-sign, coordinate, Hadamard, and Haar-random orthogonal designs. Every physical probe has total current norm \(h\), including dense sign patterns. In the notation below, \(U=[u_1,\ldots,u_m]\) has probe columns, \(y\) stacks scalar measurements, and \(d=y-U^\mathsf Tq\). The code stores probes as rows.

| Method | Estimate or update |
|---|---|
| Monte Carlo | \(\widehat\lambda=q+(n/m)Ud\) |
| Minimum-change least squares | \(\widehat\lambda=q+(U^\mathsf T)^+d\) |
| Ridge about EqProp | Minimize \(\|y-U^\mathsf T\ell\|^2+\alpha\|\ell-q\|^2\); equivalently \(q+U(U^\mathsf TU+\alpha I)^{-1}d\) for \(\alpha>0\) |
| Kaczmarz | Start \(\ell_0=q\); one ordered pass of \(\ell_{k+1}=\ell_k+u_k(y_k-u_k^\mathsf T\ell_k)\) |
| Orthogonal projection | \(\widehat\lambda=q+Ud\), requiring \(U^\mathsf TU=I\) |

The \(n/m\) factor in MC compensates for \(\mathbb E[u u^\mathsf T]=I/n\). LS and ridge use SVD filtering, not matrix inversion. Weighted regression is available, but the experiments use equal weights and the predeclared ridge \(\alpha=0.1m/n\). This is a regularization control, not a noise-calibrated or oracle-tuned optimum.

For exact independent random-sign measurements, MC is unbiased and has squared estimation error \((n-1)\|\lambda-q\|^2/m\) in expectation. Exact Kaczmarz projections cannot increase Euclidean adjoint error; fresh independent sign probes leave a factor \((1-1/n)^m\) in expected squared error. A randomized subset of \(m\) orthonormal directions leaves a factor \(1-m/n\). These are mathematical calculations for this measurement model. At \(n=32,m=8\), the respective factors are 3.875, approximately 0.776, and 0.75. Noisy or finite-amplitude observations need not obey the projection monotonicity guarantee.

Kaczmarz is an established projection method. [Strohmer and Vershynin](https://arxiv.org/pdf/math/0702226) prove an expected convergence rate for consistent full-rank overdetermined systems; applying it to these physical adjoint equations is an adaptation. Repeating a fixed underdetermined probe set does not recover its unmeasured subspace. LS also retains the baseline in that subspace, and one Kaczmarz pass generally differs from batch LS.

Full unrestricted response identification needs at least \(n\) independent input directions; each direction can measure all outputs. `fit_response` requires full weighted design rank, even when a prior is provided, and returns \(\widehat R\) without inverting it to recover \(J\). Sparse coupling does not imply sparse response. Square random-sign designs can be poorly conditioned. Complete orthogonal bases have condition number one; above \(n\) probes, the generator concatenates orthonormal blocks and fits them with LS.

One noisy \(q\) is shared across the measured equations, so the errors in \(d\) are correlated. The fitting interface treats \(y\) as data and \(q\) as the prior. Full orthogonal coverage cancels \(q\) algebraically, although the matched implementation still counts its extra pair. Lower Euclidean adjoint error is not a guarantee of lower parameter-gradient error: the local parameter-force derivatives determine the relevant weighting.

## Completed fixed-equilibrium measurements

[compare_adjoint_measurements.py](../labs/tools/compare_adjoint_measurements.py) completed all 12 declared network cases and 1,680 result rows: sizes 16/32, asymmetries 0/1.5, seeds 0/1/2, cubic coefficient 0.25, probe budgets 4/8/16/32/64, amplitudes 0.003/0.03, and per-voltage-read noise standard deviations 0/\(10^{-4}\). The four random-sign estimators and three orthogonal designs give seven comparisons per setting. Each noisy setting averages eight independent read-noise trials; these are not eight additional networks. Sixteen unseen probe directions test response predictions. The maximum force residual was \(9.993\times10^{-13}\), below the declared \(10^{-12}\) tolerance.

The table reports mean relative L2 adjoint error across three networks, with eight read trials per network, for **32 states, asymmetry 1.5, amplitude 0.03, and read noise \(10^{-4}\)**. The measured EqProp baseline error was 0.933.

| Design / estimator | 8 probe pairs | 32 probe pairs | 64 probe pairs |
|---|---:|---:|---:|
| Random-sign MC | 1.318 | 0.850 | 0.615 |
| Random-sign Kaczmarz | 0.877 | 0.589 | 0.339 |
| Random-sign LS | 0.872 | 0.195 | 0.0174 |
| Random-sign ridge | 0.872 | 0.403 | 0.202 |
| Coordinate projection / block LS | 0.743 | 0.0185 | 0.0132 |
| Hadamard projection / block LS | 0.832 | 0.0185 | 0.0130 |
| Random orthogonal projection / block LS | 0.784 | 0.0185 | 0.0129 |

At 32 probes, the random-sign design condition number averaged 142.5, compared with one for the orthogonal bases. Mean held-out full-response prediction error was 0.218 for random-sign LS and 0.0168 for Hadamard. The corresponding hidden input-gradient errors were 0.372 and 0.0359. These gradient diagnostics use hypothetical directed off-diagonal recurrent weights and hidden input weights; this sweep performs no parameter training. Full-response prediction always uses its separate unregularized LS fit, even on CSV rows containing a different adjoint estimator.

The two amplitudes expose the expected bias/noise tradeoff. Full Hadamard adjoint error without read noise rose from \(4.76\times10^{-8}\) at \(h=0.003\) to \(4.76\times10^{-6}\) at \(h=0.03\). With read noise \(10^{-4}\), it instead fell from 0.185 to 0.0185. Both amplitudes remain close to linear response in this example; this two-point comparison does not locate an optimal amplitude. [Shi et al., 2022](https://epubs.siam.org/doi/10.1137/21M1452470) provide adaptive finite-difference interval selection given a noise estimate. That adaptive algorithm is not implemented here.

These results support full-rank orthogonal measurements and expose the bias of an overly strong fixed ridge. Eight probes leave substantial unresolved feedback error; the three partial orthogonal designs should not be ranked generally from only three network realizations.

Artifacts under `simulation_results/adjoint_measurement_comparison/` are [config.json](../simulation_results/adjoint_measurement_comparison/config.json), [measurements.csv](../simulation_results/adjoint_measurement_comparison/measurements.csv), [completion.json](../simulation_results/adjoint_measurement_comparison/completion.json), `status.json`, `run.log`, and [measurement_comparison.png](../simulation_results/adjoint_measurement_comparison/measurement_comparison.png) / `.svg`. Completion records 15,564 clean forward equilibrations and 139,980 modeled state reads. Clean states are reused computationally across estimators, budgets, and noisy reads. Per-estimate physical accounting is \(1+2(m+1)\) equilibrations/state-vector reads and \(2(m+1)h^2\) summed squared applied currents, excluding the separately recorded 16 held-out diagnostic pairs. This current sum is an excitation proxy, not dissipated energy or measurement time.

## Harder recurrent digits comparison

[recurrent_eqprop.py](../labs/recurrent_eqprop.py) and [train_recurrent_eqprop_digits.py](../labs/tools/train_recurrent_eqprop_digits.py) implement a harder experiment. The task is classification of the ten handwritten digits from sklearn's bundled 8-by-8 dataset, with fixed stratified splits of 1,149 training, 288 validation, and 360 test examples. Feature normalization uses training statistics only. The 32 continuous states comprise 22 hidden and 10 output nodes; inputs drive hidden nodes only. This removes the previous matched teacher construction and introduces recurrent parameter learning: 496 symmetric recurrent couplings, 1,408 input weights and 32 biases, for 1,936 trainable degrees of freedom.

The force and logits are

\[
F(s)=-s-0.25s^{\odot3}+(S+K)s+[Bx;0]+b,
\qquad \text{logits}=4s_{\rm out}.
\]

Training changes symmetric zero-diagonal \(S\), hidden input weights \(B\), and all biases \(b\), while skew-symmetric \(K\) remains fixed with initial spectral norm 1.0. Projected SGD caps \(\|S\|_2\) at 0.7; every method receives the same constraint. Fixed \(K\) does not imply constant relative Jacobian asymmetry, which the gradient audits record.

The measured error phases solve \(F(s)\mp\beta_{\rm eff}\nabla_s C(s)=0\) for cross entropy, updating the cost gradient during settling. The positive coefficient magnitude is \(\beta_{\rm eff}=h/\max(\|c(s^0)\|,1)\), bounding the initial error current by \(h\) without an unbounded repulsive-phase coefficient when the error vanishes. Random probes use constant currents \(\pm h u\). A forward IMEX solver handles local leak/cubic forces implicitly and coupling explicitly; it does not construct a Jacobian.

The symmetric parameter uses a projected full-matrix convention. The exact or estimated adjoint gives

\[
G_S=-\operatorname{sym}(\lambda s^\mathsf T),\quad
G_B=-\lambda_h x^\mathsf T,\quad G_b=-\lambda,
\]

with the recurrent diagonal removed and examples averaged. The `contrastive_ep` control instead uses the actual paired energy-partial contrast, including

\[
G_S^{\rm EP}=-\frac{s^+s^{+\mathsf T}-s^-s^{-\mathsf T}}{4\beta_{\rm eff}}.
\]

For this model, the antisymmetric part of \(J\) is exactly the known \(K\). `known_skew_asymep` settles \(F(s)-2K(s-s^0)-\beta_{\rm eff}\nabla_sC(s)\) and uses the same paired contrastive update. Its tangent is \(J^\mathsf T\), giving a strong known-structure control without reconstructing a Jacobian. Hardware would need to implement the corrective couplings; fewer settling phases alone do not prove a lower hardware cost. Generic probe methods use measured \(q,y\) to replace the feedback in the local parameter-force rule. Thus both actual contrastive EqProp and measured-adjoint learning are present, with their different operations explicit.

The declared full comparison has nine methods: `adjoint`, `contrastive_ep`, `known_skew_asymep`, `mc8`, `kaczmarz8`, `lstsq8`, `ridge8`, `orthogonal8`, and `orthogonal32`. Defaults are seeds 0/1/2, batch size 96, 15 epochs, \(h=0.01\), and read noise \(10^{-5}\). `orthogonal` here uses Hadamard directions. An eight-epoch, seed-zero, exact-adjoint-only validation screen over 0.01/0.03/0.1/0.3 selected a common learning rate of **0.3**, with validation accuracy 96.53%; the complete [calibration summary](../simulation_results/digits_lr_calibration/summary.json) contains all four trials. This avoids test-set tuning but does not optimize each estimator separately. No claim of a best-tuned training-method ranking follows from it.

A separately declared exploratory follow-up, `orthogonal_mc8`, uses eight randomized Hadamard directions with the \(n/m\) scaling from MC. It preserves unbiasedness over the randomized subspace while matching the iid MC current norm and probe count. For exact measurements its squared-error factor is \((n-m)/m\), equal to 3 at \(n=32,m=8\), compared with 3.875 for iid sign MC. This is a derived expectation, not a training result. The follow-up was added after inspecting early main-run trajectories and must remain separate from the original 27-trajectory comparison.

The declared local CPU execution and monitoring plan is:

| Output directory under `simulation_results/` | Declared coverage | Expected duration before launch |
|---|---|---|
| `digits_smoke/` | All nine methods, seed 0, two epochs; 192/64/64 examples | Seconds |
| `digits_lr_calibration/` | Four exact-adjoint learning-rate trajectories, seed 0, eight epochs, full train/validation splits; no test evaluation | Seconds to minutes |
| `digits_comparison/` | Nine methods by three seeds, 15 epochs, common selected learning rate; 27 trajectories and 432 epoch rows | Minutes |
| `digits_orthogonal_mc/` | Separate `orthogonal_mc8` follow-up, seeds 0/1/2, 15 epochs at learning rate 0.3; three trajectories and 48 epoch rows | Minutes |

The launcher is a foreground CPU Python process retained through an exec session. Check `status.json`, `run.log`, and `current.csv` for an epoch heartbeat and real metric progress; `metrics.csv` accumulates completed trajectories. If an expected heartbeat stops, inspect the process and exception before changing execution settings. Preserve a failed directory and rerun the same scientific configuration in a fresh directory after diagnosis. Completion requires declared trajectory/row coverage and semantic artifacts, not only process exit.

Each digits directory stores `run.json` (configuration, command, versions, commit/dirty status, PID), `status.json`, `run.log`, per-trajectory `.npz` checkpoints including split indices and preprocessing, `metrics.csv`, and `summary.json`. Non-calibration runs additionally produce `report.md` and `learning.png` / `.svg`. Training metrics include loss/accuracy, modeled equilibrations, state reads, excitation sums, residuals, projection counts, and first-minibatch gradient audits. Test metrics are evaluated only at the final epoch. Current counters sum phase amplitude squared; the error-current counter uses its free-phase initial value, while the actual cost force changes during settling. Neither counter integrates physical energy over time. Wall time includes validation and audits that physical-operation counts exclude, so it is not a measured hardware timing comparison.

## Collected digits results

The original 27 trajectories completed in 617.8 seconds; the separate three-run
orthogonal-MC follow-up completed in 72.7 seconds. Both used the declared data,
15 epochs, learning rate 0.3, amplitude 0.01 and read noise 1e-5. All 30 final
checkpoints were independently loaded and replayed, exactly reproducing test
cross entropy and accuracy. The merged dataset has 480 epoch rows, no duplicated
method/seed/epoch tuples, and maximum force residual `9.9996e-10`, below the 1e-9
training tolerance. Initialization, dataset splits and normalization match across
methods for each seed. Every final recurrent matrix differs from initialization;
the smallest Frobenius change is 1.734. The focused mathematical and pipeline suite
has **65 passing tests**.

The table reports held-out test accuracy, mean ± population SD across the three
seeds. These are descriptive spreads, not confidence intervals. The phase count
includes one free equilibrium and excludes evaluation and Jacobian audits.

| Learning method | Test accuracy | Equilibrations per example/update |
|---|---:|---:|
| Exact adjoint oracle | 95.65% ± 0.13 pp | 1, plus a dense adjoint solve |
| Ordinary contrastive EP under fixed asymmetry | 18.43% ± 11.06 pp | 3 |
| Known-skew AsymEP, contrastive update | 95.65% ± 0.13 pp | 3, with corrective couplings |
| Independent-sign MC, 8 probes | 95.09% ± 0.57 pp | 19 |
| Kaczmarz, 8 probes | 42.87% ± 24.92 pp | 19 |
| Minimum-change LS, 8 probes | 45.74% ± 28.67 pp | 19 |
| Fixed ridge, 8 probes | 44.26% ± 27.99 pp | 19 |
| Orthogonal projection, 8 probes | 46.76% ± 24.45 pp | 19 |
| Full orthogonal reconstruction, 32 probes | 95.65% ± 0.13 pp | 67 |
| Unbiased orthogonal MC, 8 probes (follow-up) | 95.46% ± 0.35 pp | 19 |

Three observations follow from this particular experiment.

1. **Improving individual adjoint MSE is insufficient for reliable learning.**
   Low-budget LS/projection methods performed much less consistently than MC,
   despite their better ideal individual-error guarantees. For example, the
   first seed's initial relative adjoint errors were 1.56 for MC and 0.711 for
   LS, but their batch parameter-gradient cosines were 0.558 and 0.102. Final
   accuracies were 94.44% and 19.17%. Across seeds LS ranged from 19.17% to
   85.56%, so it should be described as inconsistent here, rather than uniformly
   incapable of learning. The estimates' unmeasured-direction bias, the local
   parameter mapping, minibatch averaging, and the common projected optimizer
   all matter. This is not a controlled isolation of each contribution.

2. **Debiasing a randomized subspace changes the learning result at the same
   measurement budget.** For partial orthogonal projection,
   `E[lambda_hat] = q + (m/n)*(lambda-q)` in exact linear response. With 8 of 32
   directions, most of the baseline bias remains in the expectation. Multiplying
   the random subspace correction by `n/m` removes that bias; the separate
   follow-up reached 95.46% mean accuracy instead of 46.76%. Its ideal variance
   is also lower than independent-sign MC at this budget, but the 0.37 pp
   observed accuracy difference between the two MC variants is too small to
   establish a general performance advantage from these three seeds.

3. **Known physical structure is the cheapest successful phase-count control.**
   Known-skew AsymEP matches the exact baseline here with three equilibrations,
   compared with 19 for eight-probe correction and 67 for full reconstruction.
   That result assumes the skew coupling is known and its corrective dynamics
   can be implemented. It is not evidence that arbitrary circuit asymmetry is
   known, cheaply calibrated, or correctable without additional hardware.

Gradient plots compare each method with the true gradient **at its own current
parameters**, before the recurrent spectral projection. High cosine late in a
poor run does not imply the same trajectory or projected step as the successful
oracle. One common learning rate was selected using exact-adjoint validation;
individual estimator learning rates, ridge strength, and probe budgets were not
optimized. Thus this is a matched exploratory comparison, not a best-tuned
benchmark ranking. The ordinary EP result pertains to imposed non-conservative
coupling, not to conservative EqProp in general.

The [combined report](../simulation_results/digits_combined/report.md),
[plots](../simulation_results/digits_combined/learning.png),
[summary](../simulation_results/digits_combined/summary.json), and
[checkpoint verification](../simulation_results/digits_combined/verification.json)
include both groups. `sources.json` and the merged CSV retain the distinction
between the primary 27 runs and the later three-run follow-up. The original
artifacts remain under `digits_comparison/` and `digits_orthogonal_mc/`.

## Reproduction

Run from `/home/filip/server_code_random_nudge` on branch
`codex/hopfield-random-nudge-adjoint`, in an environment with NumPy, sklearn,
Matplotlib and pytest. The commands below use the existing `py312` environment.
The shown output names are the original runs and already exist; choose fresh
names when rerunning, because the launchers deliberately reject overwrites.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1
DRN_PYTHON=/home/filip/miniconda3/envs/py312/bin/python
"$DRN_PYTHON" -m pytest -q \
  labs/tests/test_random_nudge_hopfield.py \
  labs/tests/test_adjoint_estimators.py \
  labs/tests/test_adjoint_measurements.py \
  labs/tests/test_recurrent_eqprop.py \
  labs/tests/test_recurrent_eqprop_digits.py
"$DRN_PYTHON" -m labs.tools.compare_adjoint_measurements \
  --output simulation_results/adjoint_measurement_comparison
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits --calibrate \
  --output simulation_results/digits_lr_calibration
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits --learning-rate 0.3 \
  --output simulation_results/digits_comparison
"$DRN_PYTHON" -m labs.tools.train_recurrent_eqprop_digits \
  --methods orthogonal_mc8 --learning-rate 0.3 \
  --output simulation_results/digits_orthogonal_mc
"$DRN_PYTHON" -m labs.tools.summarize_recurrent_eqprop_digits \
  --runs simulation_results/digits_comparison \
  --follow-up-runs simulation_results/digits_orthogonal_mc \
  --output simulation_results/digits_combined
```

The summary command validates complete coverage, matched initialization and
splits, finite metrics, actual recurrent learning and physical-operation counts,
then replays every checkpoint before generating the consolidated plots.

## Scope of the evidence

Read noise affects measured nudged states/responses; free equilibria, targets, cost derivatives, and parameter-force factors remain exact. Neither experiment includes actuator errors, baseline drift, process noise, inaccessible states, or nonlinear readout calibration. Response estimation is repeated at each input and parameter setting rather than assuming a globally valid Jacobian.

Lock-in measurements, fluctuation-based inference, and gain-metric calibration from the supplied note remain distinct future experiments. The correlation identity is valid for a fully observed stationary stable linear stochastic model; capacitance-weighted physical dynamics require converting the inferred drift adjoint back to the force convention. Noise added to solver iterations would identify that solver's stochastic transition law without an additional physical correspondence. These DC experiments establish neither fluctuation inference nor a DRN hardware speedup.
