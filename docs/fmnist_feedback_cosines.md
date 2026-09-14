# Fashion-MNIST feedback directions on frozen checkpoints

Completed 14 September 2026 after the user redirected the task from accuracy
training to cosine similarity. Training was stopped; no further classifier
training or accuracy evaluation was performed for this analysis.

Script: `labs/tools/compare_fmnist_feedback_cosines.py`.
Artifacts: `simulation_results/fmnist_homeostasis_20260914/cosines/`.
The estimator replay took 13.83 seconds after cohort screening, compared with
several minutes per training trajectory. No optimizer or predictor updates occur.

## Design

Seven actual checkpoints are used: initialization, VF epoch 10, and homeostasis
epoch 10 for seeds 0 and 1, plus the completed learned-four-probe epoch-10
checkpoint at seed 0. No intermediate parameters are reconstructed from logs.
The paired VF and homeostasis networks have identical initialization per seed.
They follow the adjacent-layer Figure 4a-d model with 522 dynamical states and
independent forward/backward weights. See the preceding replication note for
source parity, optimizer, normalization and shortened training details.

Sample four examples per class from the official test set using seed 20260915.
Of 40 candidates, 27 meet the same force-residual criterion at **every** checkpoint
(float64 L2 residual below 1e-7 after 150 free steps). This shared intersection
is frozen for all comparisons; indices, labels, hashes, exclusions and residuals
are saved in `cohort.npz` and `run.json`. Results apply to this screened cohort,
not to all Fashion-MNIST inputs. Batches for parameter-gradient analysis have
size 10, including the final seven-example batch.

At each checkpoint compute the true force adjoint
`lambda=(W D-I)^-T D c_r` as a digital reference. Compare:

- Paper VF: exact forward derivative of its 20-step response trajectory.
- Paired VF: real +/-0.01 cost nudges, measured activity response after 20 steps.
- Monte Carlo correction of paired VF, using 1, 4 or 16 fresh unit-norm random-sign
  probes, eight independent trials per budget.
- Minimum-change projection correction, using the **same measured probes**:
  `q + Z.T solve(Z Z.T, y-Z q)`, with one probe per row of Z. The small solve
  uses the known injection design, not an estimated network Jacobian. It is a
  biased correction that retains unmeasured components of q.
- The saved learned predictor and its MC/projection corrections, only on its own
  seed-0 checkpoint. It is not transferred to other networks or refitted.

All physical probe responses use the same free states, beta, 20-step response
budget, directions and examples. Finite response bias remains part of the test;
a finite-time adjoint control separates it from the inverse-versus-transposed
inverse discrepancy. Homeostasis is assessed through its trained weights; its
regularizer gradient is not added to the task-gradient reference.

## Results

Mean first-hidden-layer cosine to the true adjoint, averaged across seeds 0 and 1
and the common examples (MC/projection also average eight probe trials):

| Frozen weights | VF response | MC correction, 4 probes | Projection, 4 probes |
|---|---:|---:|---:|
| Initialization | -0.043 | 0.010 | -0.041 |
| VF, epoch 10 | 0.354 | 0.027 | 0.353 |
| Homeostasis, epoch 10 | 0.965 | 0.169 | 0.964 |

Paired VF agrees with the exact-response VF cosine to the displayed precision.
For the input-weight gradient on fixed minibatches, VF feedback yields mean
cosine 0.412 on VF-trained checkpoints and 0.951 on homeostasis-trained checkpoints.
The task-gradient CSV contains every parameter block, not just the output/readout
that can dominate a flattened global score.

On the separate learned-predictor checkpoint, the first-hidden cosine is 0.454
for the saved predictor alone, 0.030 after its four-probe MC correction, and 0.453
after its four-probe projection correction. These are **same-checkpoint**
comparisons on one seed. No claim is made that its weights are matched to the
VF or homeostasis endpoints.

These results show that homeostasis produced strongly aligned feedback on the
checked equilibria. The unshrunk low-probe MC estimate is noisy, despite its ideal
unbiasedness. Projection mostly preserves the baseline at these small budgets;
it does not establish a substantial cosine improvement. Euclidean-error decrease
does not guarantee cosine improvement, especially per layer.

## Evidence and limits

`feedback_cosines.csv` records all-state and layerwise cosine, relative error,
norm ratio and counts of nonzero vectors. `parameter_gradients.csv` records
cosine, RMS, near-zero fraction, scale relative to initialization and reference
gradient change on identical minibatches. A high cosine can coexist with an
incorrect magnitude; the finite-time reference illustrates this distinction.
No zero vector is silently assigned a perfect cosine.

Checkpoint hashes before and after replay and parameter tensor equality passed.
Only epochs 0 and 10 are available in this comparison. Plot:
`cosine_comparison.png` (also PDF); per-checkpoint table: `report.md`.
The two seeds, screened cohort, and eight probe trials support a cheap diagnostic,
not a conclusion about training accuracy, generalization, or hardware efficiency.

The earlier digits predictor cosine of 0.985 was measured on a whole-minibatch
parameter gradient; it is not directly comparable to the per-example hidden
feedback cosines above. The earlier untied MNIST study also showed that noisy
four-probe feedback could train better despite much lower per-example cosine
than homeostasis. Different models, batches and optimization/settling protocols
prevent attributing the changed results to the probe method alone. See the
[reconciled analysis](on_chip_asymmetry_direction.md) for the positive training
evidence and the proposed matched minibatch audit.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DISABLE_SHM=1 \
  /home/filip/miniconda3/envs/py312/bin/python -m labs.tools.compare_fmnist_feedback_cosines \
  --root simulation_results/fmnist_homeostasis_20260914/main \
  --output simulation_results/fmnist_homeostasis_20260914/cosines_new
```
