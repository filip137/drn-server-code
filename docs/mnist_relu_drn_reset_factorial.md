# Controlled RESET Factorial

## Question

The historical 95.97% test result and the newer 85.52% teacher-KL result do
not isolate a single cause. They jointly changed the hidden digital bias,
supervision loss, amplifier indexing, and minibatch input-reset behavior.
Consequently, neither result can by itself establish whether the historical
number is optimistic or the newer number is pessimistic.

## Design

`mnist_relu_drn_reset_factorial.v1` is a closed 2 x 2 x 2 experiment:

- hidden digital bias: disabled or enabled;
- supervision: historical paired-output squared error or teacher KL;
- amplifier indexing: model-local logical indices or the archived
  process-global behavior.

All eight arms use the same frozen teacher checkpoint (SHA-256
`42b0526c4a433057b0ad0f09a7b36ede64afd7d47a3a8985724cac0e46358d54`),
the same cohort-A device source (SHA-256
`207b143fc1a63710c944db74bc573f16b616ce50e22c6605c12068bf6a3f230d`),
seed 42, `voltage_amp = 4`, and `current_amp = 0.25`. Each arm explicitly
resets input state between minibatches and performs the same reseed/rebuild
between learning-rate selection and production training. Thus input-state
carryover and construction lifecycle are controls rather than hidden proxies
for the bias factor.

The legacy-index arms intentionally train the rebuilt production model with
resolved edges 3->4 and 4->5. Logical arms remain 0->1 and 1->2 across both
constructions. Fresh-process validation resolves 0->1 and 1->2 in either
case, matching how the archived indexing defect affected training but not a
fresh deployment process.

Execution is parallelized over the local RTX 3090, Trex RTX 5090, and Akib
RTX 3080. Each host receives complementary factorial cells, so both levels of
bias, loss, and indexing occur equally often within that host. This blocks a
host-wide accuracy offset out of every main-effect contrast. With one run per
cell, host-by-interaction effects remain inseparable from factorial
interactions; the cross-host reduced smoke is a compatibility gate rather than
an uncertainty estimate.

## Results

All eight production runs and all eight fresh-process tests completed. The
analyzer verified the common teacher and device-source hashes, identical dense
device assignments, explicit minibatch input reset, one controlled production
restart per arm, the intended training indices, and fresh-process test indices
0->1 and 1->2.

| Hidden bias | Amplifier indexing | Paired MSE test accuracy | Teacher-KL test accuracy |
| --- | --- | ---: | ---: |
| off | logical | 93.90% | 86.28% |
| off | legacy process-global | 95.72% | 77.59% |
| on | logical | 93.94% | 85.49% |
| on | legacy process-global | 95.70% | 80.45% |

The factorial main effects on test accuracy are:

- teacher KL minus paired MSE: `-12.36` percentage points;
- legacy minus logical amplifier indexing: `-2.54` percentage points;
- bias enabled minus disabled: `+0.52` percentage points.

The indexing main effect must not be interpreted on its own because indexing
and loss interact strongly. Legacy indexing raises the paired-MSE result by
`+1.82` points without bias and `+1.76` points with bias, but lowers the KL
result by `-8.69` and `-5.04` points, respectively. Under corrected logical
indexing, switching from paired MSE to KL costs `7.62` points without bias and
`8.45` points with bias. Bias itself changes the logical paired-MSE result by
only `+0.04` points and the logical KL result by `-0.79` points.

Consequently, the historical `95.97%` number was modestly optimistic for the
corrected circuit: the matching legacy-index/MSE factorial cells reach
`95.70-95.72%`, whereas corrected logical indexing reaches `93.90-93.94%`.
The newer approximately `85.52%` KL result was not made artificially poor by
the indexing fix; the corrected logical KL cells reach `85.49-86.28%`. Most of
the old-versus-new gap is therefore attributable to the supervision objective,
with about `1.8` points of additional uplift from the archived indexing defect
in the MSE setting. Hidden bias is not a material explanation in this run.

The validated machine-readable output is under
`results/factorial_parallel_analysis_20260817_v3/`: `factorial_summary.json`,
`factorial_arms.csv`, and `factorial_accuracy.png`.

## Interpretation Limits

This campaign has one independent run per cell. Its factorial contrasts
separate the three effects at seed 42 but do not estimate run-to-run
uncertainty. Any scientific claim about effect stability requires replication
with additional declared seeds. The old 95.97% result remains a historical
reference, not an additional factorial replicate, because its input-reset
semantics differ.

The complete sequential campaign manifest is
`campaigns/manifests/mnist_relu_drn_reset_factorial.json`. The executed
parallel shards use the same eight configs from artifact-recorded source
snapshot commit `1af0d2f6a7bf831d2d4a69f1ec3359963f4b0f76`, which is
not present in this checkout. The recorded source archive SHA-256 is
`d400859984eb550bac7cf1bc3e22b41c8f5ac171006c434fc9bec8c211c3910c`.
Their launch contract and cross-host parity result are local ignored
artifacts under
`results/factorial_parallel_control_20260817_v3/`. Collected raw output and
the generated JSON/CSV/PNG analysis are intentionally ignored by Git.
