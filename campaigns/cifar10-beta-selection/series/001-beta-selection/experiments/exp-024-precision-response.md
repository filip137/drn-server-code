---
id: "exp-024"
title: "Matched float32/float64 CIFAR displacement–gradient response"
status: "complete"
hypotheses: ["H-018"]
---
# Does float64 move the small-response failure boundary?

Filip explicitly authorizes the matched precision replay proposed after exp-022.
Use [H-018](../../../hypotheses/H-018-float32-response-floor.md) to distinguish
gradient arithmetic from precision in the forward/phase computation.

## Frozen cases and measurements

Same ours fixed-0.3× EqProp-trained checkpoints0/1/5, ordered256 training images,
batch32, no augmentation/noise, V4/C1, perfect diodes with saved explicit diode
dictionaries, finite T/K=[6,6,4], loss, frozen weights, BN minibatch statistics
and frozen running buffers as exp-022. Promote the existing checkpoint values
exactly; float64 does not reconstruct a trajectory trained in float64. Reuse
exp-022 input/config/checkpoint hashes and its frozen numerical source.

Three arms on each checkpoint: float32 states+arithmetic; identical float32
endpoints with float64 arithmetic; float64 model/states/forcing+arithmetic.
Cast the already preprocessed float32 image tensors to float64 so image values
remain identical. Include all analog states/conductances/biases, dense head,
BN and gain parameters in conversion checks. Same float32 endpoints serve
both arithmetic arms, with one phase solve. Both references (native finite-T
BPTT and matched frozen-current local BPTT) use the arm's state precision.
Also compare float32/float64 reference directions and gradient scales.

Original17 multipliers from1e-4 through1e4, plus1e-10,1e-8,1e-6 to probe the
lower boundary; same saved block base betas. Measure all8 convolutions,
absolute/relative half-difference RMS, cosine, norm ratio, unchanged-coordinate
fractions, and active-coordinate changes relative to float32 spacing. Pool
gradients by example count and displacements by raw squared sums. Retain all
eight batches and report the existing cosine>=0.95/norm0.9–1.1 criterion as
descriptive. No interpolation of exact thresholds. Lower crossings outside
the grid stay unbracketed. Reconcile overlapping float32 rows with exp-022.

Improvement only after full promotion implicates errors before the gradient
arithmetic; mixed arithmetic improvement implicates that arithmetic. Full
promotion can also alter free states, BN and forcing, which limits attribution.
No optimizer, training, test-set read or solver-iteration change is permitted.

## Execution and monitoring handoff

Root: `results/cifar-precision-response-20260930-v1/`, with frozen `source/`,
configs/launch commands, CPU `smoke/`, checkpoint `cells/e{0,1,5}/`, collection
and analysis. Runner `experiments/replay_cifar_precision_response.py`.
One local same-runner smoke on two images/one beta/epoch0 exercises all arms.
Three sequential queue jobs on Loulou RTX5090, expected20–60 minutes total,
maximum7200 GPU seconds across all cases/retries/validation. Initial cap2400s
per checkpoint; reconcile unused budget across cases if needed. CPU smoke
cap300s. First production-batch timings refine the estimate without duplicating
or changing scientific cases. Deadline October1 08:00 Paris.

Bound float64 peak at22GiB plus2GiB headroom; actual prior float32 peak10.21GiB.
Share only with identified Ben workloads when measured memory permits and
respect the weekday fleet cap. The run-watch service owns execution monitoring;
root collects/validates/reviews. Stop on nonfinite results, wrong precision,
source/input/restore mismatch, changed parameters/BN, or budget exhaustion.
Preserve every failed/partial attempt; no automatic scientific retuning.

Expected coverage: 3 checkpoints × 3 arms × 20 betas × 8 layers =1440 pooled
rows, plus11520 minibatch rows. Both arithmetic arms must have identical state
measurements. Record runtime and source identities, successful bundle validation,
pooling checks, reference changes and float32-reproduction checks before review.

Preparation: Riri became occupied by Ben/Kellian jobs. Loulou had26802MiB
free with identified Ben workloads, and the identical torch2.11.0+cu128/CUDA12.8
runtime and CIFAR training data. Stage unchanged checkpoints/reference rows
there and preserve the same RTX5090 device class. Four numerical CPU tests
passed, including exact reproduction of the original float32 phase-pair
estimator, full-model dtype/reset preservation and sub-float32-spacing changes.
The initial CPU `smoke/` succeeded but exposed an inspection issue in the mixed
control: scalar amplification/normalization dtype also follows layer objects.
Corrected this with private promoted interaction/layer views, preserving actual
float32 endpoints, and reran all4 tests. `smoke-source-v0/` preserves that initial
smoke's source. The corrected same-runner smoke is `smoke-v1/`.

Corrected smoke completed in about90s and passed canonical validation. Queue
jobs `cifar-precision-response-e{0,1,5}-20260930` are submitted. The service's
fresh receipt confirms epoch0 running on Loulou and epochs1/5 pending behind
it. Commands/prepared identities are in `launch/e*.json`; inspect with
`gpu_queue.py status JOB_ID`. Root collects each completed `cells/eEPOCH/`
from `/home/filip/server_code/results/cifar-precision-response-20260930-v1/`
on Loulou, validates using `python -m experiments.reporting validate-run DIR`,
then runs `python -m experiments.analyze_cifar_precision_response RESULTS_ROOT`.
Use `--epochs 0` for explicitly partial analysis until all cases are collected.
The queue owns routine polling; root owns timing assessment and scientific
review, with no separate polling agent or competing launch mechanism.

Placement update: Fifi became idle with32039MiB free and the same runtime.
Held then cancelled the unstarted Loulou epoch1 queue entry (zero attempts,
zero charged compute), and replaced it with
`cifar-precision-response-e1-fifi-20260930`, command `launch/e1-fifi.json`.
Epoch0 and epoch5 retain Loulou. This uses at most two RTX5090s concurrently,
preserves all cases/source/input hashes, and keeps the same7200s total budget.
Collect epoch1 from Fifi; the other checkpoint cells from Loulou. The first
initialization batch took226s including startup, subsequent batches about171s;
the run fits its2400s allocation. First-four-batch observations are preliminary:
at multiplier1e-4, block1 Conv1 cosine is0.328–0.511 in float32 and the
arithmetic-only control, versus0.9990–0.9997 in full float64. No pooled or
trained-checkpoint conclusion is claimed until collection and review.

Terminal handoff: all three checkpoint cells are locally collected and
validated, and all three production queue jobs are complete. Coverage is
1440 pooled rows,11520 batch rows and3264 overlapping float32-reference
checks. Queue-charged total4479.22s, below7200s; peak reserved CUDA21.51GiB
per case. No production failures, exclusions or retries. Terminal receipts
are `launch/*-terminal.json`; the cancelled original epoch1 job had zero
attempts and zero charge. Final `analysis/summary.json` covers epochs0/1/5
with `complete_coverage: true`; earlier partial analysis logs are preserved.
All source/input/dtype/restore/immutability and displacement-pooling checks
passed. Root completed scientific review in the
[result note](../results/exp-024-precision-response.md). No active monitoring
handoff or remaining cases. The training runs remain stopped.
