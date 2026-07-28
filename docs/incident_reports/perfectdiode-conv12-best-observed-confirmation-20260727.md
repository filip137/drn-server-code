# Incident Report: Conv1/Conv2 Confirmation Launch on Jean Zay

Incident window: 2026-07-27 through 2026-07-28
Experiment: `perfectdiode-conv12-best-observed-confirmation-20260727-v1`
Scope: ordinary-MNIST diagnostic
Severity: launch-blocking; no scientific data loss
Current state: blocked with no active successor job or supervisor

## Executive summary

The requested six Conv1 confirmations at 10 epochs and six Conv2
confirmations at 20 epochs did not start production training on Jean Zay.
Production remained correctly blocked because no live canary completed.

Four distinct operational failures occurred:

1. The first scheduler dry-run rejected explicit host-memory options that
   Jean Zay does not permit for this GPU job type.
2. Canary `311193` received a V100, but the wrapper incorrectly rejected
   Jean Zay's post-allocation logical-CPU accounting representation.
3. Canary `311705` received and verified a V100, then failed before training
   while the source-integrity check expanded a 116 MiB archive in the
   compute node's size-limited `/tmp`.
4. After those integration defects were repaired, canary `319506_[0]`
   remained pending on scheduler priority and was cancelled by the
   supervisor after the mandatory 900-second scheduler-start deadline.

There was also one login-node contract failure between the third and fourth
items: the approved-plan validator could not resolve its repo-relative
manifest path in the staged checkout. No Slurm job was created for that
attempt. Mirroring the exact hash-verified manifest into the checkout fixed
the problem.

The first three failures were defects or omissions in our Jean Zay launch
integration. The final failure was external capacity: all compliant
production-QOS probes predicted starts after the requested morning deadline.
The tempting H100 development-QOS route was deliberately rejected because
IDRIS reserves that QOS for development or execution tests, and the required
canary must use the same QOS as the result-producing batch.

No training entry ran, no official test data was read, and no production
parent or task was submitted.

## Requested experiment

The unchanged scientific payload contains twelve entries:

| Architecture | Scheme | Optimizer | `(rho_conv, rho_dense)` | Epochs |
|---|---|---|---:|---:|
| Conv1 | baseline | SGD | `(0.009, 0.03)` | 10 |
| Conv1 | baseline | Adam | `(0.009, 0.03)` | 10 |
| Conv1 | ours | SGD | `(0.009, 0.03)` | 10 |
| Conv1 | ours | Adam | `(0.009, 0.03)` | 10 |
| Conv1 | legacy | SGD | `(0.003, 0.03)` | 10 |
| Conv1 | legacy | Adam | `(0.003, 0.01)` | 10 |
| Conv2 | baseline | SGD | `(0.027, 0.09)` | 20 |
| Conv2 | baseline | Adam | `(0.027, 0.09)` | 20 |
| Conv2 | ours | SGD | `(0.009, 0.03)` | 20 |
| Conv2 | ours | Adam | `(0.027, 0.03)` | 20 |
| Conv2 | legacy | SGD | `(0.003, 0.03)` | 20 |
| Conv2 | legacy | Adam | `(0.009, 0.01)` | 20 |

Conv1 retained `T/K=4/4` and 34,380 optimizer steps. Conv2 retained
`T/K=6/6` and 68,760 optimizer steps. The execution plan packed the entries
into six fixed pairs, with two ordinary training processes sharing each
allocated GPU.

No rho, learning-rate vector, seed, initialization, dataset split, epoch
budget, or `T/K` value changed during the incident.

## Timeline

| Stage | Outcome | Training reached? |
|---|---|---:|
| Initial `sbatch --test-only` | Rejected explicit Slurm host-memory options | No job |
| Canary `311193` | Failed `2:0` after 12 seconds on CPU-accounting audit | No |
| Canary `311705` | Failed `1:0` after 25 seconds during source-archive verification | No |
| Attempt 04 | `$JOBSCRATCH` correction validated locally; launch preparation interrupted | No job |
| Attempt 05-r3 | Approved-plan validator could not resolve staged manifest | No job |
| Canary `319506_[0]` | Pending on `Priority`; cancelled after 903.305 seconds | No allocation |
| Production | Blocked on the live canary throughout | No jobs |

## Detailed failures

### 1. Invalid host-memory request

The first immutable launch candidate explicitly requested 64 GiB of host
memory. Jean Zay rejected `--mem`, `--mem-per-cpu`, and `--mem-per-gpu` at
the scheduler test stage.

This was corrected by binding the site-managed host-memory policy and
emitting no explicit Slurm memory option. The GPU, CPU, walltime, packing,
and scientific settings did not change.

### 2. Canary `311193`: CPU accounting was interpreted too narrowly

The task correctly requested:

- one node and one task;
- one V100;
- `CPUs/Task=16`;
- `ReqTRES cpu=16`;
- one thread per physical core.

After allocation, Jean Zay exposed those 16 requested physical CPUs as
`NumCPUs=32` logical processors on a two-thread V100 node. The wrapper
required `NumCPUs=16` exactly and rejected the otherwise valid allocation.

The corrected audit independently verifies requested CPU TRES and task
shape, while accepting only the documented pending or allocated accounting
representations. Training remained GPU-only; these CPUs were host resources
supporting the GPU processes.

### 3. Canary `311705`: source verification used compute-node `/tmp`

This canary passed the corrected allocation audit and verified the V100/CUDA
environment. It then failed the archive-fingerprint check before a training
process started.

The source archive was approximately 116 MiB compressed and 214 MiB
expanded. The safe verifier used Python's default temporary directory, which
resolved to the compute node's limited `/tmp`. The runtime collapsed the
underlying extraction exception into
`source_archive_fingerprint_valid=false`, so the exact errno was lost.
Archive size, timing, login-node success, and IDRIS storage guidance make
temporary-space exhaustion the high-confidence cause.

The correction requires the scheduler-created, job-unique `$JOBSCRATCH`,
validates that it is writable and user-scoped, and exports it as `TMPDIR`
before the live runtime begins.

### 4. Attempt 05-r3: approved-plan manifest path was not staged

The final plan bound the immutable manifest by its repository-relative path:

```text
results/.launch_staging/<experiment-id>/<bundle-id>/manifest.json
```

The full bundle existed in FMU result staging, outside the staged checkout.
The official plan validator ran with the checkout as its repository root and
therefore reported the manifest as missing. Scheduled source and bundle
preflight had already passed, but the supervisor stopped before `sbatch`.

The repair mirrored only the exact hash-verified manifest at the path the
plan validator expected. No scientific artifact or bundle identity changed.

### 5. Canary `319506_[0]`: scheduler-start deadline

Attempt 06-r3 passed:

- focused local tests;
- exact-ID tracker and approved-plan validation;
- scheduled-run preflight;
- source archive, checkout fingerprint, environment, and bundle validation;
- live allocation/account audit;
- `sbatch --test-only`;
- post-submission Slurm contract readback.

The live canary was submitted at 2026-07-27T23:41:14+02:00 on
`fmu@v100`, `gpu_p13`, `qos_gpu-t3`, with `v100-32g`. It remained
`PENDING (Priority)` and had no start estimate. At
2026-07-27T23:56:17+02:00, the persistent supervisor enforced the
900-second start deadline and cancelled it.

The preserved receipt records:

```text
failure_kind=scheduler_start_timeout
elapsed_seconds=903.305
job_id=319506
cancellation.status=cancelled
production.status=blocked_on_canary
```

The job never received a node or GPU and ran zero training steps.

## Why the incident repeated

The incident was not one recurring error. It was a chain of independent
launch-contract assumptions discovered serially:

- site-specific Slurm memory syntax;
- requested versus allocated CPU representation;
- compute-node temporary-storage policy;
- local versus staged path resolution;
- scheduler availability within a hard start deadline.

Every wrapper or contract correction correctly invalidated the previous
source archive, bundle, plan, authorization, and receipts. That protected
scientific integrity, but made basic environment discovery expensive.

The underlying process failures were:

1. No single minimal Jean Zay compatibility probe exercised scheduler
   options, allocation readback, `$JOBSCRATCH`, CUDA, and archive extraction
   before freezing the production contract.
2. Local plan validation did not reproduce the exact staged checkout and
   external-bundle topology.
3. The source archive was much larger than the execution closure.
4. Archive extraction errors lost their original exception details.
5. The scheduler test's predicted late start was recorded but not compared
   with the supervisor's 900-second start deadline before live submission.
6. An existing failure report could mask a later retry exception when the
   same attempt path was reused.

## Impact

- Production parent jobs submitted: **0**
- Production tasks submitted: **0**
- Scientific entries completed: **0 of 12**
- Training optimizer steps executed: **0**
- Official test-set reads: **0**
- Canaries allocated and failed before training: **2**
- Canary cancelled before allocation: **1**
- Total allocated GPU time from failed canaries: **37 seconds**
- Training artifacts produced: **none**
- Scientific data lost or corrupted: **none**

## What was successfully repaired

The final V100 attempt had working contracts for:

- Jean Zay site-managed host memory;
- physical/logical CPU accounting;
- one-GPU CUDA verification;
- `$JOBSCRATCH`-backed archive verification;
- immutable source, environment, bundle, and authorization hashes;
- durable preparation and scheduler-start deadlines;
- automatic canary cancellation and fail-closed production blocking;
- a persistent local controller retaining the SSH session.

Last valid immutable identities:

- source commit:
  `367abf8d0f9a76b562a65a0b80c0ebbaef6624dc`;
- source archive SHA-256:
  `228d17636e036e49e0e16302f52bcf7f85efb4c150d8786220eb204e9d9e310e`;
- bundle:
  `pdconfirmbundle_c56239abf6f84bc74fdc0ccb5c0054371cd43ad66631f4b3ce2e61d094568e5d`;
- manifest SHA-256:
  `63bf6496d7e790fda7a5e8c3f8a677a25263c9aded65ac2eb373c51b707b7f61`.

## Corrective and preventive actions

### Required before another Jean Zay attempt

1. Add one short same-wrapper environment probe covering Slurm request
   syntax, allocation accounting, `$JOBSCRATCH`, CUDA/GPU identity, and
   archive extraction.
2. Run plan validation from the exact staged checkout with the real remote
   bundle topology.
3. Parse the `sbatch --test-only` predicted start and reject a live
   submission when it already exceeds the scheduler-start deadline.
4. Preserve the underlying archive/tar/OSError details in every failed
   receipt.
5. Use a new attempt directory for every terminal failure; never let a prior
   failure report mask the current exception.
6. Minimize the source archive to the reviewed execution closure.

### Automated regression coverage

Add tests for:

- pending `NumCPUs=16` versus allocated `NumCPUs=32`;
- absence of forbidden Slurm memory flags;
- required `$JOBSCRATCH`/`TMPDIR` behavior;
- staged manifest resolution from the remote checkout;
- scheduler estimates later than the start deadline;
- same-QOS canary and production enforcement;
- duplicate-safe supervisor restart with distinct failure receipts.

## Current disposition

At the final audit:

- canary `319506_[0]` was terminal `CANCELLED`;
- no `pd-successor-canary` or `pd-successor-production` job remained;
- no successor supervisor was active;
- production was never submitted.

Non-submitting scheduler probes showed that compliant production QOS could
not meet the requested morning window. The remaining operational decision is
therefore outside this repaired contract: accept a later Jean Zay
production-QOS window, or explicitly authorize a different compute host.

Operational state is tracked in
[`current_experiments.md`](../current_experiments.md).
