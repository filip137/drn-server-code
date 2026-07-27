# Incident Report: Conv1/Conv2 Multi-Stage Launch Loop

Date: 2026-07-27
Experiment: `perfectdiode-conv12-best-observed-confirmation-20260727-v1`
Repair branch: `codex/multistage-failstop`
Operational result: production was never submitted

## Impact

- Production parent jobs submitted: **0**
- Production tasks submitted: **0**
- Scientific training entries completed: **0 of 12**
- Failed live canaries: **2**
- GPU allocation: **one V100 per canary**
- Training artifacts: **none**

Both live failures occurred before Python training. The 16 requested CPUs were
host resources accompanying the V100; the workload remained explicitly
GPU-only.

## Required stop point

The earliest strict stop was when staged validation output became truncated
and gate success could no longer be observed. The first indisputable
execution failure came immediately afterward: remote staged-source validation
used the nonexistent Jean Zay Python path
`/lustre/fswork/projects/rech/umg/ucy17uy/conda/envs/server-code/bin/python`.
The pasted transcript does not preserve that command's exact return code or
stderr.

At that first failure the attempt should have ended with:

- stage: remote staged-source validation;
- error: wrong/nonexistent pinned interpreter path;
- completed work: prior local gates plus immutable-input preparation and
  remote hash matches; the remote staged-source gate itself had not passed;
- launched jobs: zero;
- last valid evidence: prior local gate outputs and the observed remote hash
  matches, not a passing remote validation log;
- next action: ask whether to diagnose and create a new attempt.

Instead, the agent repaired and retried in the same turn, beginning the slow
failure loop.

## Failure ledger

1. Staged validation output was truncated, making success unobservable.
2. The reconstructed remote command used the wrong Python path.
3. The correct Python was run without its required PyTorch module, causing
   MKL failure during test collection.
4. The noninteractive SSH shell had not initialized the `module` function.
5. Six staged tests contained workstation-only `.codex` paths and were later
   deselected instead of being made portable before the gate.
6. A reconstructed command used the wrong validator filename.
7. The plan validator could not resolve a manifest stored outside the
   extracted source topology.
8. `sbatch --test-only` rejected explicit `--mem*` options on Jean Zay.
9. Isolated Git-object and worktree creation hit shared-metadata permissions.
10. Cross-filesystem hard-link staging failed.
11. The persistent supervisor exited unexpectedly.
12. Canary `311193` received a V100 but the wrapper rejected Jean Zay's
    post-allocation `NumCPUs=32` logical-CPU representation even though
    `ReqTRES cpu=16` and `CPUs/Task=16` were correct.
13. The second canary received and verified a V100, then failed after about
    25 seconds while source verification expanded the 116 MiB compressed
    archive into limited `/tmp` instead of job-specific `$JOBSCRATCH`.
14. Archive verification swallowed the underlying extraction exception, so
    the likely `ENOSPC` errno was reduced to one false boolean.

Material source or launcher corrections invalidated and regenerated immutable
source, bundle, plan-binding, authorization, or preflight identities. That
made serially discovered cluster assumptions expensive.

## Systemic causes

- Long remote commands were reconstructed manually after context compaction
  instead of using one canonical environment/bootstrap runner.
- Jean Zay memory, CPU-accounting, and temporary-storage semantics were
  discovered serially in scientific canaries rather than one minimal
  environment compatibility probe.
- The source archive included substantially more than the runtime execution
  closure.
- The supervisor permitted unbounded retry behavior inside the long-running
  simulation workflow.
- Repository guidance protected production from a failed gate but did not
  define and stop one bounded, explicitly armed long-run launch attempt.

## Remediation in this worktree

- Explicit Slurm memory flags remain prohibited.
- The live audit independently requires `ReqTRES cpu=16` and
  `CPUs/Task=16`, while accepting Jean Zay's validated 16/32 `NumCPUs`
  representations.
- Live workers require `$JOBSCRATCH` and export it as `TMPDIR` before source
  verification.
- Archive failures preserve exception type, message, filename, and OS errno.
- Paired children now fail fast, terminate the sibling process group,
  escalate to group kill after a bounded wait, and finish before a
  stage-specific deadline with a Slurm safety margin.
- The supervisor defaults to one reconciliation pass; explicit following is
  bounded.
- Remote source/bootstrap validation now has one canonical, bounded entry
  point:
  `experiments/validate_mnist_conv_perfectdiode_successor_staged_jeanzay.sh`.
- Every supervisor subprocess has a hard deadline and process-group cleanup.
- A failed stage/state and first-write-wins failure report are terminal and
  cannot be resumed.
- The dedicated Jean Zay supervisor does nothing without
  `--arm-long-run`; the approved launcher supplies that flag and the
  supervisor emits the full `LONG-RUN ATTEMPT ARMED` marker before entering
  its terminal fail-stop boundary.
- Failure reports perform one bounded live scheduler readback and preserve
  worker-log tails, including extraction errno such as `ENOSPC`.
- The exact experiment tracker must be `preflighting` for the canary and
  `launch-ready` for production; remote submission consumes a fresh,
  hash-bound canonical-validator receipt bound to the exact state and output
  paths rather than a staged tracker copy. Canary and production use distinct
  immutable receipts and separate supervisor invocations.
- Canary, tracker-gate, and failure receipts use first-write-wins publication;
  a new attempt cannot overwrite prior receipt evidence.
- Root, nested, and skill guidance now requires stop, report, and a new user
  message before repairing or retrying an explicitly armed long-run launch or
  active-run monitoring failure. Ordinary development and short diagnostics
  remain repairable within a bounded retry budget.

## Current operational state

This repair did not query the live scheduler and launched no remote command or
job. The pasted history shows both canaries terminal and no production
submission. Attempt-04 transfer was still in progress when the pasted history
ended, so only the local frozen source and partial staging are known to exist.
This worktree intentionally has no replacement approved plan; any successor
must start from a new user-approved lifecycle plan and new immutable attempt
paths.
