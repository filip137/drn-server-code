# Jean Zay Notes

## Access

- From this machine, use the configured SSH alias: `ssh jean-zay`.
- The alias resolves to `ucy17uy@jean-zay.idris.fr` through `ProxyJump integnano2`.
- A direct `ssh ucy17uy@jean-zay.idris.fr` can close immediately from this network; use the alias unless the SSH config changes.

## Project And Account

- Default all Jean Zay runs to the R3 project unless the user explicitly requests another project/account.
- R3 project short code: `fmu`.
- V100 account: `fmu@v100`.
- Before launching R3 jobs, use `eval "$(/gpfslocalsup/bin/idrenv -d fmu)"` when available.
- Keep source under `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`.
- Put R3 run outputs under `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results`.
- Before reusing an older Slurm file that names `umg@v100` or writes results under `umg`, update the account/project/output root to `fmu` first.

## GPU Packing Policy

Do not default to one simulation per V100 for small diagnostic screens. For MNIST Conv hard-sigmoid training, one run often does not need a full V100, so the preferred Jean Zay pattern is:

- Allocate one V100 per Slurm task with `#SBATCH --gres=gpu:1`.
- Inside that task, run `2-4` independent configurations concurrently on the same GPU when memory/utilization allows.
- Start conservatively with `2` concurrent processes per V100 for new launchers, then increase to `3-4` only after checking `nvidia-smi` memory and utilization from a pilot.
- Keep each process writing to a distinct output directory.
- If GPU memory is high, utilization is already near saturation, or failures look like CUDA OOM/contention, fall back to one simulation per V100.

One Slurm array task per simulation is still acceptable for expensive final runs, memory-heavy runs, or when fast turnaround is more important than GPU-hour efficiency. For broad diagnostic LR/saturation sweeps, pack multiple runs per V100 unless there is evidence that packing is unsafe.

## Singleton Validation Fast Path

For an approved fixed continuation that needs one real-batch Jean Zay
compatibility check, use:

```bash
python -m experiments.jeanzay_validation launch \
  --request docs/experiment_requests/REQUEST.md \
  --experiment-id EXPERIMENT_ID \
  --attempt-id NEW_IMMUTABLE_ATTEMPT_ID
```

The cached capability profile is
[`../configs/dispatch/functional_jeanzay.json`](../configs/dispatch/functional_jeanzay.json).
The command:

1. validates the approved request and resolves its existing parent and exact
   case without adding a catalog entry for the validation;
2. creates and hashes one package containing source, parent bundle, resolved
   request study, and launch specification;
3. runs one remote import-only preflight in the reviewed module;
4. runs `sbatch --test-only`;
5. submits exactly one `0-0` validation array task and immediately reads it
   back;
6. monitors without retry; and
7. after `COMPLETED`, invokes the metadata-first and full
   `sync-remote-results` passes.

The central experiment tracker is updated automatically for visibility, but
it is not launch authority and cannot block this validation path. The launcher
allows no production mode or successor expansion. On the first failure it
writes one failure receipt, runs no broad tests, performs no retry, and asks
for user direction.

## Monitoring

- Use `squeue -u "$USER"` for active state.
- Use `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES%40 -P` after completion.
- Check logs under the run-specific `slurm/` directory in the source checkout.
- Count completions with `find RESULT_ROOT -name metrics.json | wc -l`.

## Result Repatriation

Jean Zay is a compute and staging location, not the permanent result archive.
A run is not complete until its complete bundle and Slurm logs have been
copied back and validated on local storage.

Follow the shared
[`remote_result_workflow.md`](remote_result_workflow.md) and use the
repository's `$sync-remote-results` skill. Pull its small metadata view first
so local evaluation can start immediately; let the resumable complete archive
pass continue independently. The remaining steps in this section define final
archival acceptance.

Before submission, record:

- the exact run-specific remote result directory;
- the intended local result directory;
- the Slurm job and array-task identifiers;
- the code identity and resolved-config identity.

Keep every transferable artifact under the run-specific remote directory when
possible, including resolved configs, manifests, checkpoints, metrics,
completion records, and Slurm logs. Never use a shared result root as the unit
of transfer or cleanup.

Before transfer, verify that Slurm reports a terminal state and that the remote
collector, completion marker, or experiment-specific validator passes. Copy
the whole self-contained run or study directory. Do not select individual
result files from a larger bundle.

Use the configured `jean-zay` SSH alias and copy into a new local staging
directory. Replace the example paths with one exact run identifier:

```bash
remote_run=/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/RUN_ID
local_stage=/home/filip/server_code/results/.incoming/RUN_ID

mkdir -p "$local_stage"
rsync -a --partial --protect-args \
  "jean-zay:${remote_run}/" \
  "${local_stage}/"
```

For a modern bundle with hash-bound completion manifests, first require a
quick rsync comparison. The command must exit zero and print no changed files:

```bash
rsync -a --delete --dry-run --itemize-changes --protect-args \
  "jean-zay:${remote_run}/" \
  "${local_stage}/"
```

Then run the experiment-specific collector or integrity validator against the
local staging copy. That validator must check the bundle's recorded artifact
hashes. For a canonical MNIST Conv sweep, collection is:

```bash
python -m experiments.mnist_conv collect \
  --sweep /home/filip/server_code/results/.incoming/RUN_ID/sweeps/SWEEP_DIR
```

For an older result without checksummed completion manifests, repeat the rsync
dry-run with `--checksum`, create and compare a complete file SHA-256 and
byte-count manifest before accepting the transfer, and retain its evidence
classification as legacy or unverified until curation says otherwise.

Only after the checksum comparison and local validation pass may the staging
directory be published into the local result store:

```bash
local_run=/home/filip/server_code/results/RUN_ID

test ! -e "$local_run"
mv "$local_stage" "$local_run"
```

Write a transfer receipt outside the immutable result payload, for example
under `/home/filip/server_code/results/_transfer_receipts/`. Record:

- the SSH host and fully resolved remote and local paths;
- Slurm job identifiers and terminal `sacct` states;
- sweep, study, run, and content identities where applicable;
- transferred file count and byte count;
- the copy and checksum-verification commands and outcomes;
- the local scientific-validation command and outcome;
- the transfer timestamp.

If any check fails, keep the remote directory intact and retry with `rsync`.
Never mask transfer failures with `scp -r ... || true`. After successful local
publication, perform cleanup as a separate recorded step and remove only the
exact run-specific remote directory; never remove the shared result root or
source checkout.
