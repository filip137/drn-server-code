# Jean Zay Notes

## Access And Paths

- SSH alias: `jean-zay`.
- Default allocation: `fmu@v100`.
- Partition/QoS/constraint: `gpu_p13`, `qos_gpu-t3`, `v100-16g`.
- Source checkout:
  `/lustre/fswork/projects/rech/umg/$USER/server_code`.
- Results:
  `/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.
- Use `$JOBSCRATCH` when a job needs a private extracted source tree.

Jean Zay is the preferred scheduled target for Conv3 arrays and other
production work that does not fit the local, Akib, or Trex lanes.

Do not add unsupported memory directives to V100 jobs. Before a substantial
submission, validate the exact source, config, wrapper, and resource request.
The current default is a synchronous local canary followed by one Jean Zay
production submission. Reintroduce a live Jean Zay canary for an execution
contract only if a target-specific production failure shows that the local
gate is insufficient.

## Submission Recording

Before any live `sbatch`, add a `planned` row to the Recent Experiment
Directories table in
[`current_simulations.md`](current_simulations.md). This applies to production
jobs, arrays, restored remote canaries, and retries.

Prefer a direct link to the intended result directory. When the final directory
depends on the Slurm job ID and cannot yet be linked, record enough information
to recover it: the launch name, config or Slurm wrapper, and expected remote
result root or filename pattern. Immediately after submission, add the returned
job ID, change the row to `running`, and replace the placeholder with the
result link as soon as it resolves.

## Launch

The shared target in `configs/experiment_targets.json` can run a local canary
and then submit an existing remote command:

```bash
python -m experiments.launch run jean-zay \
  --name NAME \
  --log /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/NAME/slurm-%j.log \
  --local-canary-command \
    "python -m experiments.exact_run CONFIG.json --output-root results/NAME/local-canary --device cpu --dataset-root /home/filip/datasets/mnist --smoke --summary-json results/NAME/local-canary.json" \
  --local-canary-require results/NAME/local-canary.json \
  --dry-run -- \
  python EXPERIMENT_SCRIPT.py --config CONFIG.json
```

The launcher runs the local canary synchronously. A nonzero exit or a missing
or empty required artifact blocks production. On success it performs exactly
one Jean Zay submission. Use an experiment-specific Slurm file when modules,
arrays, packing, or resource details exceed the shared target.

Keep one scientific surface on one recorded target. Scheduling must not alter
its config, batch size, initializer, cohorts, or training order.

## Monitoring

```bash
ssh jean-zay 'squeue -u "$USER"'
ssh jean-zay \
  'sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES%40 -P'
```

The launcher status path is:

```bash
python -m experiments.launch status jean-zay JOB_ID
```

## Copy Results Locally

Wait for a terminal scheduler state, then copy the exact run directory into a
staging path:

```bash
mkdir -p /home/filip/server_code/results/.incoming/RUN_ID
rsync -a --partial --protect-args \
  "jean-zay:/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/RUN_ID/" \
  "/home/filip/server_code/results/.incoming/RUN_ID/"
```

Run the experiment's collector or integrity checks on the local copy. Keep
the remote copy until local validation succeeds; remote cleanup is a separate
explicit action.
