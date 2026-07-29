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
submission, validate the exact staged source, config, environment, wrapper,
resource request, and one short live canary.

## Launch

The shared target in `configs/experiment_targets.json` submits an existing
remote command:

```bash
python -m experiments.launch run jean-zay \
  --name NAME \
  --log /lustre/fsn1/projects/rech/fmu/$USER/server_code/results/NAME/slurm-%j.log \
  --dry-run -- \
  python EXPERIMENT_SCRIPT.py --config CONFIG.json
```

Remove `--dry-run` only after the exact source/config check and live canary
pass. Use an experiment-specific Slurm file when modules, arrays, packing, or
resource details exceed the shared target.

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
