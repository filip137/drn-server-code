# Jean Zay Notes

## Access and paths

- Use the configured alias: `ssh jean-zay`.
- Default V100 account: `fmu@v100`.
- Partition/QoS/constraint: `gpu_p13`, `qos_gpu-t3`, `v100-16g`.
- Source checkout:
  `/lustre/fswork/projects/rech/umg/ucy17uy/server_code`.
- Results:
  `/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results`.
- Use `$JOBSCRATCH` for per-job staging when a job needs a private extracted
  source tree.

Do not add unsupported memory directives to V100 jobs. Validate the remote
checkout, module, paths, and a short real smoke before submitting a long
batch.

The shared target in `configs/experiment_targets.json` submits an existing
remote command:

```bash
python -m experiments.launch run jean-zay \
  --name NAME --log /lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/NAME/slurm-%j.log -- \
  python EXPERIMENT_SCRIPT.py --config CONFIG.json
```

The launcher creates the log directory before submission. Use an
experiment-specific Slurm file directly when modules, arrays, packing, or
resource details need more than the shared target.

## Monitoring

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES%40 -P
```

## Copy results locally

Wait for a terminal scheduler state, then copy the exact run directory:

```bash
mkdir -p /home/filip/server_code/results/.incoming/RUN_ID
rsync -a --partial --protect-args \
  "jean-zay:/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/RUN_ID/" \
  "/home/filip/server_code/results/.incoming/RUN_ID/"
```

Run the experiment's collector or integrity checks on the local copy. Keep
the remote copy until local validation succeeds; remote cleanup is a separate
explicit action.
