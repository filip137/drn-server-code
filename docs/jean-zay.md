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

## Monitoring

- Use `squeue -u "$USER"` for active state.
- Use `sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,AllocTRES%40 -P` after completion.
- Check logs under the run-specific `slurm/` directory in the source checkout.
- Count completions with `find RESULT_ROOT -name metrics.json | wc -l`.
