# Conv3 refined-beta ten-epoch training

Current execution: original task **2178212_0** completed; recovery array
**2178358**, indices **1-5**, throttle raised to **4** after the original
finished, at most four H100 GPUs total.
Two-hour cap per task; the scientific configs remain unchanged.

Filip requests the six missing training runs, with three concurrent runs on
one RTX 5090. This completes the training comparison after grid refinement;
calibration alone did not complete the earlier request. Exploratory selection
evidence, not official-test or full-horizon qualification.

| Scheme | Injected beta for every matrix >.95 | Injected beta for every matrix >.90 |
|---|---:|---:|
| Baseline | 147.682614594 | 404.141105702 |
| Ours | 2.49274796756 | 5.26875648112 |
| Legacy | 2.81845428732 | 4.42250110273 |

Use exactly these measured values, not the rounded display values. Keep
seed0, ten epochs, ordinary MNIST's fixed55k/5k train/validation split,
batch16 and validation batch64, zero read noise, float64 centered frozen-current
EqProp, T=K=8, wide weights[0,100], zero biases, saved matched initializer,
fresh Adam state, and unchanged scheme-specific exact LR vectors. Numerical
source is the preserved source used by the preceding layerwise training.
The accepted user-fixed T/K operating point and its finite-T residual caveat
remain unchanged. No new calibration, LR search, test read, or extra seeds.

Primary endpoint: final epoch10 validation accuracy, comparing .90 to .95
within each scheme, and the previous beta controls. Report best validation,
loss trajectories and any nonfinite failure epoch/batch. Early stability is
ten finite epochs with final accuracy less than5pp below its own best;
this is a gross-collapse screen, not a claim that a4pp loss is acceptable.
Retain every scientific outcome. Small single-seed differences are descriptive.

Execution: local same-runner one-train/one-validation-batch smoke for all six
configs using `experiments.exact_run --smoke`. Admit one idle authorized
RTX5090 after checking all inventory hosts and Jean Zay; all four accessible
5090s were occupied at preparation. Wait for the first free one, then keep
all six runs there. Confirm three concurrent full-batch smokes fit with
at least2GiB free GPU memory before production. Three independent slots each
run one scheme's .95 case followed by its .90 case. A scientific failure is
recorded and does not suppress the paired case; an operational failure stops
that slot for diagnosis. Never replace or stop unrelated jobs.

Expected training wall time4–6h after admission, updated from measured first-
epoch throughput; queue time unknown. Cap each run at6h and the whole occupied
GPU at12h including target smoke, plus at most10min local smoke. Wait for
admission up to24h while actively monitoring; no GPU time is consumed waiting.
Monitor real batches/metrics/logs and resource state at most30min apart,
with45min stale-progress detection. Copy outputs locally and validate them
before final conclusions. Reuse existing controls only with their first ten
epochs explicitly identified. Update the beta-protocol assessment with the
new training evidence rather than leaving the refined selections untested.

Configs: `configs/conv/eqprop_conv3_refined_beta_training_20260919_v1/`.
Local results: `results/eqprop-conv3-refined-beta-training-20260919-v1/`.
Remote results: `/home/filip/server_code/results/eqprop-conv3-refined-beta-training-20260919-v1/`.
Final report: `paper_ready_results/conv3_refined_beta_training_20260919.md`.

## User-directed placement change

All accessible5090s were busy. Filip explicitly requested local, Akib and
nom-cool-1 next, otherwise Jean Zay. Local3090 and Akib3080 are idle;
nom-cool-1 is busy. Six local exact-run smokes passed, peak GPU usage4970MiB.
Use three concurrent local slots for baseline and ours (four runs total),
and one Akib slot for the legacy pair. Keep each scheme pair on one host.
Run the local three-way smoke and the two Akib smokes before production.
Budget is six hours per target, twelve physical GPU-hours combined, plus
at most ten minutes local smoke. The lean direct parallel transport invokes
unchanged experiments.exact_run; no new scientific runner or calibration.

## Final user-directed target: Jean Zay

Filip then requested Jean Zay to reduce completion time. All six production
runs move to one Jean Zay array,0-5%4,one V100/task and four GPUs concurrent.
No local or Akib production was launched; Akib staging and local smokes are
retained as preparation. Ten local smoke cases (six serial plus four in the
three-way transport check) completed; peak three-way local memory14641MiB.
Use fmu@v100,gpu_p13,qos_gpu-t3,v100-16g,10CPUs/task,maximum4h/task,
24GPU-hour defensive cap. Expected1–2h/task,2–4h overall after scheduling;
queue time is unknown until submitted. Numerical config and saved source
remain unchanged. The repo's local-canary-then-production rule explicitly
supersedes the generic separate remote-canary step; restore a remote canary
only if a Jean-Zay-specific failure occurs. A direct Slurm array has no custom
supervisor or automatic resubmitter; record one parent and monitor task IDs
using squeue/sacct. Results go under the authorized fmu result root.

## Final queue contract: H100 development QoS

Live production-QoS forecasts were days away. Verified development QoS permits
six short exploratory jobs with four concurrent GPUs and a two-hour wall cap.
The H100 development forecast was near immediate (Slurm timestamps are CEST;
cluster UTC was checked). Final resource contract: umg@h100,gpu_p6,
qos_gpu_h100-dev,h100,one GPU and24CPUs per task,array0-5%4,02:00:00.
Load arch/h100 then pytorch-gpu/py3/2.5.0. Budget12GPU-hours plus local smokes.
The two-hour cap is operational, not an epoch reduction; incomplete runs will
remain explicitly incomplete and require diagnosed recovery. All six numerical
configs still specify the full ten epochs. One final local semantic smoke
per exact config is required after this finalization, then one submission.

## Diagnosed launcher recovery

Original indices1–5 failed before training because the exact runner inherited
SLURM_ARRAY_TASK_ID after the wrapper had already selected a single config.
The corrected v2 wrapper explicitly passes --index0 (as two CLI arguments).
All49 focused tests and the local nonzero-index regression smoke passed.
Live H100 recovery canary2178310 completed0:0; its collected semantic bundle
and resource contract validate. Original task0 continues uninterrupted.
The five failed startup directories remain excluded, preserved alongside the
retry-v2 replacements. The canary is smoke evidence, not a training outcome.
