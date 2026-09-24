# Conv3 T/K cosine across amplification schemes

Filip requested varying K and extending the previous T diagnostic to baseline
and legacy. Use a compact complete grid `T=[8,16,32]`, `K=[8,16,32]` for all
three schemes at training/read sigma 5e-4. The final epoch-30 seed-0 A100
checkpoints and their training injected betas remain fixed: baseline
404.141105702, ours 5.26875648112, legacy 4.42250110273. This is a cosine
diagnostic, without residual-based filtering or training qualification.

Keep the 36 batches of 16 validation examples, four matched read-noise draws,
clean controls, float64 centered frozen-current EP, original conductance
weights, exact-zero biases, explicit perfect-diode dictionaries, input gain
360 and paired 20-output squared loss. At each point EP uses K steps per
nudged phase and is compared with BPTT through exactly K zero-nudge steps
from the same post-T state. Changing K therefore changes both estimators;
do not interpret this as comparison with one fixed equilibrium gradient.
No training, accuracy evaluation or official-test read is performed.

Reuse the three T8/K8 training-beta cells from the original trained beta
study and ours at T16/K8 and T32/K8 from the subsequent T-transition study.
Their exact source paths and result hashes are listed in `cached_cells.json`
under the new result root. The other 22 cells are new. All 27 points share
the same full cohort and exact noise seeds. The three existing T64/K8 cells
may be shown separately as context, outside this declared 3×3 grid.

Use the unchanged `experiments.replay_conv3_trained_beta_noise` runner with
eight readable configs under
`configs/conv/eqprop_conv3_cosine_tk_schemes_20260922_v1/`. Each config holds
one T/K pair and the schemes still requiring that point. K=8 configs at T16
and T32 include baseline and legacy; all six higher-K configs include all
three schemes. Each beta factor is exactly 1.

Target: Akib RTX3080 via SSH `akibscomputer`; Python
`/home/filiposana/miniconda3/envs/py312/bin/python`. Every workstation GPU
except Akib is occupied at planning; Jean Zay has no user job. Akib matches
all five cached points' numerical environment. Use a new isolated workspace.
Check the most demanding T32/K32 three-scheme one-batch smoke first to verify
memory and iteration semantics, followed by one-batch smokes for the other
seven exact configs. Production begins only after these pass. Do not reduce
batch size or change precision to make a failed memory check pass. Diagnose
an operational failure and preserve its artifacts before choosing a remedy.

Estimated runtime is 45–60 minutes on one GPU. The outer launcher enforces a
5,400-second cap, including all smokes. Each config has a 900-second internal
cap including its smoke. Monitor batches/cells and GPU occupancy at least
each minute while active, with launcher PID, log and exit-code tracking.
Stop only our invalid or obsolete worker; never interfere with unrelated jobs.

Command per config:
`python -m experiments.replay_conv3_trained_beta_noise --config CONFIG`;
append `--smoke` for its semantic check. Remote outer root:
`/home/filiposana/server_code/results/eqprop-conv3-cosine-tk-schemes-20260922-v1`.
Scientific outputs are beneath its isolated `workspace/results/` tree.
Authoritative local root:
`results/eqprop-conv3-cosine-tk-schemes-20260922-v1/`.

Expected new coverage is 22 production cells, 22 smoke cells and 15,840
production layer comparisons. With the five cached cells, the matrix has
19,440 layer comparisons. Collect and validate all included bundles and
source/cohort identities. Plot layerwise clean/noisy cosine heatmaps on the
T/K grid and curves that isolate T at fixed K and K at fixed T. Report which
schemes and layers change, with median and 10–90% spread, and distinguish
finite-grid findings from convergence or minimum-T/K claims. Curate the
supported conclusion in the experimental manifest.

## Launch and smoke outcome

Akib nohup supervisor 486843 passed GPU admission and completed all 22
one-batch smokes before production. T32/K32 passed for all three schemes at
the original batch size 16 and float64 precision. Runtime iteration guards
verify that both BPTT and EqProp use the requested K and common post-T state.
No memory workaround, numerical change or failed attempt was needed.
Completed production cells are collected and validated incrementally; partial
plots are explicitly labeled until all 27 included matrix cells are present.

## Completed result

All 22 new production cells and 22 smokes completed successfully. Including
the five cached anchors, all 27 grid cells and 19,440 layer comparisons are
present and validate locally. All 664 collected output file hashes match
Akib; all 12 checkpoint source files remain unchanged. All 324 scheme/T/batch
groups preserve identical post-T state and force hashes across K, and every
included batch replay verifies the requested BPTT and EP iteration lengths.
There were no missing points, failed attempts, retries or scientific exclusions.

The launcher exited 0 and released its worker. Replay including smokes took
2,400.39 seconds, total wall time 2,462 seconds, within the 5,400-second cap.

Baseline and legacy noisy layer medians vary by at most 1.61e-4 and 6.55e-6
across the grid. Ours has a strong interaction: larger K makes the readout
more negative at T8 but improves it at T16 and T32. At T32, K8→K32 raises
readout cosine .82046→.85758 while lowering clean Conv1 .96967→.47268;
there is no uniform gain across layers. The readout pattern is also present
without read noise. These are fixed-beta, fixed-checkpoint results with a
matched finite-K BPTT reference; they do not establish a general optimum.

[Curated interpretation](experimental_manifest.md#conv3-tk-cosine-interaction-across-amplification-schemes-september-22)
· [Final report and plots](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/analysis/report.md)
· [Collection validation](../results/eqprop-conv3-cosine-tk-schemes-20260922-v1/collection_validation.json).
