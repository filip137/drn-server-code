# Per-matrix beta selection and ten-epoch training

User request: Conv2/Conv3, all three amplification schemes, choose beta so
every weight-matrix gradient cosine exceeds .90 or .95, then compare ten-epoch
training performance. This is exploratory selection evidence, not paper test
accuracy. Use the largest passing **tested** injected beta; no interpolation
or extra safety decade. The request specifies cosine, so no norm constraint
is added. Report symmetric norm mismatch separately (all selected points
exceed .10 somewhere). Apply strict `>` to every matrix, every one of the
36 fixed calibration batches from the validation partition, at both initialization and the saved BPTT best
checkpoint. Reuse the completed September 18 calibration measurements.

| Architecture/scheme | beta at .90 | beta at .95 | Work |
|---|---:|---:|---|
| Conv2 baseline | 500 | 500 | One new run, Fifi |
| Conv2 ours | 50 | 30 | Two concurrent new runs, local RTX3090; Loulou attempts superseded as recorded below |
| Conv2 legacy | 30 | 30 | Reuse prior pilot, nonfinite epoch 5 |
| Conv3 baseline | 300 | 100 | New beta300 on Riri; reuse beta100 control first ten epochs |
| Conv3 ours | 3 | .9 | Reuse beta3 control first ten epochs; new beta.9 on Trex |
| Conv3 legacy | 1 | 1 | Reuse completed ten-epoch pilot |

Twelve rule conditions map to nine distinct scientific settings and five new
runs. Legacy selections are open upper grid edges, not established maxima.
Equal selections are not independent replications. Reused evidence is named
explicitly, including the failed Conv2 legacy outcome. Historical 30-epoch
control runs contribute only epochs 1–10: fixed learning rates, no horizon
schedule, identical initialization and epochwise data order. Validate their
bundles and comparison fields; do not relabel full-run checkpoints as epoch 10.

Freeze seed 0, ordinary MNIST 55k/5k split, batch16, zero read noise, centered
frozen-current float64 EP, wide weights [0,100], zero biases, existing exact
Adam vectors, and T=K=6/8. Use the same preserved numerical source and saved
initializers as `eqprop-beta-training-stability-20260918-v1`. Only beta and
study labels differ from the corresponding ten-epoch config. User-fixed T/K
and the existing Conv3 finite-T residual caveat remain unchanged. No new
rho/LR search, official-test reads, or automatic three-seed promotion.

Primary performance endpoint: final epoch-10 validation accuracy difference
in percentage points, relative to (1) the stricter .95 selection and (2) the
historical training beta for the same architecture/scheme. Also report best
validation, train/validation loss trajectories, nonfinite failure epoch/batch,
and any accuracy collapse. Early stability requires ten finite epochs and a
final drop strictly below 5pp from the run's own best. A failed candidate is
not assigned an invented epoch-10 accuracy. Single-seed small differences
are descriptive and cannot establish a statistically reliable improvement.

All authorized GPU hosts and Jean Zay were checked before placement. Riri,
Trex, Fifi and Loulou were idle RTX5090s; Loulou runs both Conv2 ours cases
concurrently, others one case each. Preserve at least 2GiB memory headroom.
Keep the two threshold values of a scheme on the same host when both are new.
Historical controls and reused pilots have recorded host differences; this
is an exploratory multi-host comparison, not a new formal matched paper grid.
Expected runtime 1.5–2h after launch, per-new-run timeout4h, 20 worker-hour
maximum and approximately16 physical-GPU-hour ceiling including small smoke
overhead. No Jean Zay jobs are needed given idle local-network 5090 capacity.

Smoke each new config locally through `experiments.exact_run --smoke`, then
on its target through the same runner before production. GPU idle checks
must still pass immediately before admission; never disturb another user's
job. Keep logs, resolved configs, canonical bundles, native exit receipts,
and sampled gradients (eight samples/epoch). Retain numerical failures without
changing parameters. Monitor processes/GPU/artifact progress at most30min
apart, with a45min stall threshold; collect and validate locally before
conclusions. All five scientific workers are independent; no queue extension
or custom runtime controller is needed.

Configs: `configs/conv/eqprop_layerwise_beta_training_20260919_v1/`.
Study: `results/eqprop-layerwise-beta-training-20260919-v1/`.
Remote roots: `/home/filip/server_code/results/eqprop-layerwise-beta-training-20260919-v1/`
on each recorded target. `cases.json` names all nine settings and reuse paths;
`selection_audit.json` records all twelve threshold selections and worst
matrix metrics. Final report: `paper_ready_results/layerwise_beta_training_20260919.md`.

## Placement revision after unrelated GPU arrivals

Four unrelated GPU clients arrived on Loulou after admission, substantially
slowing the two Conv2 ours pilots. A paired100-batch operational probe on the
idle local RTX3090 measured .1666 seconds/batch per worker, about96 minutes
for a fresh ten-epoch run plus validation. Replace both Loulou attempts with
unchanged local configs and the same saved initializers, seed and minibatch
order. This throughput-based decision is fixed before replacement training;
no outcome-based choice between attempts is permitted. Preserve the partial
Loulou bundles as superseded operational attempts, not numerical failures.
The two short throughput probes are excluded from the scientific comparison.
Both betas remain on the same physical target for the paired comparison.
The original20 worker-hour total ceiling remains sufficient, including all
superseded/probe time; each replacement retains a four-hour cap. Expect about
100 further minutes for the pair. The local same-runner smoke gate must pass
before retiring only the identified Loulou training children. All other GPU
clients remain untouched. See placement-revision.json and handoff.json.
