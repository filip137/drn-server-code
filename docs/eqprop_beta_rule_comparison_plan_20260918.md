# Beta selection rule comparison

Authorized September 18, 2026. Seed-0 diagnostic, not training qualification.

Compare clean EqProp/BPTT cosine thresholds .90, .95 and .99, applied to every
weight matrix or to the raw concatenated weight gradient, both with and without
the symmetric norm-difference limit .10. Select the largest tested passing beta
for each of 12 rules per architecture/scheme. Require every batch at both
initialization and the BPTT best-validation checkpoint to pass; also report
checkpoint-specific limits. Do not reduce selected beta by another decade.

All nine Conv1/2/3 x baseline/ours/legacy surfaces use original wide [0,100]
zero-bias Adam checkpoints and original initializers. Keep the preserved
physical-KCL runtime, float64, centered frozen-current EqProp, unchanged
preprocessing and T=K=4/6/8. BPTT differentiates through exactly K iterations
from the identical post-T state. No optimizer or official-test access.

Reuse the September 14 selection cohort, including its four regression and
32 additional batches of 16. Hash-identical examples and boundaries are shared
across betas and schemes. These are selection data, not fresh confirmation.

Injected-beta anchors (baseline/ours/legacy): Conv1 100/30/3, Conv2 100/10/.03,
Conv3 10/3/.001. Multiply each by
`[.01,.03,.1,.3,1,2,3,5,7.5,10,20,30,50,75,100,300,1000]`.
Record base and injected beta with the original current-scale mapping.
153 cases x 2 checkpoints x 36 batches = 11,016 replays, 33,048 layer comparisons.

Whole-gradient statistics sum per-layer dot products and squared norms without
layer normalization or LR scaling. Exclude frozen biases. Apply the norm gate
at the same grouping as cosine. Preserve individual layer statistics and
contributions so domination by large layers remains visible. Non-finite values
fail and undefined cosines remain unresolved. Do not assume monotonicity;
report disconnected passing regions and open upper grid edges. The projected
KKT residual-p90 threshold .01 remains separate, including the Conv3 T8 caveat.

Implementation extends the existing replay's sufficient statistics and uses
one direct preparation/execution/analysis script. Freeze readable configs and
the analyzer source before GPU work. Test aggregation against explicit vector
concatenation, dominant-layer masking, norm grouping, zero/nonfinite values,
and selection. Reproduce overlapping September 14/16 layer metrics within
1e-10, verify unchanged checkpoints/parameters and all cohort/runtime guards.

Target: local RTX3090; fallback nom-cool-1 after availability/environment checks.
Keep the entire study on one recorded host. Check the authorized GPU inventory
and Jean Zay before allocation. Same-runner Conv3 smoke precedes production.
Budget: six physical GPU-hours including smoke/retries, estimated 2–4 hours;
update from measured timings. One worker initially. Record PID, command, logs,
per-batch heartbeat and terminal bundle; inspect actual progress throughout,
with full watchdog checks at most 30 minutes apart. Retry operational errors
only after diagnosis; retain scientific failures without changing the grid.

Results: `results/eqprop-beta-rule-comparison-20260918-v1/`; each case has the
canonical manifest/status/metrics/result bundle. Publish the 108 selections,
checkpoint-specific values, raw measurements, norm-contribution summaries and
matplotlib PDF/PNG figures in `paper_ready_results/`. Curate the conclusion in
the experimental manifest. No multi-seed confirmation, training resumption,
paper edit or automatic beta promotion is included.

## Authorized transport amendment, September 18

At the user’s request to parallelize, keep all 51 Conv3 cases on local RTX3090
and move all 102 Conv1/Conv2 cases to the idle Akib RTX3080. All schemes and
betas for each architecture remain on one host. This supersedes the original
single-host placement only; numerical source, initializers, checkpoints, cohort,
and scientific configuration remain hash-identical apart from filesystem paths.
Allow up to three GPU-hours per host, six combined including prior local work
and smokes. Expected remaining wall time 1.5–2 hours, refined after remote timing.
Remote output: `/home/filiposana/server_code/results/eqprop-beta-rule-comparison-20260918-v1/`.
A Conv2 smoke on Akib must reproduce the historical batch before its 102 cases.
The local two-worker benchmark yielded 1.068x throughput, below the declared
1.10x cutoff; use one worker per GPU. Preserve the original frozen plan and all
completed local results. Collect remote canonical bundles before final analysis.
