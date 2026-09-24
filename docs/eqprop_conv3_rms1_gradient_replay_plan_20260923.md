# Conv3 gradient quality at initial output-displacement-one betas

User-requested exploratory checkpoint replay, September 23, 2026. Compare
ours injected beta1.385 with legacy .02173 at the shared saved seed0
initialization and the two p99 epoch30 final checkpoints. Keep these replay
betas fixed: the trained checkpoints were originally trained at .987333678708
and .1 respectively, with endpoint sigma5e-4 and inherited Adam learning rates.
This is not training at the new betas or a guarantee of output RMS one after
training. No optimizer steps, accuracy evaluation, or official-test reads.

Use the same36 validation batches of16 ordinary-MNIST examples as the earlier
studies, float64 centered frozen-current EqProp, input gain360, explicit
perfect diodes, exact-zero biases, and the original [0,100] weights. The
injected-beta convention is base_beta=B/(voltage_amp/current_amp)^3.
Clean plus four matched endpoint draws at sigma5e-4 use seeds
2026092101/2026092111/2026092121/2026092131. Four cells give2,880 layer rows.

All cases use T16/K8. Prior initialization T8/16/32,K8/32 and p99-trained
T8/16,K8/16/32/64 diagnostics show negligible iteration effects on these
weights; new projected-KKT residuals must still pass and are reported. BPTT
differentiates exactly K zero-nudge steps from the identical post-T state.
Compare cosine, relative gradient error, norm ratio, RMS, near-zero fraction,
actual noise/clean gradient ratio and phase displacement for every weight.
Report per-batch medians and10–90% spread; compare schemes using paired
batches, averaging the four noise draws within a batch. Noisy draws are not
independent training seeds. Saved gradient vectors are not retained; no
cross-epoch gradient-direction claim will be made.

Target: Akib RTX3080, isolated workspace. Inventory check found Akib idle;
local,nom-cool-1,Trex,Riri,Fifi,Loulou occupied; own Jean Zay queue empty.
Use a synchronous local CPU smoke because the local GPU is occupied, then a
same-config Akib GPU smoke before production to qualify memory and device.
Use the previously successful expandable allocator and256-MiB cuDNN workspace
settings. Expected5–10min production; combined smoke/replay budget1,800s.
Budgeted retries may address operational issues without changing the science.

Config: `configs/conv/eqprop_conv3_rms1_gradient_replay_20260923.json`.
Runner: `python -m experiments.replay_conv3_trained_beta_noise --config CONFIG
--device cuda`; local smoke uses `--smoke --device cpu --target local:CPU`.
Results: `results/eqprop-conv3-rms1-gradient-replay-20260923-v1`.
Remote root: `/home/filiposana/server_code/results/eqprop-conv3-rms1-gradient-replay-20260923-v1`.
Monitor process and per-batch semantic progress through a30-minute deadline;
collect and validate all four production bundles locally before interpretation.
Completion requires complete coverage, unchanged source/checkpoint bytes,
matched cohort/noise, valid smoke and production bundles, CSV/PNG/report,
and manual scientific interpretation in experimental_manifest.md.

Launch:13 focused tests pass. Four local CPU smoke cases pass in279.86s,
including all residual gates. All196 staged scientific files match their
local hashes. Akib was rechecked idle and started through the akib-ssh nohup
fallback, launcher507723. Four GPU smokes pass in10.30s. The remote wrapper
allows1,520s for GPU smoke plus production after charging the local CPU
smoke; source/config and scientific settings are identical. Remote PyTorch
is 2.5.1 with CUDA12.1 on Python3.12.12, unlike the local CPU smoke environment. All production
comparisons remain on the same RTX3080/runtime. Stage files, wrapper, logs,
and source snapshots are retained under the result root.

Completed: all four production cases, 2,880 comparisons and eight smoke
bundles validate locally. All 119 remote hashes match. All 2,304 projected-KKT
checks pass, maximum 2.82041e-7. No failed or excluded cases. Akib exited zero
and released the GPU. Production took 344.60s; total charged CPU/GPU smoke
plus replay was 634.77s within the 1,800s budget.

Ours has much better noisy Conv3 cosine at initialization (.697435 versus
.238445) and epoch 30 (.753430 versus .276865), improving on every paired
batch. Both early layers remain noise-dominated; legacy retains smaller clean
finite-beta error and a small readout advantage. These are fixed-beta replays
on existing p99-trained weights, not training qualification. See the
[report](../results/eqprop-conv3-rms1-gradient-replay-20260923-v1/analysis/report.md)
and [manifest](experimental_manifest.md#conv3-gradient-quality-at-output-displacement-one-betas-september-23).
