# Conv3 p99 beta/K checkpoint replay

User request: increase K to test whether the high-beta comparison was limited
by nudged-phase settling. This is exploratory read-only validation replay,
with no training, checkpoint changes, or official-test access.

Config: `configs/conv/eqprop_conv3_p99_beta_k_replay_20260923.json`.
Runner: `experiments/replay_conv3_trained_beta_noise.py`.
Results: `results/eqprop-conv3-p99-beta-k-replay-20260923-v1`.

- Checkpoints: completed p99 sigma=5e-4 training, legacy epoch30 final,
  ours epoch30 final, and ours epoch27 best validation. Source paths and hashes
  remain exactly those of the preceding beta replay.
- Common T=16; K=8,16,32,64; injected beta=.987333678708,3,10.
- Same36 validation batches of16 (576 examples), clean and sigma=5e-4 with
  the same four endpoint-noise seeds. Frozen-current centered float64 EqProp,
  exact-zero biases, inherited explicit diode dictionaries, preprocessing,
  output encoding, loss, and amplification remain unchanged.
- 36 production cells and25,920 layer comparisons. Local same-config smoke:
  all12 checkpoint/K combinations at beta10 on one batch.
- Compare EqProp to BPTT at matching K and separately to fixed K=64 BPTT.
  Compare each clean EqProp gradient to K=64 EqProp at the same beta, and each
  BPTT gradient to K=64 BPTT. Compute K64 first per checkpoint to retain those
  references in CPU memory. Report median and10–90% spread, gradient scale and
  error, endpoint signal, and projected-KKT/raw-output residuals.
- K32-to64 agreement assesses convergence. K64 is a tested finite reference,
  not an assumed exact equilibrium gradient. Residuals use the existing .01
  rule; this diagnostic does not requalify a training operating point or LR.
- Compare best sampled beta separately for each scheme by both minimum
  layerwise median cosine and third-convolution cosine. A boundary winner is
  best tested, not a global optimum. Accuracy at new beta/K is unmeasured.

Target: Fifi RTX5090, isolated workspace under the remote result root.
Expected40–70 minutes; total smoke plus production cap7200 seconds. All listed
GPU hosts and Jean Zay checked September23: Fifi/Trex/Akib available, local
available for smoke; nom-cool-1/Riri/Loulou occupied; no own Jean Zay jobs.
The preceding accepted p99 T16 replay is the free-state convergence evidence;
the requested K sweep itself measures the remaining gradient convergence.

Completion: all declared cells have canonical successful bundles; source and
parameter immutability and matched cohorts/noise pass; authoritative outputs
copied locally and validated; no silent failures or exclusions; report, CSVs,
matplotlib plots, dashboard handoff, and manual manifest interpretation.

Status: complete and reviewed; launched on Fifi tmux `p99-beta-k-replay-20260923`. All12 local GPU
smokes passed in55.19s;12 focused tests passed;194 staged input files match.
One sandbox-blocked smoke is preserved and excluded; no scientific work ran
in that attempt. The resolved config charges10s for it.

Completed: all 36 cells and 25,920 comparisons validate locally; all 20,736
residual records pass. Fifi exited 0 and released its GPU. Runtime was
2,200.07s, charged 2,265.25s including smoke/allowance, within 7,200s.
All 509 remote output hashes match. Every prior K8 cell reproduces exactly.
Noisy median-cosine changes from K8 to K64 remain below 9.62e-7; the best
sampled beta and scheme ranking do not change. See the
[report](../results/eqprop-conv3-p99-beta-k-replay-20260923-v1/analysis/report.md).
