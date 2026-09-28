# X-007 — Where does legacy cease to lead in gradient alignment?

On September 25, Filip judged that further training to establish legacy's advantage
was no longer useful. The next question is whether ours or baseline beats legacy
in any part of the normalized output-displacement / endpoint-read-noise plane.
He confirmed **initialization first** and the current **D/F** normalization.
This is a prospective diagnostic question motivated by observed training outcomes,
not a claim that legacy is universally superior or that cosine determines accuracy.

## Evidence and competing explanations

[Exp007](../series/004-relative-output-training/results/exp-007-relative-output-training.md)
found no average training advantage for ours in the completed Conv1/Conv3 scopes.
Exp008's completed Conv2 comparison has legacy ahead in all twelve target/noise
pairs; increasing displacement to 6 helps all three schemes. These are seed-0,
epoch-10 validation results. The remaining Conv3 target-6 training cases were
withdrawn on this change of direction; completed and active-run results are retained.

Possible explanations remain distinct: legacy could have better alignment throughout
the sampled plane; it could be better only where previous training betas lay; or
large beta could trade reduced read-noise error for nonlinear finite-beta bias.
Output alignment alone may hide poor earlier-layer gradients (H004). Existing clean
Conv3 beta replays and the recent measured D/F calibrations can reduce new compute.

## Chosen coordinates and comparison

Use r = D/F, D = pooled RMS((v+ − v−)/2), F = pooled RMS(post-T free output),
on the physical 20-node output. Match r across schemes at their shared architecture
initializer. Pool squares and counts over the same 576 validation examples. This
is not equal absolute D: also retain F, D and D/sigma, since the Gaussian endpoint
noise remains absolute and the schemes have different free-output RMS.

For every convolution weight tensor and the readout, compare noisy centered EqProp
to clean finite-K BPTT from the identical post-T free state at the same parameters.
Use T=K4/6/8 for Conv1/2/3. No T/K experiments, optimizer steps or official-test reads.
Biases remain zero and frozen. Do not compare a noisy scheme to another scheme's
BPTT vector, or average noisy gradients before taking cosine.

## Compact screen and interpretation

Five target ratios: **0.2, 0.8, 2, 6, 10**. Five noise levels:
**1e-4, 5e-4, 1e-3, 2e-3, 5e-3**, plus sigma0 as a clean reference.
Reuse measured betas at0.8/2/6. Predict0.2/10 from the measured response, then check
those predictions once; do not run a dense root-finding or log-beta sweep.
Three independent endpoint-noise draws per batch screen sensitivity; use common
random draws across schemes and beta within a depth, and scaled common draws across
sigma. BPTT and clean endpoints can be cached without rerunning physical solves for
noise levels. The seed-0 initialization is the only model seed in this first stage.

Report per-layer cosine surfaces and ours−legacy / baseline−legacy difference maps.
A layer-specific advantage is meaningful but is not an all-layer win. Also compare
each scheme's worst-layer mean cosine to expose failures concealed by the readout.
An isolated winning cell is a candidate, not proof of a region. Confirm at most one
adjacent two-cell neighborhood per architecture with fresh noise draws. If no such
neighborhood appears, stop at the declared grid; do not keep widening the search
until a preferred winner emerges. A negative result applies to this sampled domain,
initializer, noise model and finite-iteration reference, not all beta/noise values.

[H008](../hypotheses/H-008-initial-alignment-region.md) and
[series005](../series/005-initial-displacement-noise-map/README.md) define the plans.
Trained-state replay is deferred until initialization evidence warrants it; no new
training is part of this research series.
