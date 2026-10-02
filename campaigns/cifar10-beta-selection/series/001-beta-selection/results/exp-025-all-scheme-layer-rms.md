---
experiment: "exp-025"
evidence: "validated-local"
summary: "Complete float64 replay: shared block-local targets exist in every block for both checkpoint groups. Full-network comparison retains the baseline EqProp epoch1 block3 exception; local and full-network references answer different questions."
verdicts: {"H-019": "inconclusive"}
---
# Result: shared displacement targets depend on training history and block

**All18 unique checkpoint replays completed and passed collection and full
analysis.** The21 labelled views keep EqProp0/1/5 separate from BPTT0/10/30/50;
only tensor-identical initializations are shared. Evidence comprises3024 pooled
layer rows and24192 minibatch rows, using256 fixed images in eight batches of32.
All computations use full float64, native T/K=[6,6,4], frozen checkpoints/BN,
no optimizer, augmentation, noise or official test split.

Root: `results/cifar-layer-rms-all-schemes-float64-20260930-v1/`.
`analysis/summary.json` has complete coverage and no exclusions. Checkpoint,
source, runtime precision and immutability checks, target attainment within3%,
reference invariance, raw-sum pooling and final-layer/output identity all pass.
RTX5090/3090 placements share numerical code and torch2.11.0+cu128/CUDA12.8;
GPU provenance is retained. The operational time limits differ by placement.
Final queue completion was September30 21:45 Paris. Charged time including the
cancelled early3090 attempt was48790.70s (13.55GPU-hours), below the20-hour cap.
The two3090 cases took115.0/117.3minutes;5090 cases took32.5–40.8minutes.
There are no remaining assigned jobs. Superseded float32 outputs and the
cancelled short-budget3090 attempt remain excluded and preserved as recorded
in the experiment note; no completed scientific case failed or was omitted.

## Matched block-only comparison — October 1 follow-up

At Filip's request, `analysis/local_bptt/` reanalyses the same complete float64
artifacts against **block-only BPTT**. Both methods start from the same saved
block input and free states and use the same fixed output loss force. This
isolates local gradient estimation under those conditions; full-network BPTT
advances all blocks before evaluating its loss. No new replay or training ran.
This is a requested follow-up analysis, not a retrospective replacement of
H-019's original full-network criterion or its verdict.

Largest passing **sampled block-output d/f target**, shared by every layer,
scheme and checkpoint within each group:

| Checkpoint group / local cosine criterion | Block1 | Block2 | Block3 |
|---|---:|---:|---:|
| EqProp0/1/5: pooled >=0.95 | 0.01 | 0.035 | 0.00003 |
| EqProp0/1/5: pooled >=0.99 | 0.005 | 0.01 | 0.00003 |
| EqProp0/1/5: every minibatch >=0.99 | 0.001 | 0.005 | 0.00001 |
| BPTT0/10/30/50: pooled >=0.95 | 0.01 | 0.005 | 0.001 |
| BPTT0/10/30/50: pooled >=0.99 | 0.005 | 0.0025 | 0.0005 |
| BPTT0/10/30/50: every minibatch >=0.99 | 0.0025 | 0.001 | 0.0005 |

Every sampled target from1e-6 through each endpoint passes that row's criterion;
gaps are not certified. At the every-minibatch0.99 target vectors, all pooled
and minibatch gradient norm ratios also lie within0.9–1.1. For EqProp targets
**[0.001,0.005,0.00001]**, worst minibatch cosines by block are
**0.993074/0.994415/0.999311**. For BPTT targets
**[0.0025,0.001,0.0005]**, they are0.994442/0.996975/0.996527.
These are diagnostic candidates requiring separately calibrated beta, not
demonstrated training or read-noise prescriptions.

The baseline epoch1 block3 exception disappears locally: at target1e-6 its
Conv2 pooled cosine is0.999999989. The remaining block3 restriction is real
in this local comparison: all three EqProp epoch1 schemes have a pooled0.99
upper sampled target of3e-5, versus0.125 at initialization. By epoch5 these
upper targets are0.001/0.00025/0.0001 for baseline/ours/legacy. Thus training
changes the useful response range substantially, but not monotonically.
Across all EqProp cases and layers, even the lowest tested output target1e-6
has minimum pooled cosine0.999959 and minibatch cosine0.999895. This grid
does not identify a lower failure boundary in float64.

Use the local curves to assess beta/nudging quality; retain the full-network
comparison below to study the remaining discrepancy relevant to BPTT training.
The source of that discrepancy has not yet been isolated.

### Proposed trained-checkpoint operating targets

Filip requests fixed RMS targets with local cosine strictly **>0.95** over
the available epoch1–50 evidence, followed by beta search ranges. Use
**[0.01,0.005,0.00001]** for blocks1–3: these are the largest sampled targets
passing every pooled layer and minibatch independently in both trained groups
(EqProp1/5 and BPTT10/30/50), across all schemes. This comparison of passing
decisions does not join the two histories into one training trajectory or
establish performance at unmeasured intermediate epochs. Initialization is
excluded from the beta ranges below.

| Block | Fixed output d/f | Working window: sampled endpoints | Worst EqProp minibatch cosine | Worst BPTT minibatch cosine |
|---|---:|---:|---:|---:|
| 1 | 0.01 | 0.005–0.01 | 0.952304 | 0.973186 |
| 2 | 0.005 | 0.0025–0.005 | 0.994415 | 0.957159 |
| 3 | 0.00001 | 0.000003–0.00001 | 0.999311 | 0.999940 |

These upper targets honor the requested cosine criterion; block1/2 have
limited margin. Block1's pooled/minibatch norm ratios span0.8257–1.1093, so
this choice does not also meet the earlier0.9–1.1 norm criterion. The windows
use the selected target and preceding sampled target; intermediate values are
not certified. No lower failure boundary has been established.

| Training history | Observed beta envelope, block1 | Block2 | Block3 |
|---|---:|---:|---:|
| EqProp1/5 | 0.1531–22.0771 | 19.1913–522.8088 | 0.026626–1.64560 |
| BPTT10/30/50 | 0.002645–0.049888 | 0.045154–5.46730 | 0.00010462–0.0135093 |

For a common logarithmic calibration search over these observed histories,
rounded outer brackets are **[0.002,30]**, **[0.04,600]**, and
**[0.0001,2]**. These bracket the checkpoint-specific betas producing the
working-window endpoints; they are not intervals in which every beta works
at every checkpoint. Recalibrate against measured RMS and expand if a new
state requires it. BPTT-history bounds are not forecasts of later EqProp
training, and these brackets omit initialization's much smaller betas.

`analysis/local_bptt/trained_target_proposal.json` records the exact selection,
group-specific passing targets and limiting layers; `trained_target_beta_brackets.csv`
contains all45 trained checkpoint/block pairs with beta and achieved RMS at
both endpoints. The decision uses the existing float64 replay only; no training
or new GPU diagnostic was launched.

## Original full-network comparison: common sampled ranges

The table gives the **largest passing sampled output d/f target** against
full-network BPTT, shared by
all schemes and all checkpoints within each group, including initialization.
Every sampled point from1e-6 through the listed endpoint passes. This does
not certify the gaps between grid points or locate the true boundary.

| Checkpoint group / criterion | Block1 | Block2 | Block3 |
|---|---:|---:|---:|
| BPTT0/10/30/50: pooled cosine >=0.95 | 0.01 | 0.005 | 0.001 |
| BPTT0/10/30/50: pooled cosine >=0.99 | 0.005 | 0.0025 | 0.0005 |
| BPTT0/10/30/50: pooled cosine >=0.95 and norm ratio0.9–1.1 | 0.005 | 0.005 | 0.001 |
| EqProp0/1/5: pooled cosine >=0.95 | 0.02 | 0.035 | None |
| EqProp0/1/5: pooled cosine >=0.99 | 0.005 | 0.01 | None |
| EqProp0/1/5: pooled cosine >=0.95 and norm ratio0.9–1.1 | 0.0025 | 0.035 | None |

Trained-only intersections have the same endpoints. BPTT block1/2 upper bounds
are limited by legacy epoch50; block3's0.95 bound is limited by legacy epoch10.
Block3 has two convolutions whereas blocks1/2 have three, so depth alone does
not explain the relative-displacement limits observed here.

For further **BPTT-checkpoint diagnostics**, candidate targets
**[0.005,0.0025,0.0005]** retain pooled cosine >=0.99 for all layers and cases.
Worst pooled cosines by block are0.992567/0.991738/0.998852; worst minibatch
cosines are0.989979/0.989629/0.996527. Every minibatch also passes cosine>=0.95
and norm0.9–1.1 at these targets. If >=0.99 is required in every minibatch,
use sampled upper targets[0.0025,0.001,0.0005]. These are noise-free candidate
responses; each requires its own block/checkpoint beta calibration.

## The early EqProp block3 exception against full-network BPTT

At epoch1, ours and legacy block3 pass pooled cosine>=0.99 from1e-6 through
3e-5. Baseline's only sampled pooled0.95 pass is1e-4, with no0.99 pass.
Consequently no common sampled target meets0.95 across the EqProp group.
At epoch5 all three schemes have excellent agreement at1e-6: minimum pooled
cosine across all eight layers is >=0.999980, and minimum minibatch cosine
is >=0.999927. The restriction is not a monotonic worsening with epoch.

For baseline epoch1 block3 Conv2 at target1e-6, EqProp/local frozen-current
BPTT cosine is0.999999989, local/native BPTT cosine is0.945834551, and
EqProp/native cosine is0.945834892. The mismatch therefore persists after
small-nudge EqProp agrees with the local reference. This points to a
finite-iteration/reference difference beyond the float32 precision problem;
it does not identify which solver/forward operation causes that difference.

Raising baseline's target to1e-4 gives worst pooled native cosine0.962508,
but worst layer/minibatch cosine is only0.770202. The best shared EqProp
block3 compromise on this grid is1e-5, with worst pooled cosine0.945858 and
worst minibatch cosine0.647383, both limited by baseline epoch1. Treat neither
as a reliable all-case prescription. A narrow unsampled passing interval is
not excluded, and would still need minibatch assessment.

## Interpretation and decision

Under its original full-network criterion, H-019 has **scoped support for
every BPTT block and for EqProp blocks1/2**.
Its unrestricted transfer claim remains inconclusive: no shared sampled
EqProp block3 target passes, while the grid does not exhaust all possibilities.
Use separate training-history recommendations and keep the baseline epoch1
exception explicit. Good BPTT-checkpoint gradients do not establish recovery
of a deteriorated EqProp training trajectory, nor training accuracy gains.

Full float64 permits very small responses with high cosine across nearly all
cases; increasing displacement beyond the useful range still misaligns
trained gradients. Output displacement remains a useful calibration variable,
but it does not fully specify native-gradient agreement. The minimum tested
1e-6 is not an established lower limit. Read-noise tolerance was not tested.
The next focused diagnostic would separate free-state and nudged-iteration
sensitivity of baseline epoch1 block3 at one fixed small beta. No further
simulation or training is launched by this review.

## Plots and reproducibility

[Local EqProp0/1/5 overview](../../../../../results/cifar-layer-rms-all-schemes-float64-20260930-v1/analysis/local_bptt/eqprop_output_worst_layer_and_common_targets.jpg)
and [local BPTT0/10/30/50 overview](../../../../../results/cifar-layer-rms-all-schemes-float64-20260930-v1/analysis/local_bptt/bptt_output_worst_layer_and_common_targets.jpg)
show the matched block-only comparison. Six per-scheme eight-layer JPGs and
the same CSV tables are under `analysis/local_bptt/`; curve solid/dotted/dashed
lines show pooled cosine/minibatch minimum/minibatch median against each
layer's own d/f. Reproduce with
`python experiments/analyze_cifar_layer_rms_transfer.py --reference local_bptt`.
All6993 local sampled decisions were cross-checked against raw rows; the
analyzer's default full-network intersections remain identical to the original.

[EqProp0/1/5 block-output overview](../../../../../results/cifar-layer-rms-all-schemes-float64-20260930-v1/analysis/eqprop_output_worst_layer_and_common_targets.jpg)
and [BPTT0/10/30/50 block-output overview](../../../../../results/cifar-layer-rms-all-schemes-float64-20260930-v1/analysis/bptt_output_worst_layer_and_common_targets.jpg)
show worst-layer cosine and sampled intersections. Each scheme also has its
own eight-layer cosine-versus-own-d/f JPG, separately for both groups, under
`analysis/{eqprop,bptt}_{baseline,ours,legacy}_own_layer_cosine.jpg`.
`layer_metrics.csv`, `batch_metrics.csv`, `common_target_points.csv` and
`candidate_betas.csv` retain all measurements and checkpoint-specific betas.

Reproduce with `python experiments/analyze_cifar_layer_rms_transfer.py`.
The full CPU analysis and eight JPGs completed automatically after the last
collection; root reviewed them on October1. Earlier partial analysis remains
labelled as such. Detailed case/source/checkpoint identities and timing remain
in the manifests, collection receipts and `analysis/summary.json`.
