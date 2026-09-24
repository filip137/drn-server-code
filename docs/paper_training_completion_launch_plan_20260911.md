# Paper training completion launch plan

Approved 2026-09-11; replication scope extended by Filip on 2026-09-12.
**New admissions paused by Filip on September 14 at 12:48 CEST.** Let the
current single-case workers finish and stop at their run boundaries, as the
stated default interpretation of the pause. Collect Main's next completion
and review coverage with Filip. Do not launch any prepared follow-up or
resume automatically. The proposed budget/placement extension is deferred
to that review. Training and validation only.
The [completion manifest](paper_ready_results_manifest.md) tracks all 216 cells;
38 validated seed-0 bundles are already in `paper_ready_results/`.

## Authorized campaign and limits

| Block | Original missing runs | Authorized work | Release condition |
|---|---:|---|---|
| Wide BPTT | 18 | All 18 | Frozen settings and smoke |
| Wide centered EP | 19 | Pilot and all 18 replications | Complete wide pilot gate, now passed |
| Bounded BPTT | 60 | All 60 | Frozen settings and smoke |
| Bounded centered EP | 81 | Seed 0, then seeds 1/2 for qualified groups | Common beta and all three full seed-0 ceiling pilots pass |
| Total | 178 | All eligible cases within the hard GPU-hour ceiling | Failed scientific groups remain held |

On September 12 Filip explicitly authorized continuing to all three EqProp
seeds when a suitable beta is found. For bounded EP, release each
architecture/scheme group only after its common beta passes the numerical
checks and all three seed-0 ceiling pilots pass full-training stability.
The former requirement to wait for all 27 pilots and another replication
approval is replaced by this group release. There is no new GPU-hour budget:
the existing 300-hour hard ceiling remains in force.

The campaign has a **hard 300 allocated GPU-hour ceiling**, including
smokes, checks, failed attempts, and retries. Before each admission sum settled
GPU time and the full outstanding reservations, including the proposed jobs.
Admission fails above 300 hours. Pending scheduler time costs zero; an admitted
job retains its full reservation until authoritative accounting settles it.
Each Conv1 training has a 2-hour limit; Conv2/3 have 8-hour limits. Shell timeout
and Slurm wall limits enforce these reservations. Budget exhaustion may leave
some eligible cells unfinished; it does not authorize a budget increase.

Initial ETA is 2–4 days with available resources. Historical duration proxies
span 241–309 GPU hours for this tranche; bounded EP timing is an extrapolation.
Update estimates from observed times. Review the campaign by 2026-09-18.
Results, logs, initializers, and frozen source are under
`results/paper-training-completion-20260911-v1/`; raw bundles remain outside Git.

## Frozen science

### September 13 authorized follow-up

At 23:40 CEST Filip directs applying the revised baseline T/K to **both
BPTT and EqProp**, prioritizing Main, akibscomputer and nom-cool-1. Preserve
the 18 existing baseline BPTT results as the original operating-point
evidence; new matched baseline training replaces them only after validation.
No unrelated group changes. For RTX 5090 packing, measure both memory and
compute headroom. The two tested Conv3 EP workers reach 99% GPU utilization
but reduce throughput and threaten the unchanged eight-hour case cap, so
that particular packing is rejected. Other mixtures may be tested if they
have real compute headroom.

The first axis controls motivate a bounded refinement before full training:
retain native K=6/8 and test Conv2 T=12 and Conv3 T=20/24/32, first at beta
.1 and then .01 if needed. Require the unchanged full gate at all three
ceilings, and stop at the smallest tested passing T. These read-only checks
share the existing two-GPU-hour diagnostic cap and absolute deadline; no
additional training or budget increase is implied by the timing refinement.

The revised matched-baseline phase consists of 36 trainings: 18 BPTT
replacements plus the 18 previously missing EqProp runs. The original
216-cell collection stays intact while replacement results are collected
under a separately named baseline revision. The remaining original eight
qualified repetitions continue unchanged. Thus 44 full trainings remain
after the matched-baseline scope change, including the eight admitted cases.

First launch the three seed-0 BPTT ceilings per architecture at the newly
qualified T/K, with unchanged Adam vectors. Prefer Akib for Conv2 and Main
for Conv3 while nom-cool-1 completes its admitted wave. Recheck the chosen
beta at their new matched BPTT best checkpoints before the three full EqProp
pilots; then release EqProp seeds 1/2 after all three pilots are stable.
Admit only waves that fit the remaining hard 300-GPU-hour cap. Stage/configure
later seeds now, but do not expand that budget without Filip's instruction.

Filip requests larger T/K diagnostics for the unresolved bounded Conv2 and
Conv3 baseline beta qualification, and explicitly directs moving the five
waiting Conv3 seed-1 repetitions from occupied Trex to another RTX 5090.
Loulou is verified idle at 23:12 CEST and becomes the sole primary campaign
5090. Filip subsequently requested concurrent Loulou workers, otherwise
nom-cool-1. A measured two-worker check fits memory but reduces aggregate
throughput to .775x, so three ours cases run serially on Loulou (24-hour
reservation), alongside two legacy cases on nom-cool-1 (16-hour reservation).
Eight-hour case limits and source-v8 scientific configs are unchanged.
The already completed legacy tight
seed-1 case stays on Trex; record this authorized split of the comparison
group in the final analysis. No rerun of completed evidence is required.

The T/K follow-up is a validation-only numerical diagnostic on the same
seed-0 initializers, BPTT best checkpoints and four fixed batches. Preserve
all prior failed measurements and qualification thresholds. Choose a bounded
iteration ladder after inspecting the runner; record its cases and compute
cap before launch. The recorded diagnostic uses Conv2 T=K=12/24/48 and
Conv3 16/32/64, with native controls at beta .01, then the original larger
beta ladder at the first passing iteration count. Axis controls compare
doubled T/native K and native T/doubled K at the previously failing ceiling.
The Main GPU diagnostic is capped at two GPU-hours. Larger-T/K evidence does not silently replace the original
T/K=6/8 comparison or authorize an LR search. The 300-GPU-hour ceiling and
one-weekday-5090 restriction remain unchanged.

Import the exact named and ordered Adam rate vectors from the original configs.
Use corrected physical KCL, perfect diode with explicit parameter dictionaries,
zero frozen biases, shared initialization, and fresh optimizer state.
Conv1/2/3 retain 10/30/30 epochs, T=K=4/6/8, input gains 40/100/360,
and their existing architecture and preprocessing. Model and shuffle seeds are
0/1/2; split seed remains 0. Ordinary MNIST uses 55,000 training and 5,000
validation examples, batches 16/64. Official-test access is disabled.

Wide bounds remain [0,100] with normal Kaiming initialization. Bounded bounds
are [1e-5,Gmax], Gmax in {1e-4,5e-4,1e-3}. Each architecture/seed shares one
Uniform[1e-5,1e-4) initializer across all ceilings, schemes, and algorithms.
Preserve the three original bounded seed-0 files and generate seeds 1/2 from
the normal initialization path. BPTT is float32; EP loads those same numerical
weights then casts the complete runtime to float64.

Use `python -m experiments.prepare_paper_training_completion` to write plain
configs and lists. Use `python -m experiments.exact_run` for both same-config
one-batch smokes and full training. Accepted unchanged T/K points are cited;
there is no new rho or learning-rate search.

## EqProp gates

Centered frozen-current EP uses current_scale=auto, normalized gradients and
zero read noise. The injected beta triplets (baseline/ours/legacy) remain
Conv1 100/30/3, Conv2 100/10/.03, Conv3 100/3/.001. Base beta divides by
(Av/Ai)^L. Retain the known wide Conv3 one-decade gradient and residual caveats;
this training completion does not seal paper eligibility.

Run the corrected Conv1 legacy numerical check at injected beta 3, then its
full seed-0 training. Release all 18 wide EP replications only after all nine
wide pilot conditions pass, using the eight collected stable pilots.
Unexpected failure of the corrected wide pilot holds this block for a
scientific decision; do not silently change its beta.

For bounded EP, use four fixed validation batches of 16 examples at the shared
initializer and the matched trained BPTT best checkpoint for each of 27
conditions. Finish the six missing corrected Conv1/2 bounded legacy BPTT seed-0
references early. Test injected beta B × {1,.1,.01,.001,.0001}; select the
largest passing tested beta shared by all three ceilings of an architecture
and scheme. Every scored weight-layer/batch cosine must be >=.99 and symmetric
norm difference <=.10, with finite phases. Undefined or zero-reference
comparisons remain unresolved. Report residual p90 against .01 separately;
new unresolved equilibrium conditions hold for a scientific decision.

After numerical qualification, train the qualified full seed-0 pilots. A stable pilot
completes its full epoch budget with finite metrics/checkpoints, finite phases,
zero biases, a valid canonical bundle, and a final validation drop **strictly
less than 5 percentage points** from its own best. No accuracy floor applies.
For numerical instability, use the next qualifying beta in the declared ladder
and repeat all three ceilings of that architecture/scheme. Preserve every
attempt and charge its time. Ladder exhaustion holds the condition.
All three ceiling pilots of an architecture/scheme must pass before its
bounded seeds 1/2 are released at the same beta. An unresolved group does not
block other qualified groups. Admit replications in batches that fit the
remaining 300-hour budget, keeping one active Jean Zay array at a time.
Never tune separately for an unlucky replication seed.

## Placement and launch order

| Lane | Initial work | Maximum concurrent training |
|---|---|---:|
| Main RTX 3090 | Conv1 BPTT, local preparation/checks | 1 |
| Akib RTX 3080 via `akibscomputer` | Conv2 BPTT | 1 |
| Trex RTX 5090 | Conv3 BPTT | 1 |
| Jean Zay fmu@v100 | EP pilot, then gated replications/pilots | 4 total allocations |

Filip's September 12 placement preference puts Conv1 on available RTX
3080/3090 hosts, keeps Conv2 flexible, and prefers verified RTX 5090s or
Jean Zay for Conv3. Nom-cool-1 now hosts the complete unstarted Conv1 EP
replication groups. Existing partly completed matched groups retain their
recorded targets. Current admissions and any whole-group replacements are
recorded in the continuing manifest.

Check all authorized hosts and Jean Zay before allocating, and never use an
occupied lane. Initial wave: wide BPTT and corrected wide EP seed0; prioritize
six corrected bounded BPTT seed0 next; overlap bounded BPTT on spare lanes.
Keep each architecture/seed/weight-contract comparison group on one target;
move only whole unstarted groups. Do not make host choice depend on scheme.

**At most one campaign RTX 5090 GPU on weekdays**, including smokes/checks.
Weekend is Saturday 00:00 through Monday 00:00 Europe/Paris. Extra idle Fifi or
Loulou GPUs need matched Trex environment/data, and full job wall limit plus
one hour must fit before Monday. Enforce a hard Monday boundary for extra-GPU
overruns, record interruption, and restart unchanged later. The primary GPU
may continue. Riri is excluded pending verified access; nom-cool-1 is fallback.

Jean Zay uses one active array at a time, concurrency `%4`; all other campaign
V100 allocations count toward four. Use gpu_p13, qos_gpu-t3, fmu@v100, one
V100-16GB and 10 CPUs per task unless measured fit requires 32GB. Module is
pytorch-gpu/py3/2.5.0. Freeze a valid source directory explicitly; the configured
historical working directory is absent. Remote output root is
`/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/paper-training-completion-20260911-v1/`.
Repository/user policy requires synchronous local semantic smoke immediately
before production, with no separate remote canary unless a production failure
demonstrates a Jean Zay-specific execution problem.

A placement check is planned for Nom after its current original EqProp wave
is complete, collected and accounted. Time a short Conv3 baseline EqProp
T=24/K=8, beta .1, seed-0 largest-ceiling diagnostic through the same
scientific runner: 256 training batches and one validation batch, no official
test. Use a separate maximum .25-GPU-hour admission and 15-minute operational
bound; expected duration is a few minutes. Output will be campaign
`checks/nom_conv3_baseline_ep_timing_20260914/`. No timing work is admitted yet.
The current estimate for this revised case on a 3090 exceeds the eight-hour
case maximum, but that estimate is unmeasured. Use the measured rate, including
validation/launch overhead and a safety margin, to decide whether an entire
unstarted Conv3 EqProp replication seed group can fit on Nom after all three
Loulou seed-0 pilots pass. Keep the full seed group on one recorded host and
preserve its scientific configs. If it cannot fit, retain the Loulou placement.

## Jean Zay acceleration authorized September14

Filip authorized Jean Zay acceleration at 10:10 CEST September 14, with **at most four campaign GPUs concurrently across all Jean Zay jobs**. A fresh SSH attempt at 10:11 CEST timed out during banner exchange. The last scheduler record lists all-node electrical maintenance through **September16 18:00 CEST**; no new job has been submitted or reserved. The three revised Conv3 EqProp seed0 pilots are prepared as candidate array `0-2%4`, source v11, common beta .1, T24/K8, 30 epochs, eight hours per task (24 GPU-hours maximum; approximately 6–8 elapsed hours if scheduled together, extrapolated). [Prepared command and admission conditions](../results/paper-training-completion-20260911-v1/launch/jz_baseline_tk_conv3_ep_seed0.preparation.json).

Continue the lab queue while Jean Zay is unavailable. If access returns before any revised Conv3 pilot starts, the entire unstarted seed0 group may transfer after fresh source/account/scheduler checks, full budget admission, and synchronous exact local smoke plus semantic validation. Once that group starts on Loulou, keep it there; Jean Zay then takes an entire unstarted Conv3 repetition seed group only after all three full pilots pass and the new guard is frozen. Never duplicate a case or combine array throttles above four total GPUs. Host authorization does not raise the current 300-GPU-hour cap.

At11:07 CEST September14, coverage is **195/216; 21 full training runs remain**.
Main seed1 middle ceiling is collected at87.36/87.36% validation and settled
2.2725GPUh with both receipts zero. Main seed1 largest ceiling now advances
under sourcev9, a three-hour cap and a2.3-hour estimate; supervisor736645,
worker736654 and GPU736677, two exact semantic checks pass. Remaining Main
seed2 cases remain unadmitted. Settled228.841717 plus52reserved gives
**280.841717/300 committed**. Three Jean Zay SSH attempts through11:03CEST
time out; the four-GPU-capped alternative is prepared, unreserved and unsubmitted.

At12:17 CEST September14, coverage is **198/216;18 training runs remain**.
All original ours/legacy Conv3 repetitions are collected. The new active lanes
are Main Conv3 BPTT seed1/largest, Akib Conv2 EqProp seed1/middle, Nom Conv2
EqProp seed2/tight, and Loulou Conv3 EqProp seed0/tight; each launch is verified
twice. Settled255.889138 plus28reserved gives **283.889138/300 committed**.
Nom's collected timing projects9.788h (11.256h with margin) for Conv3 EqProp,
so that placement exceeds the current8h limit. A separate
[375-hour/twelve-hour parallel proposal](paper_training_parallel_acceleration_proposal_20260914.md)
is awaiting Filip's answer; no changed limit or pilot host split is applied.
The prepared Jean Zay pilot array is superseded by the started Loulou group.
Jean Zay remains authorized for eligible unstarted work at no more than four
concurrent GPUs, but SSH remains unavailable during scheduled maintenance.

## Monitoring and collection

Record source/config hashes, commands, limits, launch handles and reservations
before production. Verify worker/resource truth and semantic artifacts
immediately after launch, then inspect logs, metrics and artifacts every 30
minutes. Diagnose no progress after 45 minutes. Retry only understood
operational failures with unchanged science and new attempt directories.
Collect remote bundles locally preserving links, verify hashes and canonical
validation, then update `paper_ready_results` and its continuing manifest.
Do not seal checkpoints or read the official test split during this tranche.

September14 at 09:40 CEST: **194/216 training results are collected; 22 remain**.
All three revised Conv2 EqProp pilots pass at beta .1 and T12/K6; the last
gives 93.92/93.54% best/final validation. Source v11 freezes their proofs,
releases six Conv2 repetitions and includes qualified Conv3 pilot configs.
It is staged on Akib, Nom and Loulou. Six exact Conv2 repetition CPU smokes,
three Akib seed1 GPU smokes and three Conv3 pilot CPU smokes pass.

The first Conv2 EqProp seed1 tight-ceiling run started on Akib at 09:36:26
CEST and is verified twice with exact source/config/initializer and batches.
Keep its other ceilings on Akib and the seed2 group on Nom after its original
wave is collected/accounted. Each case requires an eight-hour reservation,
expected about 2.24 hours. Conv3 BPTT seed1 middle continues on Main, followed
by four prepared three-hour cases. Loulou and Nom each have one original
EqProp repeat left. Loulou then takes all three revised Conv3 seed0 pilots
after GPU smokes/admission. Conv3 seeds1/2 remain held for pilot stability.
The planned Nom timing diagnostic tests whether a complete later Conv3 seed
group fits its eight-hour limit.

Settled cost **226.569217 GPU-hours** plus **52 reserved** gives
**278.569217/300 committed**. The proposed350-hour extension is unapproved.
Whole waves remain reserved through completion, collection and validation.
The prior access gap recovered without training retries; fresh checks cover
all four hosts. See the [remaining-run manifest](../paper_ready_results/current_contract_remaining_runs.md)
and [continuing tracker](paper_ready_results_manifest.md).

Jean Zay electrical maintenance runs September 14 09:00 through September
16 18:00 CEST. Cancelled eight-hour request2117328 consumed zero GPU time.
Replacement seed-1 array2117432 and seed-2 array2118922 are fully collected,
validated and accounted at 6.693056 and 7.028333 GPU-hours. Both used unchanged
30-epoch configs with three-hour Slurm caps, a 10,740-second TERM timeout and
20-second KILL allowance. Seed2 completed at 08:08 CEST, before maintenance.
No active campaign allocation remains. Source, configurations, initializers,
canonical results and all terminal logs validate locally. Further submissions require restored access and a usable scheduler window after maintenance; continue the preferred lab GPUs while access is unavailable.

The separate [36-run revision tracker](../paper_ready_results/baseline_tk_revision/README.md)
preserves all 18 native-T/K BPTT baselines. Baseline T differs from the
unchanged ours/legacy groups, so cross-scheme analysis must explicitly report
the different relaxation counts and measured training cost.

After each new BPTT reference is complete and collected, run
`python -m experiments.check_baseline_revision_beta conv2 1em4 --smoke`
(adjust architecture and ceiling), then the same command without `--smoke`.
CPU is preferred while all training GPUs are occupied. The checker refuses
unfinished or original-T/K references. Candidate beta .1 is tested across all
three ceilings; .01 is the only lower candidate remaining in the declared
baseline ladder. A numerically failed group remains held. Full EqProp pilots
and subsequent replication keep the existing numerical and stability gates.
