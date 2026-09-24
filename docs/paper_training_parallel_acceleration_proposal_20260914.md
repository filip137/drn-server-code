# Proposed parallel completion under larger compute limits

Status: **deferred to the requested review; no increase authorized or reserved**. New admissions are paused. The calendar estimates below predate the pause and must be revised after a resume decision. Prepared September14 after collecting and validating the Nom timing diagnostic. This supersedes the earlier unapproved350-GPU-hour proposal only if Filip accepts it.

Nom measured0.326809 seconds per training batch on the qualified Conv3 EqProp configuration. Including full validation and conservative per-epoch overhead, the30-epoch projection is9.788hours; adding15% gives11.256hours. The current eight-hour cap prevents this placement. Sourcev11, T24/K8, beta.1, float64, all scientific settings and30epochs remain unchanged.

Request a375-GPU-hour campaign cap and up to12hours for a Conv3 EqProp case on a3090, plus permission to split the seed0 pilot ceilings between Nom and Loulou. Other full-case limits remain unchanged. No extra weekday5090 is used; Jean Zay retains at most four concurrent campaign GPUs when accessible.

| Work | Target and order | Bound / estimate | Condition |
|---|---|---|---|
| Already running tight Conv3 EqProp pilot | Loulou | Existing8h / about6.9h | Continue unchanged |
| Middle Conv3 EqProp pilot, seed0/Gmax5e-4 | Nom after its first currently planned Conv2 seed2 case | Proposed12h / about9.8h | Approval, exact target smoke, fresh source/idle check and full12h reservation |
| Largest Conv3 EqProp pilot, seed0/Gmax1e-3 | Loulou after tight pilot | Existing8h / about6.9h | Skip the Nom-assigned middle case in Loulou queue; admit individually |
| Remaining Conv2 repetitions | Akib seed1 and Nom seed2 | Existing8h/case / about2.24h | Continue, with Nom middle/largest Conv2 cases after its Conv3 pilot if approved |
| Conv3 EqProp repetition groups | One complete seed group on Loulou, one on Nom; choose by actual availability after all3pilots pass | Loulou up to24h/group; Nom up to36h/group | Freeze new full-pilot guard, admit each case, preserve full seed group on one host |

The first two completed Loulou epochs now project 6.618 hours per Conv3 case. With Nom's remaining Conv2 cases included in its queue, this plan projects completion around September 16 at 10:06 CEST, compared with about 23:27 CEST for serial Loulou placement: approximately 13 hours earlier. These are preliminary estimates, conditional on all pilots passing, immediate handoffs and stable runtimes. No Jean Zay speedup is assumed.

Four Conv3 cases moving from 6.618 hours on Loulou to 9.788 hours on Nom add about 12.7 GPU-hours. The central campaign forecast becomes about 348.4 GPU-hours before future checks, retries and official-test evaluation. The proposed 375-hour cap provides margin; it does not authorize spending beyond that cap. The earlier 16-hour saving was provisional and is superseded by this queue-aware estimate.

The split pilot host is a recorded transport deviation. Initialization, deterministic train/validation membership, all minibatch orders, resolved configs, epochs and checkpoint rules must match the frozen contract. Preserve all seeds and report the host/software metadata. No official test is read during training. If approval does not arrive, keep the existing300-hour/eight-hour limits and continue the currently authorized queue.

The first proposed Nom command is `results/paper-training-completion-20260911-v1/launch/nom_baseline_tk_conv3_ep_seed0_gmax5em4_12h_proposed.command.txt`, executed from remote`source-v11` through the existing detached launcher. It calls the same scientific exact runner directly because the ordinary list wrapper enforces the old eight-hour cap. Its source check, duplicate guard, GPU vacancy check, twelve-hour outer timeout, launch records and canonical run artifacts remain required. The planned remote output is`/home/filip/server_code/results/paper-training-completion-20260911-v1/production/nom_baseline_tk_conv3_ep_seed0_gmax5em4_12h_proposed/`. No target staging, allocation or production launch of this proposed case has occurred.

[Collected timing evidence](../results/paper-training-completion-20260911-v1/checks/nom_conv3_baseline_ep_timing_20260914/timing_estimate.json) · [Current authorized launch plan](paper_training_completion_launch_plan_20260911.md)
