---
id: "exp-017"
title: "Baseline noisy training matched to the epoch30 ours/legacy comparisons"
status: "complete"
hypotheses: ["H-011"]
---
# Exp017 — Baseline relative-noise comparisons

Filip requested baseline noise tests to match ours and legacy. Add10seed0 cases:
Conv2 seed0-reference D/F4, Conv3 D/F6, eta1e-6,1e-5,1e-4,3e-4,1e-3.
Use nodewise Gaussian relative acquisition noise eta*abs(clean endpoint)*xi,
same eta every noninput layer, independentphase/layer/node draws, nofloor.
Cleaninputs/physicalrelaxation/nudgingforce/validation. No1e-8/1e-7 cases.

Filip explicitly instructed30epochs inonego. Train baseline uninterrupted from
originalseed0initialization with freshAdam only atinitialization. Ours/legacy
exp015 reset Adam/RNG at epoch10; record this optimizer-history confound when
comparing and do notattribute crossscheme differences solely to amplification.
Use baseline's acceptedbeta and own frozenLRvector; samebatchsize,T/K,float64,
diodes, frozenbiases, inputgain, weightbounds, dataset and split ascomparison.
No newbeta/LR/TK sweeps; reuse accepted576example seed0matching evidence.

Primary: fixed cumulativeepoch30validation, baselinevsoursvslegacy at eachdepth/eta;
secondary epoch10 comparison toexp014. Single-seed exploratory evidence, not
significance or officialtest. No independentlyselectedbestepoch comparison.

Result root registered before creation:
`results/conv23-baseline-relative-noise-20260927-v1/`.
Configroot: `configs/conv/baseline_relative_noise_20260927/`.

## Execution and monitoring handoff

Allowance60GPUh including preparation/overhead/retries;72hdeadline
fromfirstadmission. Expected30–45GPUh, pluswaiting for currentauthorizedruns.
Prefer Conv2local/nom1 and Conv3verified5090s, preserving scientificsettings.
Queues wait for priorcleanbaseline work ontheirhost beforeGPUcapacityadmission;
no preemption or CPUfallback. Samecampaign Ben/Kellian sharingpermission,
Conv3measured8192MiBfloor, otherownersprotected. Onepersistentwatch ownsnew10cases.
No tests/canaries peruserstandingwaiver; retain identity, actualCUDAfiniteprogress,
andruntimeguards. Frozen numericalsource iscorrectedexp015v2 with existingexact_run. Configs/initializer identities explicit.
Save modelandoptimizercheckpointseachepoch. Finalcollectlocally, validatecomplete30epochs, reconcile10completedtrajectories, comparewithexp014/015 fixedmetrics,
write resultnote/ledger. Do notmodifyexistingclean/noisycontrols.

### Prepared and scheduled

All10configs ready and staged, same2seed0initializers and acceptedbaselinebeta
ascleanbaselinecontrols. No newcalibration. Expliciteta inconfigs andprovenance
preparation.json; acquisitionseed2026081601 andshuffle/split0 preserved.

Detachedcontrollers: local4179060,nom13780275,fifi682614,loulou3874140,riri829197.
Exactcommands/paths intransport/launches.json. Allocations14/10/14/14/8GPUh.
Conv2localeta1e-6/1e-4/1e-3,nom1eta1e-5/3e-4; Conv3fifieta1e-6/1e-4,
louloueta1e-5/3e-4,ririeta1e-3. Eachlane waitsits priorcleanbaselinequeueexit
beforeliveGPUadmission; existingjobs remainunchanged.
Frozen numericalsource SHA256
`f337f482deb8a14614f5e8b8197613a06e365891d8d4ee2069810be29ffb8477`.

Harddeadline 2026-09-30T14:34:46.811153+00:00.

Latest request verification (2026-09-27 14:54 UTC): all five controllers are alive,
all ten cases remain queued behind their clean-baseline predecessors, and no
production case has started (charged GPU time 0). Persistent monitor PID4179582
is alive with dispatch enabled; observation has no controller or probe errors.
The repeated baseline-noise request is covered by these existing queues; no
duplicate submission. Evidence: execution-root
`monitor/request-observation-20260927.json`. Scientific settings are unchanged.

Finalanalysis: `python -m experiments.analyze_baseline_relative_noise --study-root results/conv23-baseline-relative-noise-20260927-v1` usingpy312. Writes10casecoverage, fixedepoch10/30 table, comparisonwithavailableexp015CSV, andJPG. Crossschemeepoch30interpretation must retain uninterruptedbaseline versusresetourslegacy qualifier.

### Fifi queued work relocated at user request

Conv3baselineeta1e-6/1e-4 movedfromFifitoTrex; controller545276. It waitsforthe actualTrexcleanbaselinecontroller545237 tofinish. Otherplacementsunchanged.
Onlyunstarted childlesscontrollerswerestopped;zerochargedGPUtime. OldFifireceiptsarepreservedlocally underrelocated-from-fifi; exactreplacementcommands intransport/launches.json. GPUverifiedTrexRTX5090 with31960MiBfree andBen-onlysharing. Numericalsource/configs/initializers unchanged;originalcaps andharddeadlinespreserved. ActiveFifitrainingnotinterrupted. Monitorlanes/collectionnowfollowTrex.

### Reviewed completion — September 28

All ten cases collected and validated with epochs1–30 and all31 model/optimizer checkpoint pairs per case. Measured launcher charge31.750735GPUh of60GPUh, with no production failures or exclusions. [Result and scoped review](../results/exp-017-baseline-relative-noise.md) records fixedepoch10/30 measurements and the optimizer-history confound. Evidence: registered root monitor/coverage-validation.json and monitor/FINAL_SUMMARY.md. REVIEW_COMPLETE; no further exp017 compute.
