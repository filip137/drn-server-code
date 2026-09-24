# Confirmed Conv3 p99 diagnostic request

Experiment: Conv3 p99 read noise 5e-4
Type: derived comparison
Parent: eqprop-conv3-p95-read-noise-5em4-20260921-v1
Purpose: Assess full 30-epoch training stability of p99 beta nominations.
Cases/subset: legacy and ours, seed0.
Scientific changes: injected beta0.1 legacy; beta0.987333678708 ours.
Keep fixed: inherit everything else from exact parent configs.
Budget / seeds / sweep:30epochs each; seed0; sigma5e-4 only; no expansion.
Decision style: independent stability checks;30 finite epochs and final drop under5pp.
Evidence / official-test policy: ordinary diagnostic; official test forbidden.
Compute / distribution: Jean Zay two V100s, one case per GPU.
Executor: adapt operations yes; launch after checks yes; stop both on any run failure.

User review: User said “yes” on2026-09-22 after the complete scientific summary,
inherited settings, beta verification, and two-V100 execution proposal were shown.
Scientific summary approved: yes.
Execution proposal authorized: yes.
