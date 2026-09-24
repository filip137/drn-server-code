# Matched baseline T/K revision

Updated 2026-09-14T21:13:31.801521+00:00. **22/36 revised training results collected and validated.** Official-test evaluations: zero.

Filip authorized revised T/K for both BPTT and EqProp. Conv2 uses T=12, K=6; Conv3 uses T=24, K=8. The three weight ceilings and seeds 0/1/2, exact Adam vectors, initializers, minibatch order and 30-epoch budgets are unchanged.

The common candidate beta is 0.1, supported by the [61 completed numerical replays](../baseline_tk_beta_qualification.md). New seed-0 BPTT best checkpoints must pass the same numerical gate. Each architecture then requires all three full seed-0 EqProp pilots to pass before its seeds 1/2 are released.

The [new-reference beta report](beta_qualification.md) tracks the collected checkpoint rechecks and distinguishes partial ceiling coverage from a qualified common beta.

The [full-pilot stability report](pilot_stability.md) tracks all six revised seed-0 EqProp pilots and the gate before seeds 1/2.

The 18 earlier baseline BPTT runs remain in the original collection as native-T/K evidence. They do not satisfy this revision. This directory has its own 36-cell ledger and bundles; it does not overwrite the original 216-cell ledger. Eight original non-baseline repetitions were still missing when this revision started, giving 44 further full training completions in total.

Current campaign coverage is **202/216**, with **14** full training completions still missing. See the [current contract ledger](../current_contract_run_status.csv) and [remaining runs](../current_contract_remaining_runs.csv), which overlay this revision onto the unchanged other cells.

Preferred resources are local/Main, Akib and nom-cool-1. Loulou is the sole weekday campaign RTX 5090. Packing requires measured throughput improvement; the completed two-EqProp benchmark was slower. Each full wave must fit the unchanged 300 GPU-hour ceiling, with an eight-hour maximum per training case. Explicit shorter operational ceilings and reservations are recorded in the continuing campaign manifest.

| Cell | T/K | Status | Target | Best/final validation | Result |
|---|---:|---|---|---:|---|
| T3_BPTT_conv2_baseline_gmax0.0001_seed0 | 12/6 | complete_collected_validated | akibscomputer | 85.52/85.08% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0001_seed0/result.json) |
| T3_BPTT_conv2_baseline_gmax0.0001_seed1 | 12/6 | complete_collected_validated | jean-zay | 85.42/85.06% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0001_seed1/result.json) |
| T3_BPTT_conv2_baseline_gmax0.0001_seed2 | 12/6 | complete_collected_validated | jean-zay | 85.24/85.24% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0001_seed2/result.json) |
| T3_BPTT_conv2_baseline_gmax0.0005_seed0 | 12/6 | complete_collected_validated | akibscomputer | 92.38/92.16% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0005_seed0/result.json) |
| T3_BPTT_conv2_baseline_gmax0.0005_seed1 | 12/6 | complete_collected_validated | jean-zay | 91.52/91.46% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0005_seed1/result.json) |
| T3_BPTT_conv2_baseline_gmax0.0005_seed2 | 12/6 | complete_collected_validated | jean-zay | 92.22/92.22% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.0005_seed2/result.json) |
| T3_BPTT_conv2_baseline_gmax0.001_seed0 | 12/6 | complete_collected_validated | akibscomputer | 93.54/93.36% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.001_seed0/result.json) |
| T3_BPTT_conv2_baseline_gmax0.001_seed1 | 12/6 | complete_collected_validated | jean-zay | 93.62/93.62% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.001_seed1/result.json) |
| T3_BPTT_conv2_baseline_gmax0.001_seed2 | 12/6 | complete_collected_validated | jean-zay | 93.50/93.40% | [bundle](bundles/T3_BPTT_conv2_baseline_gmax0.001_seed2/result.json) |
| T3_BPTT_conv3_baseline_gmax0.0001_seed0 | 24/8 | complete_collected_validated | main | 77.68/77.68% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.0001_seed0/result.json) |
| T3_BPTT_conv3_baseline_gmax0.0001_seed1 | 24/8 | complete_collected_validated | main | 77.50/76.70% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.0001_seed1/result.json) |
| T3_BPTT_conv3_baseline_gmax0.0001_seed2 | 24/8 | planned_replacement | not admitted | pending | pending |
| T3_BPTT_conv3_baseline_gmax0.0005_seed0 | 24/8 | complete_collected_validated | main | 87.12/87.12% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.0005_seed0/result.json) |
| T3_BPTT_conv3_baseline_gmax0.0005_seed1 | 24/8 | complete_collected_validated | main | 87.36/87.36% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.0005_seed1/result.json) |
| T3_BPTT_conv3_baseline_gmax0.0005_seed2 | 24/8 | planned_replacement | not admitted | pending | pending |
| T3_BPTT_conv3_baseline_gmax0.001_seed0 | 24/8 | complete_collected_validated | main | 89.38/89.38% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.001_seed0/result.json) |
| T3_BPTT_conv3_baseline_gmax0.001_seed1 | 24/8 | complete_collected_validated | main | 89.58/89.58% | [bundle](bundles/T3_BPTT_conv3_baseline_gmax0.001_seed1/result.json) |
| T3_BPTT_conv3_baseline_gmax0.001_seed2 | 24/8 | planned_replacement | not admitted | pending | pending |
| T3_EP_conv2_baseline_gmax0.0001_seed0 | 12/6 | complete_collected_validated | akibscomputer | 85.42/85.00% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0001_seed0/result.json) |
| T3_EP_conv2_baseline_gmax0.0001_seed1 | 12/6 | complete_collected_validated | akibscomputer | 85.36/85.04% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0001_seed1/result.json) |
| T3_EP_conv2_baseline_gmax0.0001_seed2 | 12/6 | complete_collected_validated | nom-cool-1 | 85.22/85.22% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0001_seed2/result.json) |
| T3_EP_conv2_baseline_gmax0.0005_seed0 | 12/6 | complete_collected_validated | akibscomputer | 92.54/92.30% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0005_seed0/result.json) |
| T3_EP_conv2_baseline_gmax0.0005_seed1 | 12/6 | complete_collected_validated | akibscomputer | 91.50/91.50% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.0005_seed1/result.json) |
| T3_EP_conv2_baseline_gmax0.0005_seed2 | 12/6 | qualified_stable_pilot_group_pending_admission | nom-cool-1: whole three-ceiling seed group; cases not admitted | pending | pending |
| T3_EP_conv2_baseline_gmax0.001_seed0 | 12/6 | complete_collected_validated | akibscomputer | 93.92/93.54% | [bundle](bundles/T3_EP_conv2_baseline_gmax0.001_seed0/result.json) |
| T3_EP_conv2_baseline_gmax0.001_seed1 | 12/6 | qualified_stable_pilot_group_pending_admission | akibscomputer: whole three-ceiling seed group; cases not admitted | pending | pending |
| T3_EP_conv2_baseline_gmax0.001_seed2 | 12/6 | qualified_stable_pilot_group_pending_admission | nom-cool-1: whole three-ceiling seed group; cases not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.0001_seed0 | 24/8 | complete_collected_validated | loulou | 77.68/77.68% | [bundle](bundles/T3_EP_conv3_baseline_gmax0.0001_seed0/result.json) |
| T3_EP_conv3_baseline_gmax0.0001_seed1 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.0001_seed2 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.0005_seed0 | 24/8 | numerically_qualified_seed0_pilot_pending | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.0005_seed1 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.0005_seed2 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.001_seed0 | 24/8 | numerically_qualified_seed0_pilot_pending | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.001_seed1 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |
| T3_EP_conv3_baseline_gmax0.001_seed2 | 24/8 | awaiting_all_three_full_revised_seed0_pilots | loulou after its original wave; not admitted | pending | pending |

[Machine-readable ledger](run_status.csv) · [Original baseline evidence snapshot](original_native_tk_rows.json) · [Continuing campaign manifest](../../docs/paper_ready_results_manifest.md)
