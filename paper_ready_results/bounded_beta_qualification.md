# Bounded EqProp beta qualification

**Original T/K qualification record.** The two held baseline groups now have
an authorized matched training revision; see the [larger-T/K diagnostic](baseline_tk_beta_qualification.md)
and [revision manifest](baseline_tk_revision/README.md). This record preserves
the original 99 replays and does not relabel their failures.

All nine declared numerical groups were resolved by 2026-09-12: seven qualify and two remain held after ladder exhaustion. These read-only replays use four fixed validation batches of 16 examples at the shared initializer and matched BPTT best checkpoint for each ceiling. They do not read official MNIST test data.

A beta qualifies only when every weight-layer/batch cosine is at least .99, symmetric norm difference is at most .10, phases are finite, and the separate equilibrium residual gate passes. The selected injected beta is the largest passing tested point common to all three ceilings. Base beta is injected beta divided by `(A_V/A_I)^L`.

| Architecture | Scheme | State | Selected injected beta | Worst cosine at selected beta | Largest norm difference at selected beta |
|---|---|---|---:|---:|---:|
| conv1 | baseline | Three seeds complete and stable | 0.1 | 0.9986855 | 0.0110828 |
| conv1 | ours | Three seeds complete and stable | 0.03 | 0.9985390 | 0.0134596 |
| conv1 | legacy | Three seeds complete and stable | 0.03 | 0.9994320 | 0.0046104 |
| conv2 | baseline | Held after declared ladder | — | — | — |
| conv2 | ours | Three seeds complete and stable | 0.001 | 0.9951980 | 0.0377301 |
| conv2 | legacy | Three seeds complete and stable | 0.003 | 0.9979283 | 0.0283384 |
| conv3 | baseline | Held after declared ladder | — | — | — |
| conv3 | ours | Three pilots stable; seed 2 running, seed 1 waiting | 0.003 | 0.9969936 | 0.0850763 |
| conv3 | legacy | 7/9 complete; two seed-1 ceilings waiting | 0.001 | 0.9974632 | 0.0516333 |

The [99 tested beta/ceiling rows](bounded_beta_qualification.csv) link to the complete collected replay bundles, including failed numerical conditions. All 99 bundles validate and their files and symlinks match their sources; see the [collection audit](provenance/bounded_beta_collection.json) and [selection report](provenance/bounded_beta_selection.json). Conv2 legacy qualifies at .003 after .03 fails the cosine threshold; its largest observed norm difference at the selected beta is 0.0283384.

Numerical qualification is followed by full seed-0 training. All nine Conv1
pilots and all eighteen Conv1 repetitions are now collected and stable; see
the [complete three-seed comparison](bounded_conv1_three_seed_validation.md).
All nine Conv2 legacy runs are now collected and stable across all three
seeds, with maximum best-to-final decline .12 pp. All nine Conv2 ours runs are collected and stable at beta .001 (maximum
decline .34 pp); see the [complete comparison](bounded_conv2_ours_three_seed_validation.md). All three Conv3
legacy pilots pass at beta .001 (zero final decline). Their repetitions
are admitted at the same frozen betas. The tight Conv3 ours pilot is collected
and stable at beta .003: best/final 85.20/84.64%, decline .56 pp, all 30
epochs finite, float64 and zero biases. The middle pilot also passes the
full gate at 94.68/94.50%, decline .18 pp, with exact local/remote checksums.
The largest pilot also passes at 95.72/95.72%, zero decline, completing
the same-beta group. All 21 qualified pilots are now collected and stable;
source v8 releases the six ours repetitions. Trex's seed-1 wave awaits its
occupied GPU, while JZ seed 2 is running as array 2110632 after all launch checks pass. Keep the existing
300-GPU-hour ceiling and weekday limit of one active campaign 5090.

The two baseline groups have no qualified common beta within the declared
ladders; their six pilots and twelve repetitions remain held. See the
[current run manifest](../docs/paper_ready_results_manifest.md).
