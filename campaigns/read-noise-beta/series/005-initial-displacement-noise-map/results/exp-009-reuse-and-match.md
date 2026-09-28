---
experiment: "exp-009"
evidence: "validated-local"
summary: "All45 initialization D/F matches accepted:27 reused and18 new checks; no new-point corrections."
verdicts: {"H-008": "not-tested"}
---
# Exp009 — Beta lookup ready

All45 architecture/scheme/target combinations satisfy the1% matching tolerance on
the frozen576-example cohort. Reused27 matches at0.8/2/6; measured18 at0.2/10.
Largest relative error across all45 is5.259384e-6 (0.000526%). Both new calibration
bundles pass canonical validation; all45 lookup provenance hashes match their files.
New calibration64.59s plus same-path parity smoke19.22s =0.0233 GPUh, below0.5h cap.
No optimizer, training or official-test reads; no T/K change or additional sweep.

[Lookup](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/beta_lookup/beta_lookup.csv)
SHA256: `4946df1fd9c62ea94f2bd41e6172b216bba79c9f82b35f05000245b89d270dc3`.
[Raw new matches](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/beta_lookup/calibration/all/matches.json).
[Resolved cases](../../../../../results/conv123-initial-displacement-noise-map-20260925-v1/beta_lookup/matched_cases.json).
Cohort SHA256: `95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd`.
Source/model hashes and actual injected/internal betas accompany each resolved case.

Matching establishes comparable initial relative output displacement, not equal
absolute displacement/SNR or gradient fidelity. H008 is not tested at this stage.
Proceed to the already-authorized exp010 screen once the replay implementation and
same-path smokes pass; no training is implied.
