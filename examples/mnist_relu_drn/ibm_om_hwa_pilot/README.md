# IBM OM HWA pilot configs

These strict configs implement the predeclared
[`mnist-ibm-om-hwa-program-verify-pilot-20260822-v1`](../../../studies/mnist-ibm-om-hwa-program-verify-pilot-20260822-v1.json)
study. The scientific contract is in
[`docs/ibm_om_hwa_program_verify_pilot.md`](../../../docs/ibm_om_hwa_program_verify_pilot.md).

Training configs:

- `clean.json`: clean off-chip BPTT;
- `gaussian_3pct.json`: 3% per-output-channel Gaussian HWA;
- `om_repaired.json`: fixed IBM OM bounds with corrupt sites replaced; and
- `om_published.json`: fixed IBM OM bounds with published corruption retained.

All four use the same full-corrupt pulse-resolved selection modifier. The four
`heldout_assignment_*.json` configs provide the predeclared test assignments.

Every run using these configs requires an explicit calibrated device-model
artifact:

```text
--device-model data/ibm_reram_om_pv128_hwa_v1.json
```

The artifact must be built from the two adequate adaptive conditions in the
focused `hwa_production_cap128` study
`ibm-reram-om-hwa-device-model-20260822-v1`; the runtime will not substitute
the broader cap-128 study, the older cap-512 artifact, or a Gaussian surrogate.
Training also requires the exact supervised bounded-DRN checkpoint as
`--teacher-weights`. Validation requires that same teacher plus one selected
training checkpoint as `--weights`.
