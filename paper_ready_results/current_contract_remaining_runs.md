# Remaining training for the current paper contract

Updated 2026-09-14T21:13:31.795841+00:00. **202/216 training results collected and validated; 14 full runs remain.**

This list is regenerated from the current contract after each collection. All remaining cases use 30 epochs. These are training-completion requirements; official-test evaluations remain zero and require the later contract/checkpoint seal.

| Architecture | Algorithm | Scheme | T/K | Missing seeds by weight ceiling | Runs |
|---|---|---|---:|---|---:|
| conv2 | EP | baseline | 12/6 | 0.0005: 2; 0.001: 1, 2 | 3 |
| conv3 | BPTT | baseline | 24/8 | 0.0001: 2; 0.0005: 2; 0.001: 2 | 3 |
| conv3 | EP | baseline | 24/8 | 0.0001: 1, 2; 0.0005: 0, 1, 2; 0.001: 0, 1, 2 | 8 |

The revised baseline EqProp groups require a common beta to pass on every new seed-0 BPTT reference, followed by all three full seed-0 EqProp pilots passing stability before seeds 1/2. See the [new-reference beta report](baseline_tk_revision/beta_qualification.md) and [revision manifest](baseline_tk_revision/README.md) for gate status.

[Live targets and launch records](../docs/paper_ready_results_manifest.md) · [Per-run CSV](current_contract_remaining_runs.csv) · [Current validation tables](current_contract_validation_tables.md)
