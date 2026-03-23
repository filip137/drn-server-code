# Tables

Generated from the current CPU timing bundles under `figures_for_paper_digits/timings/*/cpu_timing/combined_latest_by_hidden.csv`.

- `table1_accuracy.csv` / `.md`: 34 matched rows with available SPICE error summaries
- `table2_runtime_decomposition.csv` / `.md`: 34 matched rows with available SPICE timings
- `table3_cpu_only_runs.csv` / `.md`: 8 CPU-only rows with no SPICE timing, reporting batch size 1 validation runtime and validation accuracy
- Explicitly skipped from the matched SPICE tables: `single_diode_exponential`, `3H/256` (included in `table3_cpu_only_runs`)
- Rows with missing SPICE timing or missing error summaries are excluded from Tables 1 and 2 and summarized separately in Table 3 when the CPU validation run is available.
