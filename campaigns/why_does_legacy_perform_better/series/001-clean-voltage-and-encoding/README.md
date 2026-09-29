# Clean voltage and encoding evidence

This series imports two completed September 28 pilots without moving or rerunning
them. The campaign was adopted September 29, after their measurements.

| Record | Instrument | Evidence |
|---|---|---|
| [exp-001](experiments/exp-001-conv2-target-encoding.md) | Six clean Conv2 BPTT runs; target amplitudes 1/16 and 1/4 | [Encoding-sensitive accuracy](results/exp-001-conv2-target-encoding.md) |
| [exp-002](experiments/exp-002-layerwise-voltage.md) | Paired initialization/trained replay of clean EqProp checkpoints | [RMS voltage over layers](results/exp-002-layerwise-voltage.md) |

Keep the algorithms and evidence roles separate. The voltage plot does not
measure voltages in the BPTT encoding runs; the encoding intervention does not
by itself identify the causal role of hidden voltages or conductance bounds.
The published figures and small CSVs are copies of existing evidence; source
paths and SHA256 values are in `results/figures/provenance.json`.
