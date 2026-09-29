# Clean voltage and encoding evidence

This series imports two completed September 28 pilots without moving or rerunning
them. The campaign was adopted September 29, after their measurements. A subsequent
read-only replay measures voltages in the BPTT encoding checkpoints themselves.

| Record | Instrument | Evidence |
|---|---|---|
| [exp-001](experiments/exp-001-conv2-target-encoding.md) | Six clean Conv2 BPTT runs; target amplitudes 1/16 and 1/4 | [Encoding-sensitive accuracy](results/exp-001-conv2-target-encoding.md) |
| [exp-002](experiments/exp-002-layerwise-voltage.md) | Paired initialization/trained replay of clean EqProp checkpoints | [RMS voltage over layers](results/exp-002-layerwise-voltage.md) |
| [exp-003](experiments/exp-003-trained-bptt-encoding-voltage.md) | Final-epoch-30 replay of the six BPTT encoding checkpoints | [Trained encoding voltage magnitudes](results/exp-003-trained-bptt-encoding-voltage.md) |

Keep the algorithms and evidence roles separate. Exp-002 measures EqProp;
exp-003 measures the BPTT encoding runs. The encoding intervention and voltage
profiles do not by themselves identify the causal role of hidden voltages or
conductance bounds.
The published figures and small CSVs are copies of existing evidence; source
paths and SHA256 values for the imports are in `results/figures/provenance.json`;
new BPTT replay identities are in `results/figures/conv2_encoding_voltage_provenance.json`.
