# Digital ReLU MSE references for Table 1

Completed 2026-09-17: **18/18 trainings**, three seeds per depth/loss. Values are maximum-validation-checkpoint accuracy, mean ± sample standard deviation (%), on the same 5,000-example validation cohort used by the DRN tables. The official test set remains unread.

Manuscript inclusion: **Table 1 only** displays the MSE references (nine trainings, three aggregates). The current Overleaf version was fetched and updated in commit `8a889d7`; Tables 2 and 3 are unchanged from that version. Cross-entropy results remain below as supporting evidence. The DRNs use squared error on paired-output differences (`0.5 * sum(class error²)` per example); the digital MSE averages over ten classes, so output encoding and loss normalization differ.

| Architecture | Channels | MSE | Cross-entropy | Epochs |
|---|---|---:|---:|---:|
| Conv1 | 32 | 96.47 ± 0.12 | 97.59 ± 0.09 | 10 |
| Conv2 | 32/64 | 98.19 ± 0.16 | 98.38 ± 0.11 | 30 |
| Conv3 | 32/64/128 | 99.01 ± 0.03 | 98.77 ± 0.08 | 30 |

Cross-entropy has the higher observed mean for Conv1 and Conv2; MSE has the higher mean for Conv3. There is no uniformly better loss under this fixed learning rate and training budget. These are descriptive three-seed results, not a significance claim or a tuned comparison of loss functions.

All biases, including the output bias, were initialized at zero and retained explicit Adam learning rate zero. Weights used a constant Adam learning rate of .001, no decay, and PyTorch default signed Kaiming-uniform initialization (`a=sqrt(5)`). Batch size was 16. MSE used one-hot 0/1 targets with a mean over examples and classes; cross-entropy used raw logits. Networks have one signed input channel and ten logits, kernel 3, padding 1, strides 2/2/1 by depth, and no pooling. Inputs use `.3*(x-.1307)/.3081` at gain one. Total parameter counts, including frozen biases, are 63,050 / 50,186 / 155,402.

The digital weights are unconstrained; these references appear only in Table 1. This is a conventional digital performance reference, not a signed-weight clipping experiment. The DRN versus digital comparison also changes initialization, output/loss representation, input gain, signed-weight support, and dynamics. Halving channels follows the requested signed-weight representation convention and does not establish equal parameter counts or identical expressive capacity. No digital learning-rate search was performed.

Validation completed:

- All 18 canonical bundles and source/checkpoint/artifact hashes validate; all 420 epochs and 1,443,960 optimizer steps are present.
- Initial, best and final checkpoints are finite, have the requested geometry, and have exactly zero biases. Final Adam state is finite with the exact expected step counts.
- Each loss pair shares identical initialization tensors, environment, source, data and minibatch order. Split and every epoch order match the corresponding DRN table runs.
- All 18 best checkpoints were replayed on the local GPU; every saved validation accuracy was reproduced exactly.
- Both remote output trees are checksum-identical to their authoritative local copies. All three launchers exited zero; no full training failed, was excluded or retried.
- Ten successful same-path smokes and seven CPU regression checks passed. The initial local py312 smoke failed in cuDNN before an optimizer step; it is retained under `smoke/conv1_mse_seed0`, superseded operationally by the six passing local py309 smokes. The scientific config was unchanged.

Production occupied **0.497500 aggregate GPU-hours** across the three lanes (including startup/serialization), below the four-GPU-hour cap. Training-loop recorded time was 0.493065 GPU-hours. Conv1 ran locally on RTX3090/torch2.5.1, Conv2 on Akib RTX3080/torch2.5.1, and Conv3 on Trex RTX5090/torch2.11.0. Cross-depth environments differ, while each depth’s two losses and all seeds share a target. An unrelated Trex workload started after our idle allocation and caused intermittent runtime contention; it was left untouched.

The current Overleaf manuscript source and Table 1 input were updated and pushed. Tables 2 and 3 retain their remote contents. No TeX engine is installed locally, so source structure and numerical cells were checked without compiling a PDF.

Evidence:

- [Frozen config](../configs/digital_relu_table12_20260917.json)
- [Plan, source, smoke and launch record](../docs/digital_relu_table12_plan_20260917.md)
- [Six aggregate rows](digital_relu_table12_20260917.csv)
- [All per-seed values and source result hashes](digital_relu_table12_per_seed_20260917.csv)
- [Validated study summary](../results/digital-relu-table12-20260917-v1/analysis/summary.json)
- [Independent checkpoint replay](../results/digital-relu-table12-20260917-v1/analysis/checkpoint_replay.json)
- [Remote checksum verification](../results/digital-relu-table12-20260917-v1/analysis/collection_verification.json)
- [Exit receipts, budget and smoke reconciliation](../results/digital-relu-table12-20260917-v1/analysis/closeout.json)
- [Manuscript](../papers/amplification_overleaf/bidir_paper_theory_revised.tex)
