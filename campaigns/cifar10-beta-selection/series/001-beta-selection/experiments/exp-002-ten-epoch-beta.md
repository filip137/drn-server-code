---
id: "exp-002"
title: "Ten-epoch CIFAR training across beta values"
status: "planned"
hypotheses: []
---
# exp-002 — Ten-epoch CIFAR training across beta values

Deferred in favor of the approved [exp-010](exp-010-epoch-beta-adaptation.md):
two fresh ours runs at initial RMS0.09, comparing epoch-wise beta recalibration
with fixed beta. The six-arm proposal below is preserved as a later comparison.

Requested next stage: compare fresh 10-epoch runs at different betas on the same
CIFAR-10 architecture. Select a bounded candidate set after
[exp-001](exp-001-smaller-beta-cosine.md), including a larger-beta comparator.
Judge by epoch10 validation accuracy and CE, with training CE and time as context;
use the same initializer, data order, optimizer, LRs, batch size and BN treatment.
Do not select each arm's best epoch or read the official test split.

Before execution, specify the actual blockwise EqProp training update across
pooling/BN boundaries, trainable BN/gains and the dense readout; preserve the
explicit physical current coefficient and loss normalization. The existing
CIFAR BPTT trainer does not consume beta. Candidate values, exact training
contract, common schedule/seed, cases, runtime budget and results path remain to
be frozen after the diagnostic. This record plans the requested training stage;
it is not a claim that those runs are already prepared or queued.

Current authorized comparison (September 29): six fresh 10-epoch arms,
baseline/ours/legacy at initialization relative RMS 0.03 and 0.09 in every block,
using two RTX 5090s. This supersedes the earlier multiplier candidate sets.
Filip's immediate follow-up asks what happens with unchanged beta; first complete
[exp-009](exp-009-fixed-beta-transfer.md), which freezes the initialization
coefficients across saved checkpoints. Preserve the distinction between fixed
beta and a controller that recalibrates beta to maintain RMS. No six-arm training
jobs have been launched; the beta-dependent training update still needs the
implementation and contract described above.
