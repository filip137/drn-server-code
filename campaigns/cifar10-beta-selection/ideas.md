# Ideas — choosing beta for CIFAR-10

We have seen in the MNIST experiments that beta can be relatively large and
still maintain good performance. We do not know whether this is also the case
for CIFAR-10; it could be more sensitive.

First check the cosine similarity between EqProp and BPTT gradients across beta.
Then compare short training runs of 10 epochs using different betas. Good gradient
alignment can guide candidate selection, but training results decide which beta
works well.

Beta selection must balance small-beta read-noise sensitivity against large-beta
gradient bias. The same beta may give different RMS during training because
effective response and loss-gradient forcing change. First replay frozen
initialization betas, then compare the requested 0.03/0.09 initialization-RMS
training arms in all three schemes.

| Idea | Priority | Exploration | Hypothesis / experiment |
|---|---|---|---|
| RMS displacement can change greatly during training; responses that become too small may explain poor EqProp gradients and learning | Current interpretation: fixed-checkpoint layer sweeps completed; see linked review | [X-001](explorations/X-001-beta-sensitivity.md) | [H-017](hypotheses/H-017-small-trained-layer-response.md), [exp-022](series/001-beta-selection/experiments/exp-022-epoch-layer-response.md) |
| Complete the missing lower-ceiling/larger-block3-target cell to separate target and ceiling effects | Current: one fresh epoch, queued below baseline extensions | [X-001](explorations/X-001-beta-sensitivity.md) | [H-016](hypotheses/H-016-low-ceiling-larger-block3-rms.md), [exp-021](series/001-beta-selection/experiments/exp-021-low-cap-block3-rms.md) |
| Trained-checkpoint RMS targets may fail to preserve the actual gradients used in early learning | Current: short exact-prefix comparison of failed10,000/30,000-ceiling trajectories | [X-001](explorations/X-001-beta-sensitivity.md) | [H-015](hypotheses/H-015-early-selected-gradients.md), [exp-019](series/001-beta-selection/experiments/exp-019-early-selected-gradients.md) |
| The larger block3 RMS target recommended across trained checkpoints may improve fresh learning | Completed:32.88% improves on28.04% at the same ceiling, but still misses matched-BPTT gates | [X-001](explorations/X-001-beta-sensitivity.md) | [H-014](hypotheses/H-014-block3-rms-training.md), [exp-018](series/001-beta-selection/experiments/exp-018-block3-rms-training.md) |
| Early beta clipping may limit current-batch response selection | Completed: all RMS misses removed, but accuracy worsened32.64→28.04%; no extension | [X-001](explorations/X-001-beta-sensitivity.md) | [H-013](hypotheses/H-013-current-beta-ceiling.md), [exp-017](series/001-beta-selection/experiments/exp-017-current-beta-ceiling.md) |
| Select beta from the current minibatch before its update to avoid response-delay overshoots | Completed: better CE and RMS tracking, but32.64% missed the accuracy gate | [X-001](explorations/X-001-beta-sensitivity.md) | [H-012](hypotheses/H-012-current-batch-beta.md), [exp-015](series/001-beta-selection/experiments/exp-015-current-batch-beta.md) |
| Updating beta from each training minibatch may track response changes missed by fixed or epoch-wise beta | Completed: median RMS tracked, but lagged34.34% versus fixed34.22% and worse CE; continuation gate failed | [X-001](explorations/X-001-beta-sensitivity.md) | [H-011](hypotheses/H-011-minibatch-beta-feedback.md), [exp-014](series/001-beta-selection/experiments/exp-014-minibatch-beta-feedback.md) |
| Trained-checkpoint betas may transfer to fresh learning, or require a different early-stage policy | Current: all three schemes, full-epoch screens and conditional extensions toward50 epochs | [X-001](explorations/X-001-beta-sensitivity.md) | [H-010](hypotheses/H-010-fresh-trained-beta-transfer.md), [exp-012](series/001-beta-selection/experiments/exp-012-fresh-beta-staged.md), [exp-013](series/001-beta-selection/experiments/exp-013-all-scheme-long-eqprop.md) |
| Betas aligned at epoch10 may preserve short BPTT-like learning | Current: five beta vectors and two autograd controls, identical half-epoch continuations | [X-001](explorations/X-001-beta-sensitivity.md) | [H-009](hypotheses/H-009-trained-beta-short-learning.md), [exp-011](series/001-beta-selection/experiments/exp-011-trained-beta-half-epoch.md) |
| Recalibrating beta each epoch may preserve effective nudging and improve actual learning | Stopped early by Filip: epoch5 adaptive51.82%, fixed53.56%, historical BPTT66.68%; epoch10 verdict inconclusive | [X-001](explorations/X-001-beta-sensitivity.md) | [H-008](hypotheses/H-008-epoch-beta-adaptation.md), [exp-010](series/001-beta-selection/experiments/exp-010-epoch-beta-adaptation.md) |
| Fixed initialization beta may drift away from its useful RMS/alignment range during training | Current: freeze initialization RMS 0.03/0.09 betas across epochs 10/30/50 | [X-001](explorations/X-001-beta-sensitivity.md) | [H-007](hypotheses/H-007-fixed-beta-transfer.md), [exp-009](series/001-beta-selection/experiments/exp-009-fixed-beta-transfer.md) |
| Large nudging may distort CIFAR gradients more than it does on MNIST | First: fixed-T/K cosine sweep | [X-001](explorations/X-001-beta-sensitivity.md) | [H-001](hypotheses/H-001-large-nudging-error.md), [exp-001](series/001-beta-selection/experiments/exp-001-smaller-beta-cosine.md) |
| Compare different betas directly through 10-epoch CIFAR learning | Next: select candidates from the diagnostic and define the EqProp training update | [X-001](explorations/X-001-beta-sensitivity.md) | [exp-002](series/001-beta-selection/experiments/exp-002-ten-epoch-beta.md); training hypothesis to be frozen with its contract |
| A shared output RMS range may select different betas with high cosine across CIFAR blocks | Current: focused sweep | [X-001](explorations/X-001-beta-sensitivity.md) | [H-002](hypotheses/H-002-common-output-displacement.md), [exp-003](series/001-beta-selection/experiments/exp-003-common-rms.md) |
| A common RMS target may transfer between ours, baseline and legacy | Current: repeat the focused sweep in the other schemes | [X-001](explorations/X-001-beta-sensitivity.md) | [H-003](hypotheses/H-003-displacement-transfer-across-schemes.md), [exp-004](series/001-beta-selection/experiments/exp-004-baseline-legacy-rms.md) |
| Block2's lower alignment may persist after BPTT training rather than being an initialization effect | Current: replay original epoch50 checkpoints | [X-001](explorations/X-001-beta-sensitivity.md) | [H-004](hypotheses/H-004-trained-block2-alignment.md), [exp-006](series/001-beta-selection/experiments/exp-006-trained-e50-rms.md) |
| More nudged iterations may improve block2's response at fixed beta | Current: epoch50 block2 K6/12/24/48 screen | [X-001](explorations/X-001-beta-sensitivity.md) | [H-005](hypotheses/H-005-nudged-k-block2.md), [exp-007](series/001-beta-selection/experiments/exp-007-fixed-beta-nudged-k.md) |
| Useful RMS may depend on training stage, requiring a compromise or changing target | Current: compare epochs0/10/30/50 and recommend per-block targets | [X-001](explorations/X-001-beta-sensitivity.md) | [H-006](hypotheses/H-006-intermediate-rms-transfer.md), [exp-008](series/001-beta-selection/experiments/exp-008-intermediate-rms.md) |
