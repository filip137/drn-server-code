# MNIST bounded-DRN teacher run

## Monitoring contract

- Scientific tier: formal, intended as the frozen teacher/source checkpoint
  for the CMO versus Wan-2022 deployment study.
- Coverage: one bias-free `[1568, 100, 20]` perfect-diode DRN, seed 17,
  trained from scratch for exactly 10 epochs on the 55,000-example MNIST
  training split; clean model selection uses the fixed 5,000-example
  validation split. The 10,000-example MNIST test split is untouched during
  training and will be evaluated after collection.
- Conductance constraint: every weight remains in
  `[9 / 88.199997, 1] = [0.1020408197973068, 1]` from initialization onward.
  Deployment is multiplicative only: CMO uses `88.199997 uS * w`; Wan-2022
  uses `40 uS * w`.
- Frozen source commit: `6aa32237ea5894939af9a9a271b3081fcb983fc9`.
- Frozen source tree: `6e70efc440b4db953d7559c8344d21018842b6f9`.
- Source bundle SHA-256:
  `2af61ccb8ac35ce73ad563a82d42c6e172509e0d7b229eb96b7c22436b449501`.
- Config SHA-256:
  `29fc5094f997fa0670e67c1635f5accc21dde749a940463c955fa4888cdd6040`.
- Akib source:
  `/home/filiposana/staged/mnist_bounded_drn_teacher_6aa32237_v2`.
- Akib result root:
  `/home/filiposana/server_code/results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237`.
- Akib launcher state:
  `/home/filiposana/server_code/results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237.launcher`.
- Combined log: `<launcher state>/job.log`; lifecycle evidence is
  `<launcher state>/exit_code`, `pid`, `started_at`, and `finished_at`.
- Required terminal artifacts: one successful run directory containing
  `result.json`, `metrics.jsonl`, `checkpoints/weights.pt`, and
  `checkpoints/resume.pt`; then a separate clean test-validation run and a
  local copy with matching hashes.
- Progress evidence: launcher/PID and GPU occupancy plus a new epoch metric
  and updated resume checkpoint. Check immediately twice, then every 30
  minutes; diagnose after 45 minutes without semantic artifact growth.
- Expected runtime: approximately 30--90 minutes on the Akib RTX 3080. A
  retry is safe only into a new result and launcher path using the same commit
  and config; do not alter the learning rates, seed, bounds, or epoch budget.

## Pre-launch evidence

- Local clean-worktree gates: 117 focused tests passed; legacy `labs/tests`
  passed 42 tests with 1 skipped.
- Local numerical canary: one epoch horizon, two training minibatches and one
  validation minibatch completed; named weights and exact-resume checkpoints
  were written with all conductances inside the configured interval.
- Akib environment: passwordless SSH succeeded; PyTorch 2.5.1 reported CUDA
  available on an NVIDIA GeForce RTX 3080; no compute process occupied the
  GPU at preflight.
- Staging recovery: an initial shallow-bundle clone failed before launch due
  to an omitted parent object. The complete bundle was checked at the `v2`
  path above and preserved the intended commit and tree exactly.

## Terminal result

- Training run ID: `20260820T140222.931115Z-809aec60-8da7bc60`.
- Launcher: `nohup`, PID `281272`, exit code `0`; repository-native status is
  `complete`. Runtime was 149.214 s.
- All 10 epochs completed (34,380 minibatch updates). The selected checkpoint
  is the final epoch: 93.58% validation accuracy and validation cost
  0.08011558. Final-epoch training accuracy was 93.9964%.
- Independent test run ID:
  `20260820T140600.942755Z-c40671d3-e2c7d4b0`; launcher exit code `0` and
  repository-native status `complete`.
- Untouched 10,000-example MNIST test accuracy: **94.29%**; mean paired
  squared-error cost: `0.07546469`. All evaluations used exactly four
  coordinate iterations.
- Selected named-weights SHA-256:
  `f0036b36cebe970c5105c22ebe703d53fec99b7716215cca0e55ee09cfaf70cf`.
  Exact-resume SHA-256:
  `aceb970c287430e641f1de0d2c32565959ba3c6f77be3a391ccab6b0a2a4ee24`.
- First layer realized logical range: `0.10204082--0.23940003`; CMO mapping:
  `9.0--21.1151 uS`; Wan mapping: `4.08163--9.57600 uS`.
- Second layer realized logical range: `0.10204082--0.56101835`; CMO mapping:
  `9.0--49.4818 uS`; Wan mapping: `4.08163--22.4407 uS`.
- No value reached the upper bound. The fraction at the finite lower floor was
  1.007% in the first layer and 4.95% in the second layer.
- Remote and collected local copies match file-by-file SHA-256 for the
  training run, test run, both checkpoints, validation artifacts, and both
  launcher sidecars. Local artifacts are under
  `results/mnist_bounded_drn_teacher_seed17_10ep_6aa32237*`.
