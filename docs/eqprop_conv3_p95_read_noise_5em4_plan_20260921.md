# Conv3 p95 read-noise 5e-4 repeat

Authorized by Filip: “launch 5e-4 runs”, September21,2026.
Three fresh seed0 cases: baseline, ours, legacy; sigma5e-4,30epochs each.
This request releases this level independently of the failed1e-3 condition.
No other noise levels or extra seeds. Existing clean-control plans remain
separate; no duplicate clean production in this launch. Three extra clean
configs are staged only because the unchanged wrapper's canary contract
exercises all six config slots; they are not production cases here.

Inherit all scientific settings from the p95 sigma1e-3 parent: injected beta
147.682614594/2.49274796756/2.81845428732 for baseline/ours/legacy, base beta
unchanged; same saved initializer, fresh Adam, exact scheme-specific weight
rates, zero frozen biases, [0,100] weights, input gain360, perfect diode,
float64 centered frozen-current EqProp, T=K=8, reset-each-batch. Ordinary
MNIST fixed55k/5k split, batch16/validation64, no official-test read. Noise
seed2026081601, independent Gaussian non-input endpoint readout perturbations;
relaxation and validation remain clean. Only sigma and descriptive IDs change.
Reuse the accepted T/K operating point with its finite-T residual caveat.
Same frozen parent scientific source archive74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819.

Use unchanged tested V100 wrapper and environment, fmu@v100/gpu_p13/
qos_gpu-dev/v100-16g,1GPU/4CPUs per task. Three GPUs concurrently, within the
user's six-GPU total cap. Queue three arrays with corresponding-task aftercorr
links and automatic cancellation of successors whose predecessor fails.
Ten epochs per allocation,120min Slurm cap/119min payload timeout; no manual
scientific validation pause at epoch10 or20. Native guards preserve exact
weights, Adam, RNGs, noise counts, history and best checkpoints.
Nine production allocations,18GPUh cap,19GPUh including local/remote gates.
Expected5–6h plus queues; deadline2026-09-23T10:30:09.440114Z.

Before production: exact config-difference check, inherited focused wrapper/
continuation tests, local real-runner CPU smokes for all three new cases and
one same-contract live V100 canary through both restores. Preserve all failed
receipts; fail closed on admission errors. No new scientific calibration.

Monitor operational progress; NonFiniteTrainingError is an independent
scientific negative outcome, terminates that case, and does not suppress the
other cases. No numerical retry or retuning. Unexpected operational errors
follow the armed-attempt stop/report policy. Final scientific review after
termination:30 finite epochs and final validation strictly less than5pp below
own best; report trajectories, transient drawdown, losses and exact failure
locations. Clean-relative comparison requires the matched clean controls;
missing clean coverage remains explicitly unavailable. Collect terminal
snapshots locally and register the physical inventory before final analysis.

Local results: results/eqprop-conv3-p95-read-noise-5em4-20260921-v1/.
Remote results: /lustre/fsn1/projects/rech/fmu/ucy17uy/server_code/results/eqprop-conv3-p95-read-noise-5em4-20260921-v1/.

September21 user override: add 1e-5 and 1e-4, and cancel all nine noise cases if any run fails. This replaces the independent scientific-failure continuation rule above. No retries. Shared observer: stop-all-02 in the 5e-4 result directory.
