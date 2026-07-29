# Current Perfect-Diode Conv State

Updated: 2026-07-29

This is a dashboard, not a second protocol. Selection rules remain in the
[active protocol index](conv_paper_hyperparameter_protocol.md). Live running
state is generated separately in
[`current_simulations.md`](current_simulations.md).

## Current Focus

The active question is perfect-diode BPTT training for:

- Conv1, Conv2, and Conv3;
- strides `[2]`, `[2,2]`, and `[2,2,1]`;
- kernel `3`, padding `1`, and no pooling;
- baseline `v1/c1`, proposed/ours `v4/c1`, and legacy `v4/c0.25`;
- plain SGD and Adam; and
- wide-range reference and bounded hardware weight contracts.

Rho and learning-rate selection uses ordinary MNIST. Paper training uses
deterministic medium-affine MNIST.

## Wide-Range Rho Status

The wide-range reference projects conductance weights to `[0,100]`.

### Conv1/Conv2

The fixed ordinary-MNIST operating points are:

| Architecture | Input gain | Operational `T/K` |
|---|---:|---:|
| Conv1 | `40` | `4/4` |
| Conv2 | `100` | `6/6` |

All six fixed-`T/K` security rows passed comparison with `(T=64,K=64)`.
The 12 scheme x optimizer core rho surfaces completed and produced generally
strong three-epoch diagnostic candidates. Filip has now authorized all 12
explicit parameter-wise LR vectors for current unbounded result rows. Use
those vectors unchanged, at the supplied precision, from the machine-readable
handoff below.

All six Conv1 vectors completed ten-epoch confirmations with ordinary-MNIST
validation accuracy between `95.60%` and `96.48%`; final review is pending.
Conv2 remains three-epoch best-observed evidence, with validation accuracy
between `95.96%` and `97.78%`. Terminal rho selection remains incomplete, and
the current handoff does not imply Conv2 long-confirmation evidence.

Authority:
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md) and
[`perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv12_unbounded_fixed_lr_handoff_20260729_v1.json).

### Conv3

The ordinary-MNIST Conv3 contract uses:

- channels `[64,128,256]`, strides `[2,2,1]`, and padding `[1,1,1]`;
- shared diagnostic input gain `360`; and
- the user-fixed `T=K=8` operating point for the completed rho study.

The six optimizer-specific rho surfaces completed on Jean Zay: 54 initial
cells plus 37 boundary-expansion cells, with all final search bounds closed.
The selected `(rho_conv,rho_dense)` pairs are baseline SGD
`(0.001,0.03)`, baseline Adam `(0.027,0.27)`, ours SGD `(0.009,0.03)`,
ours Adam `(0.081,0.09)`, legacy SGD `(0.003,0.01)`, and legacy Adam
`(0.027,0.01)`. These are one-seed, three-epoch ordinary-MNIST selection
results. The execution used `weight_min=0` and `weight_max=null`; it is
unbounded selector evidence rather than an identical `[0,100]` selection
surface. Filip explicitly authorized applying its six published vectors to
the downstream `[0,100]` paper configs; the source mismatch and missing long
confirmation remain recorded limitations.

Authority:
[`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md),
the
[`perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json`](../configs/conv/perfectdiode_conv3_unbounded_fixed_lr_handoff_20260729_v1.json)
handoff, and the
[`conv3_pd_unbounded_rho_t8k8_20260729T125734Z` manifest entry](experimental_manifest.md#conv3_pd_unbounded_rho_t8k8_20260729t125734z--conv3-perfect-diode-rho-selection).

## Bounded Hardware Status

The bounded condition projects `ConvWeight_*` and `DenseWeight_*` to
`[1e-5,1e-4]`. It independently screens:

- `bounded_uniform` sampled from `[1e-5,1e-4)`; and
- `bounded_kaiming_uniform` with gain `1`.

Both initializers require independent ordinary-MNIST rho selection for all 18
architecture x scheme x optimizer surfaces. A single initializer is then
chosen globally by matched 2% loss-plateau wins.

The repository retains an immutable Conv1/Conv2 predecessor config, but no
completed all-depth selector or global winner is recorded. Bounded
medium-affine paper runs therefore remain pending.

Authority:
[`perfectdiode_bounded_weight_protocol.md`](perfectdiode_bounded_weight_protocol.md).

## Paper-Run Status

The deterministic medium-affine paper grid has 36 rows per model seed:

```text
3 architectures x 3 schemes x 2 optimizers x 2 weight contracts
```

Each row must import its matching ordinary-MNIST LR handoff unchanged. The
12 user-directed Conv1/Conv2 vectors may be used for current unbounded result
rows. Ordinary-MNIST validation accuracy is never reported as paper evidence.

Current epoch decisions:

| Architecture | Paper budget |
|---|---:|
| Conv1 | 10 epochs |
| Conv2 | 30 epochs |
| Conv3 | 30 epochs |

The first wide-range batch uses model seed `0`, loader seed `0`, and affine
seed `1729`. It trains on the deterministic medium-affine 55,000/5,000
train/validation split, selects the maximum-validation-accuracy checkpoint,
and reads the official 10,000-example medium-affine test split exactly once
from that checkpoint. Every predeclared row completing its exact epoch budget
with finite metrics and verified artifacts is included; operational failures
may only be retried unchanged.

Filip authorized the Conv3 vectors selected under `weight_max=null` for use
with the `[0,100]` paper contract. That source-contract mismatch remains
explicit provenance rather than being silently erased. The 18 exact seed-0
wide-range configs are staged under
`configs/conv/paper_medium_affine_perfectdiode_wide_seed0_20260729_v1/`.
Bounded medium-affine rows remain blocked on the global bounded initializer and
LR handoff.

## Compute Targets

This table records stable access and routing information, not live occupancy.
Always check the target immediately before launch.

| Resource | Launcher target | Access / workdir | Preferred work |
|---|---|---|---|
| Local foreground | `local` | current checkout | unit tests and foreground smokes |
| Local GPU tmux | `main` | session `main`; `/home/filip/server_code` | Conv1, smokes, diagnostics |
| Akib | `akib` | SSH alias `akib`; `/home/filiposana/server_code` | Conv1/Conv2 surfaces |
| Trex | `trex` | `filip@trex`; `/home/filip/server_code` | Conv2/Conv3 surfaces and long confirmations |
| Jean Zay | `jean-zay` | SSH alias `jean-zay`; Slurm `fmu@v100` | Conv3, arrays, and scheduled production |

Jean Zay source checkout:
`/lustre/fswork/projects/rech/umg/$USER/server_code`.

Jean Zay result root:
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results`.

Keep one scientific surface on one recorded target. Host availability must not
change the surface's config, batch size, initialization, cohorts, or training
order.

## Next Actions

1. Pass the fail-closed local, Jean Zay test-only, and one-task live-canary
   gates for the staged 18-row seed-0 wide-range paper batch, then launch and
   collect its Conv1/Conv2/Conv3 arrays.
2. Finish Conv1 confirmation review and terminal wide-range rho publication;
   run the missing Conv2 long confirmations if terminal evidence is required.
3. Complete any required Conv3 long confirmation while retaining the explicit
   `weight_max=null` selection versus `[0,100]` downstream limitation.
4. Execute both bounded initializer families across all 18 surfaces and
   publish the global selector.
5. Decide later multi-seed scope after the seed-0 wide-range batch is analyzed.
6. Generate the bounded medium-affine configs only after the successful
   all-depth bounded handoff.
