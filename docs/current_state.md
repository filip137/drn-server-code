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
strong three-epoch diagnostic candidates. Filip has now authorized nine
explicit parameter-wise LR vectors for current unbounded result rows. Use
those vectors unchanged as the interim handoff recorded in the authority
below.

The three Conv1/Conv2 rows still lacking an interim vector are:

- Conv1 legacy Adam;
- Conv2 ours SGD; and
- Conv2 ours Adam.

Do not fill those gaps from older configs or diagnostic winners. Terminal rho
selection and long-confirmation evidence remain incomplete even for the nine
interim-authorized rows.

Authority:
[`perfectdiode_learning_protocol.md`](perfectdiode_learning_protocol.md).

### Conv3

The ordinary-MNIST Conv3 contract uses:

- channels `[64,128,256]`, strides `[2,2,1]`, and padding `[1,1,1]`;
- shared diagnostic input gain `360`; and
- per-scheme `T/K` selection followed by six optimizer-specific rho surfaces.

The scientific protocol is authorized, but no terminal all-surface `T/K` and
LR handoff is recorded in the active documentation.

Authority:
[`perfectdiode_conv3_learning_protocol.md`](perfectdiode_conv3_learning_protocol.md).

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
nine user-directed interim Conv1/Conv2 vectors may be used for current
unbounded result rows; missing surfaces remain unresolved. Ordinary-MNIST
validation accuracy is never reported as paper evidence.

Current epoch decisions:

| Architecture | Paper budget |
|---|---:|
| Conv1 | 10 epochs |
| Conv2 | 30 epochs |
| Conv3 | pending |

Final model seeds, Conv3 epoch budget, checkpoint rule, and inclusion rule
remain pending. No complete paper grid is authorized until the corresponding
wide-range and bounded LR handoffs are terminal.

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

1. Run only the nine interim-authorized Conv1/Conv2 unbounded rows as needed;
   keep the three missing rows unresolved.
2. Finish terminal wide-range rho selection and confirmation for Conv1/Conv2.
3. Complete Conv3 `T/K` selection and its six wide-range rho surfaces.
4. Execute both bounded initializer families across all 18 surfaces and
   publish the global selector.
5. Freeze Conv3 budget, final seeds, checkpoint rule, and inclusion rule.
6. Generate exact medium-affine configs from the successful handoffs, smoke
   on each target, and launch the wide-range and bounded paper grids.
