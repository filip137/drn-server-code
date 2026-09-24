# Baseline EqProp: larger-T/K qualification

The original Conv2/Conv3 baseline failures are reproduced at T=K=6/8 on the local RTX 3090. Larger iteration counts resolve the gradient and equilibrium gates on the same checkpoints and fixed validation batches. No training or official-test evaluation is included.

| Architecture | Qualified T/K | Largest passing tested beta | Worst cosine | Maximum symmetric norm difference |
|---|---:|---:|---:|---:|
| conv2 | 12/12 | 0.1 | 0.997672 | 0.019521 |
| conv3 | 32/32 | 0.1 | 0.997096 | 0.047384 |

The follow-up keeps native K and reduces the iteration cost. These are the selected settings for the newly authorized matched BPTT/EqProp baseline training:

| Architecture | Selected T/K | Common beta | Worst cosine | Maximum symmetric norm difference |
|---|---:|---:|---:|---:|
| conv2 | 12/6 | 0.1 | 0.997674 | 0.019507 |
| conv3 | 24/8 | 0.1 | 0.997096 | 0.071091 |

Conv3 T=20/K=8 still fails the norm threshold at the largest ceiling; T=24/K=8 passes everywhere. Larger beta candidates 100, 10 and 1 are rejected once a ceiling fails; those deliberately short-circuited failed candidates do not have all-three-ceiling coverage. Every selected candidate does.

Filip requested the new T/K for both algorithms. First train new seed-0 BPTT references and repeat the beta qualification at their best checkpoints, then run the three full EqProp pilots before releasing seeds 1/2. The previous 18 baseline BPTT results remain intact as original-operating-point evidence.

All 61 full diagnostic replays validate locally: both checkpoint roles, all four batches, finite float64 calculations, exact-zero biases and unchanged source/checkpoint bytes. The native GPU controls reproduce the prior CPU failure measurements. No layer or batch is excluded.

| Architecture | T/K | Previously failing ceiling | Worst cosine | Maximum symmetric norm difference | Combined gate |
|---|---:|---:|---:|---:|---|
| conv2 | 12/6 | 1em4 | 0.999422 | 0.020955 | pass |
| conv2 | 6/12 | 1em4 | 0.952220 | 0.260173 | fail |
| conv3 | 16/8 | 1em3 | 0.936254 | 0.272664 | fail |
| conv3 | 8/16 | 1em3 | -0.130989 | 1.068223 | fail |

For Conv2 at the tight ceiling, increasing only the free-phase T fixes the mismatch; increasing only the gradient-phase K does not. This supports insufficient free-phase relaxation as the cause in that case. The Conv3 controls show what doubling one phase achieves, without establishing a globally optimal T/K choice.

The gate requires every weight-layer/batch cosine ≥ .99, symmetric norm difference ≤ .10, and all projected-residual p90 checks ≤ .01. The selected beta is shared across all three ceilings. Selection tested larger betas in descending order at the first passing equal-T/K setting; higher iteration counts were not searched once a setting passed.

The source BPTT checkpoints were trained at T/K=6/6 and 8/8; these replays evaluate both gradients at the explicitly larger diagnostic T/K. New full training would change the baseline operating-point contract and still requires three full seed-0 pilots before releasing seeds 1/2. Existing BPTT runs cannot be relabeled as trained at these larger counts.

Individual replay reports inherit a generic “shared paper T/K” introductory phrase from the frozen analyzer. Their T/K tables, execution manifests and source-native T/K fields record the actual override; this collection is diagnostic evidence, not a replacement of the original paper contract.

![Gradient agreement versus relaxation iterations](figures/baseline_tk_beta_qualification.png)

[All diagnostic measurements](baseline_tk_beta_qualification.csv) · [Selection and source provenance](provenance/baseline_tk_20260913/selection.json)
