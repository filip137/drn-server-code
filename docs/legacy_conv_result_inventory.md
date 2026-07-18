# Legacy Conv result inventory

Historical Conv result roots are immutable inputs. The canonical MNIST Conv
runner and collector never add manifests, statuses, checksums, or summaries to
them and never treat their directory names as proof of completion.

Two read-only classifications are used:

- `legacy_unverified`: a historical bundle whose scientific configuration,
  checkpoint rule, batch-state behavior, completion state, or artifact hashes
  have not all been verified.
- `curated`: a historical bundle reviewed under
  `docs/amplification_experiment_curation.md`. Curated rows keep their stated
  diagnostic/final/superseded classification; curation does not convert them
  into canonical run bundles.

New results live only below the content-addressed `runs/`, `sweeps/`, and
`attempts/` layout. To compare a legacy result with a new run, record its
original relative path and classification in analysis output; do not copy it
into the canonical store or rewrite the historical files.

Carried-state Conv results produced by the former within-epoch state reuse are
diagnostic. Paper-facing replacements must restart from initialization with
`batch_state_policy=reset_each_batch` and the frozen multi-seed comparison
contract.
