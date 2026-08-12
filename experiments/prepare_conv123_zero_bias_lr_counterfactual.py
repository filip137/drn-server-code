from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / (
        "configs/conv/"
        "perfectdiode_conv123_zero_bias_lr_counterfactual_ordinary_mnist_"
        "seed0_20260807_v1"
    )
)
STUDY_ID = (
    "perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-"
    "seed0-20260807-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_lr_counterfactual"

GRADIENT_SUMMARY_PATH = (
    "results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/"
    "analysis/mechanism/gradient_summary.csv"
)
GRADIENT_SUMMARY_SHA256 = (
    "0519ea3125f4bee5c57b4429f5cfc2fff2b65b9278f61cdb63106133b51d5848"
)
REPLAY_RECEIPT_PATH = (
    "results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1/"
    "analysis/mechanism/mechanism_replay_receipt.json"
)
REPLAY_RECEIPT_SHA256 = (
    "3bc1d09ec6101dfd05309f358796434a9a5b3535cdac440821d13072e1f12fb6"
)
COHORT_SHA256 = "a9b8b890b187fb5023fa93ce6fe754229ad94b083d33670857d05df1f8d28074"
PROPOSAL_METRIC = "lr_times_gradient_over_parameter_rms_median"
PROPOSAL_FORMULA = (
    "eta_baseline_to_target_i = eta_baseline_i * "
    "proposal_target_i / proposal_baseline_i"
)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    parameter_order: tuple[str, ...]
    source_epochs: int
    proposals: dict[str, dict[str, float]]
    exact_match_rates: dict[str, dict[str, float]]


ARCHITECTURES = (
    ArchitectureSpec(
        name="conv1",
        parameter_order=("ConvWeight_0", "DenseWeight_0", "Bias_0"),
        source_epochs=10,
        proposals={
            "baseline": {
                "ConvWeight_0": 0.007318314292361338,
                "DenseWeight_0": 0.02595955130405961,
            },
            "ours": {
                "ConvWeight_0": 0.007189495737495173,
                "DenseWeight_0": 0.02399481191250834,
            },
            "legacy": {
                "ConvWeight_0": 0.0024523720428708336,
                "DenseWeight_0": 0.025126888285143402,
            },
        },
        exact_match_rates={
            "ours": {
                "ConvWeight_0": 0.14018423406991842,
                "DenseWeight_0": 0.01111378298085222,
            },
            "legacy": {
                "ConvWeight_0": 0.047817525600773715,
                "DenseWeight_0": 0.011638131792965963,
            },
        },
    ),
    ArchitectureSpec(
        name="conv2",
        parameter_order=(
            "ConvWeight_0",
            "ConvWeight_1",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
        ),
        source_epochs=30,
        proposals={
            "baseline": {
                "ConvWeight_0": 0.01984737151245336,
                "ConvWeight_1": 0.021937469890476118,
                "DenseWeight_0": 0.08064787178479493,
            },
            "ours": {
                "ConvWeight_0": 0.006162791523796401,
                "ConvWeight_1": 0.007333206025710122,
                "DenseWeight_0": 0.026860646198426333,
            },
            "legacy": {
                "ConvWeight_0": 0.0028768346079271374,
                "ConvWeight_1": 0.0026390670932861003,
                "DenseWeight_0": 0.027405752577325,
            },
        },
        exact_match_rates={
            "ours": {
                "ConvWeight_0": 2.455705508720658,
                "ConvWeight_1": 1.2751890331155598,
                "DenseWeight_0": 0.2733196996026676,
            },
            "legacy": {
                "ConvWeight_0": 1.1463406748526415,
                "ConvWeight_1": 0.4589137961235322,
                "DenseWeight_0": 0.27886641320856775,
            },
        },
    ),
    ArchitectureSpec(
        name="conv3",
        parameter_order=(
            "ConvWeight_0",
            "ConvWeight_1",
            "ConvWeight_2",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
            "Bias_2",
        ),
        source_epochs=30,
        proposals={
            "baseline": {
                "ConvWeight_0": 0.0006556054641313252,
                "ConvWeight_1": 0.0008241507398236009,
                "ConvWeight_2": 0.0008950076048284659,
                "DenseWeight_0": 0.027286645693103014,
            },
            "ours": {
                "ConvWeight_0": 0.00515077307138097,
                "ConvWeight_1": 0.00747213599730087,
                "ConvWeight_2": 0.007966611142005073,
                "DenseWeight_0": 0.027268337761241077,
            },
            "legacy": {
                "ConvWeight_0": 0.002029970575781221,
                "ConvWeight_1": 0.002509664256561181,
                "ConvWeight_2": 0.002682528952416708,
                "DenseWeight_0": 0.009292070240297805,
            },
        },
        exact_match_rates={
            "ours": {
                "ConvWeight_0": 17.44444006505199,
                "ConvWeight_1": 56.97686316961596,
                "ConvWeight_2": 41.566527660089776,
                "DenseWeight_0": 1.1914265319047057,
            },
            "legacy": {
                "ConvWeight_0": 6.875026244078034,
                "ConvWeight_1": 19.136803318276744,
                "ConvWeight_2": 13.99634197177058,
                "DenseWeight_0": 0.4059953752057774,
            },
        },
    ),
)


@dataclass(frozen=True)
class CaseSpec:
    filename: str
    role: str
    scheme: str
    parent_index: int
    target_scheme: str | None


CASES = (
    CaseSpec(
        filename="00_baseline_sgd_current.json",
        role="baseline_current",
        scheme="baseline",
        parent_index=0,
        target_scheme=None,
    ),
    CaseSpec(
        filename="01_baseline_sgd_conv3x_stress.json",
        role="baseline_conv3x_stress",
        scheme="baseline",
        parent_index=0,
        target_scheme=None,
    ),
    CaseSpec(
        filename="02_baseline_sgd_ours_proposal_exact.json",
        role="baseline_ours_proposal_exact",
        scheme="baseline",
        parent_index=0,
        target_scheme="ours",
    ),
    CaseSpec(
        filename="03_baseline_sgd_legacy_proposal_exact.json",
        role="baseline_legacy_proposal_exact",
        scheme="baseline",
        parent_index=0,
        target_scheme="legacy",
    ),
    CaseSpec(
        filename="04_ours_sgd_current.json",
        role="ours_current",
        scheme="ours",
        parent_index=2,
        target_scheme=None,
    ),
    CaseSpec(
        filename="05_legacy_sgd_current.json",
        role="legacy_current",
        scheme="legacy",
        parent_index=4,
        target_scheme=None,
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_reference(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _optional_verify_measurement_artifacts() -> None:
    for relative_path, expected_digest in (
        (GRADIENT_SUMMARY_PATH, GRADIENT_SUMMARY_SHA256),
        (REPLAY_RECEIPT_PATH, REPLAY_RECEIPT_SHA256),
    ):
        path = REPO_ROOT / relative_path
        if path.is_file():
            actual = _sha256(path.read_bytes())
            if actual != expected_digest:
                raise ValueError(
                    f"Frozen proposal artifact changed: {relative_path}: "
                    f"expected {expected_digest}, got {actual}"
                )


def _source_filename(case: CaseSpec) -> str:
    label = {0: "baseline", 2: "ours", 4: "legacy"}[case.parent_index]
    return f"{case.parent_index:02d}_{label}_sgd_bias_zero_seed0.json"


def _weight_names(architecture: ArchitectureSpec) -> tuple[str, ...]:
    return tuple(
        name for name in architecture.parameter_order if not name.startswith("Bias_")
    )


def _bias_names(architecture: ArchitectureSpec) -> tuple[str, ...]:
    return tuple(
        name for name in architecture.parameter_order if name.startswith("Bias_")
    )


def _validate_parent(
    architecture: ArchitectureSpec,
    case: CaseSpec,
    config: dict[str, Any],
    path: Path,
) -> None:
    if tuple(config.get("parameter_order", ())) != architecture.parameter_order:
        raise ValueError(f"Unexpected parameter order in {path}")
    if config.get("optimizer", {}).get("name") != "SGD":
        raise ValueError(f"Expected an SGD parent config: {path}")
    if config.get("seed") != 0:
        raise ValueError(f"Expected seed zero in {path}")
    if config.get("lab", {}).get("epochs") != architecture.source_epochs:
        raise ValueError(f"Unexpected source epoch budget in {path}")
    official_test_policy = (
        config.get("evaluation", {}).get("official_test", {}).get("policy")
    )
    if official_test_policy != "disabled":
        raise ValueError(f"Expected the official test to be disabled in {path}")
    rates = config.get("learning_rates_by_parameter", {})
    if any(
        float(rates.get(name, math.nan)) != 0.0
        for name in _bias_names(architecture)
    ):
        raise ValueError(f"Expected exact-zero bias rates in {path}")
    expected_scheme = {0: "baseline", 2: "ours", 4: "legacy"}[case.parent_index]
    if case.scheme != expected_scheme:
        raise ValueError(f"Case/parent scheme mismatch for {path}")


def _proposal_derivation(
    architecture: ArchitectureSpec,
    case: CaseSpec,
    parent_rates: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    weights = _weight_names(architecture)
    baseline_proposals = architecture.proposals["baseline"]

    if case.role == "baseline_conv3x_stress":
        multipliers = {
            name: 3.0 if name.startswith("ConvWeight_") else 1.0
            for name in weights
        }
        rates = {name: parent_rates[name] * multipliers[name] for name in weights}
        target_proposals = {
            name: baseline_proposals[name] * multipliers[name] for name in weights
        }
        derivation = {
            "formula": "eta_i = eta_parent_i * declared_multiplier_i",
            "kind": "conv_learning_rate_times_three_dense_unchanged_stress",
            "multipliers_by_parameter": multipliers,
            "source_proposals_by_parameter": baseline_proposals,
            "target_proposals_by_parameter": target_proposals,
            "target_scheme": None,
        }
        return rates, derivation

    if case.target_scheme is not None:
        target = architecture.proposals[case.target_scheme]
        multipliers = {
            name: target[name] / baseline_proposals[name] for name in weights
        }
        computed = {
            name: parent_rates[name] * multipliers[name] for name in weights
        }
        rates = architecture.exact_match_rates[case.target_scheme]
        for name in weights:
            if not math.isclose(
                computed[name], rates[name], rel_tol=2e-15, abs_tol=0.0
            ):
                raise ValueError(
                    f"Frozen exact rate does not reproduce the proposal formula: "
                    f"{architecture.name}/{case.target_scheme}/{name}: "
                    f"computed={computed[name]!r}, frozen={rates[name]!r}"
                )
        derivation = {
            "formula": PROPOSAL_FORMULA,
            "kind": "exact_fixed_cohort_initial_relative_proposal_match",
            "multipliers_by_parameter": multipliers,
            "source_proposals_by_parameter": baseline_proposals,
            "target_proposals_by_parameter": target,
            "target_scheme": case.target_scheme,
        }
        return dict(rates), derivation

    scheme_proposals = architecture.proposals[case.scheme]
    rates = {name: parent_rates[name] for name in weights}
    derivation = {
        "formula": "eta_i = eta_parent_i",
        "kind": "published_current_rate_control",
        "multipliers_by_parameter": {name: 1.0 for name in weights},
        "source_proposals_by_parameter": scheme_proposals,
        "target_proposals_by_parameter": scheme_proposals,
        "target_scheme": case.scheme,
    }
    return rates, derivation


def _arm_id(architecture: ArchitectureSpec, case: CaseSpec) -> str:
    suffix = {
        "baseline_current": "baseline_sgd_bias_zero_lr_current_seed0",
        "baseline_conv3x_stress": "baseline_sgd_bias_zero_conv3x_stress_seed0",
        "baseline_ours_proposal_exact": (
            "baseline_sgd_bias_zero_ours_proposal_exact_seed0"
        ),
        "baseline_legacy_proposal_exact": (
            "baseline_sgd_bias_zero_legacy_proposal_exact_seed0"
        ),
        "ours_current": "ours_sgd_bias_zero_lr_current_seed0",
        "legacy_current": "legacy_sgd_bias_zero_lr_current_seed0",
    }[case.role]
    return f"{architecture.name}_{suffix}"


def _config_bytes(
    architecture: ArchitectureSpec,
    case: CaseSpec,
) -> tuple[bytes, dict[str, Any]]:
    source_path = SOURCE_ROOT / architecture.name / _source_filename(case)
    source_bytes = source_path.read_bytes()
    config: dict[str, Any] = deepcopy(json.loads(source_bytes))
    _validate_parent(architecture, case, config, source_path)

    arm_id = _arm_id(architecture, case)
    parent_rates = {
        name: float(config["learning_rates_by_parameter"][name])
        for name in _weight_names(architecture)
    }
    weight_rates, derivation = _proposal_derivation(
        architecture, case, parent_rates
    )
    all_rates = {
        **weight_rates,
        **{name: 0.0 for name in _bias_names(architecture)},
    }
    ordered_rates = [all_rates[name] for name in architecture.parameter_order]

    config["arm_id"] = arm_id
    config["study_id"] = STUDY_ID
    config["lab"]["epochs"] = 30
    config["lr"] = ordered_rates
    config["optimizer"]["learning_rate"] = ordered_rates
    config["learning_rates_by_parameter"] = all_rates
    config["reporting"] = {
        "arm_id": arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "study_id": STUDY_ID,
    }
    config["evaluation"]["inclusion_rule"] = (
        "Include this predeclared ordinary-MNIST LR-counterfactual arm only if "
        "it completes all 30 epochs with finite metrics, all 31 model and "
        "optimizer checkpoints and all 150 traced optimizer transitions are "
        "complete, every hidden-bias learning rate and checkpoint tensor is "
        "exactly zero, the result bundle validates, and the official test is "
        "not read."
    )

    bias_ablation = config["handoff"]["diagnostic_bias_ablation"]
    rates_match_parent = all(
        float(config_rate) == float(parent_rate)
        for config_rate, parent_rate in zip(
            ordered_rates,
            json.loads(source_bytes)["lr"],
            strict=True,
        )
    )
    bias_ablation["all_weight_learning_rates_unchanged"] = rates_match_parent
    bias_ablation["scientific_question"] = (
        "Can a 30-epoch, layer-specific baseline learning-rate counterfactual "
        "recover the zero-bias SGD trajectory of ours or legacy, and how does "
        "that intervention depend on convolutional depth?"
    )
    config["handoff"]["lr_counterfactual"] = {
        "architecture": architecture.name,
        "parent_config": str(source_path.relative_to(REPO_ROOT)),
        "parent_config_sha256": _sha256(source_bytes),
        "proposal_derivation": derivation,
        "proposal_measurement": {
            "bptt_gradient_batches": 2,
            "checkpoint_role": "initialization",
            "cohort_sha256": COHORT_SHA256,
            "cohort_examples": 128,
            "gradient_summary": _artifact_reference(
                GRADIENT_SUMMARY_PATH, GRADIENT_SUMMARY_SHA256
            ),
            "metric": PROPOSAL_METRIC,
            "optimizer_label_in_replay": "SGD",
            "performance_metrics_used_for_rate_derivation": False,
            "replay_receipt": _artifact_reference(
                REPLAY_RECEIPT_PATH, REPLAY_RECEIPT_SHA256
            ),
            "values_frozen_in_config": True,
        },
        "rates_selected_post_hoc_from_this_study": False,
        "role": case.role,
        "weight_rates_match_parent": rates_match_parent,
    }

    payload = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    row = {
        "architecture": architecture.name,
        "arm_id": arm_id,
        "config": str(
            (OUTPUT_ROOT / architecture.name / case.filename).relative_to(REPO_ROOT)
        ),
        "config_sha256": _sha256(payload),
        "epoch_budget": 30,
        "optimizer": "SGD",
        "proposal_derivation": derivation,
        "role": case.role,
        "scheme": case.scheme,
        "source_config": str(source_path.relative_to(REPO_ROOT)),
        "source_config_sha256": _sha256(source_bytes),
        "weight_rates_by_parameter": weight_rates,
        "weight_rates_match_parent": rates_match_parent,
    }
    return payload, row


def _readme_bytes() -> bytes:
    text = """# Conv1/2/3 zero-bias long-LR counterfactual

Status: deterministic 18-arm configuration set prepared; no production job is
implied by materialization.

## Question

Can a baseline perfect-diode convolutional network recover the zero-bias SGD
trajectory of ours or legacy merely by changing its layer-specific learning
rates over a full 30-epoch run, and does that answer change with depth?

This is an ordinary-MNIST learning-rate selection and mechanism study. Its
validation metrics are diagnostic and are not paper-facing accuracy evidence.
Any selected vector must be frozen before a separate deterministic
medium-affine confirmation.

## Frozen design

Conv1, Conv2, and Conv3 each contain the same six scientific roles:

1. baseline at its current accepted weight rates;
2. baseline with every Conv rate multiplied by three and Dense unchanged;
3. baseline exactly matched to ours' fixed-cohort initial relative proposals;
4. baseline exactly matched to legacy's fixed-cohort initial relative proposals;
5. ours at its current accepted weight rates; and
6. legacy at its current accepted weight rates.

All 18 arms use ordinary MNIST, seed 0, the canonical architecture-specific
operating point and input gain, `[0,100]` weight projection, plain SGD, exact
zero hidden biases, and 30 epochs. Every run is intended to save initialization
plus all 30 epoch model/optimizer checkpoints and five real optimizer-transition
traces per epoch.

The exact proposal-matched baseline rate for weight `i` is

```text
eta_baseline_to_target_i =
    eta_baseline_i * proposal_target_i / proposal_baseline_i
```

where `proposal_i` is the frozen initialization value of
`LR * RMS(gradient) / RMS(weight)`. The complete source proposal values,
per-parameter multipliers, rates, parent-config hashes, and replay provenance
are embedded in every generated config and in `study_manifest.json`.

The proposal evidence comprises two BPTT-gradient batches from the fixed
128-example replay cohort. The rate derivation consumes no target performance
metric. Exact proposal matching is limited to per-weight initialization RMS
magnitudes on that cohort: it does not match gradient direction, coordinate
distributions, output scaling, the later optimizer trajectory, or establish
global learning-rate optimality.

The frozen measurement sources are recorded by hash even though `results/` is
not part of a staged source archive. Materialization is self-contained; when
the local source artifacts are present, the generator additionally verifies
their bytes.

## Selection

Baseline selection is independent for Conv1, Conv2, and Conv3. For each
architecture, rank only its four baseline candidates by mean validation loss
over epochs 26--30, form an inclusive 2% loss plateau, and apply the fully
declared deterministic tie-breaks in `study_manifest.json`. Ours and legacy
are comparison controls, not selector candidates. Missing, non-finite, or
contract-incomplete arms are ineligible.

This selects among a predeclared mechanistic set; it does not prove a globally
optimal baseline learning rate.

## Prepared Jean Zay envelope

No job is submitted by this package. The planned execution uses
`experiments/run_conv123_zero_bias_lr_counterfactual_jeanzay.slurm` on
`fmu@v100`, `gpu_p13`, `qos_gpu-t3`, `v100-16g`, with one GPU, ten CPUs,
`06:00:00` per task, module `pytorch-gpu/py3/2.5.0`, and array `0-17%6`.
The remote result pattern is
`/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/` followed by the
study ID. The conservative budgets are 6, 12, and 24 V100-hours for Conv1,
Conv2, and Conv3 respectively: 42 V100-hours total and roughly seven hours of
ideal compute wall time at concurrency six, plus queueing.

## Conditional paper sequence

Review the SGD mechanism panel before expanding it. If optimizer-general
evidence is needed, first make a new frozen-initialization fresh-state Adam
proposal replay for all three architectures; Adam moment state means that raw
gradient ratios cannot be substituted for optimizer-step proposal ratios.
Only then materialize the analogous Adam counterfactual panel.

After ordinary-MNIST selection, freeze one selected baseline vector per tested
architecture and optimizer. The paper-facing deterministic medium-affine
confirmation uses affine seed 1729, paired model/loader seeds 0, 1, and 2, and
four roles per architecture: baseline current, baseline selected, ours
current, and legacy current. This is 36 arms per optimizer across Conv1/2/3,
with a conservative 84 V100-hour budget per optimizer. Select the best
validation checkpoint, read the official medium-affine test split exactly
once, report paired seed differences and spread, and never retune from the
medium-affine result.

## Materialization

```bash
python -m experiments.prepare_conv123_zero_bias_lr_counterfactual
python -m experiments.prepare_conv123_zero_bias_lr_counterfactual --check
```

The generator reads only the canonical checked-in zero-bias parent configs.
It does not require ignored result files.

## Local preparation gate

On 2026-08-07 the final 18-config set passed a synchronous local CPU smoke
through `experiments.exact_run` with the local MNIST path used only as a
transport override, one real train batch, one validation batch, gradient
tracing, initialization/epoch checkpoints, and the official test disabled.
All 18 canonical bundles validated; their trace and checkpoint inventories
were complete, their split and first-epoch order hashes matched, and every
configured and checkpointed hidden bias remained exactly zero. The ephemeral
smoke root was `/tmp/pd-c123-zb-lr30-smoke-g6TZCT`; it is gate evidence, not a
scientific result directory. The final static suite passed 67 focused tests,
Python compilation, and `bash -n` on the Slurm wrapper.
"""
    return text.encode("utf-8")


def _build_outputs() -> dict[Path, bytes]:
    _optional_verify_measurement_artifacts()
    outputs: dict[Path, bytes] = {}
    rows: list[dict[str, Any]] = []
    config_digests: list[str] = []
    selectors: dict[str, Any] = {}

    for architecture in ARCHITECTURES:
        architecture_candidates: list[str] = []
        for local_index, case in enumerate(CASES):
            payload, row = _config_bytes(architecture, case)
            path = OUTPUT_ROOT / architecture.name / case.filename
            row["global_index"] = len(rows)
            row["index_within_architecture"] = local_index
            outputs[path] = payload
            rows.append(row)
            config_digests.append(row["config_sha256"])
            if case.scheme == "baseline":
                architecture_candidates.append(row["arm_id"])

        selectors[architecture.name] = {
            "baseline_candidates": architecture_candidates,
            "eligibility": (
                "exactly 30 completed epochs; finite metrics; complete 31-entry "
                "model/optimizer checkpoint series; complete 150-transition "
                "gradient trace; exact-zero configured and checkpoint biases; "
                "validated result bundle; official_test_read=false"
            ),
            "primary_metric": "mean_validation_loss_epochs_26_through_30",
            "relative_loss_plateau": 0.02,
            "tie_breaks": [
                "higher_mean_validation_accuracy_epochs_26_through_30",
                "lower_maximum_target_conv_proposal",
                "lower_sum_target_weight_proposals",
                "lower_target_dense_proposal",
                "arm_id",
            ],
        }

    ordered_digest_payload = "".join(f"{digest}\n" for digest in config_digests)
    manifest = {
        "axes": {
            "architectures": [item.name for item in ARCHITECTURES],
            "optimizer": ["SGD"],
            "seed": [0],
            "schemes": ["baseline", "ours", "legacy"],
        },
        "bias_contract": {
            "initialization": "zero",
            "learning_rate": 0.0,
            "required_checkpoint_value": 0.0,
        },
        "conditional_follow_up": {
            "adam_mechanism_panel": {
                "prerequisite": (
                    "new frozen-initialization fresh-state Adam proposal replay "
                    "for Conv1/2/3; do not substitute raw-gradient ratios"
                ),
                "status": "not_materialized_pending_sgd_review",
            },
            "medium_affine_confirmation": {
                "affine_seed": 1729,
                "arms_per_optimizer": 36,
                "expected_gpu_hours_per_optimizer": 84,
                "model_and_loader_seeds": [0, 1, 2],
                "official_test_policy": (
                    "evaluate once from the selected validation checkpoint; "
                    "never retune on medium affine"
                ),
                "roles_per_architecture": [
                    "baseline_current",
                    "baseline_selected_counterfactual",
                    "ours_current",
                    "legacy_current",
                ],
                "status": "after_ordinary_mnist_selection",
            },
        },
        "configs": rows,
        "created_at": "2026-08-07",
        "dataset": {
            "affine_corruption": False,
            "name": "MNIST",
            "official_test_read": False,
            "source_split": "train",
            "train_validation_split": "deterministic_stratified_55000_5000",
            "variant": "ordinary",
        },
        "diagnostics": {
            "checkpoint_every_epoch": True,
            "expected_checkpoint_epochs": list(range(31)),
            "gradient_trace_samples_per_epoch": 5,
            "expected_gradient_trace_transitions": 150,
        },
        "epoch_budget": {item.name: 30 for item in ARCHITECTURES},
        "evidence_class": EVIDENCE_CLASS,
        "launch_plan": {
            "account": "fmu@v100",
            "array": "0-17%6",
            "constraint": "v100-16g",
            "cpus_per_task": 10,
            "expected_compute_wall_hours_at_concurrency_six": 7,
            "expected_gpu_hours_by_architecture": {
                "conv1": 6,
                "conv2": 12,
                "conv3": 24,
            },
            "expected_gpu_hours_total": 42,
            "gpus_per_task": 1,
            "module": "pytorch-gpu/py3/2.5.0",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "remote_result_pattern": (
                "/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/"
                + STUDY_ID
            ),
            "status": "prepared_not_submitted",
            "target": "jean-zay",
            "time_limit": "06:00:00",
            "wrapper": (
                "experiments/run_conv123_zero_bias_lr_counterfactual_jeanzay.slurm"
            ),
        },
        "ordered_config_set_sha256": _sha256(
            ordered_digest_payload.encode("utf-8")
        ),
        "paper_facing": False,
        "proposal_measurement": {
            "checkpoint_role": "initialization",
            "bptt_gradient_batches": 2,
            "cohort_sha256": COHORT_SHA256,
            "cohort_examples": 128,
            "formula": PROPOSAL_FORMULA,
            "gradient_summary": _artifact_reference(
                GRADIENT_SUMMARY_PATH, GRADIENT_SUMMARY_SHA256
            ),
            "metric": PROPOSAL_METRIC,
            "performance_metrics_used_for_rate_derivation": False,
            "replay_receipt": _artifact_reference(
                REPLAY_RECEIPT_PATH, REPLAY_RECEIPT_SHA256
            ),
            "values_frozen_in_generator": True,
        },
        "proposal_match_limitations": [
            (
                "Matches only per-weight initialization RMS proposal magnitudes "
                "on the frozen cohort."
            ),
            "Does not match gradient direction or coordinate distributions.",
            "Does not match output scaling or the later optimizer trajectory.",
            "Does not establish global learning-rate optimality.",
        ],
        "promotion_rules_by_architecture": selectors,
        "promotion_use": (
            "Freeze each architecture's selected baseline vector before a new "
            "deterministic medium-affine zero-bias confirmation; never report "
            "ordinary-MNIST accuracy as paper evidence."
        ),
        "schema_version": (
            "perfectdiode-conv123-zero-bias-lr-counterfactual-study/v1"
        ),
        "scientific_question": (
            "Over a full 30-epoch budget, can baseline SGD match the zero-bias "
            "optimization of ours or legacy after exact per-weight proposal "
            "matching, and how does that intervention depend on Conv depth?"
        ),
        "study_id": STUDY_ID,
        "weight_contract": {"projection": [0.0, 100.0]},
    }
    outputs[OUTPUT_ROOT / "study_manifest.json"] = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    outputs[OUTPUT_ROOT / "README.md"] = _readme_bytes()
    return outputs


def prepare(*, check: bool) -> None:
    outputs = _build_outputs()
    if check:
        mismatches = [
            str(path.relative_to(REPO_ROOT))
            for path, expected in outputs.items()
            if not path.is_file() or path.read_bytes() != expected
        ]
        if mismatches:
            raise SystemExit(
                "Materialized Conv1/2/3 LR-counterfactual files are stale or "
                "missing: " + ", ".join(mismatches)
            )
        return

    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact Conv1/2/3 zero-bias SGD LR counterfactuals."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files differ from the deterministic build.",
    )
    args = parser.parse_args()
    prepare(check=args.check)


if __name__ == "__main__":
    main()
