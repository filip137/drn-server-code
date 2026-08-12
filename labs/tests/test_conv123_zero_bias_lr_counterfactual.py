from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments.exact_run import load_exact_config
from experiments.prepare_conv123_zero_bias_lr_counterfactual import prepare


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_lr_counterfactual_ordinary_mnist_seed0_20260807_v1"
)
SOURCE_ROOT = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1"
)
WRAPPER = (
    REPO_ROOT / "experiments/run_conv123_zero_bias_lr_counterfactual_jeanzay.slurm"
)
PROPOSAL_SOURCE = (
    REPO_ROOT
    / "results/perfectdiode-conv123-zero-bias-ordinary-mnist-seed0-20260805-v1"
    / "analysis/mechanism/gradient_summary.csv"
)
PROPOSAL_SOURCE_SHA256 = (
    "0519ea3125f4bee5c57b4429f5cfc2fff2b65b9278f61cdb63106133b51d5848"
)
REPLAY_RECEIPT_RELATIVE_PATH = (
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
STUDY_ID = (
    "perfectdiode-conv123-zero-bias-lr-counterfactual-ordinary-mnist-"
    "seed0-20260807-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_lr_counterfactual"
EXPECTED_ORDERED_CONFIG_SET_SHA256 = (
    "fef39c6899821ab187839415947a4869dbe04f16fd59458b2d475fce510b1f01"
)

CASE_FILENAMES = (
    "00_baseline_sgd_current.json",
    "01_baseline_sgd_conv3x_stress.json",
    "02_baseline_sgd_ours_proposal_exact.json",
    "03_baseline_sgd_legacy_proposal_exact.json",
    "04_ours_sgd_current.json",
    "05_legacy_sgd_current.json",
)
ARCHITECTURES = ("conv1", "conv2", "conv3")
EXPECTED_CONFIG_ORDER = tuple(
    f"{architecture}/{filename}"
    for architecture in ARCHITECTURES
    for filename in CASE_FILENAMES
)

ARCHITECTURE_CONTRACT = {
    "conv1": {
        "parameter_order": ["ConvWeight_0", "DenseWeight_0", "Bias_0"],
        "iterations": 4,
        "input_gain": 40.0,
        "bias_count": 1,
        "layer_shapes": [[2, 28, 28], [64, 14, 14], [20]],
        "strides": [2],
    },
    "conv2": {
        "parameter_order": [
            "ConvWeight_0",
            "ConvWeight_1",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
        ],
        "iterations": 6,
        "input_gain": 100.0,
        "bias_count": 2,
        "layer_shapes": [
            [2, 28, 28],
            [64, 14, 14],
            [128, 7, 7],
            [20],
        ],
        "strides": [2, 2],
    },
    "conv3": {
        "parameter_order": [
            "ConvWeight_0",
            "ConvWeight_1",
            "ConvWeight_2",
            "DenseWeight_0",
            "Bias_0",
            "Bias_1",
            "Bias_2",
        ],
        "iterations": 8,
        "input_gain": 360.0,
        "bias_count": 3,
        "layer_shapes": [
            [2, 28, 28],
            [64, 14, 14],
            [128, 7, 7],
            [256, 7, 7],
            [20],
        ],
        "strides": [2, 2, 1],
    },
}

SCHEME_AMPLIFICATION = {
    "baseline": (1.0, 1.0),
    "ours": (4.0, 1.0),
    "legacy": (4.0, 0.25),
}

SOURCE_CONFIGS = {
    "conv1": {
        "baseline": ("00_baseline_sgd_bias_zero_seed0.json", "d20dac2f0bcffb2c4a2e2869f2fdcbe8847f3826d556e6fde8f3930fca507b96"),
        "ours": ("02_ours_sgd_bias_zero_seed0.json", "9a4a43b95c7e9f4641ca60cfea5a939b1c48441465b626c409d95f857b4e63ff"),
        "legacy": ("04_legacy_sgd_bias_zero_seed0.json", "27b665cebc33e5bd5a0d7ead5c84417fc04ac73a31c12d73a7f9b7a5bc3cb58b"),
    },
    "conv2": {
        "baseline": ("00_baseline_sgd_bias_zero_seed0.json", "71f0cf64f30fad9599dd435548d900cc03067c84f80ab4f2fea13f14211e3b54"),
        "ours": ("02_ours_sgd_bias_zero_seed0.json", "ad36f8ea697fcc303d80570be011261a9d4375de36c7dfe3912192486e03b8ad"),
        "legacy": ("04_legacy_sgd_bias_zero_seed0.json", "910ac8941acce850f20328308eac4af9d0a17f41eb6fb06c990044216891cb7f"),
    },
    "conv3": {
        "baseline": ("00_baseline_sgd_bias_zero_seed0.json", "c1ec06e658fafc7bce7653decfc7e37315745bd99fc5abd341782b847c6140cf"),
        "ours": ("02_ours_sgd_bias_zero_seed0.json", "51b6a77a11f2b8ac06551a2f6054449a5ffccb34e8be01a54f7ce61c22bab5a8"),
        "legacy": ("04_legacy_sgd_bias_zero_seed0.json", "361effc3a289f8a5466021f9bef3e3c9fe33eb75808a96328bceecca8696e68d"),
    },
}

# Frozen replay measurements used by the predeclared exact proposal transform.
# Each value is LR * median_batch(RMS(gradient)) / RMS(initial parameter).
FROZEN_PROPOSALS = {
    "conv1": {
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
    "conv2": {
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
    "conv3": {
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
}

BASELINE_RATES = {
    "conv1": {"ConvWeight_0": 0.142696, "DenseWeight_0": 0.0120238},
    "conv2": {
        "ConvWeight_0": 7.90864,
        "ConvWeight_1": 3.81476,
        "DenseWeight_0": 0.82063,
    },
    "conv3": {
        "ConvWeight_0": 2.2203793618679493,
        "ConvWeight_1": 6.284350813613316,
        "ConvWeight_2": 4.669784642297713,
        "DenseWeight_0": 1.1922264543625982,
    },
}

ALLOWED_SOURCE_DIFF_PREFIXES = (
    ("arm_id",),
    ("study_id",),
    ("lab", "epochs"),
    ("lr",),
    ("optimizer", "learning_rate"),
    ("learning_rates_by_parameter",),
    ("reporting",),
    ("evaluation", "inclusion_rule"),
    (
        "handoff",
        "diagnostic_bias_ablation",
        "all_weight_learning_rates_unchanged",
    ),
    ("handoff", "diagnostic_bias_ablation", "scientific_question"),
    ("handoff", "lr_counterfactual"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    return json.loads((CONFIG_ROOT / "study_manifest.json").read_text(encoding="utf-8"))


def _row_config_path(row: dict[str, Any]) -> Path:
    return REPO_ROOT / row["config"]


def _relative_config_path(row: dict[str, Any]) -> str:
    return str(_row_config_path(row).relative_to(CONFIG_ROOT))


def _scheme_for_case(filename: str) -> str:
    if filename.startswith("04_"):
        return "ours"
    if filename.startswith("05_"):
        return "legacy"
    return "baseline"


def _source_scheme_for_case(filename: str) -> str:
    return _scheme_for_case(filename)


def _expected_rates(architecture: str, filename: str) -> list[float]:
    order = ARCHITECTURE_CONTRACT[architecture]["parameter_order"]
    source_scheme = _source_scheme_for_case(filename)
    source_filename = SOURCE_CONFIGS[architecture][source_scheme][0]
    source = load_exact_config(SOURCE_ROOT / architecture / source_filename)
    rates = dict(source["learning_rates_by_parameter"])

    if filename.startswith("01_"):
        for name in order:
            if name.startswith("ConvWeight_"):
                rates[name] = 3.0 * BASELINE_RATES[architecture][name]
    elif filename.startswith(("02_", "03_")):
        target = "ours" if filename.startswith("02_") else "legacy"
        for name in order:
            if name.startswith(("ConvWeight_", "DenseWeight_")):
                rates[name] = (
                    BASELINE_RATES[architecture][name]
                    * FROZEN_PROPOSALS[architecture][target][name]
                    / FROZEN_PROPOSALS[architecture]["baseline"][name]
                )
    return [float(rates[name]) for name in order]


def _diff_paths(left: Any, right: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(_diff_paths(left[key], right[key], path))
        return result
    if left != right:
        return {prefix}
    return set()


def _allowed_diff(path: tuple[str, ...]) -> bool:
    return any(path[: len(prefix)] == prefix for prefix in ALLOWED_SOURCE_DIFF_PREFIXES)


def test_materialized_configs_match_builder_and_manifest_hashes() -> None:
    prepare(check=True)
    manifest = _manifest()
    rows = manifest["configs"]

    assert len(rows) == 18
    assert [row["global_index"] for row in rows] == list(range(18))
    assert [row["index_within_architecture"] for row in rows] == [
        index for _architecture in ARCHITECTURES for index in range(6)
    ]
    assert [_relative_config_path(row) for row in rows] == list(
        EXPECTED_CONFIG_ORDER
    )

    digests = []
    for row in rows:
        path = _row_config_path(row)
        digest = _sha256(path)
        assert digest == row["config_sha256"]
        digests.append(digest)
    ordered_payload = "".join(f"{digest}\n" for digest in digests).encode()
    assert hashlib.sha256(ordered_payload).hexdigest() == (
        manifest["ordered_config_set_sha256"]
    )
    assert manifest["ordered_config_set_sha256"] == (
        EXPECTED_ORDERED_CONFIG_SET_SHA256
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize(
    ("filename", "target"),
    (
        ("02_baseline_sgd_ours_proposal_exact.json", "ours"),
        ("03_baseline_sgd_legacy_proposal_exact.json", "legacy"),
    ),
)
def test_exact_proposal_matched_rates_follow_the_frozen_formula(
    architecture: str,
    filename: str,
    target: str,
) -> None:
    config = load_exact_config(CONFIG_ROOT / architecture / filename)
    expected = _expected_rates(architecture, filename)

    assert config["lr"] == pytest.approx(expected, rel=1.0e-15, abs=0.0)
    for name, value in zip(config["parameter_order"], expected, strict=True):
        assert config["learning_rates_by_parameter"][name] == pytest.approx(
            value, rel=1.0e-15, abs=0.0
        )
        if name.startswith("Bias_"):
            assert value == 0.0
        else:
            baseline_rate = BASELINE_RATES[architecture][name]
            expected_multiplier = (
                FROZEN_PROPOSALS[architecture][target][name]
                / FROZEN_PROPOSALS[architecture]["baseline"][name]
            )
            assert value == pytest.approx(
                baseline_rate * expected_multiplier,
                rel=1.0e-15,
                abs=0.0,
            )


def test_optional_local_proposal_csv_matches_frozen_hash_and_rows() -> None:
    if not PROPOSAL_SOURCE.is_file():
        pytest.skip(
            "ignored results/ proposal CSV is unavailable; self-contained proposal "
            "algebra remains covered above"
        )
    assert _sha256(PROPOSAL_SOURCE) == PROPOSAL_SOURCE_SHA256

    with PROPOSAL_SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {
        (row["architecture"], row["scheme"], row["parameter_name"]): float(
            row["lr_times_gradient_over_parameter_rms_median"]
        )
        for row in rows
        if row["architecture"] in ARCHITECTURES
        and row["optimizer"] == "SGD"
        and row["checkpoint_role"] == "initialization"
        and row["parameter_type"] in {"ConvWeight", "DenseWeight"}
    }
    expected = {
        (architecture, scheme, parameter): value
        for architecture, schemes in FROZEN_PROPOSALS.items()
        for scheme, parameters in schemes.items()
        for parameter, value in parameters.items()
    }
    assert observed == expected


@pytest.mark.parametrize("relative_path", EXPECTED_CONFIG_ORDER)
def test_every_arm_freezes_the_30_epoch_zero_bias_contract(
    relative_path: str,
) -> None:
    architecture, filename = relative_path.split("/", maxsplit=1)
    config = load_exact_config(CONFIG_ROOT / relative_path)
    contract = ARCHITECTURE_CONTRACT[architecture]
    scheme = _scheme_for_case(filename)
    params = config["datasets"]["mnist"]["params"]
    model = config["model_base"]
    override = config["model_overrides"]["mnist_bp_conv_amp"]
    expected_rates = _expected_rates(architecture, filename)

    assert config["study_id"] == STUDY_ID
    assert config["reporting"] == {
        "arm_id": config["arm_id"],
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "study_id": STUDY_ID,
    }
    assert config["lab"]["epochs"] == 30
    assert config["seed"] == 0
    assert config["training_algorithm"] == "BP"
    assert config["batch_state_policy"] == "reset_each_batch"
    assert config["parameter_order"] == contract["parameter_order"]
    assert config["lr"] == pytest.approx(expected_rates, rel=1.0e-15, abs=0.0)
    assert config["optimizer"] == {
        "learning_rate": config["lr"],
        "lr_decay": 1.0,
        "momentum": 0.0,
        "name": "SGD",
        "weight_decay": 0.0,
    }

    bias_names = [name for name in config["parameter_order"] if name.startswith("Bias_")]
    assert bias_names == [f"Bias_{index}" for index in range(contract["bias_count"])]
    assert all(config["learning_rates_by_parameter"][name] == 0.0 for name in bias_names)

    assert config["datasets"]["mnist"]["factory"] == (
        "labs.datasets.MnistTrainValidationDataset"
    )
    assert params["affine_config"] is None
    assert params["batch_size"] == 16
    assert params["validation_batch_size"] == 64
    assert params["split_seed"] == 0
    assert params["shuffle_seed"] == 0
    assert params["normalize_mean"] == 0.1307
    assert params["normalize_std"] == 0.3081
    assert params["normalize_scale"] == 0.3
    assert config["evaluation"]["official_test"]["policy"] == "disabled"

    assert model["num_iterations_training"] == contract["iterations"]
    assert model["num_iterations_inference"] == contract["iterations"]
    assert model["input_gain"] == contract["input_gain"]
    assert model["weight_init_mode"] == "kaiming_uniform"
    assert (model["weight_min"], model["weight_max"]) == (0.0, 100.0)
    assert model["non_linearity"] == "perfect_diode"
    assert (model["voltage_amp"], model["current_amp"]) == SCHEME_AMPLIFICATION[
        scheme
    ]
    for diode_key in (
        "quadratic_diode_param",
        "exponential_diode_param",
        "hard_sigmoid_param",
    ):
        assert diode_key in model
        assert isinstance(model[diode_key], dict)
    assert override["layer_shapes"] == contract["layer_shapes"]
    assert [stage["stride"] for stage in override["conv_pipeline"]] == contract[
        "strides"
    ]


@pytest.mark.parametrize("relative_path", EXPECTED_CONFIG_ORDER)
def test_parent_hashes_and_structural_diff_whitelist(relative_path: str) -> None:
    architecture, filename = relative_path.split("/", maxsplit=1)
    config = load_exact_config(CONFIG_ROOT / relative_path)
    source_scheme = _source_scheme_for_case(filename)
    source_filename, source_digest = SOURCE_CONFIGS[architecture][source_scheme]
    source_path = SOURCE_ROOT / architecture / source_filename
    source = load_exact_config(source_path)

    assert _sha256(source_path) == source_digest
    counterfactual = config["handoff"]["lr_counterfactual"]
    assert REPO_ROOT / counterfactual["parent_config"] == source_path
    assert counterfactual["parent_config_sha256"] == source_digest
    assert counterfactual["rates_selected_post_hoc_from_this_study"] is False

    differences = _diff_paths(config, source)
    unexpected = sorted(path for path in differences if not _allowed_diff(path))
    assert unexpected == []


def test_manifest_diagnostics_and_promotion_contract() -> None:
    manifest = _manifest()

    assert manifest["study_id"] == STUDY_ID
    assert manifest["evidence_class"] == EVIDENCE_CLASS
    assert manifest["paper_facing"] is False
    assert manifest["epoch_budget"] == {
        architecture: 30 for architecture in ARCHITECTURES
    }
    assert manifest["diagnostics"] == {
        "checkpoint_every_epoch": True,
        "expected_checkpoint_epochs": list(range(31)),
        "gradient_trace_samples_per_epoch": 5,
        "expected_gradient_trace_transitions": 150,
    }
    assert manifest["bias_contract"] == {
        "initialization": "zero",
        "learning_rate": 0.0,
        "required_checkpoint_value": 0.0,
    }

    assert "medium-affine" in manifest["promotion_use"]
    for architecture in ARCHITECTURES:
        promotion = manifest["promotion_rules_by_architecture"][architecture]
        assert promotion["primary_metric"] == (
            "mean_validation_loss_epochs_26_through_30"
        )
        assert promotion["relative_loss_plateau"] == 0.02
        assert len(promotion["tie_breaks"]) == 5
        assert "official_test_read=false" in promotion["eligibility"]
        candidates = []
        for filename in CASE_FILENAMES[:4]:
            arm_id = load_exact_config(CONFIG_ROOT / architecture / filename)["arm_id"]
            candidates.append(arm_id)
        assert promotion["baseline_candidates"] == candidates


def test_manifest_and_configs_freeze_proposal_measurement_provenance() -> None:
    manifest = _manifest()
    measurement = manifest["proposal_measurement"]
    assert measurement == {
        "bptt_gradient_batches": 2,
        "checkpoint_role": "initialization",
        "cohort_examples": 128,
        "cohort_sha256": COHORT_SHA256,
        "formula": PROPOSAL_FORMULA,
        "gradient_summary": {
            "path": str(PROPOSAL_SOURCE.relative_to(REPO_ROOT)),
            "sha256": PROPOSAL_SOURCE_SHA256,
        },
        "metric": PROPOSAL_METRIC,
        "performance_metrics_used_for_rate_derivation": False,
        "replay_receipt": {
            "path": REPLAY_RECEIPT_RELATIVE_PATH,
            "sha256": REPLAY_RECEIPT_SHA256,
        },
        "values_frozen_in_generator": True,
    }
    limitations = " ".join(manifest["proposal_match_limitations"]).lower()
    for limitation in (
        "initialization",
        "gradient direction",
        "output scaling",
        "later optimizer trajectory",
        "global learning-rate optimality",
    ):
        assert limitation in limitations

    expected_embedded = {
        key: value
        for key, value in measurement.items()
        if key not in {"formula", "values_frozen_in_generator"}
    }
    expected_embedded.update(
        {
            "optimizer_label_in_replay": "SGD",
            "values_frozen_in_config": True,
        }
    )
    for relative_path in EXPECTED_CONFIG_ORDER:
        config = load_exact_config(CONFIG_ROOT / relative_path)
        counterfactual = config["handoff"]["lr_counterfactual"]
        assert counterfactual["proposal_measurement"] == expected_embedded
        assert (
            counterfactual["proposal_derivation"]
            == next(
                row["proposal_derivation"]
                for row in manifest["configs"]
                if _relative_config_path(row) == relative_path
            )
        )


def test_wrapper_freezes_resources_order_and_study_contract() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    manifest = _manifest()

    assert manifest["launch_plan"] == {
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
            "/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/" + STUDY_ID
        ),
        "status": "prepared_not_submitted",
        "target": "jean-zay",
        "time_limit": "06:00:00",
        "wrapper": str(WRAPPER.relative_to(REPO_ROOT)),
    }

    for directive in (
        "#SBATCH --account=fmu@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t3",
        "#SBATCH --constraint=v100-16g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --cpus-per-task=10",
        "#SBATCH --time=06:00:00",
        "#SBATCH --array=0-17%6",
    ):
        assert directive in text
    assert "#SBATCH --mem" not in text
    assert "pytorch-gpu/py3/2.5.0" in text
    assert f'STUDY_ID="{STUDY_ID}"' in text
    assert f"--evidence-class {EVIDENCE_CLASS}" in text
    assert "--gradient-trace-samples-per-epoch 5" in text
    assert "--checkpoint-every-epoch" in text
    assert "--skip-terminal-official-test" in text
    assert "experiments.reporting validate-run" in text
    assert "LR_COUNTERFACTUAL_CONFIG_SET_SHA256" in text
    assert "SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]}" in text

    entries = re.findall(
        r'"\$\{CONFIG_ROOT\}/(conv[123]/\d\d_[^"]+\.json)"', text
    )
    assert entries == list(EXPECTED_CONFIG_ORDER)

    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
