#!/usr/bin/env python3
"""Materialize and run the Conv1/Conv3 Adam EqProp sigma-5e-4 study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import exact_run
from experiments import run_conv123_zero_bias_adam_eqprop_one_decade as clean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv13_zero_bias_adam_eqprop_one_decade_"
    "sigma_5em4_ordinary_mnist_10_30ep_seed0_20260817_v1.json"
)
SCHEMA = "perfectdiode-conv13-zero-bias-adam-eqprop-one-decade-read-noise-study/v1"
STUDY_ID = (
    "perfectdiode-conv13-zero-bias-adam-centered-float64-eqprop-one-decade-"
    "sigma-5em4-ordinary-mnist-10-30ep-seed0-20260817-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_zero_bias_read_noise_training_diagnostic"
ARCHITECTURES = ("conv1", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
NOISE_STD = 5e-4


@dataclass(frozen=True)
class CaseSpec:
    index: int
    architecture: str
    scheme: str

    @property
    def arm_id(self) -> str:
        return (
            f"{self.architecture}_{self.scheme}_adam_centered_float64_eqprop_"
            "beta_one_decade_sigma_5em4_seed0"
        )


# Each amplification scheme occupies one V100 and runs its Conv1/Conv3 pair
# concurrently, as requested.
CASE_SPECS = (
    CaseSpec(0, "conv1", "baseline"),
    CaseSpec(1, "conv3", "baseline"),
    CaseSpec(2, "conv1", "ours"),
    CaseSpec(3, "conv3", "ours"),
    CaseSpec(4, "conv1", "legacy"),
    CaseSpec(5, "conv3", "legacy"),
)
PACKS = ((0, 1), (2, 3), (4, 5))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _clean_spec(spec: CaseSpec) -> clean.CaseSpec:
    return next(
        row
        for row in clean.CASE_SPECS
        if row.architecture == spec.architecture and row.scheme == spec.scheme
    )


def _source_row(study: Mapping[str, Any], spec: CaseSpec) -> Mapping[str, Any]:
    return study["source"]["configs"][spec.architecture][spec.scheme]


def load_and_validate_study(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    study = _read_json(path)
    if study.get("schema_version") != SCHEMA or study.get("study_id") != STUDY_ID:
        raise ValueError(f"Unexpected study identity in {path}.")
    if study.get("evidence_class") != EVIDENCE_CLASS or study.get("paper_facing"):
        raise ValueError("Unexpected evidence classification.")

    contract = study.get("scientific_contract")
    expected_contract = {
        "architectures": list(ARCHITECTURES),
        "schemes": list(SCHEMES),
        "optimizer": "Adam",
        "dataset": "ordinary_mnist_deterministic_55000_train_5000_validation",
        "official_test_read": False,
        "model_seed": 0,
        "split_seed": 0,
        "shuffle_seed": 0,
        "batch_size": 16,
        "validation_batch_size": 64,
        "epochs": {"conv1": 10, "conv3": 30},
        "T": {"conv1": 4, "conv3": 8},
        "K": {"conv1": 4, "conv3": 8},
        "output_row_exponent_L": {"conv1": 1, "conv3": 3},
        "injected_beta_B": {
            "conv1": {"baseline": 100.0, "ours": 30.0, "legacy": 3.0},
            "conv3": {"baseline": 100.0, "ours": 3.0, "legacy": 0.001},
        },
        "base_beta": {
            "conv1": {"baseline": 100.0, "ours": 7.5, "legacy": 0.1875},
            "conv3": {
                "baseline": 100.0,
                "ours": 0.046875,
                "legacy": 2.44140625e-7,
            },
        },
        "runtime_dtype": "float64",
        "training_algorithm": "centered_frozen_current_eqprop",
        "current_scale": "auto",
        "normalize_current_scale": True,
        "endpoint_read_noise_std": NOISE_STD,
        "endpoint_read_noise_seed": 2026081601,
        "input_read_noise": False,
        "noise_applied_after_equilibrium_before_local_energy_gradient": True,
        "phase_noise_correlation": "independent",
        "layer_noise_correlation": "independent",
        "matched_standard_normal_draws_across_cases": True,
        "expected_endpoint_read_noise_draws": {
            "smoke": {"conv1": 4, "conv3": 8},
            "production": {"conv1": 137520, "conv3": 825120},
        },
        "checkpoint_selection": "maximum_validation_accuracy",
        "retain_all_scientific_terminal_outcomes": True,
        "clean_control_study_id": clean.STUDY_ID,
        "bias_learning_rate": 0.0,
        "weight_min": 0.0,
        "weight_max": 100.0,
        "weight_initialization": "kaiming_uniform",
        "non_linearity": "perfect_diode",
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in expected_contract.items()
    ):
        raise ValueError("The frozen scientific contract changed.")
    if contract.get("stable_definition") != {
        "finite_full_epoch_completion": True,
        "final_validation_drop_from_best_pp_strictly_less_than": 5.0,
    }:
        raise ValueError("The frozen stability definition changed.")

    deviations = study.get("protocol_deviations", {})
    gradient = deviations.get("one_decade_gradient_gate", {})
    residual = deviations.get("conv3_baseline_t8_free_state_residual", {})
    if (
        gradient.get("passing_layer_batch_rows") != 213
        or gradient.get("total_layer_batch_rows") != 216
        or gradient.get("user_directed_override") is not True
        or gradient.get("reclassified_as_gradient_passing") is not False
        or residual.get("present") is not True
        or residual.get("reclassified_as_residual_passing") is not False
        or deviations.get("paper_ready") is not False
    ):
        raise ValueError("The explicit qualification deviations changed.")

    parent = study["source"]["parent_clean_study_config"]
    parent_path = _root_path(parent["path"])
    if _sha256(parent_path) != parent["sha256"]:
        raise ValueError("The parent clean study bytes changed.")
    parent_study = clean.load_and_validate_study(parent_path)

    environment = study["source"]["jean_zay_environment"]
    environment_path = _root_path(environment["path"])
    if _sha256(environment_path) != environment["sha256"]:
        raise ValueError("The frozen Jean Zay environment bytes changed.")
    environment_config = _read_json(environment_path)
    allocation = environment_config["allocation"]
    resources = environment_config["resources"]
    execution = study["execution"]
    for key in ("account", "partition", "qos", "constraint"):
        if execution.get(key) != allocation.get(key):
            raise ValueError(f"Study and UMG allocation disagree for {key}.")
    expected_execution = {
        "target": "jean-zay",
        "canary_array": "0-0",
        "production_array": "0-2%3",
        "logical_case_count": 6,
        "pack_count": 3,
        "runs_per_gpu": 2,
        "cpus_per_task": 16,
        "canary_walltime": "02:00:00",
        "production_walltime": "08:00:00",
        "packs": [list(pack) for pack in PACKS],
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("The packed execution contract changed.")
    if int(resources["cpus_per_task"]) != 16:
        raise ValueError("The UMG environment no longer supplies 16 CPUs per task.")

    if tuple(spec.index for spec in CASE_SPECS) != tuple(range(6)):
        raise ValueError("Logical case indices are not contiguous.")
    for spec in CASE_SPECS:
        source = _source_row(study, spec)
        source_path = _root_path(source["path"])
        if _sha256(source_path) != source["sha256"]:
            raise ValueError(f"Source config bytes changed for {spec.arm_id}.")
        parent_source = parent_study["source"]["configs"][spec.architecture][
            spec.scheme
        ]
        if source != parent_source:
            raise ValueError(f"Source row differs from the clean parent for {spec.arm_id}.")

    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            digest = study["source"]["clean_controls"]["result_sha256"][
                architecture
            ][scheme]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("A clean-control result digest is malformed.")

    study["_path"] = str(path)
    study["_parent_study"] = parent_study
    return study


def _resolved_case_config(
    study: Mapping[str, Any], spec: CaseSpec
) -> dict[str, Any]:
    config = copy.deepcopy(
        clean._resolved_case_config(study["_parent_study"], _clean_spec(spec))
    )
    contract = study["scientific_contract"]
    source = _source_row(study, spec)
    config["study_id"] = study["study_id"]
    config["arm_id"] = spec.arm_id
    config["eqprop"]["endpoint_read_noise_std"] = NOISE_STD
    config["eqprop"]["endpoint_read_noise_seed"] = int(
        contract["endpoint_read_noise_seed"]
    )
    config["reporting"] = {
        "study_id": study["study_id"],
        "arm_id": spec.arm_id,
        "evidence_class": study["evidence_class"],
        "evidence_tier": study["evidence_tier"],
        "paper_facing": False,
    }
    config["qualification_source"].update(
        {
            "study_config": str(Path(study["_path"]).relative_to(ROOT)),
            "source_config": source["path"],
            "source_config_sha256": source["sha256"],
            "clean_control_study_id": contract["clean_control_study_id"],
            "clean_control_run_root": study["source"]["clean_controls"][
                "result_root"
            ],
            "clean_control_result_sha256": study["source"]["clean_controls"][
                "result_sha256"
            ][spec.architecture][spec.scheme],
            "read_noise_policy": (
                "endpoint sigma 5e-4 after each free/nudged equilibrium; "
                "independent phase/layer draws with one shared fixed seed"
            ),
            "paper_ready": False,
        }
    )
    return config


def materialize_configs(study: Mapping[str, Any], directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    paths: list[Path] = []
    for spec in CASE_SPECS:
        path = directory / f"{spec.index:02d}_{spec.arm_id}.json"
        _write_json(path, _resolved_case_config(study, spec))
        exact_run.load_exact_config(path)
        paths.append(path)
    return paths


def ordered_config_set_sha256(paths: Sequence[Path]) -> str:
    payload = "".join(f"{_sha256(path)}\n" for path in paths).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def plan(study: Mapping[str, Any]) -> dict[str, Any]:
    contract = study["scientific_contract"]
    cases: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        source_config = _read_json(_root_path(_source_row(study, spec)["path"]))
        cases.append(
            {
                "index": spec.index,
                "pack_index": next(
                    index for index, pack in enumerate(PACKS) if spec.index in pack
                ),
                "architecture": spec.architecture,
                "scheme": spec.scheme,
                "optimizer": "Adam",
                "epochs": int(contract["epochs"][spec.architecture]),
                "T": int(contract["T"][spec.architecture]),
                "K": int(contract["K"][spec.architecture]),
                "injected_beta_B": float(
                    contract["injected_beta_B"][spec.architecture][spec.scheme]
                ),
                "base_beta": float(
                    contract["base_beta"][spec.architecture][spec.scheme]
                ),
                "endpoint_read_noise_std": NOISE_STD,
                "learning_rates_by_parameter": source_config[
                    "learning_rates_by_parameter"
                ],
            }
        )
    return {
        "study_id": study["study_id"],
        "target": study["execution"]["target"],
        "account": study["execution"]["account"],
        "canary_array": study["execution"]["canary_array"],
        "production_array": study["execution"]["production_array"],
        "runtime_dtype": contract["runtime_dtype"],
        "case_count": len(cases),
        "pack_count": len(PACKS),
        "packs": [list(pack) for pack in PACKS],
        "protocol_deviations": study["protocol_deviations"],
        "cases": cases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--directory", type=Path, required=True)
    for name in ("smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--target", default="local")
        command.add_argument("--indices", type=int, nargs="*")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study = load_and_validate_study(args.study)
    if args.command == "plan":
        print(json.dumps(plan(study), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "materialize":
        paths = materialize_configs(study, args.directory)
        print(
            json.dumps(
                {
                    "paths": [str(path) for path in paths],
                    "ordered_config_set_sha256": ordered_config_set_sha256(paths),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_root = args.output_root.expanduser().resolve()
    configs = materialize_configs(study, output_root / "resolved_configs")
    indices = list(range(len(configs))) if args.indices is None else list(args.indices)
    if len(indices) != len(set(indices)) or any(
        index < 0 or index >= len(configs) for index in indices
    ):
        raise ValueError(f"Invalid run indices: {indices!r}.")
    result = exact_run.run(
        [configs[index] for index in indices],
        output_root=output_root,
        device=args.device,
        smoke=args.command == "smoke",
        study_id=study["study_id"],
        evidence_class=study["evidence_class"],
        target=args.target,
        dataset_root=args.dataset_root,
        skip_terminal_official_test=True,
        environ={},
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
