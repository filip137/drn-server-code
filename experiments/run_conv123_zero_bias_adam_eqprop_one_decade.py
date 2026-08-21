#!/usr/bin/env python3
"""Materialize and run the zero-bias Adam EqProp one-decade study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import exact_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_"
    "one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json"
)
SCHEMA = "perfectdiode-conv123-zero-bias-adam-eqprop-one-decade-training-study/v1"
STUDY_ID = (
    "perfectdiode-conv123-zero-bias-adam-centered-float64-eqprop-"
    "one-decade-ordinary-mnist-10-30-30ep-seed0-20260816-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_zero_bias_training_qualification"
ARCHITECTURES = ("conv1", "conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")


@dataclass(frozen=True)
class CaseSpec:
    index: int
    architecture: str
    scheme: str

    @property
    def arm_id(self) -> str:
        return (
            f"{self.architecture}_{self.scheme}_adam_centered_float64_eqprop_"
            "beta_one_decade_sigma_0_seed0"
        )


# Conv1/Conv3 are paired by amplification scheme; Conv2 baseline/ours share a
# GPU and Conv2 legacy is the requested singleton fifth pack.
CASE_SPECS = (
    CaseSpec(0, "conv1", "baseline"),
    CaseSpec(1, "conv3", "baseline"),
    CaseSpec(2, "conv1", "ours"),
    CaseSpec(3, "conv3", "ours"),
    CaseSpec(4, "conv1", "legacy"),
    CaseSpec(5, "conv3", "legacy"),
    CaseSpec(6, "conv2", "baseline"),
    CaseSpec(7, "conv2", "ours"),
    CaseSpec(8, "conv2", "legacy"),
)
PACKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8,))


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


def _source_row(study: Mapping[str, Any], spec: CaseSpec) -> Mapping[str, Any]:
    return study["source"]["configs"][spec.architecture][spec.scheme]


def _amplification_factor(config: Mapping[str, Any], depth: int) -> float:
    model = config["model_base"]
    return (float(model["voltage_amp"]) / float(model["current_amp"])) ** depth


def _beta_values(
    study: Mapping[str, Any], spec: CaseSpec, config: Mapping[str, Any]
) -> tuple[float, float, float]:
    contract = study["scientific_contract"]
    injected = float(contract["injected_beta_B"][spec.architecture][spec.scheme])
    base = float(contract["base_beta"][spec.architecture][spec.scheme])
    factor = _amplification_factor(
        config, int(contract["output_row_exponent_L"][spec.architecture])
    )
    return injected, base, factor


def _validate_case_contract(
    study: Mapping[str, Any], spec: CaseSpec, source_config: Mapping[str, Any]
) -> None:
    contract = study["scientific_contract"]
    if source_config.get("seed") != 0:
        raise ValueError(f"Seed changed for {spec.arm_id}.")
    if source_config.get("training_algorithm") != "BP":
        raise ValueError(f"Source is not the frozen BPTT LR config for {spec.arm_id}.")
    if source_config.get("optimizer", {}).get("name") != "Adam":
        raise ValueError(f"Optimizer changed for {spec.arm_id}.")
    parameter_order = list(source_config["parameter_order"])
    named_rates = source_config["learning_rates_by_parameter"]
    ordered_rates = [float(named_rates[name]) for name in parameter_order]
    if ordered_rates != [float(value) for value in source_config["lr"]]:
        raise ValueError(f"Named and ordered learning rates disagree for {spec.arm_id}.")
    if ordered_rates != [
        float(value) for value in source_config["optimizer"]["learning_rate"]
    ]:
        raise ValueError(f"Optimizer learning-rate vector changed for {spec.arm_id}.")
    bias_names = [name for name in parameter_order if name.startswith("Bias_")]
    expected_bias_count = int(spec.architecture[-1])
    if len(bias_names) != expected_bias_count or any(
        float(named_rates[name]) != 0.0 for name in bias_names
    ):
        raise ValueError(f"Bias learning rates are not exact zero for {spec.arm_id}.")

    model = source_config["model_base"]
    expected_t = int(contract["T"][spec.architecture])
    expected_k = int(contract["K"][spec.architecture])
    if (
        int(model["num_iterations_inference"]) != expected_t
        or int(model["num_iterations_training"]) != expected_k
    ):
        raise ValueError(f"Source T/K changed for {spec.arm_id}.")
    if int(source_config["lab"]["epochs"]) != int(
        contract["epochs"][spec.architecture]
    ):
        raise ValueError(f"Source epoch budget changed for {spec.arm_id}.")
    if (
        model.get("exponential_diode_param")
        != contract["exponential_diode_param"]
        or model.get("quadratic_diode_param")
        != contract["quadratic_diode_param"]
    ):
        raise ValueError(f"Explicit diode parameters changed for {spec.arm_id}.")
    if (
        float(model["weight_min"]) != float(contract["weight_min"])
        or float(model["weight_max"]) != float(contract["weight_max"])
        or model["weight_init_mode"] != contract["weight_initialization"]
        or model["non_linearity"] != contract["non_linearity"]
    ):
        raise ValueError(f"Weight or nonlinearity contract changed for {spec.arm_id}.")

    dataset = source_config["datasets"]["mnist"]
    params = dataset["params"]
    if (
        dataset.get("factory") != "labs.datasets.MnistTrainValidationDataset"
        or params.get("affine_config", "missing") is not None
        or int(params["batch_size"]) != int(contract["batch_size"])
        or int(params["validation_batch_size"])
        != int(contract["validation_batch_size"])
        or int(params["split_seed"]) != int(contract["split_seed"])
        or int(params["shuffle_seed"]) != int(contract["shuffle_seed"])
    ):
        raise ValueError(f"Ordinary-MNIST cohort contract changed for {spec.arm_id}.")

    injected, base, factor = _beta_values(study, spec, source_config)
    if not math.isclose(base * factor, injected, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError(f"Base/injected beta relation changed for {spec.arm_id}.")


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
        "epochs": {"conv1": 10, "conv2": 30, "conv3": 30},
        "T": {"conv1": 4, "conv2": 6, "conv3": 8},
        "K": {"conv1": 4, "conv2": 6, "conv3": 8},
        "output_row_exponent_L": {"conv1": 1, "conv2": 2, "conv3": 3},
        "runtime_dtype": "float64",
        "training_algorithm": "centered_frozen_current_eqprop",
        "current_scale": "auto",
        "normalize_current_scale": True,
        "endpoint_read_noise_std": 0.0,
        "input_read_noise": False,
        "checkpoint_selection": "maximum_validation_accuracy",
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
    stable = contract.get("stable_definition", {})
    if stable != {
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
        "array": "0-4%5",
        "logical_case_count": 9,
        "pack_count": 5,
        "runs_per_gpu": 2,
        "cpus_per_task": 16,
        "walltime": "08:00:00",
        "packs": [list(pack) for pack in PACKS],
    }
    if any(execution.get(key) != value for key, value in expected_execution.items()):
        raise ValueError("The packed execution contract changed.")
    if int(resources["cpus_per_task"]) != 16:
        raise ValueError("The UMG environment no longer supplies 16 CPUs per task.")

    if tuple(spec.index for spec in CASE_SPECS) != tuple(range(9)):
        raise ValueError("Logical case indices are not contiguous.")
    for spec in CASE_SPECS:
        source = _source_row(study, spec)
        source_path = _root_path(source["path"])
        if _sha256(source_path) != source["sha256"]:
            raise ValueError(f"Source config bytes changed for {spec.arm_id}.")
        source_config = exact_run.load_exact_config(source_path)
        _validate_case_contract(study, spec, source_config)

    study["_path"] = str(path)
    return study


def _resolved_case_config(
    study: Mapping[str, Any], spec: CaseSpec
) -> dict[str, Any]:
    source = _source_row(study, spec)
    source_path = _root_path(source["path"])
    config = copy.deepcopy(_read_json(source_path))
    contract = study["scientific_contract"]
    injected_beta, base_beta, factor = _beta_values(study, spec, config)
    epochs = int(contract["epochs"][spec.architecture])
    t_value = int(contract["T"][spec.architecture])
    k_value = int(contract["K"][spec.architecture])

    dataset = config["datasets"]["mnist"]
    dataset["factory"] = "labs.datasets.MnistTrainValidationDataset"
    params = dataset["params"]
    params["affine_config"] = None
    params.pop("official_test_batch_size", None)
    params["batch_size"] = int(contract["batch_size"])
    params["validation_batch_size"] = int(contract["validation_batch_size"])
    params["split_seed"] = int(contract["split_seed"])
    params["shuffle_seed"] = int(contract["shuffle_seed"])

    config["schema_version"] = "mnist-conv-zero-bias-adam-eqprop-exact-run/v1"
    config["study_id"] = study["study_id"]
    config["arm_id"] = spec.arm_id
    config["training_algorithm"] = "EP"
    config["runtime_dtype"] = "float64"
    config["beta"] = base_beta
    config["lab"]["epochs"] = epochs
    config["model_base"]["num_iterations_inference"] = t_value
    config["model_base"]["num_iterations_training"] = k_value
    config["evaluation"] = {
        "checkpoint_selection": "maximum_validation_accuracy",
        "epoch_split": "validation",
        "inclusion_rule": (
            "Retain every scientific terminal outcome. A stable arm completes its "
            "full predeclared epoch budget with finite metrics and finishes less "
            "than 5.00 percentage points below its own best validation accuracy."
        ),
        "official_test": {"policy": "disabled"},
    }
    config["eqprop"] = {
        "variant": "centered",
        "nudging_mode": "current",
        "current_scale": "auto",
        "normalize_current_scale": True,
        "injected_beta_B": injected_beta,
        "amplification_factor": factor,
        "beta_tier": "one_decade_lower",
        "endpoint_read_noise_std": 0.0,
        "endpoint_read_noise_seed": int(contract["endpoint_read_noise_seed"]),
        "input_read_noise": False,
        "noise_applied_after_equilibrium_before_local_energy_gradient": True,
        "phase_noise_correlation": "independent",
        "layer_noise_correlation": "independent",
        "matched_standard_normal_draws_across_cases": True,
    }
    config["reporting"] = {
        "study_id": study["study_id"],
        "arm_id": spec.arm_id,
        "evidence_class": study["evidence_class"],
        "evidence_tier": study["evidence_tier"],
        "paper_facing": False,
    }
    config["qualification_source"] = {
        "study_config": str(Path(study["_path"]).relative_to(ROOT)),
        "source_config": source["path"],
        "source_config_sha256": source["sha256"],
        "learning_rate_policy": (
            "exact inherited BPTT Adam vector; all Bias_* rates remain zero"
        ),
        "gradient_gate_passed_rows": 213,
        "gradient_gate_total_rows": 216,
        "gradient_gate_override": "user_directed_exploratory_launch",
        "conv3_t8_residual_deviation": (
            study["protocol_deviations"]["conv3_baseline_t8_free_state_residual"]
            if spec.architecture == "conv3" and spec.scheme == "baseline"
            else None
        ),
        "paper_ready": False,
    }
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
        injected, base, factor = _beta_values(study, spec, source_config)
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
                "injected_beta_B": injected,
                "base_beta": base,
                "amplification_factor": factor,
                "learning_rates_by_parameter": source_config[
                    "learning_rates_by_parameter"
                ],
            }
        )
    return {
        "study_id": study["study_id"],
        "target": study["execution"]["target"],
        "account": study["execution"]["account"],
        "array": study["execution"]["array"],
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
    selected = [configs[index] for index in indices]
    result = exact_run.run(
        selected,
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
