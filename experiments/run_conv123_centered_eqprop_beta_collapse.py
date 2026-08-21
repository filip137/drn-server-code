#!/usr/bin/env python3
"""Materialize and run the Conv1/2/3 EqProp beta-collapse study."""

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
    / "configs/conv/perfectdiode_conv123_centered_float64_eqprop_"
    "beta_collapse_boundary_vs_1decade_sgd_adam_10ep_seed0_20260815_v1.json"
)
SCHEMA = "perfectdiode-conv123-centered-float64-eqprop-beta-collapse-study/v1"
STUDY_ID = (
    "perfectdiode-conv123-centered-float64-eqprop-beta-collapse-"
    "boundary-vs-1decade-sgd-adam-10ep-seed0-20260815-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_beta_collapse_diagnostic"
ARCHITECTURES = ("conv1", "conv2", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
OPTIMIZERS = ("SGD", "Adam")
BETA_TIERS = ("boundary", "one_decade_lower")


@dataclass(frozen=True)
class CaseSpec:
    index: int
    architecture: str
    scheme: str
    optimizer: str
    beta_tier: str

    @property
    def arm_id(self) -> str:
        optimizer = self.optimizer.lower()
        return (
            f"{self.architecture}_{self.scheme}_{optimizer}_"
            f"beta_{self.beta_tier}_sigma_0"
        )


def _case_specs() -> tuple[CaseSpec, ...]:
    """Order cases so each adjacent pair is one two-run GPU pack."""
    rows: list[CaseSpec] = []
    for beta_tier in BETA_TIERS:
        for scheme in SCHEMES:
            for optimizer in OPTIMIZERS:
                for architecture in ("conv1", "conv3"):
                    rows.append(
                        CaseSpec(
                            len(rows), architecture, scheme, optimizer, beta_tier
                        )
                    )
    for scheme in SCHEMES:
        for optimizer in OPTIMIZERS:
            for beta_tier in BETA_TIERS:
                rows.append(
                    CaseSpec(len(rows), "conv2", scheme, optimizer, beta_tier)
                )
    return tuple(rows)


CASE_SPECS = _case_specs()
PACKS = tuple((index, index + 1) for index in range(0, len(CASE_SPECS), 2))


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


def _handoff_entries(study: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for source in study["source"]["learning_rate_handoffs"].values():
        path = _root_path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"Fixed LR handoff bytes changed: {path}.")
        for row in _read_json(path).get("entries", []):
            entry_id = str(row.get("entry_id", ""))
            if entry_id:
                entries[entry_id] = row
    return entries


def _source_row(study: Mapping[str, Any], spec: CaseSpec) -> Mapping[str, Any]:
    return study["source"]["scheme_configs"][spec.architecture][spec.scheme][
        spec.optimizer
    ]


def _amplification_factor(config: Mapping[str, Any], depth: int) -> float:
    model = config["model_base"]
    return (float(model["voltage_amp"]) / float(model["current_amp"])) ** depth


def _beta_values(
    study: Mapping[str, Any], spec: CaseSpec, source_config: Mapping[str, Any]
) -> tuple[float, float, float]:
    architecture = study["architectures"][spec.architecture]
    injected = float(architecture["injected_beta"][spec.beta_tier][spec.scheme])
    factor = _amplification_factor(source_config, int(architecture["depth"]))
    return injected, injected / factor, factor


def _validate_pack_contract() -> None:
    if len(CASE_SPECS) != 36 or len(PACKS) != 18:
        raise ValueError("Expected exactly 36 logical cases in 18 packs.")
    if tuple(spec.index for spec in CASE_SPECS) != tuple(range(36)):
        raise ValueError("Case indices must be contiguous.")
    for pack_index, (left_index, right_index) in enumerate(PACKS):
        left = CASE_SPECS[left_index]
        right = CASE_SPECS[right_index]
        if pack_index < 12:
            checks = (
                left.architecture == "conv1",
                right.architecture == "conv3",
                left.scheme == right.scheme,
                left.optimizer == right.optimizer,
                left.beta_tier == right.beta_tier,
            )
        else:
            checks = (
                left.architecture == right.architecture == "conv2",
                left.scheme == right.scheme,
                left.optimizer == right.optimizer,
                (left.beta_tier, right.beta_tier) == BETA_TIERS,
            )
        if not all(checks):
            raise ValueError(f"Pack {pack_index} violates the frozen pairing contract.")


def load_and_validate_study(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    study = _read_json(path)
    if study.get("schema_version") != SCHEMA or study.get("study_id") != STUDY_ID:
        raise ValueError(f"Unexpected study identity in {path}.")
    if study.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("Unexpected evidence class.")
    _validate_pack_contract()

    contract = study.get("scientific_contract")
    expected = {
        "architectures": list(ARCHITECTURES),
        "schemes": list(SCHEMES),
        "optimizers": list(OPTIMIZERS),
        "beta_tiers": list(BETA_TIERS),
        "dataset": "ordinary_mnist_deterministic_55000_train_5000_validation",
        "official_test_read": False,
        "model_seed": 0,
        "shuffle_seed": 0,
        "batch_size": 16,
        "validation_batch_size": 64,
        "epochs": 10,
        "T": 8,
        "K": 8,
        "runtime_dtype": "float64",
        "training_algorithm": "centered_frozen_current_eqprop",
        "endpoint_read_noise_std": 0.0,
        "input_read_noise": False,
        "checkpoint_selection": "maximum_validation_accuracy",
        "collapse_definition": {
            "nonfinite_training": True,
            "final_validation_drop_from_best_pp_at_least": 5.0,
        },
        "weight_min": 0.0,
        "weight_max": 100.0,
        "weight_initialization": "kaiming_uniform",
        "non_linearity": "perfect_diode",
    }
    if not isinstance(contract, Mapping) or any(
        contract.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("The frozen scientific contract changed.")

    environment = study["source"]["jean_zay_environment"]
    environment_path = _root_path(environment["path"])
    if _sha256(environment_path) != environment["sha256"]:
        raise ValueError("The frozen Jean Zay environment contract changed.")
    allocation = _read_json(environment_path)["allocation"]
    execution = study["execution"]
    for key in ("account", "partition", "qos", "constraint"):
        if execution.get(key) != allocation.get(key):
            raise ValueError(f"Study and UMG allocation disagree for {key}.")
    execution_expected = {
        "array": "0-17%18",
        "logical_case_count": 36,
        "pack_count": 18,
        "runs_per_gpu": 2,
        "cpus_per_task": 16,
        "walltime": "04:00:00",
    }
    if any(execution.get(key) != value for key, value in execution_expected.items()):
        raise ValueError("The frozen packed execution contract changed.")

    handoff_entries = _handoff_entries(study)
    explicit_exponential = contract["exponential_diode_param"]
    explicit_quadratic = contract["quadratic_diode_param"]
    for spec in CASE_SPECS:
        architecture = study["architectures"][spec.architecture]
        source_row = _source_row(study, spec)
        source_path = _root_path(source_row["path"])
        if _sha256(source_path) != source_row["sha256"]:
            raise ValueError(f"Source config bytes changed for {spec.arm_id}.")
        source_config = exact_run.load_exact_config(source_path)
        if source_config["optimizer"]["name"] != spec.optimizer:
            raise ValueError(f"Optimizer mismatch for {spec.arm_id}.")
        if list(source_config["parameter_order"]) != list(
            architecture["parameter_order"]
        ):
            raise ValueError(f"Parameter order changed for {spec.arm_id}.")
        entry_id = f"{spec.architecture}_{spec.scheme}_{spec.optimizer.lower()}"
        handoff = handoff_entries.get(entry_id)
        if handoff is None or handoff.get(
            "learning_rates_by_parameter"
        ) != source_config.get("learning_rates_by_parameter"):
            raise ValueError(f"Source/handoff LR mismatch for {entry_id}.")
        if handoff.get("ordered_learning_rate_vector") != source_config.get("lr"):
            raise ValueError(f"Source/handoff ordered LR mismatch for {entry_id}.")
        injected, base, factor = _beta_values(study, spec, source_config)
        if not math.isclose(base * factor, injected, rel_tol=1e-15):
            raise ValueError(f"Beta scaling mismatch for {spec.arm_id}.")
        boundary = float(architecture["injected_beta"]["boundary"][spec.scheme])
        control = float(
            architecture["injected_beta"]["one_decade_lower"][spec.scheme]
        )
        if not math.isclose(control * 10.0, boundary, rel_tol=1e-15):
            raise ValueError(f"One-decade beta relation changed for {spec.arm_id}.")
        model = source_config["model_base"]
        if model.get("exponential_diode_param") != explicit_exponential or model.get(
            "quadratic_diode_param"
        ) != explicit_quadratic:
            raise ValueError(f"Explicit diode contract changed for {spec.arm_id}.")
        if (
            float(model["weight_min"]) != 0.0
            or float(model["weight_max"]) != 100.0
            or model["weight_init_mode"] != "kaiming_uniform"
        ):
            raise ValueError(f"Weight contract changed for {spec.arm_id}.")

    study["_path"] = str(path)
    return study


def _resolved_case_config(
    study: Mapping[str, Any], spec: CaseSpec
) -> dict[str, Any]:
    source_row = _source_row(study, spec)
    source_path = _root_path(source_row["path"])
    config = copy.deepcopy(_read_json(source_path))
    contract = study["scientific_contract"]
    architecture = study["architectures"][spec.architecture]
    injected_beta, base_beta, factor = _beta_values(study, spec, config)

    dataset = config["datasets"]["mnist"]
    dataset["factory"] = "labs.datasets.MnistTrainValidationDataset"
    params = dataset["params"]
    params["affine_config"] = None
    params.pop("official_test_batch_size", None)
    params["batch_size"] = int(contract["batch_size"])
    params["validation_batch_size"] = int(contract["validation_batch_size"])
    params["split_seed"] = int(contract["model_seed"])
    params["shuffle_seed"] = int(contract["shuffle_seed"])

    config["schema_version"] = "mnist-conv-eqprop-beta-collapse-exact-run/v1"
    config["study_id"] = study["study_id"]
    config["arm_id"] = spec.arm_id
    config["training_algorithm"] = "EP"
    config["runtime_dtype"] = "float64"
    config["beta"] = base_beta
    config["lab"]["epochs"] = int(contract["epochs"])
    config["model_base"]["num_iterations_inference"] = int(contract["T"])
    config["model_base"]["num_iterations_training"] = int(contract["K"])
    config["evaluation"] = {
        "checkpoint_selection": "maximum_validation_accuracy",
        "epoch_split": "validation",
        "inclusion_rule": (
            "Exploratory matched beta-collapse study. Retain every terminal run; "
            "an explicit NonFiniteTrainingError is a scientific endpoint, and a "
            "finite run collapses when final validation is at least 5.00 pp below "
            "its best epoch."
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
        "beta_tier": spec.beta_tier,
        "beta_boundary_injected_B": float(
            architecture["injected_beta"]["boundary"][spec.scheme]
        ),
        "beta_boundary_status": architecture["boundary_status"][spec.scheme],
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
    handoff_key = "conv3" if spec.architecture == "conv3" else "conv12"
    handoff = study["source"]["learning_rate_handoffs"][handoff_key]
    config["exploratory_source"] = {
        "study_config": str(Path(study["_path"]).relative_to(ROOT)),
        "source_config": source_row["path"],
        "source_config_sha256": source_row["sha256"],
        "learning_rate_handoff": handoff["path"],
        "learning_rate_handoff_sha256": handoff["sha256"],
        "learning_rate_handoff_entry": (
            f"{spec.architecture}_{spec.scheme}_{spec.optimizer.lower()}"
        ),
        "dataset_change": (
            "deterministic_medium_affine_source_template_to_ordinary_mnist_"
            "selection_split"
        ),
        "collapse_threshold_pp": 5.0,
        "conv3_tk8_residual_deviation": (
            contract["conv3_tk8_residual_deviation"]
            if spec.architecture == "conv3"
            else None
        ),
    }
    return config


def materialize_configs(study: Mapping[str, Any], directory: Path) -> list[Path]:
    directory = directory.expanduser().resolve()
    paths: list[Path] = []
    for spec in CASE_SPECS:
        config = _resolved_case_config(study, spec)
        path = directory / f"{spec.index:02d}_{spec.arm_id}.json"
        _write_json(path, config)
        exact_run.load_exact_config(path)
        paths.append(path)
    return paths


def ordered_config_set_sha256(paths: Sequence[Path]) -> str:
    payload = "".join(f"{_sha256(path)}\n" for path in paths).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def plan(study: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        source_config = _read_json(_root_path(_source_row(study, spec)["path"]))
        injected, base, factor = _beta_values(study, spec, source_config)
        rows.append(
            {
                "index": spec.index,
                "pack_index": spec.index // 2,
                "architecture": spec.architecture,
                "scheme": spec.scheme,
                "optimizer": spec.optimizer,
                "beta_tier": spec.beta_tier,
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
        "epochs_per_case": study["scientific_contract"]["epochs"],
        "T": study["scientific_contract"]["T"],
        "K": study["scientific_contract"]["K"],
        "runtime_dtype": study["scientific_contract"]["runtime_dtype"],
        "collapse_definition": study["scientific_contract"][
            "collapse_definition"
        ],
        "case_count": len(rows),
        "pack_count": len(PACKS),
        "packs": [list(pack) for pack in PACKS],
        "cases": rows,
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
