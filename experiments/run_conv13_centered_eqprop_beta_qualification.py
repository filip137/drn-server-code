#!/usr/bin/env python3
"""Materialize and run the Conv1/Conv3 clean EqProp beta qualification."""

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
    / "configs/conv/perfectdiode_conv13_centered_float64_eqprop_"
    "beta_qualification_1to2decades_10ep_seed0_20260814_v1.json"
)
SCHEMA = "perfectdiode-conv13-centered-float64-eqprop-beta-qualification/v1"
STUDY_ID = (
    "perfectdiode-conv13-centered-float64-eqprop-beta-qualification-"
    "1to2decades-10ep-seed0-20260814-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_eqprop_clean_beta_qualification_diagnostic"
ARCHITECTURES = ("conv1", "conv3")
SCHEMES = ("baseline", "ours", "legacy")
TIERS = ("one_decade_lower", "two_decades_lower")


@dataclass(frozen=True)
class CaseSpec:
    index: int
    architecture: str
    scheme: str
    beta_tier: str

    @property
    def arm_id(self) -> str:
        return f"{self.architecture}_{self.scheme}_adam_beta_{self.beta_tier}_sigma_0"


CASE_SPECS = tuple(
    CaseSpec(index, architecture, scheme, tier)
    for index, (tier, scheme, architecture) in enumerate(
        (tier, scheme, architecture)
        for tier in TIERS
        for scheme in SCHEMES
        for architecture in ARCHITECTURES
    )
)
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
    for architecture in ARCHITECTURES:
        source = study["source"][architecture]
        path = _root_path(source["learning_rate_handoff"])
        if _sha256(path) != source["learning_rate_handoff_sha256"]:
            raise ValueError(f"Fixed LR handoff bytes changed for {architecture}.")
        handoff = _read_json(path)
        for row in handoff.get("entries", []):
            entry_id = str(row.get("entry_id", ""))
            if entry_id:
                entries[entry_id] = row
    return entries


def _amplification_factor(config: Mapping[str, Any], depth: int) -> float:
    model = config["model_base"]
    return (float(model["voltage_amp"]) / float(model["current_amp"])) ** depth


def _beta_values(
    study: Mapping[str, Any], spec: CaseSpec, source_config: Mapping[str, Any]
) -> tuple[float, float, float]:
    architecture = study["architectures"][spec.architecture]
    boundary = float(
        architecture["cosine_0p99_injected_beta_boundaries"][spec.scheme]
    )
    divisor = float(study["beta_tiers"][spec.beta_tier]["boundary_divisor"])
    injected = boundary / divisor
    factor = _amplification_factor(source_config, int(architecture["depth"]))
    return injected, injected / factor, factor


def load_and_validate_study(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    study = _read_json(path)
    if study.get("schema_version") != SCHEMA or study.get("study_id") != STUDY_ID:
        raise ValueError(f"Unexpected study identity in {path}.")
    if study.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("Unexpected evidence class.")

    contract = study.get("scientific_contract")
    expected = {
        "architectures": list(ARCHITECTURES),
        "schemes": list(SCHEMES),
        "beta_tiers": list(TIERS),
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
        "optimizer": "Adam",
        "endpoint_read_noise_std": 0.0,
        "input_read_noise": False,
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
    environment_config = _read_json(environment_path)
    allocation = environment_config["allocation"]
    execution = study["execution"]
    resource_checks = {
        "account": allocation["account"],
        "partition": allocation["partition"],
        "qos": allocation["qos"],
        "constraint": allocation["constraint"],
    }
    if any(execution.get(key) != value for key, value in resource_checks.items()):
        raise ValueError("Study and UMG allocation contracts disagree.")
    if execution.get("array") != "0-5%6" or execution.get("packs") != [
        list(pack) for pack in PACKS
    ]:
        raise ValueError("The two-runs-per-GPU pack contract changed.")
    if int(execution.get("logical_case_count", -1)) != len(CASE_SPECS):
        raise ValueError("Expected exactly twelve logical cases.")

    handoff_entries = _handoff_entries(study)
    for spec in CASE_SPECS:
        architecture = study["architectures"][spec.architecture]
        source_row = study["source"][spec.architecture]["scheme_configs"][spec.scheme]
        source_path = _root_path(source_row["path"])
        if _sha256(source_path) != source_row["sha256"]:
            raise ValueError(f"Source config bytes changed for {spec.architecture}/{spec.scheme}.")
        source_config = exact_run.load_exact_config(source_path)
        if source_config["optimizer"]["name"] != "Adam":
            raise ValueError(f"Expected Adam for {spec.architecture}/{spec.scheme}.")
        if list(source_config["parameter_order"]) != list(
            architecture["parameter_order"]
        ):
            raise ValueError(f"Parameter order changed for {spec.architecture}.")
        entry_id = f"{spec.architecture}_{spec.scheme}_adam"
        handoff = handoff_entries.get(entry_id)
        if handoff is None or handoff.get("learning_rates_by_parameter") != source_config.get(
            "learning_rates_by_parameter"
        ):
            raise ValueError(f"Source/handoff LR mismatch for {entry_id}.")
        injected, base, factor = _beta_values(study, spec, source_config)
        if not math.isclose(base * factor, injected, rel_tol=1e-15):
            raise ValueError(f"Beta scaling mismatch for {spec.arm_id}.")
        model = source_config["model_base"]
        explicit = contract["exponential_diode_param"], contract["quadratic_diode_param"]
        if model.get("exponential_diode_param") != explicit[0] or model.get(
            "quadratic_diode_param"
        ) != explicit[1]:
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
    source_row = study["source"][spec.architecture]["scheme_configs"][spec.scheme]
    source_path = _root_path(source_row["path"])
    config = copy.deepcopy(_read_json(source_path))
    contract = study["scientific_contract"]
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

    config["schema_version"] = "mnist-conv-eqprop-clean-beta-exact-run/v1"
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
            "Exploratory matched clean-beta qualification; retain every finite run "
            "completing ten epochs and report scientific failures without outcome-based exclusion."
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
        "beta_boundary_cosine_threshold": 0.99,
        "beta_boundary_injected_B": float(
            study["architectures"][spec.architecture][
                "cosine_0p99_injected_beta_boundaries"
            ][spec.scheme]
        ),
        "beta_boundary_status": study["architectures"][spec.architecture][
            "boundary_status"
        ][spec.scheme],
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
    config["exploratory_source"] = {
        "study_config": str(Path(study["_path"]).relative_to(ROOT)),
        "source_config": source_row["path"],
        "source_config_sha256": source_row["sha256"],
        "learning_rate_handoff": study["source"][spec.architecture][
            "learning_rate_handoff"
        ],
        "learning_rate_handoff_sha256": study["source"][spec.architecture][
            "learning_rate_handoff_sha256"
        ],
        "learning_rate_handoff_entry": f"{spec.architecture}_{spec.scheme}_adam",
        "dataset_change": (
            "deterministic_medium_affine_source_template_to_ordinary_mnist_selection_split"
        ),
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
        source_row = study["source"][spec.architecture]["scheme_configs"][spec.scheme]
        source_config = _read_json(_root_path(source_row["path"]))
        injected, base, factor = _beta_values(study, spec, source_config)
        rows.append(
            {
                "index": spec.index,
                "pack_index": spec.index // 2,
                "architecture": spec.architecture,
                "scheme": spec.scheme,
                "beta_tier": spec.beta_tier,
                "injected_beta_B": injected,
                "base_beta": base,
                "amplification_factor": factor,
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
