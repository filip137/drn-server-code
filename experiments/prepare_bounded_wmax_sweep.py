from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_COMMIT = "08dd52e44818adf43b011542748e140dc0e01aeb"
PARENT_CONFIG_ROOT = (
    "configs/conv/"
    "ordinary_mnist_signed_scaled_bias_current_rho_t12_seed0_20260806_v1"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_bounded_wmax_sweep_signed_scaled_bias_conv123_seed0_20260809_v1"
)
STUDY_ID = (
    "perfectdiode-bounded-wmax-sweep-signed-scaled-bias-conv123-"
    "seed0-20260809-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_bounded_wmax_sensitivity"
WEIGHT_MIN = 1e-5
PARENT_STUDY_ID = (
    "perfectdiode-ordinary-mnist-signed-scaled-bias-current-rho-t12-"
    "conv123-seed0-20260806-v1"
)


@dataclass(frozen=True)
class WmaxCase:
    tag: str
    value: float


WMAX_CASES = (
    WmaxCase("1em4", 1e-4),
    WmaxCase("3em4", 3e-4),
    WmaxCase("1em3", 1e-3),
    WmaxCase("3em3", 3e-3),
    WmaxCase("1em2", 1e-2),
)

ARCHITECTURE_TARGETS = {
    "conv1": "akib",
    "conv2": "trex",
    "conv3": "jean-zay",
}

PARENT_FILENAMES = (
    "00_bounded_baseline_sgd_seed0.json",
    "01_bounded_baseline_adam_seed0.json",
    "02_bounded_ours_sgd_seed0.json",
    "03_bounded_ours_adam_seed0.json",
    "04_bounded_legacy_sgd_seed0.json",
    "05_bounded_legacy_adam_seed0.json",
)

PARENT_NAME_RE = re.compile(
    r"^\d+_bounded_(baseline|ours|legacy)_(sgd|adam)_seed0\.json$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _git_bytes(relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{PARENT_COMMIT}:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Could not read {relative_path} from {PARENT_COMMIT}: {detail}"
        )
    return completed.stdout


def _git_tree() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{PARENT_COMMIT}^{{tree}}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_parent_config(architecture: str, filename: str) -> tuple[dict[str, Any], bytes]:
    relative_path = f"{PARENT_CONFIG_ROOT}/{architecture}/{filename}"
    payload = _git_bytes(relative_path)
    config = json.loads(payload.decode("utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected an object in {relative_path}.")
    return config, payload


def _load_and_validate_parent_manifest() -> tuple[
    dict[str, Any], bytes, dict[str, tuple[dict[str, Any], bytes, dict[str, Any]]]
]:
    manifest_path = f"{PARENT_CONFIG_ROOT}/study_manifest.json"
    manifest_bytes = _git_bytes(manifest_path)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected an object in {manifest_path}.")
    if manifest.get("study_id") != PARENT_STUDY_ID:
        raise ValueError(f"Unexpected parent study_id in {manifest_path}.")

    rows = manifest.get("runs")
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError(f"Expected 36 parent rows in {manifest_path}.")
    if manifest.get("run_count") != len(rows):
        raise ValueError(f"Parent run_count disagrees in {manifest_path}.")
    if manifest.get("ordered_config_set_sha256") != _ordered_set_sha256(rows):
        raise ValueError(
            f"Parent ordered config-set hash disagrees in {manifest_path}."
        )

    configs_by_path: dict[
        str, tuple[dict[str, Any], bytes, dict[str, Any]]
    ] = {}
    architecture_indices = {architecture: 0 for architecture in ARCHITECTURE_TARGETS}
    for global_array_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"Parent row {global_array_index} is not an object.")
        config_path = row.get("config")
        if not isinstance(config_path, str) or config_path in configs_by_path:
            raise ValueError(
                f"Invalid or duplicate parent config path at row {global_array_index}."
            )
        architecture = row.get("architecture")
        if architecture not in ARCHITECTURE_TARGETS:
            raise ValueError(
                f"Unexpected parent architecture at row {global_array_index}."
            )
        if not config_path.startswith(f"{architecture}/"):
            raise ValueError(
                f"Parent config path/architecture disagree for {config_path}."
            )
        if row.get("global_array_index") != global_array_index:
            raise ValueError(f"Parent global index disagrees for {config_path}.")
        if row.get("architecture_array_index") != architecture_indices[architecture]:
            raise ValueError(
                f"Parent architecture index disagrees for {config_path}."
            )
        architecture_indices[architecture] += 1

        payload = _git_bytes(f"{PARENT_CONFIG_ROOT}/{config_path}")
        if row.get("config_sha256") != _sha256(payload):
            raise ValueError(f"Parent config hash disagrees for {config_path}.")
        config = json.loads(payload.decode("utf-8"))
        if not isinstance(config, dict):
            raise TypeError(f"Expected an object in parent config {config_path}.")
        expected_fields = {
            "K": int(config["model_base"]["num_iterations_training"]),
            "T": int(config["model_base"]["num_iterations_inference"]),
            "arm_id": config["arm_id"],
            "epochs": int(config["lab"]["epochs"]),
            "learning_rates_by_parameter": config[
                "learning_rates_by_parameter"
            ],
            "parameter_order": config["parameter_order"],
            "surface_id": config["handoff"]["surface_id"],
        }
        for key, expected in expected_fields.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"Parent manifest field {key!r} disagrees for {config_path}."
                )
        if config["lr"] != [
            row["learning_rates_by_parameter"][name]
            for name in row["parameter_order"]
        ]:
            raise ValueError(
                f"Parent manifest learning-rate order disagrees for {config_path}."
            )
        configs_by_path[config_path] = (config, payload, row)

    if architecture_indices != {"conv1": 12, "conv2": 12, "conv3": 12}:
        raise ValueError(
            f"Unexpected parent architecture coverage: {architecture_indices}."
        )
    return manifest, manifest_bytes, configs_by_path


def _validate_parent(
    config: dict[str, Any],
    *,
    architecture: str,
    filename: str,
    scheme: str,
    optimizer: str,
) -> None:
    model = config.get("model_base", {})
    if float(model.get("weight_min")) != WEIGHT_MIN:
        raise ValueError(f"Unexpected parent weight_min in {architecture}/{filename}.")
    if float(model.get("weight_max")) != 1e-4:
        raise ValueError(f"Unexpected parent weight_max in {architecture}/{filename}.")
    if model.get("weight_init_mode") != "bounded_uniform":
        raise ValueError(
            f"Expected bounded_uniform parent initialization in {architecture}/{filename}."
        )
    if config.get("init_checkpoint_path") not in (None, ""):
        raise ValueError(
            f"Expected fresh deterministic initialization in {architecture}/{filename}."
        )
    if int(config.get("seed", -1)) != 0:
        raise ValueError(f"Expected seed 0 in {architecture}/{filename}.")
    if config.get("optimizer", {}).get("name", "").lower() != optimizer:
        raise ValueError(f"Unexpected optimizer in {architecture}/{filename}.")
    surface_id = config.get("handoff", {}).get("surface_id")
    if surface_id != f"bounded__{architecture}__{scheme}__{optimizer}":
        raise ValueError(
            f"Unexpected parent surface in {architecture}/{filename}: {surface_id!r}."
        )
    rates = config.get("lr")
    if rates != config.get("optimizer", {}).get("learning_rate"):
        raise ValueError(f"Parent LR fields disagree in {architecture}/{filename}.")
    order = config.get("parameter_order")
    named = config.get("learning_rates_by_parameter")
    if not isinstance(order, list) or not isinstance(named, dict):
        raise ValueError(f"Missing named parent rates in {architecture}/{filename}.")
    if rates != [named[name] for name in order]:
        raise ValueError(f"Parent named LR order disagrees in {architecture}/{filename}.")
    official = config.get("evaluation", {}).get("official_test", {}).get("policy")
    if official != "disabled":
        raise ValueError(f"Official test is not disabled in {architecture}/{filename}.")


def _fixed_science_sha256(config: dict[str, Any]) -> str:
    normalized = deepcopy(config)
    normalized["model_base"]["weight_max"] = "VARIED_BY_WMAX_SWEEP"
    for key in ("arm_id", "study_id", "reporting", "wmax_sweep"):
        normalized.pop(key, None)
    return _sha256(_json_bytes(normalized))


def build_config(
    parent: dict[str, Any],
    *,
    parent_path: str,
    parent_sha256: str,
    architecture: str,
    scheme: str,
    optimizer: str,
    wmax_case: WmaxCase,
) -> dict[str, Any]:
    config = deepcopy(parent)
    arm_id = (
        f"{architecture}_wmax_{wmax_case.tag}_{scheme}_{optimizer}_"
        "seed0_ordinary_mnist"
    )
    config["arm_id"] = arm_id
    config["study_id"] = STUDY_ID
    config["reporting"] = {
        "arm_id": arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "study_id": STUDY_ID,
    }
    config["model_base"]["weight_max"] = wmax_case.value
    config["wmax_sweep"] = {
        "changed_scientific_fields": ["model_base.weight_max"],
        "fixed_learning_rates_unchanged": True,
        "fixed_science_sha256": _fixed_science_sha256(config),
        "initialization_policy": {
            "checkpoint_reused": False,
            "description": (
                "Rebuild bounded_uniform with the unchanged seed and parameter "
                "construction order so matched RNG quantiles map into each "
                "tested [weight_min,weight_max] interval."
            ),
            "initializer": "bounded_uniform",
            "matched_random_quantiles": True,
            "seed": 0,
        },
        "parent": {
            "config_path": parent_path,
            "config_sha256": parent_sha256,
            "source_commit": PARENT_COMMIT,
        },
        "scientific_role": (
            "whole_bounded_contract_sensitivity_not_new_learning_rate_selection"
        ),
        "user_request_date": "2026-08-09",
        "weight_max": wmax_case.value,
        "weight_min": WEIGHT_MIN,
        "weight_range_ratio": wmax_case.value / WEIGHT_MIN,
    }
    return config


def _ordered_set_sha256(rows: list[dict[str, Any]]) -> str:
    return _sha256(
        "".join(f"{row['config_sha256']}\n" for row in rows).encode("ascii")
    )


def _existing_files(output_root: Path) -> dict[str, Path]:
    if not output_root.exists():
        return {}
    return {
        path.relative_to(output_root).as_posix(): path
        for path in output_root.rglob("*")
        if path.is_file()
    }


def _verify_output(
    output_root: Path, expected_payloads: dict[str, bytes]
) -> None:
    existing = _existing_files(output_root)
    expected_paths = set(expected_payloads)
    existing_paths = set(existing)
    missing = sorted(expected_paths - existing_paths)
    stale = sorted(existing_paths - expected_paths)
    changed = sorted(
        relative_path
        for relative_path in expected_paths & existing_paths
        if existing[relative_path].read_bytes() != expected_payloads[relative_path]
    )
    if missing or stale or changed:
        detail = json.dumps(
            {"changed": changed, "missing": missing, "stale": stale},
            sort_keys=True,
        )
        raise RuntimeError(f"Generated config set does not match: {detail}")


def prepare(output_root: Path = OUTPUT_ROOT, *, check: bool = False) -> dict[str, Any]:
    output_root = Path(output_root)

    parent_manifest_path = f"{PARENT_CONFIG_ROOT}/study_manifest.json"
    (
        parent_manifest,
        parent_manifest_bytes,
        parent_configs,
    ) = _load_and_validate_parent_manifest()
    runs: list[dict[str, Any]] = []
    expected_payloads: dict[str, bytes] = {}
    architecture_rows: dict[str, list[dict[str, Any]]] = {
        architecture: [] for architecture in ARCHITECTURE_TARGETS
    }

    global_index = 0
    for architecture, target in ARCHITECTURE_TARGETS.items():
        architecture_index = 0
        for wmax_case in WMAX_CASES:
            for parent_filename in PARENT_FILENAMES:
                match = PARENT_NAME_RE.match(parent_filename)
                if match is None:
                    raise ValueError(f"Unexpected parent filename: {parent_filename}.")
                scheme, optimizer = match.groups()
                parent_relative_path = f"{architecture}/{parent_filename}"
                try:
                    parent, parent_bytes, parent_row = parent_configs[
                        parent_relative_path
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"Parent manifest does not declare {parent_relative_path}."
                    ) from exc
                _validate_parent(
                    parent,
                    architecture=architecture,
                    filename=parent_filename,
                    scheme=scheme,
                    optimizer=optimizer,
                )
                parent_path = (
                    f"{PARENT_CONFIG_ROOT}/{architecture}/{parent_filename}"
                )
                if parent_row["surface_id"] != (
                    f"bounded__{architecture}__{scheme}__{optimizer}"
                ):
                    raise ValueError(
                        f"Unexpected parent manifest surface for {parent_relative_path}."
                    )
                config = build_config(
                    parent,
                    parent_path=parent_path,
                    parent_sha256=_sha256(parent_bytes),
                    architecture=architecture,
                    scheme=scheme,
                    optimizer=optimizer,
                    wmax_case=wmax_case,
                )
                filename = (
                    f"{architecture_index:02d}_wmax_{wmax_case.tag}_"
                    f"{scheme}_{optimizer}_seed0.json"
                )
                relative_path = f"{architecture}/{filename}"
                payload = _json_bytes(config)
                expected_payloads[relative_path] = payload
                row = {
                    "K": int(config["model_base"]["num_iterations_training"]),
                    "T": int(config["model_base"]["num_iterations_inference"]),
                    "architecture": architecture,
                    "architecture_array_index": architecture_index,
                    "arm_id": config["arm_id"],
                    "config": relative_path,
                    "config_sha256": _sha256(payload),
                    "epochs": int(config["lab"]["epochs"]),
                    "global_array_index": global_index,
                    "learning_rates_by_parameter": config[
                        "learning_rates_by_parameter"
                    ],
                    "optimizer": optimizer,
                    "parameter_order": config["parameter_order"],
                    "parent_config": parent_path,
                    "parent_config_sha256": _sha256(parent_bytes),
                    "scheme": scheme,
                    "surface_id": f"{architecture}__{scheme}__{optimizer}",
                    "target": target,
                    "weight_max": wmax_case.value,
                    "weight_min": WEIGHT_MIN,
                    "weight_range_ratio": wmax_case.value / WEIGHT_MIN,
                }
                runs.append(row)
                architecture_rows[architecture].append(row)
                architecture_index += 1
                global_index += 1

    if len(runs) != 90:
        raise AssertionError(f"Expected 90 generated configs, got {len(runs)}.")

    manifest = {
        "dataset_contract": {
            "affine_corruption": False,
            "factory": "labs.datasets.MnistTrainValidationDataset",
            "official_test_read": False,
            "shuffle_seed": 0,
            "split": "deterministic_stratified_55000_5000",
            "split_seed": 0,
        },
        "epoch_budget": {"conv1": 10, "conv2": 30, "conv3": 30},
        "evidence_class": EVIDENCE_CLASS,
        "initializer_policy": {
            "checkpoint_reused": False,
            "initializer": "bounded_uniform",
            "matched_random_quantiles": True,
            "seed": 0,
        },
        "no_scientific_safety_or_rho_runs": True,
        "ordered_config_set_sha256": _ordered_set_sha256(runs),
        "ordered_config_set_sha256_by_architecture": {
            architecture: _ordered_set_sha256(rows)
            for architecture, rows in architecture_rows.items()
        },
        "parent": {
            "config_root": PARENT_CONFIG_ROOT,
            "manifest_path": parent_manifest_path,
            "manifest_sha256": _sha256(parent_manifest_bytes),
            "manifest_schema_version": parent_manifest["schema_version"],
            "ordered_config_set_sha256": parent_manifest[
                "ordered_config_set_sha256"
            ],
            "run_count": parent_manifest["run_count"],
            "source_commit": PARENT_COMMIT,
            "source_tree": _git_tree(),
            "study_id": PARENT_STUDY_ID,
        },
        "paper_facing": False,
        "run_count": len(runs),
        "runs": runs,
        "schema_version": "perfectdiode-bounded-wmax-sweep-config-set/v1",
        "scientific_question": (
            "How does the bounded conductance upper limit affect baseline, ours, "
            "and legacy amplification at fixed accepted learning vectors?"
        ),
        "study_id": STUDY_ID,
        "target_split": ARCHITECTURE_TARGETS,
        "varied_field": "model_base.weight_max",
        "weight_max_values": [case.value for case in WMAX_CASES],
        "weight_min": WEIGHT_MIN,
    }
    expected_payloads["study_manifest.json"] = _json_bytes(manifest)
    if check:
        _verify_output(output_root, expected_payloads)
        return manifest

    stale = sorted(set(_existing_files(output_root)) - set(expected_payloads))
    if stale:
        raise RuntimeError(
            "Refusing to leave stale files in generated config set: "
            + ", ".join(stale)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for relative_path, payload in expected_payloads.items():
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    _verify_output(output_root, expected_payloads)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated set is byte-for-byte current without writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare(args.output_root, check=args.check)
    print(
        json.dumps(
            {
                "config_root": str(Path(args.output_root).resolve()),
                "ordered_config_set_sha256": manifest[
                    "ordered_config_set_sha256"
                ],
                "run_count": manifest["run_count"],
                "study_id": manifest["study_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
