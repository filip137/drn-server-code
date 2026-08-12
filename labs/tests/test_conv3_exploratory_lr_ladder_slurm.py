from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from experiments.reporting import complete_run, start_run
from experiments.run_conv3_exploratory_lr_ladder import surface_specs


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "experiments/run_conv3_exploratory_lr_ladder_jeanzay.slurm"
STUDY = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv3_exploratory_lr_ladder_seed0_20260811_v1.json"
)


def _receipt_program() -> str:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", WRAPPER.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 2
    return blocks[1]


def _study_with_asset(tmp_path: Path) -> tuple[Path, dict, Path, str]:
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"test initializer checkpoint\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    study["parent"]["initializer_checkpoint_sha256"] = digest
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study), encoding="utf-8")
    return path, study, checkpoint, digest


def _write_asset(task_root: Path, checkpoint: Path, digest: str) -> None:
    asset_path = task_root / "assets/bounded_uniform/conv3/asset.json"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": digest}),
        encoding="utf-8",
    )


def _invoke_receipt(
    study_path: Path,
    task_root: Path,
    receipt_path: Path,
    *,
    task: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _receipt_program(),
            str(study_path),
            str(task_root),
            str(receipt_path),
            str(task),
            "test-v100-environment",
            "0" * 40,
            "1" * 64,
            "2" * 64,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _complete_surface(tmp_path: Path) -> tuple[Path, Path, Path]:
    study_path, study, checkpoint, digest = _study_with_asset(tmp_path)
    task_root = tmp_path / "task"
    _write_asset(task_root, checkpoint, digest)
    surface = surface_specs(study)[0]
    candidates = []
    for index in range(64):
        run_dir = task_root / "surfaces" / surface["surface_id"] / "rho/cells" / f"{index:03d}"
        manifest = {
            "study_id": study["study_id"],
            "run_id": f"cell-{index:03d}",
            "arm_id": "sgd-rho-test",
            "evidence_class": study["evidence_class"],
            "configuration": {
                "optimizer": surface["optimizer"],
                "resolved": {
                    "model_base": {
                        "weight_min": 1e-5,
                        "weight_max": 1e-4,
                        "num_iterations_inference": 8,
                        "num_iterations_training": 8,
                    }
                },
            },
            "dataset": {
                "key": "mnist",
                "variant": "ordinary",
                "evaluation_split": "validation",
                "official_test_read": False,
            },
        }
        start_run(run_dir, manifest)
        complete_run(
            run_dir,
            terminal_metrics={"final_validation_accuracy": 0.5},
            completion={"criteria_met": True, "official_test_read": False},
        )
        cell = {
            "index": index,
            "optimizer": surface["optimizer"],
            "status": "complete",
            "signature": {"evidence_class": study["evidence_class"]},
            "learning_rates_by_parameter": {
                "W_0": 1e-5,
                "Bias_0": 0.0,
            },
        }
        (run_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
        candidates.append(
            {
                "index": index,
                "status": "complete",
                "path": str(run_dir),
            }
        )
    summary_path = task_root / "surfaces" / surface["surface_id"] / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "surface": surface,
                "status": "complete",
                "complete": True,
                "expected_cells": 64,
                "terminal_cells": 64,
                "status_counts": {"complete": 64},
                "candidates": candidates,
                "best_observation": candidates[-1],
                "official_test_read": False,
            }
        ),
        encoding="utf-8",
    )
    return study_path, task_root, tmp_path / "receipt.json"


def test_wrapper_contract_and_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3%4" in text
    assert "#SBATCH --constraint=v100-16g" in text
    assert "validated_cell_bundles" in text
    assert 'summary_status == "unresolved_probe"' in text
    assert "validate_run(run_dir)" in text


def test_receipt_validates_all_64_canonical_cell_bundles(tmp_path: Path) -> None:
    study_path, task_root, receipt_path = _complete_surface(tmp_path)

    result = _invoke_receipt(study_path, task_root, receipt_path)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["summary_status"] == "complete"
    assert receipt["terminal_cells"] == 64
    assert receipt["validated_cell_bundles"] == 64
    assert receipt["semantic_pass"] is True


def test_receipt_fails_closed_on_wrong_cell_evidence_class(tmp_path: Path) -> None:
    study_path, task_root, receipt_path = _complete_surface(tmp_path)
    cell_path = next(task_root.glob("surfaces/*/rho/cells/000/cell.json"))
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["signature"]["evidence_class"] = "ordinary_mnist_selection"
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    result = _invoke_receipt(study_path, task_root, receipt_path)

    assert result.returncode != 0
    assert "Cell signature evidence class mismatch" in result.stderr
    assert not receipt_path.exists()


def test_unresolved_probe_is_a_receipted_terminal_scientific_outcome(
    tmp_path: Path,
) -> None:
    study_path, study, checkpoint, digest = _study_with_asset(tmp_path)
    task_root = tmp_path / "task"
    _write_asset(task_root, checkpoint, digest)
    surface = surface_specs(study)[0]
    surface_dir = task_root / "surfaces" / surface["surface_id"]
    probe_path = surface_dir / "rho/probe.json"
    probe_path.parent.mkdir(parents=True)
    probe_path.write_text(json.dumps({"status": "unstable"}), encoding="utf-8")
    (surface_dir / "summary.json").write_text(
        json.dumps(
            {
                "surface": surface,
                "status": "unresolved_probe",
                "complete": False,
                "expected_cells": 64,
                "terminal_cells": 0,
                "status_counts": {},
                "candidates": [],
                "best_observation": None,
                "official_test_read": False,
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"

    result = _invoke_receipt(study_path, task_root, receipt_path)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["summary_status"] == "unresolved_probe"
    assert receipt["terminal_cells"] == 0
    assert receipt["validated_cell_bundles"] == 0
    assert receipt["semantic_pass"] is True
