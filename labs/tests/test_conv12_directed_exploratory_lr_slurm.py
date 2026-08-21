from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from experiments.reporting import complete_run, start_run
from experiments.run_conv12_directed_exploratory_lr_search import (
    surface_cells,
    surface_specs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "experiments/run_conv12_directed_exploratory_lr_search_jeanzay.slurm"
LOCAL_WRAPPER = REPO_ROOT / "experiments/run_conv12_directed_exploratory_lr_search_local_target.sh"
STUDY = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv12_directed_exploratory_lr_search_seed0_20260812_v1.json"
)


def _receipt_program() -> str:
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", WRAPPER.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 2
    return blocks[1]


def _complete_surface(tmp_path: Path, task: int = 5) -> tuple[Path, Path, Path]:
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    checkpoint = tmp_path / "initializer.pt"
    checkpoint.write_bytes(b"test initializer checkpoint\n")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    architecture = study["surfaces"][task]["architecture"]
    study["parent"]["initializer_checkpoint_sha256_by_architecture"][architecture] = digest
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study), encoding="utf-8")

    task_root = tmp_path / "task"
    asset_path = task_root / "assets/bounded_uniform" / architecture / "asset.json"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": digest}),
        encoding="utf-8",
    )
    surface = surface_specs(study)[task]
    candidates = []
    expected_tk = {"conv1": (4, 4), "conv2": (6, 6)}[architecture]
    for expected in surface_cells(study, surface):
        index = expected["index"]
        run_dir = (
            task_root
            / "surfaces"
            / surface["surface_id"]
            / "rho/cells"
            / f"{index:03d}"
        )
        manifest = {
            "study_id": study["study_id"],
            "run_id": f"cell-{index:03d}",
            "arm_id": "adam-rho-test",
            "evidence_class": study["evidence_class"],
            "configuration": {
                "optimizer": surface["optimizer"],
                "resolved": {
                    "model_base": {
                        "weight_min": 1e-5,
                        "weight_max": 1e-4,
                        "num_iterations_inference": expected_tk[0],
                        "num_iterations_training": expected_tk[1],
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
            "rho_conv": expected["rho_conv"],
            "rho_dense": expected["rho_dense"],
            "status": "complete",
            "signature": {"evidence_class": study["evidence_class"]},
            "learning_rates_by_parameter": {
                "ConvWeight_0": 1e-6,
                "DenseWeight_0": 1e-6,
                "Bias_0": 0.0,
            },
        }
        (run_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
        candidates.append(
            {
                **expected,
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
                "expected_cells": len(candidates),
                "terminal_cells": len(candidates),
                "status_counts": {"complete": len(candidates)},
                "candidates": candidates,
                "best_observation": candidates[-1],
                "official_test_read": False,
            }
        ),
        encoding="utf-8",
    )
    return study_path, task_root, tmp_path / "receipt.json"


def _invoke_receipt(
    study_path: Path,
    task_root: Path,
    receipt_path: Path,
    task: int = 5,
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


def test_wrapper_contract_and_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-11%12" in text
    assert "#SBATCH --cpus-per-task=2" in text
    assert "#SBATCH --constraint=v100-16g" in text
    assert "validated_cell_bundles" in text
    assert 'summary_status == "unresolved_probe"' in text
    assert "validate_run(run_dir)" in text

    local_result = subprocess.run(
        ["bash", "-n", str(LOCAL_WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert local_result.returncode == 0, local_result.stderr
    local_text = LOCAL_WRAPPER.read_text(encoding="utf-8")
    assert "C12LR_SURFACE_INDICES" in local_text
    assert "C12LR_LAUNCHER_SEMANTIC_PASS" in local_text
    assert "validate_run(run_dir)" in local_text


def test_receipt_validates_declared_sparse_canonical_bundles(tmp_path: Path) -> None:
    study_path, task_root, receipt_path = _complete_surface(tmp_path)

    result = _invoke_receipt(study_path, task_root, receipt_path)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["summary_status"] == "complete"
    assert receipt["terminal_cells"] == 6
    assert receipt["validated_cell_bundles"] == 6
    assert receipt["semantic_pass"] is True


def test_receipt_fails_closed_on_undeclared_coordinate(tmp_path: Path) -> None:
    study_path, task_root, receipt_path = _complete_surface(tmp_path)
    cell_path = next(task_root.glob("surfaces/*/rho/cells/*/cell.json"))
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["rho_conv"] *= 2.0
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    result = _invoke_receipt(study_path, task_root, receipt_path)

    assert result.returncode != 0
    assert "Sparse coordinate mismatch" in result.stderr
    assert not receipt_path.exists()
