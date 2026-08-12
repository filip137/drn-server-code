from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_WRAPPER = (
    REPO_ROOT
    / "experiments"
    / "run_bounded_uniform_zero_bias_lr_search_local_target.sh"
)
JEAN_ZAY_WRAPPER = (
    REPO_ROOT
    / "experiments"
    / "run_bounded_uniform_zero_bias_lr_search_jeanzay.slurm"
)
STUDY_ID = (
    "perfectdiode-bounded-uniform-zero-bias-lr-search-"
    "conv123-seed0-20260810-v1"
)
STUDY_CONFIG = (
    "configs/conv/"
    "perfectdiode_bounded_uniform_zero_bias_lr_search_"
    "conv123_seed0_20260810_v1.json"
)
STUDY_CONFIG_PATH = REPO_ROOT / STUDY_CONFIG


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _python_heredocs(text: str) -> list[str]:
    return re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)


def _assert_common_contract(text: str) -> None:
    assert STUDY_ID in text
    assert STUDY_CONFIG in text
    for token in (
        "PDBLR_SOURCE_ROOT",
        "PDBLR_SOURCE_ARCHIVE",
        "PDBLR_RESULT_ROOT",
        "PDBLR_DATASET_ROOT",
        "PDBLR_ENVIRONMENT_ID",
        "PDBLR_STUDY_ID",
        "PDBLR_STUDY_CONFIG_RELATIVE",
        "PDBLR_EXPECTED_SURFACE_COUNT",
        "PDBLR_TRANSPORT_RECEIPT_SCHEMA",
        "PDBLR_STUDY_CONFIG_SHA256",
        "EXPERIMENT_SOURCE_COMMIT",
        "EXPERIMENT_SOURCE_ARCHIVE_SHA256",
    ):
        assert token in text
    assert "sha256sum -- \"${PDBLR_SOURCE_ARCHIVE}\"" in text
    assert "sha256sum -- \"${STUDY_CONFIG}\"" in text
    assert "git -C \"${PDBLR_SOURCE_ROOT}\" rev-parse HEAD" in text
    assert "source_train_images_sha256" in text
    assert "source_train_labels_sha256" in text
    assert "train-images-idx3-ubyte" in text
    assert "train-labels-idx1-ubyte" in text
    assert "torch.cuda.is_available()" in text
    assert "run_conv12_bounded_rho run-surface" in text
    for flag in (
        "--study",
        "--surface-index",
        "--device cuda",
        "--target",
        "--dataset-root",
        "--output-root",
    ):
        assert flag in text
    assert "selection.json" in text
    assert 'run_dir / "status.json"' in text
    assert 'run_status.get("state") != "complete"' in text
    assert 'artifact.get("path") == "artifacts/safety_diagnostics.json"' in text
    assert '"shared_initialization_checkpoint_sha256"' in text
    assert 'post_training_tk.get("passed") is True' in text
    assert "len(post_epochs) == 3" in text
    assert "validate_conv3_selected_post_training_tk_evidence" in text
    assert 'output_root\n        / "fixed_tk"' in text
    assert '"conv3_post_training_tk_validation"' in text
    assert 'Path(conv3_post_training_tk_validation["selected_run_dir"])' in text
    assert "PDBLR_SEMANTIC_PASS" in text
    assert "transport-receipt/v1" in text
    assert '"schema_version": transport_receipt_schema' in text
    assert "official_test_read" in text
    assert "EXPECTED_SOURCE_COMMIT" not in text
    assert not re.search(r"\b[0-9a-f]{40}\b", text)


def test_transport_wrappers_are_valid_bash() -> None:
    for wrapper in (LOCAL_WRAPPER, JEAN_ZAY_WRAPPER):
        subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_each_receipt_heredoc_imports_hashlib_where_used() -> None:
    for wrapper in (LOCAL_WRAPPER, JEAN_ZAY_WRAPPER):
        blocks = _python_heredocs(_read(wrapper))
        assert len(blocks) == 2
        for block in blocks:
            tree = ast.parse(block)
            hashlib_used = any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == "hashlib"
                for node in ast.walk(tree)
            )
            hashlib_imported = any(
                isinstance(node, ast.Import)
                and any(alias.name == "hashlib" for alias in node.names)
                for node in tree.body
            )
            assert not hashlib_used or hashlib_imported, wrapper


def test_local_transport_is_sequential_and_range_bounded() -> None:
    text = _read(LOCAL_WRAPPER)
    _assert_common_contract(text)
    assert "PDBLR_START_INDEX" in text
    assert "PDBLR_END_INDEX" in text
    assert 'PDBLR_EXPECTED_SURFACE_COUNT="${PDBLR_EXPECTED_SURFACE_COUNT:-18}"' in text
    assert "END_INDEX >= EXPECTED_SURFACE_COUNT" in text
    assert 'surface_count != expected_surface_count' in text
    assert "for (( index=START_INDEX; index<=END_INDEX; index++ ))" in text
    assert 'OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/${PDBLR_TARGET}"' in text
    assert '--target "${PDBLR_TARGET}"' in text
    assert "--smoke" not in text


def test_jean_zay_transport_has_exact_resource_and_surface_mapping_contract() -> None:
    text = _read(JEAN_ZAY_WRAPPER)
    _assert_common_contract(text)
    expected_directives = {
        "#SBATCH --account=fmu@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t4",
        "#SBATCH --constraint=v100-32g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --time=24:00:00",
        "#SBATCH --array=0-2%1",
    }
    assert expected_directives <= set(text.splitlines())
    assert 'MODULE_ID="pytorch-gpu/py3/2.5.0"' in text
    assert 'PDBLR_EXPECTED_SURFACE_COUNT="${PDBLR_EXPECTED_SURFACE_COUNT:-18}"' in text
    assert 'PDBLR_SURFACE_OFFSET="${PDBLR_SURFACE_OFFSET:-12}"' in text
    assert 'PDBLR_SURFACES_PER_TASK="${PDBLR_SURFACES_PER_TASK:-2}"' in text
    assert 'PDBLR_ISOLATE_TASK_SHARDS="${PDBLR_ISOLATE_TASK_SHARDS:-0}"' in text
    assert "SURFACE_START=$((SURFACE_OFFSET + SURFACES_PER_TASK * SLURM_ARRAY_TASK_ID))" in text
    assert "SURFACE_END=$((SURFACE_START + SURFACES_PER_TASK - 1))" in text
    assert "for (( index=SURFACE_START; index<=SURFACE_END; index++ ))" in text
    assert (
        'OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/jean-zay"'
    ) in text
    assert (
        'OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/jean-zay-task_${SLURM_ARRAY_TASK_ID}"'
    ) in text
    assert (
        'RECEIPT_ROOT="${PDBLR_RESULT_ROOT}/transport_receipts/jean-zay-task_${SLURM_ARRAY_TASK_ID}"'
    ) in text
    assert 'surface_count != expected_surface_count' in text
    assert '--target "${TARGET}"' in text
    assert 'TARGET="jean-zay"' in text
    assert '"V100" not in device_name.upper()' in text
    assert "device_bytes < 30 * 1024**3" in text
    assert "--smoke" not in text

    study = json.loads(STUDY_CONFIG_PATH.read_text(encoding="utf-8"))
    surfaces = list(
        product(
            study["scope"]["initializers"],
            study["scope"]["architectures"],
            study["scope"]["schemes"],
            study["scope"]["optimizers"],
        )
    )
    assert len(surfaces) == 18
    assert [
        [(surfaces[index][1], surfaces[index][2], surfaces[index][3]) for index in pair]
        for pair in ((12, 13), (14, 15), (16, 17))
    ] == [
        [("conv3", "baseline", "SGD"), ("conv3", "baseline", "Adam")],
        [("conv3", "ours", "SGD"), ("conv3", "ours", "Adam")],
        [("conv3", "legacy", "SGD"), ("conv3", "legacy", "Adam")],
    ]


def _mapping_trace(
    tmp_path: Path,
    *,
    task_id: int,
    surface_count: int | None = None,
    surface_offset: int | None = None,
    surfaces_per_task: int | None = None,
    isolate: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PDBLR_SOURCE_ROOT": str(tmp_path / "missing-source"),
            "PDBLR_SOURCE_ARCHIVE": str(tmp_path / "missing-source.tar.gz"),
            "PDBLR_RESULT_ROOT": str(tmp_path / "results"),
            "PDBLR_DATASET_ROOT": str(tmp_path / "mnist"),
            "PDBLR_ENVIRONMENT_ID": "test-jz-v100-32g",
            "PDBLR_STUDY_CONFIG_SHA256": "0" * 64,
            "EXPERIMENT_SOURCE_COMMIT": "0" * 40,
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256": "0" * 64,
            "SLURM_ARRAY_TASK_ID": str(task_id),
        }
    )
    if surface_count is not None:
        env["PDBLR_EXPECTED_SURFACE_COUNT"] = str(surface_count)
    if surface_offset is not None:
        env["PDBLR_SURFACE_OFFSET"] = str(surface_offset)
    if surfaces_per_task is not None:
        env["PDBLR_SURFACES_PER_TASK"] = str(surfaces_per_task)
    if isolate is not None:
        env["PDBLR_ISOLATE_TASK_SHARDS"] = "1" if isolate else "0"
    task_count = (
        (surface_count - surface_offset) // surfaces_per_task
        if None not in (surface_count, surface_offset, surfaces_per_task)
        else 3
    )
    env.update(
        {
            "SLURM_ARRAY_TASK_COUNT": str(task_count),
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": str(task_count - 1),
        }
    )
    return subprocess.run(
        ["bash", "-x", str(JEAN_ZAY_WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_jean_zay_default_mapping_is_backward_compatible(tmp_path: Path) -> None:
    result = _mapping_trace(tmp_path, task_id=2)
    assert result.returncode == 3
    assert "+ SURFACE_START=16" in result.stderr
    assert "+ SURFACE_END=17" in result.stderr
    assert f"+ OUTPUT_ROOT={tmp_path}/results/shards/jean-zay" in result.stderr
    assert (
        f"+ RECEIPT_ROOT={tmp_path}/results/transport_receipts/jean-zay/task_2"
        in result.stderr
    )


def test_jean_zay_conv3_only_mapping_uses_isolated_task_shards(
    tmp_path: Path,
) -> None:
    for task_id, expected_surface in ((0, 4), (1, 5)):
        result = _mapping_trace(
            tmp_path,
            task_id=task_id,
            surface_count=6,
            surface_offset=4,
            surfaces_per_task=1,
            isolate=True,
        )
        assert result.returncode == 3
        assert f"+ SURFACE_START={expected_surface}" in result.stderr
        assert f"+ SURFACE_END={expected_surface}" in result.stderr
        assert (
            f"+ OUTPUT_ROOT={tmp_path}/results/shards/jean-zay-task_{task_id}"
            in result.stderr
        )
        assert (
            f"+ RECEIPT_ROOT={tmp_path}/results/transport_receipts/jean-zay-task_{task_id}"
            in result.stderr
        )
