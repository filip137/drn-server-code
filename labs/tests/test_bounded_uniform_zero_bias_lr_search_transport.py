from __future__ import annotations

import json
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


def _assert_common_contract(text: str) -> None:
    assert STUDY_ID in text
    assert STUDY_CONFIG in text
    for token in (
        "PDBLR_SOURCE_ROOT",
        "PDBLR_SOURCE_ARCHIVE",
        "PDBLR_RESULT_ROOT",
        "PDBLR_DATASET_ROOT",
        "PDBLR_ENVIRONMENT_ID",
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
    assert "PDBLR_SEMANTIC_PASS" in text
    assert "transport-receipt/v1" in text
    assert "official_test_read" in text
    assert "EXPECTED_SOURCE_COMMIT" not in text
    assert not re.search(r"\b[0-9a-f]{40}\b", text)


def test_transport_wrappers_are_valid_bash() -> None:
    for wrapper in (LOCAL_WRAPPER, JEAN_ZAY_WRAPPER):
        subprocess.run(["bash", "-n", str(wrapper)], check=True)


def test_local_transport_is_sequential_and_range_bounded() -> None:
    text = _read(LOCAL_WRAPPER)
    _assert_common_contract(text)
    assert "PDBLR_START_INDEX" in text
    assert "PDBLR_END_INDEX" in text
    assert "END_INDEX >= 18" in text
    assert "for (( index=START_INDEX; index<=END_INDEX; index++ ))" in text
    assert 'OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/${PDBLR_TARGET}"' in text
    assert '--target "${PDBLR_TARGET}"' in text
    assert "--smoke" not in text


def test_jean_zay_transport_has_exact_resource_and_pair_contract() -> None:
    text = _read(JEAN_ZAY_WRAPPER)
    _assert_common_contract(text)
    expected_directives = {
        "#SBATCH --account=fmu@v100",
        "#SBATCH --partition=gpu_p13",
        "#SBATCH --qos=qos_gpu-t4",
        "#SBATCH --constraint=v100-32g",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --time=24:00:00",
        "#SBATCH --array=0-2",
    }
    assert expected_directives <= set(text.splitlines())
    assert 'MODULE_ID="pytorch-gpu/py3/2.5.0"' in text
    assert "PAIR_START=$((12 + 2 * SLURM_ARRAY_TASK_ID))" in text
    assert "PAIR_END=$((PAIR_START + 1))" in text
    assert "for (( index=PAIR_START; index<=PAIR_END; index++ ))" in text
    assert (
        'OUTPUT_ROOT="${PDBLR_RESULT_ROOT}/shards/jean-zay/'
        'task_${SLURM_ARRAY_TASK_ID}"'
    ) in text
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
