from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "experiments" / "run_paper_gradient_trace_jeanzay.slurm"


def test_gradient_trace_wrapper_freezes_umg_v100_resource_contract() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "#SBATCH --account=umg@v100" in text
    assert "#SBATCH --partition=gpu_p13" in text
    assert "#SBATCH --qos=qos_gpu-t3" in text
    assert "#SBATCH --constraint=v100-16g" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=10" in text
    assert "#SBATCH --time=03:00:00" in text
    assert "--gradient-trace-samples-per-epoch 5" in text
    assert "--checkpoint-every-epoch" in text
    assert "--skip-terminal-official-test" in text
    assert "--epoch-override 10" in text


def test_gradient_trace_wrapper_declares_exact_twelve_arm_surface() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    configs = re.findall(r'"\$\{CONFIG_ROOT\}/(conv[123]/\d\d_[^"]+\.json)"', text)

    assert len(configs) == 12
    assert len(set(configs)) == 12
    assert len([value for value in configs if value.startswith("conv1/")]) == 6
    for architecture in ("conv2", "conv3"):
        selected = [value for value in configs if value.startswith(f"{architecture}/")]
        assert selected == [
            f"{architecture}/00_baseline_sgd_seed0.json",
            f"{architecture}/02_ours_sgd_seed0.json",
            f"{architecture}/04_legacy_sgd_seed0.json",
        ]


def test_gradient_trace_wrapper_guards_array_index_and_artifacts() -> None:
    text = WRAPPER.read_text(encoding="utf-8")

    assert "SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]}" in text
    assert 'case "${TRACE_MODE}" in' in text
    assert 'metadata.get("status") != "complete"' in text
    assert 'experiments.reporting validate-run "${RUN_DIR}"' in text
