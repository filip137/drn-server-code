"""Check the saved initializer against training and diagnostic hash conventions."""
import json
from pathlib import Path

import run_section43_initialization as comparison


def test_saved_initializer_matches_both_training_precisions():
    path = Path(__file__).with_name("section43_initialization_config.json")
    config = json.loads(path.read_text())
    shadow = comparison.replay.load_shadow(path)
    entry = config["cases"][0]
    trained = {**entry, "source_config": json.loads((Path(entry["run_dir"])/"config.used.json").read_text())}
    case = comparison.initialization_case(config, trained, shadow)
    assert case["replay_initial_float32_tensor_sha256"] != config["initializer"]["float32_parameter_state_sha256"]
    runtime, guard = shadow._checkpoint_runtime(case, role="reconstructed_initialization", device=shadow.torch.device("cpu"),
                                               gradient_iterations=8, nudging_mode="current")
    assert guard["native_float32_parameter_state_sha256_before_cast"] == case["replay_initial_float32_tensor_sha256"]
    shadow._convert_runtime_dtype(runtime, shadow.torch.float64)
    assert shadow.base._parameter_state_sha256(runtime["parameters"]) == case["replay_initial_float64_tensor_sha256"]
