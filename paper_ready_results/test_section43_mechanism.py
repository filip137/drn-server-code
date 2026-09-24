"""Scientific checks for the Section 4.3 aggregation, noise, and checkpoint loader."""
import json
from pathlib import Path

import numpy as np
import torch

import run_section43_mechanism as replay
import plot_section43_mechanism as plot


def test_undefined_cosines_and_individual_vector_average():
    assert replay.compare_vectors([0., 0.], [1., 0.])["cosine"] is None
    assert replay.compare_vectors([1., 0.], [-1., 0.])["cosine"] == -1.
    values = [[100., 0.], [0., 1.]]
    mean_cosine = plot.distribution(replay.compare_vectors(v, [1., 0.])["cosine"] for v in values)["mean"]
    assert mean_cosine == .5
    assert replay.compare_vectors(np.mean(values, axis=0), [1., 0.])["cosine"] > .99


def test_independent_endpoint_noise_has_expected_centered_scale():
    zs = [torch.randn((200000,), dtype=torch.float64,
                     generator=torch.Generator().manual_seed(replay.noise_seed(2026091801, 4, 2, phase, 0)))
          for phase in (0, 1)]
    assert abs(float(torch.mean(zs[0]*zs[1]))) < .01
    assert abs(float(torch.sqrt(torch.mean(((zs[1]-zs[0])/2)**2))) - 1/np.sqrt(2)) < .005
    assert len({replay.noise_seed(1, b, d, p, l) for b in range(36) for d in range(8)
                for p in range(2) for l in range(4)}) == 36*8*2*4


def test_centered_contrast_excludes_common_relaxation_drift():
    tensor = lambda value: torch.tensor([[value]], dtype=torch.float64)
    out = {"captured_phase_states": {"positive": [tensor(101)], "negative": [tensor(99)]},
           "captured_post_t_states": [tensor(0)], "captured_zero_states": [tensor(100)]}
    rows = {r["kind"]: r for r in replay.displacement_rows(out, {})}
    assert rows["centered_halfspan"]["rms"] == 1
    assert rows["positive_minus_free"]["rms"] == 101
    assert rows["positive_minus_matched_zero"]["rms"] == 1
    assert rows["zero_nudge_drift"]["rms"] == 100


def test_real_checkpoint_loader_preserves_float64_values():
    config_path = Path(__file__).with_name("section43_mechanism_config.json")
    shadow = replay.load_shadow(config_path)
    entry = json.loads(config_path.read_text())["cases"][0]
    source = Path(entry["run_dir"])
    case = {**entry, "source_config": json.loads((source/"config.used.json").read_text()),
            "best_checkpoint_path": source/"best_model.pt", "weights_best_path": source/"weights_best.npz",
            "best_checkpoint_sha256": entry["source_file_sha256"]["best_model.pt"]}
    runtime, guard = shadow._checkpoint_runtime(case, role="best_validation", device=torch.device("cpu"),
                                                 gradient_iterations=case["K"], nudging_mode="current")
    assert guard["native_checkpoint_dtype"] == "float64"
    assert guard["loaded_checkpoint_exactly_matches_npz"]
    assert all(p.state.dtype == torch.float64 for p in runtime["parameters"])
    assert any(not torch.equal(p.state, p.state.float().double()) for p in runtime["parameters"])
    before = shadow.base._parameter_state_sha256(runtime["parameters"])
    shadow._convert_runtime_dtype(runtime, torch.float64)
    assert shadow.base._parameter_state_sha256(runtime["parameters"]) == before
