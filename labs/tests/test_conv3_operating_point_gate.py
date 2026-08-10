from __future__ import annotations

import math

import pytest
import torch

import experiments.conv3_operating_point_gate as gate_module
from experiments.conv3_operating_point_gate import (
    DEFAULT_CONTRACT,
    _clamp_occupancy_per_sample,
    _projected_perfect_diode_residual,
    assess_gradient_gate,
    assess_reference_gradient_viability,
)


def _gradient_row(
    vector: list[float],
    *,
    batch_index: int,
    weight_rms: float = 1.0,
) -> dict:
    tensor = torch.tensor(vector, dtype=torch.float64)
    rms = float(torch.sqrt(torch.mean(tensor.square())))
    return {
        "batch_index": batch_index,
        "vector": tensor,
        "gradient_l2": float(torch.linalg.vector_norm(tensor)),
        "gradient_rms": rms,
        "zero_fraction": float((tensor.abs() <= 1.0e-12).to(torch.float64).mean()),
        "nominal_sgd_lr1_proposal_unit": rms / weight_rms,
    }


def test_projected_kkt_residual_uses_correct_boundary_signs() -> None:
    # Channels are [positive_0, positive_1, negative_0, negative_1].
    state = torch.tensor([[0.0, 2.0, 0.0, -2.0]])
    gradient = torch.tensor([[3.0, -4.0, -5.0, 6.0]])

    residual = _projected_perfect_diode_residual(
        state, gradient, epsilon=1.0e-8
    )

    # Positive boundary accepts positive g, negative boundary accepts negative g;
    # free coordinates always use abs(g).
    assert residual.tolist() == [[0.0, 4.0, 0.0, 6.0]]


def test_projected_kkt_residual_rejects_outward_boundary_gradients() -> None:
    state = torch.zeros((1, 4))
    gradient = torch.tensor([[-3.0, -2.0, 5.0, 7.0]])

    residual = _projected_perfect_diode_residual(
        state, gradient, epsilon=1.0e-8
    )

    assert residual.tolist() == [[3.0, 2.0, 5.0, 7.0]]
    assert _clamp_occupancy_per_sample(state, epsilon=1.0e-8).item() == 1.0


def test_residual_audit_enables_autograd_for_energy_derivative() -> None:
    class Layer:
        def __init__(self) -> None:
            self.state = torch.ones((1, 2), requires_grad=True)

    class Network:
        def set_input(self, _images, *, reset) -> None:
            assert reset is True

    class Minimizer:
        def compute_equilibrium(self) -> None:
            assert torch.is_grad_enabled() is False

    class Energy:
        @staticmethod
        def grad_layer_fn(layer):
            def gradient():
                assert torch.is_grad_enabled() is True
                energy = layer.state.square().sum()
                return torch.autograd.grad(energy, layer.state)[0]

            return gradient

    layer = Layer()
    context = {
        "parameters": [layer],
        "free_layers": [layer],
        "network": Network(),
        "minimizer_inference": Minimizer(),
        "energy_fn": Energy(),
        "inference_iterations": 8,
    }

    result = gate_module._residual_audit(
        context,
        [(torch.zeros((1, 1)), torch.zeros(1, dtype=torch.long))],
        expected_examples=1,
        threshold=3.0,
        clamp_epsilon=1.0e-12,
    )

    assert result["passed"] is True
    assert result["layers"][0]["selected_residual"]["p90"] == pytest.approx(2.0)


def test_gradient_gate_passes_identical_live_reference() -> None:
    rows = [
        _gradient_row([1.0, 2.0, 0.0, 4.0], batch_index=index)
        for index in range(8)
    ]
    operational = {name: rows for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")}
    reference = {name: rows for name in operational}

    result = assess_gradient_gate(operational, reference, DEFAULT_CONTRACT)

    assert result["reference_viable"] is True
    assert result["comparison_passed"] is True
    assert result["passed"] is True
    assert all(row["relative_gradient_l2_norm_delta"] == 0.0 for row in result["parameters"])
    assert all(row["gradient_vector_cosine_mean"] == pytest.approx(1.0) for row in result["parameters"])


def test_gradient_gate_fails_dead_k64_reference_even_when_vectors_match() -> None:
    dead = [_gradient_row([0.0, 0.0, 0.0, 0.0], batch_index=index) for index in range(8)]
    operational = {name: dead for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")}
    reference = {name: dead for name in operational}

    result = assess_gradient_gate(operational, reference, DEFAULT_CONTRACT)

    assert result["reference_viable"] is False
    assert result["passed"] is False
    assert all(row["reference_proposal_units_all_finite_positive"] is False for row in result["parameters"])


def test_reference_viability_can_be_assessed_without_a_k8_comparison() -> None:
    rows = [
        _gradient_row([1.0, 2.0, 0.0, 4.0], batch_index=index)
        for index in range(8)
    ]
    reference = {
        name: rows for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
    }

    result = assess_reference_gradient_viability(reference, DEFAULT_CONTRACT)

    assert result["reference_viable"] is True
    assert [row["parameter"] for row in result["parameters"]] == [
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
    ]


def test_gradient_gate_uses_finite_positive_median_proposal_unit() -> None:
    rows = [
        _gradient_row([1.0, 2.0, 3.0, 4.0], batch_index=index)
        for index in range(8)
    ]
    rows[0] = {**rows[0], "nominal_sgd_lr1_proposal_unit": 0.0}
    operational = {
        name: rows for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
    }
    reference = {name: rows for name in operational}

    result = assess_gradient_gate(operational, reference, DEFAULT_CONTRACT)

    assert result["reference_viable"] is True
    assert result["passed"] is True
    assert all(
        row["reference_nominal_sgd_lr1_proposal_unit_median"] > 0.0
        for row in result["parameters"]
    )
    assert all(
        row["reference_proposal_units_all_finite"] is True
        for row in result["parameters"]
    )
    assert all(
        row["reference_proposal_units_all_finite_positive"] is False
        for row in result["parameters"]
    )


def test_gradient_gate_enforces_norm_zero_fraction_and_cosine_boundaries() -> None:
    reference_rows = [
        _gradient_row([1.0, 1.0, 1.0, 1.0], batch_index=index)
        for index in range(8)
    ]
    # Orthogonal and half-zero: this violates both cosine and zero-fraction
    # deltas (and also changes the norm).
    current_rows = [
        _gradient_row([1.0, -1.0, 0.0, 0.0], batch_index=index)
        for index in range(8)
    ]
    operational = {name: current_rows for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")}
    reference = {name: reference_rows for name in operational}

    result = assess_gradient_gate(operational, reference, DEFAULT_CONTRACT)

    assert result["reference_viable"] is True
    assert result["comparison_passed"] is False
    assert result["passed"] is False
    for row in result["parameters"]:
        assert row["relative_gradient_l2_norm_delta"] > 0.10
        assert row["absolute_zero_fraction_delta"] > 0.02
        assert math.isfinite(row["gradient_vector_cosine_mean"])
        assert row["gradient_vector_cosine_mean"] < 0.90


def test_runtime_keeps_t_sentinel_out_of_k_comparison(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    checkpoint = tmp_path / "initial.pt"
    output = tmp_path / "gate.json"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(gate_module, "_model_config", lambda _config: {})
    monkeypatch.setattr(
        gate_module, "_validate_source_contract", lambda _config, _model: None
    )
    monkeypatch.setattr(
        gate_module,
        "_load_training_cohort",
        lambda *_args, **_kwargs: (
            [(torch.zeros((64, 1, 1, 1)), torch.zeros(64, dtype=torch.long))],
            {"cohort_original_indices": list(range(64))},
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "_split_prefix_batches",
        lambda *_args, **_kwargs: [
            (torch.zeros((32, 1, 1, 1)), torch.zeros(32, dtype=torch.long))
        ],
    )

    def fake_context(
        _config,
        _checkpoint,
        *,
        device,
        inference_iterations,
        training_iterations,
    ):
        del device
        calls.append((inference_iterations, training_iterations))
        return {
            "parameter_sha256": "shared",
            "inference_iterations": inference_iterations,
            "training_iterations": training_iterations,
        }

    monkeypatch.setattr(gate_module, "_build_model_context", fake_context)
    monkeypatch.setattr(
        gate_module,
        "_residual_audit",
        lambda context, *_args, **_kwargs: {
            "T": context["inference_iterations"],
            "passed": True,
            "layers": [],
        },
    )
    monkeypatch.setattr(
        gate_module,
        "_cache_free_equilibria",
        lambda *_args, **_kwargs: ([tuple()], "free"),
    )
    live_rows = [
        _gradient_row([1.0, 2.0, 3.0, 4.0], batch_index=0)
    ]
    monkeypatch.setattr(
        gate_module,
        "_gradient_observations",
        lambda context, *_args, **_kwargs: {
            "K": context["training_iterations"],
            "by_parameter": {
                name: live_rows
                for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
            },
        },
    )

    result = gate_module.run_conv3_operating_point_gate(
        source,
        checkpoint,
        device="cpu",
        output_path=output,
        smoke=True,
    )

    assert result["status"] == "complete"
    assert result["security_passed"] is True
    assert calls == [(8, 8), (64, 8), (8, 64)]
    assert result["gradient"]["operational"] == {"T": 8, "K": 8}
    assert result["gradient"]["reference"] == {"T": 8, "K": 64}
    assert output.is_file()


def test_k64_viability_runtime_replays_only_t8_k64(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    checkpoint = tmp_path / "epoch_001_model.pt"
    output = tmp_path / "k64.json"
    source.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    monkeypatch.setattr(gate_module, "_model_config", lambda _config: {})
    monkeypatch.setattr(
        gate_module, "_validate_source_contract", lambda _config, _model: None
    )
    monkeypatch.setattr(
        gate_module,
        "_load_training_cohort",
        lambda *_args, **_kwargs: (
            [(torch.zeros((32, 1, 1, 1)), torch.zeros(32, dtype=torch.long))],
            {
                "cohort_original_indices": list(range(32)),
                "cohort_original_indices_sha256": "indices",
            },
        ),
    )

    def fake_context(
        _config,
        _checkpoint,
        *,
        device,
        inference_iterations,
        training_iterations,
    ):
        del device
        calls.append((inference_iterations, training_iterations))
        return {
            "parameter_sha256": "parameters",
            "training_iterations": training_iterations,
        }

    monkeypatch.setattr(gate_module, "_build_model_context", fake_context)
    monkeypatch.setattr(
        gate_module,
        "_cache_free_equilibria",
        lambda *_args, **_kwargs: ([tuple()], "free"),
    )
    live_rows = [_gradient_row([1.0, 2.0, 3.0, 4.0], batch_index=0)]
    monkeypatch.setattr(
        gate_module,
        "_gradient_observations",
        lambda context, *_args, **_kwargs: {
            "K": context["training_iterations"],
            "by_parameter": {
                name: live_rows
                for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
            },
        },
    )

    result = gate_module.run_conv3_k64_gradient_viability_gate(
        source,
        checkpoint,
        device="cpu",
        output_path=output,
        smoke=True,
    )

    assert calls == [(8, 64)]
    assert result["status"] == "complete"
    assert result["viability_passed"] is True
    assert result["checkpoint_unchanged"] is True
    assert result["gradient"]["reference"] == {"T": 8, "K": 64}
    assert output.is_file()
