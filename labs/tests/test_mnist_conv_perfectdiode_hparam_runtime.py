from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import experiments.mnist_conv.perfectdiode_hparam_runtime as runtime_module
from experiments.mnist_conv.lr_step import optimizer_step_with_diagnostics
from experiments.mnist_conv.perfectdiode_hparam_runtime import (
    CANDIDATE_TOTAL_STEPS,
    CANARY_STEPS,
    CONV1_CONFIRM_STEPS,
    CONV2_CONFIRM_STEPS,
    PerfectDiodeRuntimeError,
    SafetyMonitor,
    compare_fixed_tk_gradient_security,
    derive_raw_learning_rates,
    epoch_validation_checkpoint_plan,
    measure_adaptive_optimizer_probe,
    require_official_test_excluded,
    rho_cell_id,
    run_canary,
    run_epoch_training,
    training_contract,
)


class ScientificParameter:
    def __init__(self, name: str, value: float = 1.0) -> None:
        self.name = name
        self.state = torch.nn.Parameter(
            torch.full((2,), value, dtype=torch.float64)
        )
        self.min_cond = 0.0
        self.max_cond = 100.0


class FakeRuntime:
    def __init__(self, optimizer_name: str = "SGD") -> None:
        self.parameters = [
            ScientificParameter("ConvWeight_0", 1.0),
            ScientificParameter("Bias_0", 0.0),
            ScientificParameter("DenseWeight_0", 2.0),
        ]
        groups = [
            {"params": [parameter.state], "lr": 1.0}
            for parameter in self.parameters
        ]
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                groups,
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
                amsgrad=False,
                foreach=False,
                fused=False,
                maximize=False,
                capturable=False,
                differentiable=False,
            )
        else:
            self.optimizer = torch.optim.SGD(
                groups, lr=1.0, momentum=0.0, weight_decay=0.0
            )
        self.optimizer_name = optimizer_name
        self.device = torch.device("cpu")

    def set_learning_rate(self, rates):
        if isinstance(rates, dict):
            by_id = {
                id(parameter.state): float(rates[parameter.name])
                for parameter in self.parameters
            }
            for group in self.optimizer.param_groups:
                group["lr"] = by_id[id(group["params"][0])]
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = float(rates)

    def save(self, path):
        Path(path).write_bytes(b"fake-checkpoint\n")


def _shadow_training_step(runtime, batch, *, learning_rate, restore, **_kwargs):
    runtime.optimizer.zero_grad()
    factor = float(batch)
    for parameter in runtime.parameters:
        parameter.state.grad = torch.full_like(parameter.state, factor)
    runtime.set_learning_rate(learning_rate)
    transition = optimizer_step_with_diagnostics(
        runtime.parameters,
        runtime.optimizer,
        restore=restore,
    )
    return SimpleNamespace(
        loss=1.0,
        accuracy=0.5,
        sample_count=16,
        source_indices=(int(factor),),
        transition=transition,
    )


def _transition(
    *,
    occupancy: float = 0.0,
    projection: float = 1.0,
    gradient: float = 1.0,
    proposal_zero: bool = False,
):
    before = torch.ones(2, dtype=torch.float64)
    after = before if proposal_zero else before + 0.01
    items = []
    for name in ("ConvWeight_0", "DenseWeight_0"):
        items.append(
            SimpleNamespace(
                name=name,
                bounded_gate=True,
                gradient_rms=gradient,
                proposed_update_rms=0.0 if proposal_zero else 0.01,
                projection_efficiency=projection,
                combined_bound_occupancy=occupancy,
                proposal_is_numerically_zero=proposal_zero,
                pre_update=before,
                post_projection=after,
            )
        )
    bias = SimpleNamespace(
        name="Bias_0",
        bounded_gate=False,
        gradient_rms=gradient,
        proposed_update_rms=0.01,
        projection_efficiency=1.0,
        combined_bound_occupancy=None,
        proposal_is_numerically_zero=False,
        pre_update=before,
        post_projection=after,
    )
    items.insert(1, bias)
    return SimpleNamespace(
        parameters=tuple(items),
        by_name={item.name: item for item in items},
    )


def test_fixed_tk_security_uses_mean_batch_norm_zero_and_cosine() -> None:
    reference = {
        "ConvWeight_0": (
            torch.tensor([3.0, 4.0]),
            torch.tensor([0.0, 2.0]),
        )
    }
    operational = {
        "ConvWeight_0": tuple(value.clone() for value in reference["ConvWeight_0"])
    }
    result = compare_fixed_tk_gradient_security(
        operational,
        reference,
        operational_t=4,
        operational_k=4,
        expected_batch_count=2,
    )
    row = result["parameter_diagnostics"][0]
    assert result["security_passed"] is True
    assert row["gradient_l2_mean"] == pytest.approx(3.5)
    assert row["relative_gradient_l2_norm_delta"] == 0.0
    assert row["absolute_zero_fraction_delta"] == 0.0
    assert row["gradient_vector_cosine_mean"] == pytest.approx(1.0)


def test_fixed_tk_security_inclusive_gates_and_dead_reference_fail_closed() -> None:
    reference = {
        "ConvWeight_0": (torch.tensor([1.0, 0.0], dtype=torch.float64),)
    }
    operational = {
        "ConvWeight_0": (torch.tensor([0.9, 0.0], dtype=torch.float64),)
    }
    result = compare_fixed_tk_gradient_security(
        operational,
        reference,
        operational_t=6,
        operational_k=6,
        expected_batch_count=1,
    )
    assert result["security_passed"] is True
    dead = compare_fixed_tk_gradient_security(
        {"ConvWeight_0": (torch.zeros(2),)},
        {"ConvWeight_0": (torch.zeros(2),)},
        operational_t=4,
        operational_k=4,
        expected_batch_count=1,
    )
    assert dead["security_passed"] is False
    assert dead["minimum_gradient_vector_cosine"] is None


def test_adaptive_adam_probe_restores_exact_empty_state(monkeypatch) -> None:
    runtime = FakeRuntime("Adam")
    initial_parameters = [
        parameter.state.detach().clone() for parameter in runtime.parameters
    ]
    initial_optimizer = copy.deepcopy(runtime.optimizer.state_dict())
    monkeypatch.setattr(runtime_module, "training_step", _shadow_training_step)
    result = measure_adaptive_optimizer_probe(runtime, [1.0] * 128)
    assert result["probe_stable"] is True
    assert result["used_batches"] == 32
    assert result["restoration"][
        "optimizer_state_exactly_restored_after_every_proposal"
    ]
    assert runtime.optimizer.state_dict() == initial_optimizer
    for parameter, initial in zip(runtime.parameters, initial_parameters):
        assert torch.equal(parameter.state, initial)


def test_adaptive_probe_rejects_warmed_adam(monkeypatch) -> None:
    runtime = FakeRuntime("Adam")
    for parameter in runtime.parameters:
        parameter.state.grad = torch.ones_like(parameter.state)
    runtime.optimizer.step()
    runtime.optimizer.zero_grad()
    monkeypatch.setattr(runtime_module, "training_step", _shadow_training_step)
    with pytest.raises(PerfectDiodeRuntimeError, match="fresh empty Adam"):
        measure_adaptive_optimizer_probe(runtime, [1.0] * 128)


def test_adaptive_sgd_probe_extends_from_32_to_64(monkeypatch) -> None:
    runtime = FakeRuntime("SGD")
    monkeypatch.setattr(runtime_module, "training_step", _shadow_training_step)
    batches = [1.0] * 16 + [2.0] * 16 + [1.0] * 16 + [2.0] * 80
    result = measure_adaptive_optimizer_probe(runtime, batches)
    assert result["probe_stable"] is True
    assert result["used_batches"] == 64
    assert result["unstable_parameters"] == []


def test_lr_derivation_uses_independent_targets_bias_cap_and_zero_fallback() -> None:
    probe = {
        "probe_stable": True,
        "parameter_names": [
            "ConvWeight_0",
            "Bias_0",
            "ConvWeight_1",
            "Bias_1",
            "DenseWeight_0",
        ],
        "normalization_unit_by_weight": {
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        "bias_to_weight": {
            "Bias_0": "ConvWeight_0",
            "Bias_1": "ConvWeight_1",
        },
        "bias_q90_unit_by_parameter": {"Bias_0": 10.0, "Bias_1": 0.0},
    }
    rates = derive_raw_learning_rates(
        probe, rho_conv=0.03, rho_dense=0.1
    )
    assert rates["ConvWeight_0"] == pytest.approx(0.015)
    assert rates["ConvWeight_1"] == pytest.approx(0.0075)
    assert rates["DenseWeight_0"] == pytest.approx(0.02)
    assert rates["Bias_0"] == pytest.approx(0.003)
    assert rates["Bias_1"] == rates["ConvWeight_1"]


@pytest.mark.parametrize(
    ("field", "failure_kind"),
    [
        ("occupancy", "bound_occupancy_increase"),
        ("projection", "projection_efficiency"),
    ],
)
def test_occupancy_and_projection_streaks_start_at_step_one(
    field: str, failure_kind: str
) -> None:
    monitor = SafetyMonitor({"ConvWeight_0": 0.0, "DenseWeight_0": 0.0})
    kwargs = {"occupancy": 0.21} if field == "occupancy" else {"projection": 0.49}
    failure = None
    for _ in range(16):
        failure = monitor.observe(loss=1.0, transition=_transition(**kwargs))
    assert failure is not None
    assert failure["kind"] == failure_kind
    assert failure["onset_step"] == 1
    assert failure["confirmed_step"] == 16


def test_zero_proposal_neither_increments_nor_clears_projection_streak() -> None:
    monitor = SafetyMonitor({"ConvWeight_0": 0.0, "DenseWeight_0": 0.0})
    for _ in range(8):
        assert monitor.observe(
            loss=1.0, transition=_transition(projection=0.49)
        ) is None
    for _ in range(4):
        assert monitor.observe(
            loss=1.0, transition=_transition(proposal_zero=True)
        ) is None
    failure = None
    for _ in range(8):
        failure = monitor.observe(
            loss=1.0, transition=_transition(projection=0.49)
        )
    assert failure["kind"] == "projection_efficiency"
    assert failure["confirmed_step"] == 20


def test_exact_training_contracts_and_epoch_plans() -> None:
    candidate = training_contract(architecture="conv1", mode="candidate")
    assert candidate.epochs == 3
    assert candidate.total_steps == CANDIDATE_TOTAL_STEPS == 10_314
    conv1 = training_contract(
        architecture="conv1", mode="long_confirmation"
    )
    conv2 = training_contract(
        architecture="conv2", mode="long_confirmation"
    )
    assert conv1.total_steps == CONV1_CONFIRM_STEPS == 34_380
    assert conv2.total_steps == CONV2_CONFIRM_STEPS == 103_140
    plan = epoch_validation_checkpoint_plan(
        architecture="conv2", mode="long_confirmation"
    )
    assert len(plan) == 30
    assert plan[-1]["step"] == 103_140
    assert plan[-1]["save_final_checkpoint"] is True


def test_canary_stops_at_exactly_640_and_reports_updates() -> None:
    runtime = FakeRuntime("SGD")
    transition = _transition()
    yielded = {"count": 0}

    class Loader:
        def __iter__(self):
            for index in range(700):
                yielded["count"] += 1
                yield index

    def step(_runtime, batch, **_kwargs):
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.5,
            sample_count=16,
            source_indices=(batch,),
            transition=transition,
        )

    result = run_canary(
        runtime,
        Loader(),
        learning_rates_by_parameter={
            parameter.name: 0.01 for parameter in runtime.parameters
        },
        step_function=step,
    )
    assert result["safety_clean"] is True
    assert result["completed_steps"] == CANARY_STEPS
    assert yielded["count"] == CANARY_STEPS
    assert result["median_projection_efficiency"] == pytest.approx(1.0)
    assert result["achieved_updates_by_parameter"]["ConvWeight_0"][
        "step_count"
    ] == CANARY_STEPS


class FakeBundle:
    def __init__(self) -> None:
        indices = tuple(range(55_000))
        self.validation_indices = tuple(range(100_000, 105_000))
        self.validation_indices_hash = "v" * 64
        self.train_indices_hash = "t" * 64
        self._batches = tuple(
            tuple(indices[start : start + 16])
            for start in range(0, len(indices), 16)
        )
        self.train_loader = self._batches
        self.validation_loader = object()

    def reset_train_shuffle(self):
        return self.train_loader

    def train_batch_indices(self, *, num_epochs=1):
        return tuple(self._batches for _ in range(num_epochs))


def test_three_epoch_candidate_exact_steps_and_inclusive_90_gate(
    tmp_path: Path,
) -> None:
    runtime = FakeRuntime("SGD")
    bundle = FakeBundle()
    transition = _transition()

    def step(_runtime, batch, **_kwargs):
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.5,
            sample_count=len(batch),
            source_indices=tuple(batch),
            transition=transition,
        )

    def validate(_runtime, _loader):
        return {
            "sample_count": 5_000,
            "loss": 0.5,
            "accuracy": 0.90,
            "source_indices": list(bundle.validation_indices),
        }

    result = run_epoch_training(
        runtime,
        bundle,
        architecture="conv1",
        mode="candidate",
        learning_rates_by_parameter={
            parameter.name: 0.01 for parameter in runtime.parameters
        },
        output_dir=tmp_path,
        step_function=step,
        validation_function=validate,
    )
    assert result["completed_steps"] == 10_314
    assert result["epochs_completed"] == 3
    assert result["inclusive_90_percent_accuracy_gate_passed"] is True
    assert result["passing_candidate"] is True
    assert result["checkpoints"]["best_validation"]["sha256"]
    assert result["checkpoints"]["final"]["sha256"]
    assert result["achieved_updates_by_parameter"]["DenseWeight_0"][
        "step_count"
    ] == 10_314


def test_official_test_exclusion_guards_study_and_results() -> None:
    require_official_test_excluded(
        {
            "dataset": {
                "official_test": {"enabled": False, "read_allowed": False}
            }
        }
    )
    require_official_test_excluded(
        {"official_test_read": False}, require_result_marker=True
    )
    with pytest.raises(ValueError, match="read_allowed"):
        require_official_test_excluded(
            {
                "dataset": {
                    "official_test": {"enabled": False, "read_allowed": True}
                }
            }
        )
    with pytest.raises(ValueError, match="official_test_read"):
        require_official_test_excluded(
            {"official_test_read": True}, require_result_marker=True
        )


def test_rho_cell_identity_is_prefixed_and_content_derived() -> None:
    first = rho_cell_id(
        study_id="pdstudy_test",
        surface_id="conv1_baseline_sgd",
        rho_conv=0.003,
        rho_dense=0.01,
    )
    second = rho_cell_id(
        study_id="pdstudy_test",
        surface_id="conv1_baseline_sgd",
        rho_conv=0.003,
        rho_dense=0.03,
    )
    assert first.startswith("pdcell_")
    assert len(first) == len("pdcell_") + 64
    assert first != second


def test_training_preamble_derives_cell_id_instead_of_admitting_on_it(
    tmp_path: Path, monkeypatch
) -> None:
    row = {
        "row_id": "conv1_baseline_v1_c1",
        "architecture": "conv1",
        "scheme": "baseline",
    }
    assets = object()
    spec = SimpleNamespace(study_id="current-study")
    payload = {
        "entry_id": "job-a",
        "row_id": row["row_id"],
        "optimizer": "sgd",
        "rho_conv": 0.009,
        "rho_dense": 0.03,
        "cell_id": "legacy-parent-cell-id",
        "raw_learning_rates_by_parameter": {"weight": 0.1},
    }

    monkeypatch.setattr(
        runtime_module,
        "_common_loaded_assets",
        lambda *_args, **_kwargs: (row, assets),
    )
    monkeypatch.setattr(
        runtime_module,
        "_load_stable_probe",
        lambda *_args, **_kwargs: {"probe_stable": True},
    )
    monkeypatch.setattr(
        runtime_module,
        "_resolve_rates",
        lambda _probe, resolved, **_kwargs: dict(
            resolved["raw_learning_rates_by_parameter"]
        ),
    )

    prepared = runtime_module._prepare_training_entry(
        {},
        spec,
        tmp_path,
        payload,
        data_root=tmp_path,
        download=False,
    )
    expected = rho_cell_id(
        study_id="current-study",
        surface_id="conv1_baseline_v1_c1--sgd",
        rho_conv=0.009,
        rho_dense=0.03,
    )
    assert prepared["payload"]["cell_id"] == expected
    assert payload["cell_id"] == "legacy-parent-cell-id"
    assert prepared["rates"] == {"weight": 0.1}


def test_failed_center_attempt_is_reused_when_it_is_in_safe_core(
    tmp_path: Path, monkeypatch
) -> None:
    row = {
        "row_id": "conv1_baseline_v1_c1",
        "architecture": "conv1",
        "scheme": "baseline",
    }
    assets = SimpleNamespace(
        initialization={
            "checkpoint_sha256": "a" * 64,
            "parameter_tensor_sha256": "b" * 64,
        }
    )
    spec = SimpleNamespace(study_id="pdstudy_test", config_sha256="c" * 64)
    calls = []

    monkeypatch.setattr(
        runtime_module,
        "_common_loaded_assets",
        lambda *_args, **_kwargs: (row, assets),
    )
    monkeypatch.setattr(
        runtime_module,
        "_load_stable_probe",
        lambda *_args, **_kwargs: {"probe_stable": True},
    )

    def fake_cell(
        _study,
        spec_value,
        _row,
        _assets,
        _probe,
        *,
        surface_id,
        rho_conv,
        rho_dense,
        cell_id=None,
        **_kwargs,
    ):
        calls.append((rho_conv, rho_dense))
        identifier = cell_id or rho_cell_id(
            study_id=spec_value.study_id,
            surface_id=surface_id,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
        )
        clean = (rho_conv, rho_dense) != pytest.approx((0.003, 0.01))
        return (
            {
                "cell_id": identifier,
                "rho_conv": rho_conv,
                "rho_dense": rho_dense,
                "safety_clean": bool(clean),
                "status": "complete" if clean else "safety_failure",
            },
            [{"step": 1, "status": "complete"}],
        )

    monkeypatch.setattr(runtime_module, "_run_canary_cell", fake_cell)
    result = runtime_module._execute_core_canary_entry(
        {},
        spec,
        tmp_path,
        tmp_path / "entry",
        {
            "entry_id": "surface",
            "row_id": row["row_id"],
            "optimizer": "sgd",
            "surface_id": "surface",
        },
        data_root=tmp_path,
        download=False,
        device="cpu",
    )
    assert len(result["center_attempts"]) == 2
    assert len(result["cells"]) == 9
    assert sum(cell["reused_from_center_attempt"] for cell in result["cells"]) == 2
    assert len(calls) == 9  # two attempts plus seven genuinely new core cells
    assert calls.count((0.003, 0.01)) == 1


def test_source_shard_and_probe_paths_are_scoped_to_aggregate_root(
    tmp_path: Path,
) -> None:
    assert runtime_module._source_shard_root(
        tmp_path, {"source_shard": "shards/akib"}
    ) == (tmp_path / "shards/akib").resolve()
    expected = (
        tmp_path
        / "shards/trex/stages/optimizer_probe/entries/surface/result.json"
    ).resolve()
    assert runtime_module._probe_path(
        tmp_path,
        {
            "probe_result_path": (
                "shards/trex/stages/optimizer_probe/entries/surface/result.json"
            )
        },
    ) == expected
    implicit = runtime_module._probe_path(
        tmp_path,
        {
            "source_shard": "shards/akib",
            "probe_entry_id": "surface",
        },
    )
    assert implicit == (
        tmp_path
        / "shards/akib/stages/optimizer_probe/entries/surface/result.json"
    ).resolve()
    with pytest.raises(ValueError, match="without traversal"):
        runtime_module._source_shard_root(
            tmp_path, {"source_shard": "shards/../outside"}
        )
    with pytest.raises(ValueError, match="traversal-free"):
        runtime_module._probe_path(
            tmp_path, {"probe_result_path": "../outside/result.json"}
        )
