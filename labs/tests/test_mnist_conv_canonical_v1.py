from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.mnist_conv.backend import ExecutionContext, build_engine_config
from experiments.mnist_conv.cli import main
from experiments.mnist_conv.collection import collect_sweep
from experiments.mnist_conv.identity import run_fingerprint
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.manifest import publish_manifest
from experiments.mnist_conv.runner import ExecutionPolicy, execute_run, validate_bundle
from experiments.mnist_conv.specs import RunSpec, SpecValidationError, SweepSpec


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
PROVENANCE = {
    "git_revision": "1" * 40,
    "dirty_source_digest": "2" * 64,
    "effective_code_fingerprint": "3" * 64,
}


def run_value(*, depth: int = 1, non_linearity: str = "hard_sigmoid", epochs: int = 2) -> dict:
    value = json.loads(EXAMPLE.read_text())
    profile = f"conv{depth}"
    architectures = {
        1: ([64], [2]),
        2: ([64, 128], [2, 2]),
        3: ([64, 128, 256], [2, 2, 1]),
    }
    channels, strides = architectures[depth]
    value["run"]["architecture"].update({
        "profile": profile, "channels": channels,
        "kernel_sizes": [3] * depth, "strides": strides,
        "paddings": [1] * depth,
    })
    value["run"]["model"]["weight_gains"] = [1.0] * (depth + 1)
    value["run"]["training"]["learning_rate"] = [0.01] * (2 * depth + 1)
    value["run"]["training"]["epochs"] = epochs
    value["run"]["training"]["lr_decay"] = 0.5
    if non_linearity == "hard_sigmoid":
        value["run"]["calibration"]["layer_measurements"] = [
            {"layer_index": index, "measured_saturation": 0.3 + 0.01 * (index - 1)}
            for index in range(1, depth + 1)
        ]
    else:
        value["run"]["model"]["non_linearity"] = "perfect_diode"
        value["run"]["model"]["hard_sigmoid_param"] = {}
        value["run"]["calibration"] = {
            "kind": "perfect_diode_clamped_occupancy",
            "calibration_id": "pd-test-calibration",
            "scope": "first_hidden_layer",
            "sample_count": 256,
            "batch_size": 64,
            "model_seed": 0,
            "affine_seed": 1729,
            "settling_iterations": 64,
            "adaptive_equilibrium": False,
            "clamp_epsilon": 1e-8,
            "target_initial_occupancy": 0.3,
            "measured_initial_occupancy": 0.3,
            "layer_measurements": [
                {"layer_index": index, "measured_clamped_occupancy": 0.3 + 0.01 * (index - 1)}
                for index in range(1, depth + 1)
            ],
        }
    return value


def sweep_value(base: dict, *, seeds=(0, 1), three_cases: bool = False) -> dict:
    cases = [{"id": "v1-c1", "set": {}}]
    varying = ["/seed"] if len(seeds) > 1 else []
    if three_cases:
        cases = [
            {"id": "v1-c1", "set": {
                "/run/model/voltage_amp": 1.0, "/run/model/current_amp": 1.0,
                "/run/calibration/calibration_id": "cal-v1-c1",
            }},
            {"id": "v4-c1", "set": {
                "/run/model/voltage_amp": 4.0, "/run/model/current_amp": 1.0,
                "/run/calibration/calibration_id": "cal-v4-c1",
            }},
            {"id": "v4-c0p25", "set": {
                "/run/model/voltage_amp": 4.0, "/run/model/current_amp": 0.25,
                "/run/calibration/calibration_id": "cal-v4-c0p25",
            }},
        ]
        varying += [
            "/run/model/voltage_amp", "/run/model/current_amp",
            "/run/calibration/calibration_id",
        ]
    return {
        "schema_version": "mnist-conv-sweep/v1",
        "name": "test-sweep",
        "base_run": base,
        "axes": [{"path": "/seed", "values": list(seeds)}] if len(seeds) > 1 else [],
        "cases": cases,
        "varying_fields": varying,
        "collection": {
            "expected_seeds": list(seeds),
            "required_cases": [case["id"] for case in cases],
            "group_by": ["/run/model/voltage_amp", "/run/model/current_amp"],
        },
    }


class FakeBackend:
    def __init__(self):
        self.calls: list[tuple[int, Path]] = []

    def __call__(self, *, spec, context, engine_config_path, output_dir):
        engine = json.loads(engine_config_path.read_text())
        assert engine["batch_state_policy"] == "reset_each_batch"
        assert context.initialization_checkpoint_path is None
        self.calls.append((spec.data["seed"], output_dir))
        schema = [{
            "name": "weight 0", "type": "model.variable.parameter.Weight",
            "shape": [2, 2], "dtype": "torch.float32",
        }]
        checkpoint = {
            "format": "drn.function.parameters", "version": 1,
            "schema": schema, "states": [torch.ones((2, 2))],
        }
        torch.save(checkpoint, output_dir / "best_model.pt")
        torch.save(checkpoint, output_dir / "final_model.pt")
        for name in ("weights_best.npz", "weights_final.npz"):
            np.savez(
                output_dir / name,
                weight_0=np.ones((2, 2), dtype=np.float32),
                param_names=np.asarray(["weight_0"]),
                param_types=np.asarray(["Weight"]),
                param_shapes_json=np.asarray(json.dumps([[2, 2]])),
                metadata_json=np.asarray("{}"),
            )
        epochs = spec.data["run"]["training"]["epochs"]
        train_loss = np.linspace(1.0, 0.5, epochs)
        test_loss = np.linspace(0.9, 0.4, epochs)
        train_accuracy = np.linspace(0.5, 0.8, epochs)
        test_accuracy = np.linspace(0.6, 0.9, epochs)
        for filename, values in (
            ("loss_train.npy", train_loss), ("loss_test.npy", test_loss),
            ("accuracy_train.npy", train_accuracy), ("accuracy_test.npy", test_accuracy),
        ):
            np.save(output_dir / filename, values)
        (output_dir / "events.out.tfevents.test").write_bytes(b"event")
        (output_dir / "config.json").write_text(json.dumps({"absolute": str(output_dir.resolve())}))
        metrics = {
            "best_epoch": epochs,
            "best_test_accuracy": float(test_accuracy[-1]),
            "final_test_accuracy": float(test_accuracy[-1]),
            "final_train_loss": float(train_loss[-1]),
            "final_test_loss": float(test_loss[-1]),
            "best_checkpoint_path": str((output_dir / "best_model.pt").resolve()),
            "checkpoint_path": str((output_dir / "final_model.pt").resolve()),
            "weights_best_path": str((output_dir / "weights_best.npz").resolve()),
            "weights_final_path": str((output_dir / "weights_final.npz").resolve()),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics))
        initial_lr = spec.data["run"]["training"]["learning_rate"]
        decay = spec.data["run"]["training"]["lr_decay"]
        return {"learning_rate": [[item * decay**epoch for item in initial_lr] for epoch in range(epochs)]}


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("non_linearity", ["hard_sigmoid", "perfect_diode"])
def test_complete_schema_accepts_only_supported_conv_profiles(depth, non_linearity):
    spec = RunSpec.from_dict(run_value(depth=depth, non_linearity=non_linearity))
    assert spec.data["run"]["architecture"]["profile"] == f"conv{depth}"
    assert spec.data["run"]["model"]["non_linearity"] == non_linearity


def test_model_schema_accepts_bounded_uniform_and_rejects_invalid_initialization():
    bounded = run_value()
    bounded["run"]["model"].update({
        "weight_min": 179e-6,
        "weight_max": 180e-6,
        "weight_init_mode": "bounded_uniform",
    })
    spec = RunSpec.from_dict(bounded)
    assert spec.data["run"]["model"]["weight_init_mode"] == "bounded_uniform"

    invalid_mode = run_value()
    invalid_mode["run"]["model"]["weight_init_mode"] = "kaiming_unifrom"
    with pytest.raises(SpecValidationError, match="weight_init_mode.*one of"):
        RunSpec.from_dict(invalid_mode)

    degenerate = run_value()
    degenerate["run"]["model"].update({
        "weight_min": 180e-6,
        "weight_max": 180e-6,
        "weight_init_mode": "bounded_uniform",
    })
    with pytest.raises(SpecValidationError, match="weight_min.*< weight_max"):
        RunSpec.from_dict(degenerate)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("protocol_id"),
        lambda value: value["run"]["solver"].pop("training_iterations"),
        lambda value: value["run"]["training"].update({"learning_rate": 0.01}),
        lambda value: value["run"]["training"].update({"batch_state_policy": "carry_within_epoch"}),
        lambda value: value["run"]["calibration"].pop("layer_measurements"),
        lambda value: value["run"]["solver"]["minimizer"].update({"overrelaxation_factor": float("nan")}),
        lambda value: value["run"]["architecture"].update({"pooling": {"mode": "avg"}}),
    ],
)
def test_schema_rejects_missing_implicit_or_nonfinite_science(mutation):
    value = run_value()
    mutation(value)
    with pytest.raises(SpecValidationError):
        RunSpec.from_dict(value)


@pytest.mark.parametrize(
    "path,value",
    [
        (("run", "dataset", "name"), "fashion_mnist"),
        (("run", "model", "type"), "dense"),
        (("run", "training", "algorithm"), "EP"),
        (("run", "solver", "adaptive_equilibrium"), True),
    ],
)
def test_schema_rejects_unsupported_dataset_model_algorithm_and_adaptive_solver(path, value):
    config = run_value()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SpecValidationError):
        RunSpec.from_dict(config)


def test_sweep_rejects_duplicate_cases_axis_values_and_resolved_fingerprints(tmp_path):
    duplicate_case = sweep_value(run_value(), seeds=(0,))
    duplicate_case["cases"].append(copy.deepcopy(duplicate_case["cases"][0]))
    duplicate_case["collection"]["required_cases"].append("v1-c1")
    with pytest.raises(SpecValidationError, match="case"):
        SweepSpec.from_dict(duplicate_case)

    duplicate_axis = sweep_value(run_value(), seeds=(0, 1))
    duplicate_axis["axes"][0]["values"] = [0, 0]
    duplicate_axis["collection"]["expected_seeds"] = [0]
    with pytest.raises(SpecValidationError):
        SweepSpec.from_dict(duplicate_axis)

    duplicate_runs = sweep_value(run_value(), seeds=(0,))
    duplicate_runs["cases"] = [
        {"id": "first", "set": {}},
        {"id": "second", "set": {}},
    ]
    duplicate_runs["collection"]["required_cases"] = ["first", "second"]
    sweep = SweepSpec.from_dict(duplicate_runs)
    with pytest.raises(SpecValidationError, match="distinct run identity"):
        publish_manifest(sweep, ResultLayout(tmp_path / "results"), PROVENANCE)


def test_sweep_requires_every_case_and_axis_assignment_to_be_declared(tmp_path):
    case_override = sweep_value(run_value(), seeds=(0, 1), three_cases=True)
    for case in case_override["cases"]:
        case["set"]["/run/model/input_gain"] = 2.0
    with pytest.raises(SpecValidationError, match="undeclared assignment paths"):
        SweepSpec.from_dict(case_override)

    case_override["varying_fields"].append("/run/model/input_gain")
    resolved = SweepSpec.from_dict(case_override)
    assert {item.spec.data["run"]["model"]["input_gain"] for item in resolved.expand()} == {2.0}
    publish_manifest(resolved, ResultLayout(tmp_path / "case-results"), PROVENANCE)

    axis_override = sweep_value(run_value(), seeds=(0,))
    axis_override["axes"] = [{"path": "/replicate_id", "values": ["replicate-a"]}]
    with pytest.raises(SpecValidationError, match="undeclared assignment paths"):
        SweepSpec.from_dict(axis_override)

    axis_override["varying_fields"] = ["/replicate_id"]
    resolved = SweepSpec.from_dict(axis_override)
    assert [item.spec.data["replicate_id"] for item in resolved.expand()] == ["replicate-a"]
    publish_manifest(resolved, ResultLayout(tmp_path / "axis-results"), PROVENANCE)


def test_identity_uses_full_source_replicate_and_checkpoint_sha_but_not_display_paths_or_label():
    base = run_value()
    spec = RunSpec.from_dict(base)
    baseline = run_fingerprint(spec, PROVENANCE)
    runtime_a = ExecutionContext("/datasets/a", device="cpu")
    runtime_b = ExecutionContext("/datasets/b", device="cuda")
    assert runtime_a != runtime_b
    assert run_fingerprint(spec, PROVENANCE) == baseline

    renamed = copy.deepcopy(base); renamed["label"] = "renamed"
    replicated = copy.deepcopy(base); replicated["replicate_id"] = "replicate-a"
    assert run_fingerprint(RunSpec.from_dict(renamed), PROVENANCE) == baseline
    assert run_fingerprint(RunSpec.from_dict(replicated), PROVENANCE) != baseline
    changed_source = {**PROVENANCE, "dirty_source_digest": "4" * 64}
    assert run_fingerprint(spec, changed_source) != baseline

    initialized_a = copy.deepcopy(base)
    initialized_a["run"]["initialization"]["checkpoint"] = {
        "path": "runs/source/checkpoints/best.pt", "sha256": "5" * 64,
        "format": "drn.function.parameters/v1", "source_run_id": None, "role": "best",
    }
    initialized_b = copy.deepcopy(initialized_a)
    initialized_b["run"]["initialization"]["checkpoint"]["path"] = "imports/same.pt"
    initialized_c = copy.deepcopy(initialized_a)
    initialized_c["run"]["initialization"]["checkpoint"]["sha256"] = "6" * 64
    assert run_fingerprint(RunSpec.from_dict(initialized_a), PROVENANCE) == run_fingerprint(RunSpec.from_dict(initialized_b), PROVENANCE)
    assert run_fingerprint(RunSpec.from_dict(initialized_a), PROVENANCE) != run_fingerprint(RunSpec.from_dict(initialized_c), PROVENANCE)


def test_expansion_preserves_case_and_axis_value_order_and_direct_identity():
    value = sweep_value(run_value(), seeds=(2, 0), three_cases=True)
    value["axes"].insert(0, {"path": "/replicate_id", "values": [None, "r1"]})
    value["varying_fields"].append("/replicate_id")
    sweep = SweepSpec.from_dict(value)
    expanded = sweep.expand()
    assert [axis["path"] for axis in sweep.data["axes"]] == ["/replicate_id", "/seed"]
    assert [item.case_id for item in expanded[:4]] == ["v1-c1"] * 4
    assert [item.axes["/seed"] for item in expanded[:4]] == [2, 0, 2, 0]
    assert all(item.spec.data["label"] == value["base_run"]["label"] for item in expanded)
    direct = RunSpec.from_dict(expanded[0].spec.to_dict())
    assert run_fingerprint(direct, PROVENANCE) == run_fingerprint(expanded[0].spec, PROVENANCE)


def test_backend_mapping_uses_canonical_iteration_names_and_explicit_batch_reset(tmp_path):
    spec = RunSpec.from_dict(run_value(depth=2))
    engine = build_engine_config(spec, ExecutionContext(tmp_path, device="cpu"))
    assert engine["batch_state_policy"] == "reset_each_batch"
    assert engine["model_base"]["num_iterations_inference"] == spec.data["run"]["solver"]["inference_iterations"]
    assert engine["model_base"]["num_iterations_training"] == spec.data["run"]["solver"]["training_iterations"]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("non_linearity", ["hard_sigmoid", "perfect_diode"])
def test_legacy_backend_contract_golden_for_all_v1_profiles_with_distinct_t_and_k(
    tmp_path,
    depth,
    non_linearity,
):
    value = run_value(depth=depth, non_linearity=non_linearity)
    value["run"]["solver"]["inference_iterations"] = 7
    value["run"]["solver"]["training_iterations"] = 3
    spec = RunSpec.from_dict(value)
    engine = build_engine_config(spec, ExecutionContext(tmp_path, device="cpu"))

    spatial = {
        1: [[2, 28, 28], [64, 14, 14], [20]],
        2: [[2, 28, 28], [64, 14, 14], [128, 7, 7], [20]],
        3: [[2, 28, 28], [64, 14, 14], [128, 7, 7], [256, 7, 7], [20]],
    }
    assert engine["lab"]["model_key"] == "mnist_resistive_conv_v1"
    assert engine["model_overrides"]["mnist_resistive_conv_v1"]["layer_shapes"] == spatial[depth]
    assert engine["model_base"]["non_linearity"] == non_linearity
    assert engine["model_base"]["num_iterations_inference"] == 7
    assert engine["model_base"]["num_iterations_training"] == 3
    assert engine["optimizer"] == {
        "name": "SGD",
        "learning_rate": [0.01] * (2 * depth + 1),
        "lr_decay": 0.5,
        "momentum": 0.0,
        "weight_decay": 0.0,
    }


def test_manifest_last_bundle_collection_and_decayed_history(tmp_path):
    spec = RunSpec.from_dict(run_value())
    layout, backend = ResultLayout(tmp_path / "results"), FakeBackend()
    result = execute_run(spec, ExecutionContext(tmp_path / "data", device="cpu"), layout, PROVENANCE, backend=backend)
    assert result.status == "complete"
    assert result.run_dir.name == f"{spec.data['label']}--{result.run_id}"
    assert validate_bundle(result.run_dir, result.run_id)["state"] == "complete"
    assert (result.run_dir / "logs/tensorboard/events.out.tfevents.test").is_file()
    assert not (result.run_dir / "logs/backend-config.json").exists()
    rows = (result.run_dir / "history.csv").read_text().splitlines()
    assert '"[0.005,0.005,0.005]"' in rows[-1]
    npz = np.load(result.run_dir / "weights/best.npz", allow_pickle=False)
    assert npz["param_types"].tolist() == ["model.variable.parameter.Weight"]
    assert execute_run(spec, ExecutionContext(tmp_path / "other-data"), layout, PROVENANCE, backend=backend).status == "already_complete"
    assert len(backend.calls) == 1

    sweep = SweepSpec.from_dict(sweep_value(spec.to_dict(), seeds=(0,)))
    manifest, manifest_path = publish_manifest(sweep, layout, PROVENANCE)
    assert manifest["entries"][0]["run_id"] == result.run_id
    assert (manifest_path.parent / "sweep.resolved.json").is_file()
    assert (manifest_path.parent / "jobs.jsonl").is_file()
    assert not (manifest_path.parent / "summary.csv").exists()
    collection = collect_sweep(manifest_path, layout)
    assert collection["complete"] is True
    assert collection["paper_eligible"] is False
    assert "protocol_not_approved" in collection["rows"][0]["eligibility_reasons"]
    assert (manifest_path.parent / "summary.csv").is_file()
    assert (manifest_path.parent / "summary_by_case.csv").is_file()


def test_cli_shards_are_disjoint_exact_union_and_workers_do_not_collect(tmp_path, capsys):
    config = tmp_path / "sweep.json"
    config.write_text(json.dumps(sweep_value(run_value(epochs=1), three_cases=True)))
    results = tmp_path / "results"
    backend = FakeBackend()
    common = [
        "sweep", "--config", str(config), "--results-root", str(results),
        "--data-root", str(tmp_path / "data"), "--device", "cpu",
        "--executor", "local", "--workers", "2", "--shard-count", "2",
    ]
    assert main(common + ["--shard-index", "0"], backend=backend, source_provenance=PROVENANCE) == 0
    even = json.loads(capsys.readouterr().out)["selected_job_indices"]
    assert main(common + ["--shard-index", "1"], backend=backend, source_provenance=PROVENANCE) == 0
    odd = json.loads(capsys.readouterr().out)["selected_job_indices"]
    assert set(even).isdisjoint(odd)
    assert sorted(even + odd) == list(range(6))
    sweep_dirs = list((results / "sweeps").iterdir())
    assert len(sweep_dirs) == 1
    assert not (sweep_dirs[0] / "summary.csv").exists()
    assert len(list((results / "runs").iterdir())) == 6
    assert main(["collect", "--sweep", str(sweep_dirs[0])], source_provenance=PROVENANCE) == 0
    collection = json.loads(capsys.readouterr().out)
    assert collection["complete"] is True
    assert collection["paper_eligible"] is False


def test_internal_slurm_job_index_and_direct_run_share_identity_and_bundle(tmp_path, capsys):
    layout = ResultLayout(tmp_path / "results")
    sweep = SweepSpec.from_dict(sweep_value(run_value(epochs=1), seeds=(0,)))
    manifest, manifest_path = publish_manifest(sweep, layout, PROVENANCE)
    run_config = tmp_path / "run.json"
    run_config.write_text(json.dumps(manifest["entries"][0]["run_spec"]))
    common = [
        "--results-root", str(layout.root),
        "--data-root", str(tmp_path / "data"),
        "--device", "cpu",
    ]
    backend = FakeBackend()

    assert main([
        "sweep", "--manifest", str(manifest_path), "--job-index", "0", *common,
    ], backend=backend, source_provenance=PROVENANCE) == 0
    worker = json.loads(capsys.readouterr().out)
    assert main([
        "run", "--config", str(run_config), *common,
    ], backend=backend, source_provenance=PROVENANCE) == 0
    direct = json.loads(capsys.readouterr().out)

    assert worker["run_id"] == direct["run_id"] == manifest["entries"][0]["run_id"]
    assert worker["status"] == "complete"
    assert direct["status"] == "already_complete"
    assert len(backend.calls) == 1


def test_pruned_attempt_is_terminal_and_never_published(tmp_path):
    def pruned_backend(**kwargs):
        output_dir = kwargs["output_dir"]
        checkpoint = {
            "format": "drn.function.parameters", "version": 1,
            "schema": [{
                "name": "weight 0", "type": "model.variable.parameter.Weight",
                "shape": [2, 2], "dtype": "torch.float32",
            }],
            "states": [torch.ones((2, 2))],
        }
        torch.save(checkpoint, output_dir / "best_model.pt")
        np.savez(
            output_dir / "weights_best.npz",
            weight_0=np.ones((2, 2), dtype=np.float32),
            param_names=np.asarray(["weight_0"]),
            param_types=np.asarray(["Weight"]),
            param_shapes_json=np.asarray(json.dumps([[2, 2]])),
            metadata_json=np.asarray("{}"),
        )
        return {"status": "pruned", "epoch": 1}

    spec, layout = RunSpec.from_dict(run_value()), ResultLayout(tmp_path / "results")
    first = execute_run(spec, ExecutionContext(tmp_path / "data"), layout, PROVENANCE, backend=pruned_backend)
    second = execute_run(
        spec, ExecutionContext(tmp_path / "data"), layout, PROVENANCE,
        backend=pytest.fail, policy=ExecutionPolicy(retry_failed=True),
    )
    assert first.status == second.status == "pruned"
    assert layout.find_run_dir(first.run_id) is None
    assert (layout.attempt_dir(first.run_id, first.attempt_id) / ".staging").is_dir()
