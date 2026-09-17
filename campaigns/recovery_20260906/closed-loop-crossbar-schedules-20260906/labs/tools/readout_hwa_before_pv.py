"""Read-only HWA test readout for the exact September 7 recovery sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/hwa_pre_pv_20260908"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_input(record):
    path = Path(record["path"])
    assert digest(path) == record["sha256"], path
    return path


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_replay(observed, expected):
    for key in ("examples", "student_correct", "teacher_correct"):
        assert observed[key] == expected[key], (key, observed[key], expected[key])
    assert abs(observed["kl_teacher_student"] - expected["kl_teacher_student"]) < 1e-7
    return {"passed": True, "expected_KL": expected["kl_teacher_student"],
            "observed_KL": observed["kl_teacher_student"], "KL_tolerance": 1e-7}


def read_crossbar():
    import torch
    sys.path.insert(0, str(ROOT / "crossbar_code"))
    from experiments.mnist_analog_relu import staged_runtime as engine
    from experiments.mnist_analog_relu import runtime as forward
    from experiments.mnist_analog_relu.staged_config import (
        parse_staged_crossbar_config, resolve_staged_crossbar_spec,
    )
    from experiments.schema import RunMode

    source_result = ROOT / "runs/healthy-exponential-open_loop/result.json"
    recovery = read_json(source_result)
    source = read_json(ROOT / "plan.json")["crossbar"]
    inputs = dict(recovery["manifest"]["inputs"])
    p0_path = checked_input(inputs["deployment"])
    teacher_path = checked_input(inputs["teacher"])
    origin = engine.load_device_state(p0_path)
    hwa_path = p0_path.parent / "hwa_master.pt"
    inputs["hwa_master"] = {"path": str(hwa_path), "sha256": origin.source_artifact_sha256}
    checked_input(inputs["hwa_master"])
    master = engine.load_hwa_master(hwa_path)
    assert master.teacher_sha256 == inputs["teacher"]["sha256"]
    assert master.digital_scales == origin.current.digital_scales
    assert master.report["hwa"]["fixed_final_epoch"] == 10
    faulted = engine.load_device_state(Path(source["faulted_p0"]))
    assert faulted.source_artifact_sha256 == origin.source_artifact_sha256
    assert faulted.teacher_sha256 == origin.teacher_sha256
    spec = resolve_staged_crossbar_spec(
        parse_staged_crossbar_config(read_json(source["config"])), RunMode.TRAIN)
    device = torch.device("cuda:0")
    teacher, _, teacher_sha = engine._load_teacher(teacher_path, spec=spec, device=device)
    assert teacher_sha == master.teacher_sha256
    data = engine.build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
    layout = engine._layout(spec)
    assert tuple(layout) == tuple(master.layout)
    population = origin.healthy_population
    raw = master.fixed_final_master_q.to(device)
    base = torch.maximum(torch.minimum(raw, population.logical_max.to(device)),
                         population.logical_min.to(device))
    noise = master.report["hwa"]["forward_noise"]
    evaluation_id = "epoch_010.test"
    seed = forward.derive_seed(noise["configured_seed"], population.assignment_seed,
                               "offchip_stochastic_apparent_hwa_held_evaluation", evaluation_id)
    evaluation = forward._evaluate_held_apparent_hwa_state(
        support_clamped_digital_master_q=base, digital_scales=master.digital_scales,
        layout=layout, teacher=teacher, loader=data.test, device=device,
        maximum_batches=None, sample_limit=None, nominal_dw_min=population.nominal_dw_min,
        write_noise_std=population.write_noise_std, relative_scale=noise["relative_scale"],
        generator=torch.Generator(device=device).manual_seed(seed),
        evaluation_id=evaluation_id, resolved_seed=seed)
    raw_evaluation = forward._evaluate(
        effective_state=raw, digital_scales=master.digital_scales, layout=layout,
        teacher=teacher, loader=data.test, device=device, maximum_batches=None, sample_limit=None)
    plant = engine._restore_current(origin, layout=layout, device=device)
    p0_before = {k: tensor_digest(v) for k, v in
                 {"apparent": plant.apparent, "persistent": plant.persistent}.items()}
    p0 = forward._evaluate(effective_state=plant.apparent,
        digital_scales=master.digital_scales, layout=layout, teacher=teacher,
        loader=data.test, device=device, maximum_batches=None, sample_limit=None)
    assert p0_before == {k: tensor_digest(v) for k, v in
                        {"apparent": plant.apparent, "persistent": plant.persistent}.items()}
    replay = check_replay(p0, recovery["initial_test"]["apparent"])
    primary = evaluation["apparent_forward"]
    assert primary["teacher_prediction_sha256"] == p0["teacher_prediction_sha256"]
    return dict(
        primary=primary, evaluation=evaluation,
        unmodified_digital_master_diagnostic=raw_evaluation,
        primary_state="held_apparent_HWA_surrogate_on_recovery_target_array",
        inputs=inputs, source_result=str(source_result),
        source_result_sha256=digest(source_result), p0_replay=replay,
        config={"path": source["config"], "sha256": digest(source["config"])},
        primary_forward="support-clamped final HWA master plus one held HWA noise draw; no P&V",
        hwa_selected_epoch=10, hwa_selection="fixed_final_epoch",
        digital_scales=list(master.digital_scales),
        hwa_training_assignment=master.metadata["assignment_seed"],
        readout_assignment=population.assignment_seed,
        readout_population_fingerprint=population.fingerprint,
        support_clamped_cells=int(torch.count_nonzero(base != raw).item()),
        seeds={"data": spec.runtime.data_seed, "configured_hwa_noise": noise["configured_seed"],
               "resolved_hwa_evaluation": seed},
        master_tensor_sha256=tensor_digest(raw),
        code_files=[forward.__file__, engine.__file__],
    )


def read_drn():
    import torch
    sys.path.insert(0, str(ROOT.parent / "ibm-om-cell-aware-quantized"))
    from experiments.mnist_relu_drn import figure6_om_256_ladder as engine

    recovery_root = ROOT.parent / "closed-loop-recovery-20260906"
    source_result = recovery_root / "runs/drn-healthy-closed_loop_pv/result.json"
    recovery = read_json(source_result)
    source = read_json(recovery_root / "plan.json")["drn"]
    inputs = dict(recovery["manifest"]["inputs"])
    deployment_path = checked_input(inputs["deployment"])
    manifest_path = deployment_path.parent.parent / "manifest.json"
    manifest = read_json(manifest_path)
    hwa_record = next(r for r in manifest["inputs"] if r["role"] == "HWA_master")
    inputs["hwa_master"] = {k: hwa_record[k] for k in ("path", "sha256")}
    for record in inputs.values():
        checked_input(record)
    deployment = engine._load_torch(deployment_path, schema="ebl.figure6_om_256_deployment_replica")
    prepared = engine._load_torch(Path(source["prepared"]), schema="ebl.figure6_om_256_ladder_prepared")
    hwa = engine._load_torch(Path(hwa_record["path"]), schema="ebl.figure6_om_256_hwa_master")
    assert prepared["teacher"]["sha256"] == inputs["teacher"]["sha256"]
    assert hwa["fixed_logit_gain"] == deployment["fixed_logit_gain"] == prepared["fixed_logit_gain"]
    for a, b in zip(hwa["selected_master"], deployment["targets"]["hwa"]["logical_master"], strict=True):
        assert torch.equal(a, b)
    config = read_json(source["config"])
    spec, teacher, runtime = engine._runtime(config,
        teacher_path=Path(inputs["teacher"]["path"]), gain=float(prepared["fixed_logit_gain"]), smoke=False)
    stack = runtime["stack"]
    field = engine._field_from_payload(deployment["endpoint_field"], device=stack.device)
    masters = engine._masters_to_device(hwa["selected_master"], stack.device)
    data = engine._loaders(spec, data_seed=int(deployment["seeds"]["data_order"]))
    full_g = engine.map_masters_to_conductance(masters, field)
    target_progress = engine.master_to_progress(masters, field)
    for a, b in zip(target_progress, deployment["targets"]["hwa"]["target_progress"], strict=True):
        assert torch.equal(a.cpu(), b)
    engine._apply_full_g(stack, full_g)
    primary, _ = engine._evaluate_detailed(stack, teacher, data.test, sample_limit=None)
    population = engine.load_om_array_population(Path(source["population"]))
    plant = engine._plant_from_state(field=field, population=population,
        state=deployment["targets"]["hwa"]["clean_adam_start"])
    state_before = [tensor_digest(v) for v in (plant.raw_a, plant.apparent_raw_a)]
    engine._apply_full_g(stack, plant.apparent_full_conductance)
    p0, _ = engine._evaluate_detailed(stack, teacher, data.test, sample_limit=None)
    assert state_before == [tensor_digest(v) for v in (plant.raw_a, plant.apparent_raw_a)]
    replay = check_replay(p0, recovery["initial_test"]["apparent"])
    assert primary["teacher_correct"] == p0["teacher_correct"]
    assert primary["fixed_logit_gain"] == p0["fixed_logit_gain"]
    return dict(
        primary=primary, primary_state="held_apparent_endpoint_view_continuous_HWA_target",
        inputs=inputs, source_result=str(source_result), source_result_sha256=digest(source_result),
        deployment_manifest={"path": str(manifest_path), "sha256": digest(manifest_path)},
        config={"path": source["config"], "sha256": digest(source["config"])},
        p0_replay=replay, primary_forward="saved HWA master mapped through the exact recovery endpoint field; no P&V",
        hwa_selected_epoch=hwa["report"]["selected_epoch"], hwa_selection=hwa["report"]["selection"],
        fixed_logit_gain=prepared["fixed_logit_gain"],
        readout_assignment=deployment["seeds"]["OM_assignment"],
        readout_population_fingerprint=population.fingerprint,
        seeds={"data": deployment["seeds"]["data_order"], "endpoint": deployment["seeds"]["endpoint"]},
        master_tensor_sha256=[tensor_digest(v) for v in masters],
        requested_conductance_sha256=[tensor_digest(v) for v in full_g],
        target_progress_matches_deployment=True,
        code_files=[engine.__file__, sys.modules[engine._evaluate_detailed.__module__].__file__,
                    sys.modules[engine.map_masters_to_conductance.__module__].__file__],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("crossbar", "drn"), required=True)
    args = parser.parse_args()
    output = OUT / f"{args.architecture}_readout.json"
    if output.exists():
        raise FileExistsError(output)
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start = time.time()
    result = (read_crossbar if args.architecture == "crossbar" else read_drn)()
    assert result["primary"]["examples"] == 10000
    for record in result["inputs"].values():
        checked_input(record)
    code_paths = [Path(p) for p in result.pop("code_files")] + [Path(__file__)]
    result.update(schema="ebl.hwa_pre_pv_readout.v1", status="complete", architecture=args.architecture,
        split="official_MNIST_test", shared_conditions=["healthy", "faulted"],
        evidence_class="exploratory_noncanonical_model_based", writes=0, optimizer_updates=0,
        selection_performed=False, calibration_refit=False, persistent_device_state_present=False,
        input_files_unchanged=True, host_cuda_canary_passed=True,
        cuda_device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        code_sha256={str(p): digest(p) for p in code_paths},
        protocol_sha256=digest(OUT / "readout_protocol.md"), elapsed_seconds=time.time() - start)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "KL": result["primary"]["kl_teacher_student"],
                      "accuracy": result["primary"]["student_accuracy"], "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
