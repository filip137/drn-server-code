"""Mean KL of the frozen stochastic HWA training models, before deployment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/stochastic_hwa_predeployment_20260908"
VIEWS = 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def status(architecture, **extra):
    path = OUT / f"{architecture}_status.json"
    record = dict(architecture=architecture, pid=os.getpid(), heartbeat=time.time(), **extra)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def cached_cohort(loader):
    batches = list(loader)
    h = hashlib.sha256()
    for inputs, labels in batches:
        h.update(inputs.numpy().tobytes())
        h.update(labels.numpy().tobytes())
    return batches, h.hexdigest()


def check_inputs(records):
    for record in records.values():
        assert digest(record["path"]) == record["sha256"], record["path"]


def read_crossbar(old):
    import torch
    sys.path.insert(0, str(ROOT / "crossbar_code"))
    from experiments.mnist_analog_relu import staged_runtime as engine
    from experiments.mnist_analog_relu import runtime as forward
    from experiments.mnist_analog_relu.staged_config import parse_staged_crossbar_config, resolve_staged_crossbar_spec
    from experiments.schema import RunMode
    from training.ibm_reram_hwa import load_om_array_population

    inputs = {k: v for k, v in old["inputs"].items() if k in ("teacher", "hwa_master")}
    population_path = OUT / "inputs/hwa_healthy_assignment_sampled_population.npz"
    inputs["hwa_training_population"] = {"path": str(population_path),
        "sha256": "e8e880e00e947a305a19ac0770d52a25ba5816bdb3b13aae2373f79abea6edba"}
    check_inputs(inputs)
    master = engine.load_hwa_master(Path(inputs["hwa_master"]["path"]))
    population = load_om_array_population(population_path)
    assert population.fingerprint == master.hwa_population_fingerprint
    assert population.assignment_seed == master.metadata["assignment_seed"] == 2090401
    spec = resolve_staged_crossbar_spec(parse_staged_crossbar_config(read_json(old["config"]["path"])), RunMode.TRAIN)
    device = torch.device("cuda:0")
    teacher, _, teacher_hash = engine._load_teacher(Path(inputs["teacher"]["path"]), spec=spec, device=device)
    assert teacher_hash == master.teacher_sha256
    data = engine.build_mnist_loaders(spec.data, data_seed=spec.runtime.data_seed)
    test, test_hash = cached_cohort(data.test)
    validation, validation_hash = cached_cohort(data.validation)
    raw = master.fixed_final_master_q.to(device)
    base = torch.maximum(torch.minimum(raw, population.logical_max.to(device)), population.logical_min.to(device))
    hwa = master.report["hwa"]
    assert forward.tensor_sha256(base) == hwa["fixed_final_realized_sha256"]
    noise = hwa["forward_noise"]
    common = dict(digital_scales=master.digital_scales, layout=master.layout,
        teacher=teacher, device=device, maximum_batches=None, sample_limit=None)

    def evaluate_draw(evaluation_id, loader):
        seed = forward.derive_seed(noise["configured_seed"], population.assignment_seed,
            "offchip_stochastic_apparent_hwa_held_evaluation", evaluation_id)
        apparent, draw_noise = forward._sample_aihwkit_om_apparent_write_noise(
            persistent_q=base, nominal_dw_min=population.nominal_dw_min,
            write_noise_std=population.write_noise_std, relative_scale=noise["relative_scale"],
            generator=torch.Generator(device=device).manual_seed(seed))
        metrics = forward._evaluate(effective_state=apparent, loader=loader, **common)
        return dict(evaluation_id=evaluation_id, seed=seed, metrics=metrics,
            apparent_sha256=forward.tensor_sha256(apparent), noise_sha256=forward.tensor_sha256(draw_noise))

    replay = evaluate_draw("epoch_010.validation", validation)
    expected = hwa["fixed_final_validation"]["apparent_forward"]
    expected_state = hwa["fixed_final_validation"]["held_apparent_state_receipt"]
    # A seed does not promise identical CUDA random tensors on another GPU.
    # Verify the exact original support-clamped model and its saved validation
    # predictions; retain both stochastic readouts explicitly for the audit.
    assert replay["seed"] == expected_state["resolved_seed"]
    base_validation = forward._evaluate(effective_state=base, loader=validation, **common)
    expected_base = hwa["fixed_final_validation"]["nonpersistent_support_clamped_digital_master_diagnostic"]
    assert abs(base_validation["kl_teacher_student"] - expected_base["kl_teacher_student"]) < 1e-7
    assert base_validation["student_prediction_sha256"] == expected_base["student_prediction_sha256"]
    views = []
    for index in range(VIEWS):
        views.append(evaluate_draw(f"frozen_stochastic_model.test.draw_{index:03d}", test))
        if (index + 1) % 8 == 0:
            status("crossbar", status="running", completed_views=index + 1, required_views=VIEWS)
            print(f"crossbar: {index + 1}/{VIEWS} test noise draws", flush=True)
    nominal = forward._evaluate(effective_state=raw, loader=test, **common)
    clamped = forward._evaluate(effective_state=base, loader=test, **common)
    return dict(inputs=inputs, views=views, primary_state="stochastic_HWA_model_original_training_population",
        validation_replay=dict(passed=True, state="original_support_clamped_master_diagnostic",
            expected_KL=expected_base["kl_teacher_student"], observed_KL=base_validation["kl_teacher_student"],
            exact_support_clamped_master_hash=True, predictions_match=True),
        original_stochastic_validation_KL=expected["kl_teacher_student"],
        stochastic_validation_seed_replay=dict(expected_KL=expected["kl_teacher_student"],
            observed_KL=replay["metrics"]["kl_teacher_student"], seed_matches=True,
            exact_apparent_state_hash=replay["apparent_sha256"] == expected_state["held_apparent_q_sha256"],
            note="Original Akib and local CUDA draws differ despite the same seed; this is a distributional evaluation, not bitwise RNG replay."),
        validation_readout=replay, unmodified_digital_master_diagnostic=nominal,
        support_clamped_digital_master_diagnostic=clamped,
        test_cohort_sha256=test_hash, validation_cohort_sha256=validation_hash,
        hwa_selected_epoch=10, hwa_training_assignment=population.assignment_seed,
        readout_population_fingerprint=population.fingerprint, sigma_q=noise["sigma_q"],
        support_clamped_cells=int(torch.count_nonzero(base != raw)),
        digital_scales=list(master.digital_scales),
        stochastic_integration="Monte Carlo mean of KL over 64 independent held Gaussian HWA views",
        independent_views=True, code_files=[engine.__file__, forward.__file__])


def read_drn(old):
    import torch
    sys.path.insert(0, str(ROOT.parent / "ibm-om-cell-aware-quantized"))
    from experiments.mnist_relu_drn import figure6_om_256_ladder as engine
    inputs = {k: v for k, v in old["inputs"].items() if k in ("teacher", "hwa_master", "prepared")}
    check_inputs(inputs)
    hwa = engine._load_torch(Path(inputs["hwa_master"]["path"]), schema="ebl.figure6_om_256_hwa_master")
    prepared = engine._load_torch(Path(inputs["prepared"]["path"]), schema="ebl.figure6_om_256_ladder_prepared")
    config = read_json(old["config"]["path"])
    spec, teacher, runtime = engine._runtime(config,
        teacher_path=Path(inputs["teacher"]["path"]), gain=prepared["fixed_logit_gain"], smoke=False)
    stack = runtime["stack"]
    masters = engine._masters_to_device(hwa["selected_master"], stack.device)
    data = engine._loaders(spec, data_seed=42)
    test, test_hash = cached_cohort(data.test)
    validation, validation_hash = cached_cohort(data.validation)
    fields = tuple(engine._field(seed, device=stack.device) for seed in config["endpoint_model"]["development_seeds"])
    replay = engine._macro_evaluate_masters(stack=stack, teacher=teacher,
        loader=validation, masters=masters, fields=fields, sample_limit=None)
    expected = hwa["report"]["selected_validation"]
    assert abs(replay["macro_kl_teacher_student"] - expected["macro_kl_teacher_student"]) < 1e-7
    for observed, saved in zip(replay["endpoint_evaluations"], expected["endpoint_evaluations"], strict=True):
        assert abs(observed["kl_teacher_student"] - saved["kl_teacher_student"]) < 1e-7
        assert observed["student_accuracy"] == saved["student_accuracy"]
    bank = engine._augmented_hwa_bank(config, device=stack.device, smoke=False)
    assert len(bank) == VIEWS == config["hwa"]["effective_bank_size"]
    views = []
    train_seeds = config["endpoint_model"]["train_seeds"]
    for index, field in enumerate(bank):
        full_g = engine.map_masters_to_conductance(masters, field)
        engine._apply_full_g(stack, full_g)
        metrics, _ = engine._evaluate_detailed(stack, teacher, test, sample_limit=None)
        views.append(dict(bank_index=index, rotation_index=index // len(train_seeds),
            base_endpoint_seed=train_seeds[index % len(train_seeds)], metrics=metrics,
            full_conductance_sha256=[tensor_digest(v) for v in full_g]))
        if (index + 1) % 8 == 0:
            status("drn", status="running", completed_views=index + 1, required_views=VIEWS)
            print(f"drn: {index + 1}/{VIEWS} test HWA bank views", flush=True)
    return dict(inputs=inputs, views=views, primary_state="stochastic_HWA_model_actual_64_view_training_bank",
        validation_replay=dict(passed=True, expected_KL=expected["macro_kl_teacher_student"],
            observed_KL=replay["macro_kl_teacher_student"], per_endpoint_replay_passed=True),
        validation_readout=replay, test_cohort_sha256=test_hash, validation_cohort_sha256=validation_hash,
        hwa_selected_epoch=hwa["report"]["selected_epoch"], fixed_logit_gain=prepared["fixed_logit_gain"],
        endpoint_training_seeds=train_seeds, rotations_per_population=config["hwa"]["rotations_per_population"],
        stochastic_integration="Uniform mean of KL over all 64 endpoint views in the actual HWA training bank",
        independent_views=False, code_files=[engine.__file__,
            sys.modules[engine._evaluate_detailed.__module__].__file__,
            sys.modules[engine.map_masters_to_conductance.__module__].__file__])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("crossbar", "drn"), required=True)
    args = parser.parse_args()
    path = OUT / f"{args.architecture}_readout.json"
    if path.exists():
        raise FileExistsError(path)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    start = time.time()
    status(args.architecture, status="initializing", required_views=VIEWS)
    old_path = ROOT / f"analysis/hwa_pre_pv_20260908/{args.architecture}_readout.json"
    old = read_json(old_path)
    try:
        result = (read_crossbar if args.architecture == "crossbar" else read_drn)(old)
        views = result["views"]
        assert len(views) == VIEWS
        assert all(v["metrics"]["examples"] == 10000 for v in views)
        kl = np.array([v["metrics"]["kl_teacher_student"] for v in views])
        assert np.all(np.isfinite(kl)) and np.all(kl >= 0)
        check_inputs(result["inputs"])
        result.update(schema="ebl.stochastic_hwa_predeployment.v1", status="complete",
            architecture=args.architecture, split="official_MNIST_test",
            primary=dict(examples=10000, stochastic_views=VIEWS,
                kl_teacher_student=float(kl.mean()), kl_sd_across_views=float(kl.std(ddof=0)),
                kl_minimum=float(kl.min()), kl_maximum=float(kl.max()),
                student_accuracy=float(np.mean([v["metrics"]["student_accuracy"] for v in views])),
                teacher_accuracy=views[0]["metrics"]["teacher_accuracy"]),
            shared_conditions=["healthy", "faulted"],
            evidence_class="exploratory_noncanonical_model_based", writes=0, optimizer_updates=0,
            deployment_array_used=False, persistent_device_state_present=False,
            selection_performed=False, calibration_refit=False, input_files_unchanged=True,
            original_config=old["config"], p0_replay=old["p0_replay"],
            matched_recovery_source_audit={"path": str(old_path), "sha256": digest(old_path)},
            protocol_sha256=digest(OUT / "readout_protocol.md"),
            cuda_device=torch.cuda.get_device_name(), torch_version=torch.__version__,
            elapsed_seconds=time.time() - start)
        result["code_sha256"] = {str(p): digest(p) for p in [Path(__file__), *map(Path, result.pop("code_files"))]}
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        status(args.architecture, status="complete", completed_views=VIEWS, result=str(path))
        print(json.dumps({"path": str(path), "primary": result["primary"], "seconds": result["elapsed_seconds"]}), flush=True)
    except BaseException as error:
        status(args.architecture, status="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
