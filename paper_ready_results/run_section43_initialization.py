#!/usr/bin/env python3
"""Matched Conv3 baseline epoch-0 versus epoch-30 gradient/noise diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import run_section43_mechanism as replay

HERE, ROOT = replay.HERE, replay.ROOT


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def initialization_case(config, trained_case, shadow):
    from labs.mnist_train import _param_schema, _parameter_state_sha256 as training_hash
    source = config["initializer"]
    path = Path(source["checkpoint_path"])
    assert replay.digest(path) == source["checkpoint_sha256"]
    saved = shadow.torch.load(path, map_location="cpu", weights_only=False)
    params = [SimpleNamespace(name=spec["name"], state=tensor) for spec, tensor
              in zip(saved["schema"], saved["states"], strict=True)]
    # The training and replay digests use different schema prefixes.
    assert training_hash(_param_schema(params)) == source["float32_parameter_state_sha256"]
    assert all(p.state.dtype == shadow.torch.float32 for p in params)
    assert all(not p.state.count_nonzero() for p in params if p.name.startswith("Bias_"))
    promoted = [SimpleNamespace(name=p.name, state=p.state.to(shadow.torch.float64)) for p in params]
    training_metrics = json.loads((Path(trained_case["run_dir"])/"metrics.json").read_text())
    assert training_hash(_param_schema(promoted)) == training_metrics["initial_parameter_state_sha256"]
    case = dict(trained_case)
    case["source_config"] = {**case["source_config"], "init_checkpoint_path": str(path)}
    case["native_checkpoint_dtype"] = "float32"
    case["checkpoint_role"] = "reconstructed_initialization"
    case["reconstructed_initialization_tensor_sha256"] = shadow.base._historical_initialization_sha256(params)
    case["replay_initial_float32_tensor_sha256"] = shadow.base._parameter_state_sha256(params)
    case["replay_initial_float64_tensor_sha256"] = shadow.base._parameter_state_sha256(promoted)
    return case


def checkpoint_statistics(output, epoch, batch_index, gradient_sink, voltage_sink, initial=None):
    import numpy as np
    vectors = {"eqprop": output["gradients"], "bptt": output["bptt_gradients"]}
    for estimator, gradients in vectors.items():
        for layer, (name, tensor) in enumerate(gradients.items()):
            value = tensor.numpy()
            norm = float(np.sqrt(np.sum(value*value)))
            reference = initial[estimator][name].numpy() if initial is not None else value
            metrics = replay.compare_vectors(value, reference)
            gradient_sink.add({"epoch": epoch, "checkpoint_role": "saved_initialization" if epoch == 0 else "best_validation",
                               "batch_index": batch_index, "estimator": estimator, "layer_index": layer, "parameter": name,
                               "element_count": value.size, "gradient_l2": norm, "gradient_rms": norm/np.sqrt(value.size),
                               "abs_le_1e_minus12_fraction": float(np.mean(np.abs(value) <= 1e-12)),
                               "exact_zero_fraction": float(np.mean(value == 0)),
                               "norm_relative_to_initial": metrics["norm_ratio"],
                               "cosine_to_initial_same_batch": metrics["cosine"],
                               "relative_l2_error_to_initial": metrics["relative_l2_error"]})
    states = {"post_T_free": output["captured_post_t_states"], "zero": output["captured_zero_states"],
              **output["captured_phase_states"]}
    for phase, values in states.items():
        for layer, tensor in enumerate(values):
            x = tensor.numpy()
            residual = next(r for r in output["residual_rows"] if r["phase"] == phase and int(r["layer_index"]) == layer)
            voltage_sink.add({"epoch": epoch, "batch_index": batch_index, "phase": phase, "layer_index": layer,
                              "state_layer": f"H{layer+1}" if layer < 3 else "Output", "element_count": x.size,
                              "state_sum": float(np.sum(x)), "state_squared_sum": float(np.sum(x*x)),
                              "state_mean": float(np.mean(x)), "state_rms": float(np.sqrt(np.mean(x*x))),
                              "state_std": float(np.std(x)), "state_min": float(np.min(x)), "state_max": float(np.max(x)),
                              "clamp_occupancy": residual["clamp_occupancy_mean"]})
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"section43_initialization_config.json")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shadow = replay.load_shadow(args.config)
    import experiments.reporting as reporting
    torch = shadow.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained = replay.inventory(config, shadow)[0]
    initial = initialization_case(config, trained, shadow)
    previous = Path(config["trained_reference"]["run"])
    assert not reporting.validate_run(previous)
    prior_clean = {(int(r["batch_index"]), r["parameter"]): r for r in read_csv(previous/"gradients.csv")
                   if r["architecture"] == "conv3" and r["scheme"] == "baseline" and float(r["sigma"]) == 0}
    prior_guards = {r["batch_index"]: r for r in map(json.loads, (previous/"batch_guards.jsonl").read_text().splitlines())
                    if r["architecture"] == "conv3" and r["scheme"] == "baseline"}
    dataset = config["dataset"]
    batches, cohort = shadow.base._build_validation_cohort(trained["source_config"], data_root=Path(dataset["root"]),
                           batch_size=16, example_count=576, source_indices=dataset["source_indices"],
                           excluded_source_indices=dataset["excluded_source_indices"])
    assert cohort["validation_indices_sha256"] == dataset["validation_indices_sha256"]
    for batch, expected in zip(batches, dataset["expected_batch_guards"], strict=True):
        assert all(batch[k] == value for k, value in expected.items())
    study = ROOT/"results"/config["study_id"]
    assert study.is_dir()
    if args.mode == "smoke":
        batches = batches[:1]
    else:
        assert not reporting.validate_run(study/"smoke")
        assert json.loads((study/"smoke/result.json").read_text())["terminal_metrics"]["passed"]
    used = sum(json.loads(p.read_text())["gpu_seconds"] for p in study.glob("*/usage.json"))
    remaining = config["execution"]["gpu_seconds_cap_including_smokes"]-used
    assert remaining > 0
    start = time.time()
    deadline = start+remaining
    run = study/args.mode
    sources = [Path(__file__), HERE/"run_section43_mechanism.py", args.config,
               *[ROOT/"experiments"/n for n in ("audit_eqprop_float64_shadow.py", "analyze_conv_eqprop_bptt_beta_tk_displacement.py",
                                                "analyze_conv_eqprop_bptt_checkpoint_gradients.py", "reporting.py")]]
    reporting.start_run(run, {"study_id": config["study_id"], "run_id": args.mode, "arm_id": "conv3_baseline_init_vs_trained",
                      "smoke": args.mode == "smoke", "evidence_class": "ordinary_mnist_validation_mechanism",
                      "configuration": config, "dataset": dataset, "command": [sys.executable, *sys.argv],
                      "git": {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                              "dirty": True, "exact_source_sha256": {str(p): replay.digest(p) for p in sources}},
                      "runtime": {"host": socket.gethostname(), "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
                                  "torch": torch.__version__, "python": sys.version, "remaining_gpu_seconds": remaining}})
    replay.write_json(run/"config.used.json", config)
    replay.write_json(run/"cohort.json", cohort)
    (run/"replay_source").mkdir()
    for p in sources:
        shutil.copyfile(p, run/"replay_source"/p.name)
    sinks = {key: replay.Rows(run/f"{key}.csv") for key in ("gradients", "displacement", "residuals", "phase_displacement")}
    sinks["guards"] = (run/"batch_guards.jsonl").open("w")
    stats = replay.Rows(run/"checkpoint_gradients.csv")
    voltages = replay.Rows(run/"checkpoint_voltages.csv")
    residuals = replay.Rows(run/"trained_residuals.csv")
    trained_guards = (run/"trained_clean_guards.jsonl").open("w")
    all_sinks = [*sinks.values(), stats, voltages, residuals, trained_guards]
    completed = 0
    try:
        for batch in batches:
            assert time.time() < deadline, "Budget exhausted"
            started_batch = time.monotonic()
            captured = []
            def observer(output, context):
                captured.append(checkpoint_statistics(output, 0, batch["batch_index"], stats, voltages))
            guard = replay.replay_batch(initial, batch, config, shadow, torch.device("cuda:0"), deadline, sinks,
                                        clean_observer=observer)
            assert guard["native_float32_parameter_state_sha256_before_cast"] == initial["replay_initial_float32_tensor_sha256"]
            assert guard["parameter_state_sha256_after_cast"] == initial["replay_initial_float64_tensor_sha256"]
            assert guard["noise_draws"] == prior_guards[batch["batch_index"]]["noise_draws"]
            selected = {"architecture": "conv3", "scheme": "baseline", "checkpoint_role": "best_validation",
                        "T": 8, "K": 8, "source_native_T": 8, "source_native_K": 8, "native_context": True,
                        "tk_source": "unchanged source checkpoint training config", "actual_beta": 10.}
            output = shadow._run_precision(case=trained, selected=selected, batch=batch, precision="float64",
                       device=torch.device("cuda:0"), residual_threshold=.01, amplification_exponent=3,
                       nudging_mode="current", eqprop_variant="centered", capture_endpoint_states=True,
                       record_zero_nudge_displacement=True)
            checkpoint_statistics(output, 30, batch["batch_index"], stats, voltages, initial=captured[0])
            for name, value in output["gradients"].items():
                metrics = replay.compare_vectors(value.numpy(), output["bptt_gradients"][name].numpy())
                previous_row = prior_clean[batch["batch_index"], name]
                for key, result in metrics.items():
                    assert result == float(previous_row[f"bptt_{key}"]), f"Trained clean replay differs: {name}/{key}"
            for key in ("common_post_T_state_sha256", "negative_state_sha256", "positive_state_sha256", "zero_state_sha256"):
                assert output["guard"][key] == prior_guards[batch["batch_index"]][key]
            for row in output["residual_rows"]:
                residuals.add(row)
            trained_guards.write(json.dumps({"batch_index": batch["batch_index"], "reproduces_prior_clean_metrics_and_states": True,
                                             **output["guard"]}, sort_keys=True)+"\n")
            del output, captured
            completed += 1
            for sink in all_sinks:
                sink.flush()
            progress = {"stage": "paired_replay", "completed_batches": completed, "total_batches": len(batches),
                        "seconds": time.monotonic()-started_batch, "gpu_seconds": time.time()-start}
            reporting.append_metric(run/"metrics.jsonl", progress)
            reporting.update_status_progress(run, progress)
            print(json.dumps(progress), flush=True)
        assert replay.digest(Path(config["initializer"]["checkpoint_path"])) == config["initializer"]["checkpoint_sha256"]
        for name, expected in trained["source_file_sha256"].items():
            assert replay.digest(Path(trained["run_dir"])/name) == expected
        for sink in all_sinks:
            sink.close()
        elapsed = time.time()-start
        replay.write_json(run/"usage.json", {"gpu_seconds": elapsed})
        terminal = {"passed": True, "batch_count": completed, "initialization_gradient_rows": completed*164,
                    "trained_clean_batches_reproduced": completed, "noise_draws_match_trained": True,
                    "initializer_loaded_exactly": True, "checkpoint_bytes_unchanged": True,
                    "official_test_read": False, "optimizer_steps": 0, "gpu_seconds": elapsed}
        if args.mode == "full":
            assert terminal["initialization_gradient_rows"] == config["completion"]["initialization_gradient_rows"]
        assert used+elapsed < config["execution"]["gpu_seconds_cap_including_smokes"]
        reporting.complete_run(run, terminal_metrics=terminal,
                               completion={"coverage_complete": True, "roles": ["saved_initialization_epoch0", "best_validation_epoch30"]})
        assert not reporting.validate_run(run)
        print(json.dumps({"completed": str(run), **terminal}), flush=True)
    except BaseException as error:
        for sink in all_sinks:
            sink.close()
        replay.write_json(run/"usage.json", {"gpu_seconds": time.time()-start})
        reporting.fail_run(run, error=error)
        raise


if __name__ == "__main__":
    main()
