#!/usr/bin/env python3
"""Read-only, native-float64 EqProp checkpoint mechanism replay for Section 4.3.

The clean endpoints and matched finite-K BPTT reference are calculated once per
checkpoint/batch. Noise affects only endpoint measurements, never the dynamics.
Run --mode smoke before --mode full; both consume the same recorded GPU budget.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def noise_seed(seed, batch, draw, phase, layer):
    # Scheme and sigma deliberately absent; independent phase/layer substreams.
    return seed + batch * 100000 + draw * 1000 + phase * 100 + layer


def compare_vectors(a, b):
    import numpy as np
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    an, bn = float(np.sqrt(np.sum(a*a))), float(np.sqrt(np.sum(b*b)))
    delta = float(np.sqrt(np.sum((a-b)**2)))
    return {
        "cosine": float(np.sum(a*b)/(an*bn)) if an > 0 and bn > 0 else None,
        "estimate_l2": an, "reference_l2": bn,
        "relative_l2_error": delta/bn if bn > 0 else None,
        "norm_ratio": an/bn if bn > 0 else None,
        "estimate_zero_fraction": float(np.mean(a == 0)),
        "reference_zero_fraction": float(np.mean(b == 0)),
    }


class Rows:
    def __init__(self, path):
        self.stream = Path(path).open("w", newline="")
        self.writer = None

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def load_shadow(config):
    # The mature replay engine reads --config during import to pin its runtime.
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(config)])
    sys.path.insert(0, str(ROOT))
    import experiments.reporting  # Pin current reporting before frozen runtime imports.
    path = ROOT / "experiments/audit_eqprop_float64_shadow.py"
    spec = importlib.util.spec_from_file_location("section43_shadow", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inventory(config, shadow):
    from experiments.reporting import validate_run
    cases = []
    signature = None
    for entry in config["cases"]:
        run = Path(entry["run_dir"])
        errors = validate_run(run)
        if errors:
            raise ValueError(f"Invalid source {run}: {errors}")
        for name, expected in entry["source_file_sha256"].items():
            if digest(run/name) != expected:
                raise ValueError(f"Source hash mismatch: {run/name}")
        source = json.loads((run/"config.used.json").read_text())
        model = shadow.base._model_config(source)
        assert source["training_algorithm"] == "EP"
        assert source["runtime_dtype"] == "float64"
        assert source["seed"] == 0
        assert source["batch_state_policy"] == "reset_each_batch"
        ep = source["eqprop"]
        assert ep["variant"] == "centered" and ep["nudging_mode"] == "current"
        assert ep["endpoint_read_noise_std"] == 0 and not ep["input_read_noise"]
        assert ep["normalize_current_scale"] and ep["current_scale"] == "auto"
        assert model["non_linearity"] == "perfect_diode"
        assert model["weight_min"] == 0 and model["weight_max"] == 100
        assert model["num_iterations_inference"] == entry["T"]
        assert model["num_iterations_training"] == entry["K"]
        rates = dict(zip(source["parameter_order"], source["optimizer"]["learning_rate"], strict=True))
        assert all(v == 0 for k, v in rates.items() if k.startswith("Bias_"))
        params = dict(source["datasets"]["mnist"]["params"])
        params.pop("root", None)
        if signature is None:
            signature = params
        assert params == signature, "Different preprocessing/split across source checkpoints"
        prov = json.loads((run/"metrics.json").read_text())["dataset_provenance"]
        assert prov["validation_indices_sha256"] == config["dataset"]["validation_indices_sha256"]
        cases.append({**entry, "source_config": source,
                      "best_checkpoint_path": run/"best_model.pt",
                      "weights_best_path": run/"weights_best.npz",
                      "best_checkpoint_sha256": entry["source_file_sha256"]["best_model.pt"],
                      "voltage_amplification": model["voltage_amp"],
                      "current_amplification": model["current_amp"]})
    runtime_root = Path(config["source_contract"]["runtime_source_root"])
    hashes = config["source_contract"]["runtime_code_sha256"]
    for relative, expected in hashes.items():
        if digest(runtime_root/relative) != expected:
            raise ValueError(f"Frozen runtime code changed: {relative}")
    return cases


def displacement_rows(output, context):
    import numpy as np
    plus = output["captured_phase_states"]["positive"]
    minus = output["captured_phase_states"]["negative"]
    free = output["captured_post_t_states"]
    zero = output["captured_zero_states"]
    for layer, (p, n, f, z) in enumerate(zip(plus, minus, free, zero, strict=True)):
        p, n, f, z = [x.numpy() for x in (p, n, f, z)]
        for kind, delta in (("centered_halfspan", (p-n)/2), ("positive_minus_free", p-f),
                            ("negative_minus_free", n-f), ("positive_minus_matched_zero", p-z),
                            ("negative_minus_matched_zero", n-z), ("zero_nudge_drift", z-f)):
            yield {**context, "layer_index": layer, "state_layer": f"H{layer+1}" if layer < len(plus)-1 else "Output",
                   "kind": kind, "element_count": delta.size,
                   "sum_delta": float(np.sum(delta)), "sum_squared_delta": float(np.sum(delta**2)),
                   "rms": float(np.sqrt(np.mean(delta**2))),
                   "mean_absolute_delta": float(np.mean(np.abs(delta)))}


def replay_batch(case, batch, config, shadow, device, deadline, sinks, clean_observer=None):
    import numpy as np
    torch, base = shadow.torch, shadow.base
    context = {"architecture": case["architecture"], "scheme": case["scheme"],
               "batch_index": batch["batch_index"]}
    selected = {**context, "checkpoint_role": case.get("checkpoint_role", "best_validation"), "T": case["T"], "K": case["K"],
                "source_native_T": case["T"], "source_native_K": case["K"], "native_context": True,
                "tk_source": "unchanged source checkpoint training config", "actual_beta": case["base_beta"]}
    started = time.monotonic()
    out = shadow._run_precision(case=case, selected=selected, batch=batch, precision="float64", device=device,
                               residual_threshold=.01, amplification_exponent=case["depth"], nudging_mode="current",
                               eqprop_variant="centered", capture_endpoint_states=True, record_zero_nudge_displacement=True)
    if clean_observer is not None:
        clean_observer(out, context)
    runtime = out["captured_runtime"]
    parameters_before = base._parameter_state_sha256(runtime["parameters"])
    assert all(not bool(p.state.count_nonzero()) for p in runtime["parameters"] if str(p.name).startswith("Bias_"))
    for row in displacement_rows(out, context):
        sinks["displacement"].add(row)
    for row in out["residual_rows"]:
        sinks["residuals"].add(row)
    for row in out["displacement_rows"]:
        sinks["phase_displacement"].add(row)
    clean = {k: v.numpy() for k, v in out["gradients"].items()}
    bptt = {k: v.numpy() for k, v in out["bptt_gradients"].items()}
    names = list(clean)
    rows = 0

    def record(values, sigma, draw):
        nonlocal rows
        for layer, name in enumerate(names):
            g = values[name]
            if hasattr(g, "numpy"):
                g = g.numpy()
            a, b = compare_vectors(g, bptt[name]), compare_vectors(g, clean[name])
            sinks["gradients"].add({**context, "layer_index": layer, "parameter": name, "sigma": sigma, "draw": draw,
                                    **{f"bptt_{k}": v for k, v in a.items()},
                                    **{f"clean_ep_{k}": v for k, v in b.items()}})
            rows += 1

    record(clean, 0., -1)
    input_state = out["captured_input_state"].to(device)
    phases = {phase: [s.to(device) for s in states] for phase, states in out["captured_phase_states"].items()}
    input_hash = base._tensor_sha256(input_state)

    def energy(states):
        runtime["energy_fn"].layers()[0].state = input_state
        base._restore_states(runtime["free_layers"], states)
        return base._energy_gradients(runtime)

    def estimate(negative, positive):
        grads = {}
        for index in runtime["weight_indices"]:
            name = str(runtime["parameters"][index].name).strip()
            _, g = shadow._native_eqprop_estimate(eqprop_variant="centered", beta=case["base_beta"],
                      denominator_scale=out["guard"]["output_row_current_scale"],
                      zero=positive[index], negative=negative[index], positive=positive[index])
            grads[name] = g.detach().cpu()
        return grads

    # Full readout path at sigma=0 must reproduce the saved clean estimates exactly.
    reconstructed = estimate(energy(phases["negative"]), energy(phases["positive"]))
    assert all(np.array_equal(reconstructed[k].numpy(), clean[k]) for k in names), "Zero-noise readout changed clean EP"
    draws_proof = []
    for draw in range(config["noise"]["draws"]):
        standard = {}
        hashes = {}
        for phase_index, phase in enumerate(("negative", "positive")):
            standard[phase] = []
            for layer, state in enumerate(phases[phase]):
                seed = noise_seed(config["noise"]["seed"], batch["batch_index"], draw, phase_index, layer)
                z = torch.randn(tuple(state.shape), dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
                hashes[f"{phase}/{layer}"] = base._tensor_sha256(z)
                standard[phase].append(z.to(device))
        assert all(hashes[f"negative/{i}"] != hashes[f"positive/{i}"] for i in range(len(phases["positive"])))
        draws_proof.append({"draw": draw, "standard_normal_sha256": hashes})
        for sigma in config["noise"]["sigmas"][1:]:
            if time.time() > deadline:
                raise TimeoutError("Declared total GPU budget exhausted")
            noisy = {phase: energy([v + sigma*z for v, z in zip(phases[phase], standard[phase], strict=True)])
                     for phase in ("negative", "positive")}
            record(estimate(noisy["negative"], noisy["positive"]), sigma, draw)
        del standard
    assert parameters_before == base._parameter_state_sha256(runtime["parameters"])
    assert input_hash == base._tensor_sha256(runtime["energy_fn"].layers()[0].state)
    guard = {**context, **out["guard"], "zero_noise_readout_matches_clean_bitwise": True,
             "checkpoint_unchanged_after_noise": True, "input_read_noise": False,
             "batch_payload_sha256": batch["payload_sha256"], "batch_source_indices_sha256": batch["source_indices_sha256"],
             "noise_draws": draws_proof, "gradient_rows": rows, "seconds": time.monotonic()-started}
    sinks["guards"].write(json.dumps(guard, sort_keys=True, allow_nan=False) + "\n")
    for key, sink in sinks.items():
        sink.flush()
    del out, runtime
    return guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"section43_mechanism_config.json")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--attempt", default="1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shadow = load_shadow(args.config)
    import experiments.reporting as reporting
    torch = shadow.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    study = ROOT/"results"/config["study_id"]
    assert study.is_dir(), "Register the planned result directory before launch"
    cases = inventory(config, shadow)
    dataset = config["dataset"]
    batches, cohort = shadow.base._build_validation_cohort(cases[0]["source_config"], data_root=Path(dataset["root"]),
                       batch_size=dataset["batch_size"], example_count=dataset["example_count"],
                       source_indices=dataset["source_indices"], excluded_source_indices=dataset["excluded_source_indices"])
    assert cohort["validation_indices_sha256"] == dataset["validation_indices_sha256"]
    for batch, guard in zip(batches, dataset["expected_batch_guards"], strict=True):
        assert all(batch[k] == v for k, v in guard.items())
    if args.mode == "smoke":
        cases = [c for c in cases if c["architecture"] == "conv3"]
        batches = batches[:1]
    else:
        smoke_path = study/"smoke-1"/"result.json"
        assert smoke_path.exists(), "Same-runner Conv3 smoke must complete before full replay"
        errors = reporting.validate_run(smoke_path.parent)
        assert not errors, errors
        smoke_result = json.loads(smoke_path.read_text())
        assert smoke_result["terminal_metrics"]["budget_projection_passed"]
    # Count only occupied GPU wall time; idle time between commands does not consume GPU budget.
    used = sum(json.loads(p.read_text()).get("gpu_seconds", 0) for p in study.glob("*/usage.json"))
    remaining = config["execution"]["gpu_seconds_cap_including_smokes"] - used
    if remaining <= 0:
        raise RuntimeError("No GPU budget remains")
    start = time.time()
    deadline = start + remaining
    run = study/f"{args.mode}-{args.attempt}"
    source_files = [Path(__file__), args.config, *[ROOT/"experiments"/name for name in (
        "audit_eqprop_float64_shadow.py", "analyze_conv_eqprop_bptt_beta_tk_displacement.py",
        "analyze_conv_eqprop_bptt_checkpoint_gradients.py", "reporting.py")]]
    manifest = {"study_id": config["study_id"], "run_id": run.name, "arm_id": "nine_checkpoint_readonly_replay",
                "evidence_class": config["evidence_class"], "smoke": args.mode == "smoke", "dataset": dataset,
                "configuration": config, "command": [sys.executable, *sys.argv],
                "git": {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                        "dirty": True, "exact_source_sha256": {str(p): digest(p) for p in source_files}},
                "runtime": {"host": socket.gethostname(), "pid": os.getpid(), "device": torch.cuda.get_device_name(0),
                            "torch": torch.__version__, "python": sys.version, "budget_seconds_remaining": remaining}}
    reporting.start_run(run, manifest)
    write_json(run/"config.used.json", config)
    write_json(run/"cohort.json", cohort)
    snapshot = run/"replay_source"
    snapshot.mkdir()
    for p in source_files:
        shutil.copyfile(p, snapshot/p.name)
    sinks = {key: Rows(run/f"{key}.csv") for key in ("gradients", "displacement", "residuals", "phase_displacement")}
    sinks["guards"] = (run/"batch_guards.jsonl").open("w")
    guards = []
    try:
        for case in cases:
            for batch in batches:
                if time.time() > deadline:
                    raise TimeoutError("Declared total GPU budget exhausted")
                guard = replay_batch(case, batch, config, shadow, torch.device("cuda:0"), deadline, sinks)
                guards.append(guard)
                progress = {"stage": "checkpoint_replay", "completed_batches": len(guards),
                            "total_batches": len(cases)*len(batches), "architecture": case["architecture"],
                            "scheme": case["scheme"], "batch_index": batch["batch_index"],
                            "seconds": round(guard["seconds"], 3), "gpu_seconds": time.time()-start}
                reporting.append_metric(run/"metrics.jsonl", progress)
                reporting.update_status_progress(run, progress)
                print(json.dumps(progress), flush=True)
        # Paired RNG streams must match across schemes for every architecture/batch.
        pairs = {}
        for g in guards:
            key = (g["architecture"], g["batch_index"])
            if key in pairs:
                assert pairs[key] == g["noise_draws"]
            pairs[key] = g["noise_draws"]
        for case in cases:
            for name, expected in case["source_file_sha256"].items():
                assert digest(Path(case["run_dir"])/name) == expected
        elapsed = time.time()-start
        projection = max(g["seconds"] for g in guards) * 36 * 9 * 1.2
        terminal = {"case_count": len(cases), "batch_count": len(guards),
                    "gradient_rows": sum(g["gradient_rows"] for g in guards), "gpu_seconds": elapsed,
                    "official_test_read": False, "checkpoint_files_unchanged": True,
                    "paired_noise_verified": True, "runtime_dtype": "float64",
                    "conservative_full_projection_seconds": projection,
                    "budget_projection_passed": elapsed + used + projection < 3600 if args.mode == "smoke" else elapsed + used < 3600}
        if args.mode == "full":
            assert terminal["case_count"] == config["completion"]["cases"]
            assert terminal["gradient_rows"] == config["completion"]["gradient_rows"]
        for sink in sinks.values():
            sink.close()
        write_json(run/"usage.json", {"gpu_seconds": elapsed})
        reporting.complete_run(run, terminal_metrics=terminal, completion={"all_declared_cases_complete": True,
                               "qualifications": "Validation mechanism evidence, seed 0; finite-T/K reference, source beta/gate caveats retained."})
        errors = reporting.validate_run(run)
        assert not errors, errors
        print(json.dumps({"completed": str(run), **terminal}), flush=True)
    except BaseException as exc:
        for sink in sinks.values():
            sink.close()
        write_json(run/"usage.json", {"gpu_seconds": time.time()-start})
        reporting.fail_run(run, error=exc)
        raise


if __name__ == "__main__":
    main()
