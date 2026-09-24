#!/usr/bin/env python3
"""Fill the balanced/legacy clean initialization measurements; reuse baseline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import run_section43_initialization as initialization
import run_section43_mechanism as replay

HERE, ROOT = replay.HERE, replay.ROOT


class ContextRows:
    def __init__(self, sink, context):
        self.sink, self.context = sink, context

    def add(self, row):
        self.sink.add({**self.context, **row})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE/"section43_conv3_initialization_magnitudes_config.json")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    shadow = replay.load_shadow(args.config)
    from experiments import reporting
    torch = shadow.torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained = replay.inventory(config, shadow)
    cases = [initialization.initialization_case(config, c, shadow) for c in trained]
    baseline_run = Path(config["reuse_baseline_run"])
    for source in (baseline_run, Path(config["trained_reference"]["run"])):
        assert not reporting.validate_run(source)
    prior = {(int(r["batch_index"]), r["parameter"]): r
             for r in initialization.read_csv(baseline_run/"gradients.csv") if float(r["sigma"]) == 0}
    prior_guards = {r["batch_index"]: r for r in map(json.loads, (baseline_run/"batch_guards.jsonl").read_text().splitlines())}
    dataset = config["dataset"]
    batches, cohort = shadow.base._build_validation_cohort(trained[0]["source_config"],
        data_root=Path(dataset["root"]), batch_size=16, example_count=576,
        source_indices=dataset["source_indices"], excluded_source_indices=dataset["excluded_source_indices"])
    assert cohort["validation_indices_sha256"] == dataset["validation_indices_sha256"]
    for batch, expected in zip(batches, dataset["expected_batch_guards"], strict=True):
        assert all(batch[k] == v for k, v in expected.items())
    study = ROOT/"results"/config["study_id"]
    assert study.is_dir()
    if args.mode == "smoke":
        batches = batches[:1]
    else:
        assert not reporting.validate_run(study/"smoke")
        assert json.loads((study/"smoke/result.json").read_text())["terminal_metrics"]["passed"]
        cases = [c for c in cases if c["scheme"] in config["replay_schemes"]]
    used = sum(json.loads(p.read_text())["gpu_seconds"] for p in study.glob("*/usage.json"))
    remaining = config["execution"]["gpu_seconds_cap_including_smokes"] - used
    assert remaining > 0
    start = time.time()
    deadline = start + remaining
    run = study/args.mode
    sources = [Path(__file__), args.config, HERE/"run_section43_initialization.py",
        HERE/"run_section43_mechanism.py", *[ROOT/"experiments"/n for n in (
        "audit_eqprop_float64_shadow.py", "analyze_conv_eqprop_bptt_beta_tk_displacement.py",
        "analyze_conv_eqprop_bptt_checkpoint_gradients.py", "reporting.py")]]
    reporting.start_run(run, {"study_id": config["study_id"], "run_id": args.mode,
        "arm_id": "conv3_initialization_magnitudes", "evidence_class": "ordinary_mnist_validation_mechanism",
        "smoke": args.mode == "smoke", "configuration": config, "dataset": dataset,
        "command": [sys.executable, *sys.argv],
        "git": {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "dirty": True, "exact_source_sha256": {str(p): replay.digest(p) for p in sources}},
        "runtime": {"host": socket.gethostname(), "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "python": sys.version, "remaining_gpu_seconds": remaining}})
    replay.write_json(run/"config.used.json", config)
    replay.write_json(run/"cohort.json", cohort)
    (run/"replay_source").mkdir()
    for p in sources:
        shutil.copyfile(p, run/"replay_source"/p.name)
    sinks = {key: replay.Rows(run/f"{key}.csv") for key in (
        "gradients", "displacement", "residuals", "phase_displacement")}
    sinks["guards"] = (run/"batch_guards.jsonl").open("w")
    stats = replay.Rows(run/"checkpoint_gradients.csv")
    voltages = replay.Rows(run/"checkpoint_voltages.csv")
    all_sinks = [*sinks.values(), stats, voltages]
    completed, baseline_reproduced = 0, False
    try:
        for case in cases:
            for batch in batches:
                assert time.time() < deadline, "Declared GPU budget exhausted"
                def observer(output, context):
                    initialization.checkpoint_statistics(output, 0, batch["batch_index"],
                        ContextRows(stats, context), ContextRows(voltages, context))
                    if case["scheme"] == "baseline":
                        for name, value in output["gradients"].items():
                            metrics = replay.compare_vectors(value.numpy(), output["bptt_gradients"][name].numpy())
                            for key, result in metrics.items():
                                assert result == float(prior[batch["batch_index"], name][f"bptt_{key}"])
                guard = replay.replay_batch(case, batch, config, shadow, torch.device("cuda:0"), deadline,
                                            sinks, clean_observer=observer)
                assert guard["native_float32_parameter_state_sha256_before_cast"] == case["replay_initial_float32_tensor_sha256"]
                assert guard["parameter_state_sha256_after_cast"] == case["replay_initial_float64_tensor_sha256"]
                if case["scheme"] == "baseline":
                    for key in ("common_post_T_state_sha256", "negative_state_sha256", "positive_state_sha256", "zero_state_sha256"):
                        assert guard[key] == prior_guards[batch["batch_index"]][key]
                    baseline_reproduced = True
                completed += 1
                for sink in all_sinks:
                    sink.flush()
                progress = {"stage": "initialization_replay", "scheme": case["scheme"],
                    "completed_batches": completed, "total_batches": len(cases)*len(batches),
                    "batch_index": batch["batch_index"], "gpu_seconds": time.time()-start}
                reporting.append_metric(run/"metrics.jsonl", progress)
                reporting.update_status_progress(run, progress)
                print(json.dumps(progress), flush=True)
        assert replay.digest(Path(config["initializer"]["checkpoint_path"])) == config["initializer"]["checkpoint_sha256"]
        for case in trained:
            for name, expected in case["source_file_sha256"].items():
                assert replay.digest(Path(case["run_dir"])/name) == expected
        for sink in all_sinks:
            sink.close()
        elapsed = time.time()-start
        replay.write_json(run/"usage.json", {"gpu_seconds": elapsed})
        assert elapsed + used < config["execution"]["gpu_seconds_cap_including_smokes"]
        assert completed == (3 if args.mode == "smoke" else 72)
        if args.mode == "smoke":
            assert baseline_reproduced
        reporting.complete_run(run, terminal_metrics={"passed": True, "case_count": len(cases),
            "batch_count": completed, "clean_comparison_rows": completed*4,
            "checkpoint_statistic_rows": completed*8, "baseline_control_reproduced": baseline_reproduced,
            "checkpoint_bytes_unchanged": True, "official_test_read": False, "optimizer_steps": 0,
            "gpu_seconds": elapsed}, completion={"coverage_complete": True,
            "reused_baseline_run": str(baseline_run), "scientific_gate": config["scientific_gate"]})
        assert not reporting.validate_run(run)
        print(json.dumps({"complete": str(run), "gpu_seconds": elapsed}), flush=True)
    except BaseException as error:
        for sink in all_sinks:
            sink.close()
        replay.write_json(run/"usage.json", {"gpu_seconds": time.time()-start})
        reporting.fail_run(run, error=error)
        raise


if __name__ == "__main__":
    main()
