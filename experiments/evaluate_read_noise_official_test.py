#!/usr/bin/env python3
"""Frozen-checkpoint clean inference: validation capacity probe or one test read.

There is deliberately no retry/resume path after the official-test reservation.
An interrupted read remains consumed and must be reviewed as partial evidence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import torch

from experiments.replay_conv_checkpoint_validation import (
    _build_runtime, _git_state, _resolved_model_config,
)
from experiments.reporting import (
    append_metric, atomic_write_json, complete_run, fail_run, runtime_context,
    sha256_file, start_run, update_status_progress,
)
from labs import datasets
from labs import mnist_train as trainer

SCHEMA = "read-noise-official-test/v1"
SELECTION = "max_validation_accuracy_earliest_epoch"
FINAL_SCHEMA = "read-noise-final-official-test/v1"
FINAL_SELECTION = "final_epoch"
ROOT = Path(__file__).resolve().parents[1]


def is_final_contract(contract):
    return contract.get("schema_version") == FINAL_SCHEMA


def validate_selection_contract(contract, case):
    """Keep corrected final-point evidence separate from the original best gate."""
    if not is_final_contract(contract):
        return
    if case.get("reuse") is not None:
        raise ValueError("Case declares result reuse; do not perform another official test evaluation")
    if (contract.get("final_checkpoint_authorized") is not True
            or contract.get("prior_best_test_exposure_recorded") is not True):
        raise ValueError("Final checkpoint requires explicit authorization and prior exposure record")
    epochs = {1: 10, 2: 30, 3: 30}[int(case["depth"])]
    if int(case["epochs"]) != epochs or int(case["selected_checkpoint"]["epoch"]) != epochs:
        raise ValueError("Final checkpoint must be at the complete native epoch budget (10/30/30)")
    prior = case.get("prior_best_test", {})
    if (not prior.get("contract_sha256") or not prior.get("checkpoint_sha256")
            or not isinstance(prior.get("official_test_evaluations"), int)
            or not isinstance(prior.get("official_test_examples"), int)
            or not 0 <= prior["official_test_evaluations"] <= 1
            or not 0 <= prior["official_test_examples"] <= 10000
            or (prior["official_test_evaluations"] == 0 and prior["official_test_examples"] != 0)):
        raise ValueError("Final checkpoint requires explicit valid prior-best test exposure")
    if prior["official_test_evaluations"] and not prior.get("result_sha256"):
        if prior.get("access_status") != "partial" or not prior.get("access_record_sha256"):
            raise ValueError("Prior-best test exposure requires a result or explicit partial-access identity")
    if (prior["official_test_evaluations"]
            and prior["checkpoint_sha256"] == case["selected_checkpoint"]["sha256"]):
        raise ValueError("Final checkpoint already tested as best; reuse the prior result without reading test")


def inference_source_files(files):
    return {name: digest for name, digest in files.items()
            if name.startswith(("model/", "training/"))
            or (name.startswith("labs/") and "/tests/" not in name)
            or name == "experiments/replay_conv_checkpoint_validation.py"}


def validate_capacity_receipt(contract, contract_hash, receipt, receipt_hash, config, target):
    if (receipt.get("capacity_signature") != capacity_signature(config)
            or receipt.get("target") != target or receipt.get("official_test_read") is not False
            or not 1 <= int(receipt["batch_size"]) <= 10000):
        raise ValueError("Validation capacity receipt does not match sealed inference")
    if receipt.get("contract_sha256") == contract_hash:
        return
    if not is_final_contract(contract):
        raise ValueError("Best-checkpoint contract cannot carry forward another contract's capacity receipt")
    matches = [record for record in contract.get("capacity_carryforward", [])
               if record.get("receipt_sha256") == receipt_hash
               and record.get("contract_sha256") == receipt.get("contract_sha256")
               and record.get("capacity_signature") == receipt.get("capacity_signature")
               and record.get("target") == target]
    if len(matches) != 1:
        raise ValueError("Capacity carryforward requires one explicitly sealed receipt identity")
    old_files = inference_source_files(matches[0].get("source_files", {}))
    new_files = inference_source_files(contract["source_files"])
    if not old_files or old_files != new_files:
        raise ValueError("Capacity carryforward inference source differs from the sealed current source")


def preflight_test_files(dataset_root):
    """Check file metadata only, before the irreversible test-access reservation."""
    for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        path = dataset_root / "MNIST" / "raw" / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Official test file is missing or empty: {path}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _path(value, asset_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def load_case(contract_path, expected_sha256, case_id, asset_root):
    if sha256_file(contract_path) != expected_sha256:
        raise ValueError("Sealed contract SHA256 mismatch")
    contract = json.loads(contract_path.read_text())
    allowed = {SCHEMA: SELECTION, FINAL_SCHEMA: FINAL_SELECTION}
    if (contract.get("schema_version") not in allowed or contract.get("frozen") is not True
            or contract.get("checkpoint_selection") != allowed[contract["schema_version"]]):
        raise ValueError("Expected frozen contract with explicitly matching best or final selection schema")
    cases = contract["cases"]
    if len({case["arm_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate arm identities in sealed inclusion set")
    selected = [case for case in cases if case["arm_id"] == case_id]
    if len(selected) != 1:
        raise ValueError("Case is absent from the sealed inclusion set")
    case = selected[0]
    validate_selection_contract(contract, case)
    audit = case.get("reuse_audit", {})
    prior = case.get("prior_official_test", {})
    if (audit.get("completion_criteria_met") is not True or audit.get("smoke") is not False
            or audit.get("result_official_test_read") is not False
            or audit.get("configured_official_test_policy") != "disabled"
            or audit.get("result_evaluation_split") != "validation"
            or prior.get("official_test_evaluations") != 0
            or prior.get("official_test_examples") != 0):
        raise ValueError("Source reuse audit is incomplete or records prior official test access")
    for path, digest in contract.get("source_files", {}).items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Frozen evaluation source mismatch: {path}")
    if not contract.get("source_files"):
        raise ValueError("Frozen contract must identify evaluation source_files")
    config_path = _path(case["config_path"], asset_root)
    checkpoint = case["selected_checkpoint"]
    checkpoint_path = _path(checkpoint["path"], asset_root)
    if sha256_file(config_path) != case["config_sha256"]:
        raise ValueError("Selected config SHA256 mismatch")
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("Selected checkpoint SHA256 mismatch")
    if not 1 <= int(checkpoint["epoch"]) <= int(case["epochs"]):
        raise ValueError("Selected checkpoint epoch outside declared budget")
    if not 0 <= float(checkpoint["validation_accuracy"]) <= 1:
        raise ValueError("Invalid selected validation accuracy")
    config = trainer.load_config(config_path)
    model = _resolved_model_config(config)
    if int(model["num_iterations_inference"]) != {1: 4, 2: 6, 3: 8}[int(case["depth"])]:
        raise ValueError("Expected native inference T=4/6/8")
    if trainer._runtime_dtype_from_config(config)[0] != "float64":
        raise ValueError("Expected frozen float64 runtime")
    dataset = config["datasets"][config["lab"]["dataset_key"]]
    if dataset["factory"] != "labs.datasets.MnistTrainValidationDataset":
        raise ValueError("Only train-only ordinary MNIST validation factory is allowed")
    if dataset["params"].get("affine_config") is not None:
        raise ValueError("Official ordinary MNIST forbids affine preprocessing")
    if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
        raise ValueError("Original selection config must disable official test")
    return contract, case, config_path, checkpoint_path, config


def capacity_signature(config):
    """Same numerical inference and preprocessing may share a validation probe."""
    params = config["datasets"][config["lab"]["dataset_key"]]["params"]
    return _digest({
        "model": _resolved_model_config(config),
        "energy_minimizer": config["energy_minimizer"],
        "runtime_dtype": trainer._runtime_dtype_from_config(config)[0],
        "preprocessing": {key: params.get(key) for key in (
            "normalize", "normalize_mean", "normalize_std", "normalize_scale", "affine_config")},
    })


def reserve_test(guard_dir, checkpoint_sha256, payload):
    """O_EXCL guard survives crashes, failed construction and partial test access."""
    guard_dir.mkdir(parents=True, exist_ok=True)
    path = guard_dir / f"{checkpoint_sha256}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise RuntimeError(f"Official test already reserved; no repeated access: {path}") from error
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Persist directory entry as well as content before the test loader is built.
    directory_fd = os.open(guard_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def check_runtime(runtime, checkpoint_path):
    if runtime["runtime_dtype_name"] != "float64":
        raise ValueError("Loaded runtime is not float64")
    biases = [p for p in runtime["energy_fn"].params() if p.name.startswith("Bias_")]
    if not biases or any(bool(torch.count_nonzero(p.state)) for p in biases):
        raise ValueError("Selected checkpoint must have exactly zero biases")
    for parameter in runtime["energy_fn"].params():
        if not bool(torch.isfinite(parameter.state).all()):
            raise FloatingPointError(f"Nonfinite selected parameter: {parameter.name}")
    stored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    states = stored["states"] if isinstance(stored, dict) else stored
    parameters = runtime["energy_fn"].params()
    if len(states) != len(parameters) or any(
        state.dtype != parameter.state.dtype
        or not torch.equal(state, parameter.state.detach().cpu())
        for state, parameter in zip(states, parameters)
    ):
        raise ValueError("Loaded parameters differ from exact selected checkpoint tensors")


def evaluate_batches(runtime, loader, run_dir, split, expected_examples=None):
    """Use the trainer's clean inference path and persist every completed batch."""
    seen = correct = batches = 0
    total_loss = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            # One trainer call per batch preserves its nonfinite guards and emits
            # no optimizer calls or endpoint read-noise injection.
            loss, accuracy, count = trainer._evaluate_image_loader(
                loader=[batch], network=runtime["network"], free_layers=runtime["free_layers"],
                cost_fn=runtime["cost_fn"], minimizer_inference=runtime["minimizer"],
                device=runtime["energy_fn"]._device, max_batches=None,
                split_label=split, epoch=1, epochs=1, log_interval=1,
                input_dtype=runtime["runtime_dtype"],
            )
            if count != len(batch[1]) or count <= 0:
                raise RuntimeError("Incorrect per-batch example count")
            seen += count
            correct += round(accuracy * count)
            total_loss += loss * count
            batches += 1
            progress = {"stage": split, "split": split, "batch": batch_index + 1,
                        "examples_seen": seen, "correct": correct,
                        "running_accuracy": correct / seen, "running_loss": total_loss / seen,
                        "elapsed_seconds": time.monotonic() - started,
                        "official_test_read": split == "test"}
            append_metric(run_dir / "metrics.jsonl", progress)
            update_status_progress(run_dir, progress)
            if expected_examples is not None and seen > expected_examples:
                raise RuntimeError("Evaluation exceeded sealed example count")
    if seen == 0 or (expected_examples is not None and seen != expected_examples):
        raise RuntimeError(f"Expected {expected_examples} examples; evaluated {seen}")
    return {"examples": seen, "correct": correct, "accuracy": correct / seen,
            "loss": total_loss / seen, "batches": batches,
            "elapsed_seconds": time.monotonic() - started}


def largest_fitting_batch(attempt, maximum):
    if maximum < 1:
        raise ValueError("Maximum batch must be positive")
    low, high = 0, maximum + 1
    candidate = min(64, maximum)
    while True:
        fits = attempt(candidate)
        if fits:
            low = candidate
        else:
            high = candidate
        if high - low <= 1:
            break
        candidate = min(candidate * 2, maximum) if fits and high == maximum + 1 else (low + high) // 2
    if not low:
        raise RuntimeError("No validation batch fits the declared headroom")
    return low


def batching_parity(runtime, images, selected_batch):
    """Compare the same fixed validation cohort at baseline and selected batching."""
    def predictions(batch_size):
        parts = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                runtime["network"].set_input(images[start:start + batch_size].to(
                    device=runtime["energy_fn"]._device, dtype=runtime["runtime_dtype"]), reset=True)
                runtime["minimizer"].compute_equilibrium()
                for layer in runtime["free_layers"]:
                    if not bool(torch.isfinite(layer.state).all()):
                        raise FloatingPointError("Nonfinite validation batching-parity state")
                output = runtime["output_layer"].state
                scores = runtime["cost_fn"]._scores(output) if hasattr(runtime["cost_fn"], "_scores") else output
                parts.append(scores.detach().cpu().clone())
        return torch.cat(parts)
    baseline = predictions(64)
    selected = predictions(selected_batch)
    same_predictions = bool(torch.equal(baseline.argmax(1), selected.argmax(1)))
    same_scores = bool(torch.allclose(baseline, selected, rtol=1e-9, atol=1e-11))
    if not same_predictions or not same_scores:
        raise RuntimeError("Validation batching parity failed; official test remains disabled")
    return {"examples": len(images), "baseline_batch_size": 64,
            "selected_batch_size": selected_batch, "predictions_identical": same_predictions,
            "scores_allclose": same_scores, "rtol": 1e-9, "atol": 1e-11,
            "max_absolute_score_difference": float((baseline - selected).abs().max())}


def probe(runtime, config, args, contract_hash):
    if args.device.type != "cuda":
        raise ValueError("Memory-capacity probe requires CUDA")
    # This dataset is train-only. Repetition allows testing batches >5000 without
    # constructing, opening, or touching any official-test dataset artifact.
    validation = runtime["validation_loader"].dataset
    if len(validation) != 5000:
        raise ValueError("Expected exactly 5000 validation examples")
    samples = [validation[i] for i in range(min(len(validation), args.max_batch_size))]
    images = torch.stack([item[0] for item in samples])
    labels = torch.tensor([int(item[1]) for item in samples])
    if args.max_batch_size > len(samples):
        repetitions = math.ceil(args.max_batch_size / len(samples))
        images = images.repeat((repetitions,) + (1,) * (images.ndim - 1))[:args.max_batch_size]
        labels = labels.repeat(repetitions)[:args.max_batch_size]
    attempts = []
    headroom = int(args.headroom_mib * 1024**2)

    def attempt(size):
        runtime["network"].set_input(images[:1].to(device=args.device, dtype=runtime["runtime_dtype"]), reset=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(args.device)
        free, total = torch.cuda.mem_get_info(args.device)
        available = free + torch.cuda.memory_reserved(args.device) - headroom
        torch.cuda.reset_peak_memory_stats(args.device)
        began = time.monotonic()
        try:
            evaluate_batches(runtime, [(images[:size], labels[:size])], args.run_dir, "validation_capacity")
            torch.cuda.synchronize(args.device)
            peak = torch.cuda.max_memory_reserved(args.device)
            fits = peak <= available
            outcome = "fits" if fits else "headroom_exceeded"
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_reserved(args.device)
            fits, outcome = False, "cuda_out_of_memory"
        record = {"stage": "capacity_probe", "split": "validation", "batch_size": size,
                  "outcome": outcome, "peak_reserved_bytes": peak,
                  "usable_bytes": available, "total_bytes": total,
                  "elapsed_seconds": time.monotonic() - began, "official_test_read": False}
        attempts.append(record)
        append_metric(args.run_dir / "metrics.jsonl", record)
        update_status_progress(args.run_dir, record)
        return fits

    selected = largest_fitting_batch(attempt, args.max_batch_size)
    # Repeat the chosen endpoint so a transient/fragmented boundary cannot be
    # accepted solely from a single successful allocation.
    if not attempt(selected):
        raise RuntimeError("Selected capacity was not repeatable; test access remains disabled")
    parity = batching_parity(runtime, images[:128], selected)
    return {"contract_sha256": contract_hash, "capacity_signature": capacity_signature(config),
            "target": args.target, "device_name": torch.cuda.get_device_name(args.device),
            "batch_size": selected, "max_batch_size": args.max_batch_size,
            "headroom_mib": args.headroom_mib, "attempts": attempts,
            "batching_parity": parity, "search_assumption": "monotone memory fit; largest checked-safe integer batch",
            "official_test_read": False}


def official_loader(runtime, dataset_root, batch_size):
    params = runtime["dataset_params"]
    return datasets.build_mnist_official_test_loader(
        batch_size=batch_size, root=str(dataset_root), download=False,
        normalize=params["normalize"], normalize_std=params["normalize_std"],
        normalize_mean=params.get("normalize_mean", 0.1307),
        normalize_scale=params.get("normalize_scale", 1.0), affine_config=None,
        num_workers=0, pin_memory=bool(params.get("pin_memory", False)),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "dataset-root", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--mode", choices=("probe", "test"), required=True)
    parser.add_argument("--guard-dir", type=Path)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-batch-size", type=int, default=10000)
    parser.add_argument("--headroom-mib", type=int, default=1024)
    args = parser.parse_args(argv)
    if not 1 <= args.max_batch_size <= 10000 or args.headroom_mib < 1:
        parser.error("Require max-batch-size in [1,10000] and positive headroom")
    if args.mode == "test" and (args.guard_dir is None or args.probe_result is None):
        parser.error("Official test requires --guard-dir and --probe-result")
    args.device = torch.device(args.device)
    args.run_dir = args.run_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    contract, case, config_path, checkpoint_path, config = load_case(
        args.contract, args.contract_sha256, args.case_id,
        (args.asset_root or args.contract.parent).resolve())
    if args.mode == "test" and case.get("test_target") != args.target:
        raise ValueError("Official test target differs from sealed single-host assignment")
    receipt = None
    if args.mode == "test":
        receipt = json.loads(args.probe_result.read_text())
        validate_capacity_receipt(contract, args.contract_sha256, receipt,
                                  sha256_file(args.probe_result), config, args.target)
        if args.device.type == "cuda" and receipt["device_name"] != torch.cuda.get_device_name(args.device):
            raise ValueError("Capacity receipt GPU differs from evaluation GPU")
        preflight_test_files(args.dataset_root)
    final_mode = is_final_contract(contract)
    git_identity = ({"commit": contract.get("source_commit"), "dirty": None,
                     "identity_basis": "sealed_source_files"} if final_mode else _git_state())
    manifest = {
        "study_id": contract["study_id"], "run_id": args.run_dir.name,
        "arm_id": case["arm_id"], "seed": case["seed"],
        "evidence_class": (("ordinary_mnist_final_test_after_prior_best" if final_mode else "ordinary_mnist_paper")
                           if args.mode == "test" else "ordinary_mnist_capacity_diagnostic"),
        "dataset": {"name": "MNIST", "split": "test" if args.mode == "test" else "validation",
                    "evaluation_split": "test" if args.mode == "test" else "validation",
                    "official_test_read": args.mode == "test"},
        "source": case, "resolved_config": config, "contract_sha256": args.contract_sha256,
        "source_files": contract["source_files"], "checkpoint_selection": contract["checkpoint_selection"],
        "pristine_holdout": not final_mode,
        "prior_best_test": case.get("prior_best_test"),
        "clean_inference": True, "read_noise_inference": False, "optimizer_steps": 0,
        "batch_state_policy": "reset_each_batch", "probe_receipt": receipt,
        "probe_receipt_sha256": sha256_file(args.probe_result) if receipt else None,
        "git": git_identity, "runtime": runtime_context(target=args.target),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
    }
    start_run(args.run_dir, manifest)
    access = {"reserved": False, "official_test_may_have_been_read": False,
              "complete": False, "automatic_retry_allowed": False}
    try:
        runtime = _build_runtime(config_path=config_path, checkpoint_path=checkpoint_path,
                                 dataset_root=args.dataset_root, validation_batch_size=64,
                                 device=args.device)
        check_runtime(runtime, checkpoint_path)
        if args.mode == "probe":
            summary = probe(runtime, config, args, args.contract_sha256)
            atomic_write_json(args.run_dir / "artifacts" / "capacity.json", summary)
            metrics = {"capacity": summary}
        else:
            checkpoint_hash = case["selected_checkpoint"]["sha256"]
            guard = reserve_test(args.guard_dir, checkpoint_hash, {
                "contract_sha256": args.contract_sha256, "case_id": args.case_id,
                "checkpoint_sha256": checkpoint_hash, "target": args.target,
                "run_dir": str(args.run_dir), "automatic_retry_allowed": False})
            access.update(reserved=True, guard_path=str(guard), official_test_may_have_been_read=True)
            atomic_write_json(args.run_dir / "artifacts" / "test_access.json", access)
            loader = official_loader(runtime, args.dataset_root, int(receipt["batch_size"]))
            if len(loader.dataset) != 10000:
                raise ValueError("Official MNIST test dataset must contain exactly 10000 examples")
            access["dataset_identity"] = {name: sha256_file(args.dataset_root / "MNIST" / "raw" / name)
                                          for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")}
            atomic_write_json(args.run_dir / "artifacts" / "test_access.json", access)
            summary = evaluate_batches(runtime, loader, args.run_dir, "test", expected_examples=10000)
            access.update(complete=True, examples=summary["examples"], official_test_evaluations=1)
            atomic_write_json(args.run_dir / "artifacts" / "test_access.json", access)
            metrics = {"test": summary}
        if sha256_file(config_path) != case["config_sha256"] or sha256_file(checkpoint_path) != case["selected_checkpoint"]["sha256"]:
            raise RuntimeError("Source config/checkpoint changed during evaluation")
        complete_run(args.run_dir, terminal_metrics=metrics, completion={
            "criteria_met": True, "parameters_mutated": False, "optimizer_steps": 0,
            "official_test_read": args.mode == "test", "official_test_evaluations": int(args.mode == "test"),
            "test_examples": summary["examples"] if args.mode == "test" else 0,
            "sealed_inclusion_set": True, "checkpoint_selection": contract["checkpoint_selection"],
            "pristine_holdout": not final_mode,
            "protocol_deviations": (["Final checkpoint evaluated by explicit user correction after prior-best test exposure"]
                                    if final_mode else []),
        })
    except BaseException as error:
        if args.mode == "test":
            atomic_write_json(args.run_dir / "artifacts" / "test_access.json", access)
        fail_run(args.run_dir, error=error)
        raise
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
