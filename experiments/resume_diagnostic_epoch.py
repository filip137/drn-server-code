"""Bridge stopped diagnostic runs to epoch continuation without a fresh optimizer.

Used by ``exact_run --resume-diagnostic-run``. The source must already be stopped;
this module never controls processes or alters source artifacts. Missing private
RNG state makes this an explicitly approximate continuation, not an uninterrupted
trajectory. Only the ordinary MNIST, validation-only, constant-LR Adam contract
is supported. No data loader (especially no official-test loader) is opened here.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import torch

from experiments.epoch_continuation import continuation_contract, file_identity
from experiments.reporting import atomic_write_json, sha256_file
from model.function.interaction import decode_function_checkpoint_artifact


def _rows(path):
    # A killed writer may leave only the last JSON line incomplete.
    lines = Path(path).read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def _checked_path(root, record, kind):
    path = (root / record[f"{kind}_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint path escapes source run")
    if (path.stat().st_size != record[f"{kind}_size_bytes"]
            or sha256_file(path) != record[f"{kind}_sha256"]):
        raise ValueError(f"Diagnostic {kind} checkpoint hash/size mismatch: {path}")
    return path


def _optimizer_payload(source, record, config):
    if record["schema_version"] != "mnist-conv-diagnostic-epoch-checkpoint/v1":
        raise ValueError("Unsupported diagnostic checkpoint schema")
    model_path = _checked_path(source, record, "model")
    optimizer_path = _checked_path(source, record, "optimizer")
    # These are trusted, hash-bound project-produced checkpoints.
    saved = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    tensors, schema, _ = decode_function_checkpoint_artifact(
        torch.load(model_path, map_location="cpu", weights_only=False), path=model_path)
    names = config["parameter_order"]
    if (schema is None or [s["name"].strip() for s in schema] != names
            or saved["parameter_names"] != names):
        raise ValueError("Diagnostic parameter order mismatch")
    if (saved["schema_version"] != "mnist-conv-diagnostic-optimizer-checkpoint/v1"
            or saved["optimizer_class"] != "Adam"
            or saved["epoch"] != record["epoch"]
            or saved["model_checkpoint"] != model_path.name
            or saved["model_checkpoint_sha256"] != record["model_sha256"]):
        raise ValueError("Diagnostic optimizer/model identity mismatch")
    state = saved["optimizer_state_dict"]
    groups = state["param_groups"]
    bindings = saved["optimizer_parameter_groups"]
    if len(groups) != len(names) or len(bindings) != len(groups):
        raise ValueError("Expected one Adam parameter group per configured parameter")
    ids = []
    dataset_params = config["datasets"][config["lab"]["dataset_key"]]["params"]
    batches = math.ceil(55000 / int(dataset_params["batch_size"]))
    if config.get("max_batches") is not None:
        batches = min(batches, int(config["max_batches"]))
    for index, (group, binding, tensor) in enumerate(zip(groups, bindings, tensors, strict=True)):
        if (len(group["params"]) != 1 or binding["group_index"] != index
                or binding["parameter_names"] != [names[index]]
                or group["lr"] != config["lr"][index]
                or binding["learning_rate"] != group["lr"]):
            raise ValueError("Diagnostic optimizer parameter-group binding mismatch")
        pid = group["params"][0]
        ids.append(pid)
        for key in ("betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused"):
            if key in config["optimizer"]:
                expected = config["optimizer"][key]
                if key == "betas":
                    expected = tuple(expected)
                if group.get(key) != expected:
                    raise ValueError(f"Adam hyperparameter mismatch: {key}")
        adam = state["state"].get(pid)
        if adam is None:
            if group["lr"] != 0:
                raise ValueError("Missing trained Adam state")
            continue
        if not {"step", "exp_avg", "exp_avg_sq"} <= adam.keys():
            raise ValueError("Incomplete Adam state")
        for key, value in adam.items():
            if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid Adam tensor")
            if key == "step":
                if value.numel() != 1 or value.item() != record["epoch"] * batches:
                    raise ValueError("Invalid Adam step")
            elif value.shape != tensor.shape or value.dtype != tensor.dtype:
                raise ValueError("Adam moment tensor schema mismatch")
    if len(set(ids)) != len(ids) or not set(state["state"]) <= set(ids):
        raise ValueError("Invalid Adam parameter IDs")
    return saved, tensors


def prepare_diagnostic_resume(*, source_run, run_dir, config_path, device, epoch=None):
    """Create a continuation payload in a fresh, initialized reporting bundle.

    Select the latest hash-valid completed epoch by default. Prefix checkpoints
    are copied, never linked, and the failed/incomplete tail remains in source.
    The caller owns source-process liveness checks and remaining compute budget.
    """
    source, root = Path(source_run).resolve(), Path(run_dir).resolve()
    if source == root or root.is_relative_to(source):
        raise ValueError("Recovery must use a separate run directory")
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    epochs = int(config["lab"]["epochs"])
    dataset = config["datasets"][config["lab"]["dataset_key"]]
    params = dataset["params"]
    if (manifest["configuration"]["sha256"] != sha256_file(config_path)
            or manifest["configuration"]["resolved"] != config
            or manifest["configuration"]["epochs"] != epochs
            or manifest.get("smoke", False)):
        raise ValueError("Recovery requires the unchanged full-horizon source config")
    if (config.get("batch_state_policy") != "reset_each_batch"
            or config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled"
            or dataset["factory"] != "labs.datasets.MnistTrainValidationDataset"
            or params.get("affine_config") is not None
            or params.get("num_workers", 0) != 0
            or config["optimizer"]["name"] != "Adam"
            or config["optimizer"].get("lr_decay", 1.) != 1.
            or manifest["configuration"]["diagnostic_overrides"].get("gradient_trace_samples_per_epoch", 0)
            or manifest["dataset"].get("official_test_read", False)):
        raise ValueError("Unsupported diagnostic recovery contract (validation-only ordinary MNIST required)")
    if (root / "continuation.pt").exists() or (root / "metrics.jsonl").exists():
        raise FileExistsError("Recovery destination already contains training evidence")
    records = _rows(source / "epoch_checkpoint_index.jsonl")
    by_epoch = {row["epoch"]: row for row in records}
    if len(by_epoch) != len(records):
        raise ValueError("Duplicate diagnostic checkpoint epochs")
    rows = [row for row in _rows(source / "metrics.jsonl") if row.get("kind") == "epoch"]
    metrics = {row["epoch"]: row for row in rows}
    if len(metrics) != len(rows):
        raise ValueError("Duplicate metric epochs")
    candidates = [epoch] if epoch is not None else sorted(
        (e for e in by_epoch if 0 < e < epochs and e in metrics), reverse=True)
    skipped = []
    for selected in candidates:
        try:
            if not 0 < selected < epochs or selected not in metrics:
                raise ValueError("Resume epoch must be completed and precede original horizon")
            saved, tensors = _optimizer_payload(source, by_epoch[selected], config)
            break
        except (ValueError, OSError, KeyError) as error:
            if epoch is not None:
                raise
            skipped.append(dict(epoch=selected, reason=str(error)))
    else:
        raise ValueError(f"No safe completed diagnostic epoch: {skipped}")
    prefix = [metrics[e] for e in range(1, selected + 1)]
    history = dict(loss=[], accuracy=[], test_loss=[], test_accuracy=[], learning_rate=[])
    for row in prefix:
        if row["evaluation_split"] != "validation":
            raise ValueError("Recovery cannot import official-test evidence")
        values = row["metrics"]
        for target, key in (("loss", "train_loss"), ("accuracy", "train_accuracy"),
                            ("test_loss", "validation_loss"), ("test_accuracy", "validation_accuracy")):
            value = values[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Nonfinite/incomplete epoch history")
            history[target].append(value)
        history["learning_rate"].append(list(config["lr"]))
    best_accuracy = max(history["test_accuracy"])
    best_epoch = history["test_accuracy"].index(best_accuracy) + 1
    if prefix[-1]["metrics"]["best_epoch"] != best_epoch:
        raise ValueError("Best-checkpoint selection disagrees with metric history")
    # Verify every prefix pair before copying, including the earlier best model.
    copies = []
    for e in range(selected + 1):
        record = by_epoch[e]
        for kind in ("model", "optimizer"):
            copies.append((_checked_path(source, record, kind), root / record[f"{kind}_path"]))
    sampler = torch.Generator().manual_seed(int(params["shuffle_seed"]))
    worker = torch.Generator().manual_seed(int(params["shuffle_seed"]))
    # ResettableRandomSampler draws exactly one randperm per entered epoch;
    # num_workers=0 DataLoader draws one base seed per iterator creation.
    for _ in range(selected):
        torch.randperm(55000, generator=sampler)
        torch.empty((), dtype=torch.int64).random_(generator=worker)
    rng = saved["rng_state"]
    contract = continuation_contract(config, epochs=epochs, max_batches=config.get("max_batches"),
        max_test_batches=config.get("max_test_batches"), device=device,
        gradient_trace_samples_per_epoch=0)
    private = saved.get("continuation_state")
    if private is not None:
        if private["contract"] != contract:
            raise ValueError("Full-state diagnostic scientific/runtime contract mismatch")
        sampler.set_state(private["sampler_rng"])
        worker.set_state(private["worker_rng"])
    elif any(config.get(study, {}).get("checkpoint_recovery") == "full_state_epoch_boundary_only"
             for study in ("read_noise_batch128", "rescaled_target_noise")):
        raise ValueError("This experiment requires full-state epoch recovery; private RNG state is missing")
    provenance = dict(schema="diagnostic-epoch-recovery/v1", source_run=str(source),
        source_manifest=file_identity(source / "manifest.json"), source_git=manifest.get("git"),
        checkpoint=by_epoch[selected], completed_epochs=selected, total_epochs=epochs,
        remaining_epochs=epochs-selected, next_epoch=selected+1, skipped_candidates=skipped,
        continuation_quality="approximate_stochastic_continuation",
        uninterrupted_trajectory=False, optimizer_state_restored=True,
        rng_restored=["python", "numpy", "torch_cpu", "torch_cuda"],
        rng_reconstructed=["train_sampler: one 55000-element permutation per completed epoch",
                           "worker_generator: one base seed per completed epoch (num_workers=0)"],
        rng_missing=["endpoint_read_noise_generators", "endpoint_read_noise_draw_count"],
        missing_rng_policy="Endpoint noise restarts from the original configured seed; not bit-exact.")
    if private is not None:
        provenance.update(
            continuation_quality="exact_epoch_state_continuation",
            uninterrupted_trajectory=True,
            rng_restored=["python", "numpy", "torch_cpu", "torch_cuda", "train_sampler",
                          "worker_generator", "endpoint_read_noise_generators", "endpoint_read_noise_draw_count"],
            rng_reconstructed=[], rng_missing=[], missing_rng_policy="Fail closed; all epoch state is required.")
    root.mkdir(parents=True, exist_ok=True)
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(root / by_epoch[best_epoch]["model_path"], root / "best_model.pt")
    (root / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prefix))
    (root / "epoch_checkpoint_index.jsonl").write_text("".join(
        json.dumps(by_epoch[e], sort_keys=True) + "\n" for e in range(selected + 1)))
    atomic_write_json(root / "diagnostic_resume.json", provenance)
    state = dict(schema="mnist-epoch-continuation/v1", contract=contract, epoch=selected,
        parameter_names=saved["parameter_names"], parameters=tensors,
        optimizer=saved["optimizer_state_dict"], sampler_rng=sampler.get_state(),
        worker_rng=worker.get_state(), endpoint_noise=private["endpoint_noise"] if private is not None else None,
        rng=dict(python=rng["python"], numpy=rng["numpy"], cpu=rng["torch_cpu"], cuda=rng["torch_cuda"]),
        history=history, best_accuracy=best_accuracy, best_epoch=best_epoch, trace_counts=None,
        artifacts={name: file_identity(root / name) for name in
                   ("best_model.pt", "metrics.jsonl", "gradient_trace.jsonl")},
        diagnostic_resume=provenance)
    temporary = root / "continuation.tmp"
    torch.save(state, temporary)
    os.replace(temporary, root / "continuation.pt")
    atomic_write_json(root / "continuation.json", dict(schema=state["schema"],
        completed_epochs=selected, total_epochs=epochs,
        checkpoint_sha256=sha256_file(root / "continuation.pt"), contract=contract,
        diagnostic_resume=provenance))
    return provenance
