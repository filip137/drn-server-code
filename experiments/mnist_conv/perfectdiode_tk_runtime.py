"""Numerical runtime for the Conv3 perfect-diode ordinary-MNIST T/K study.

The runtime deliberately reuses the canonical MNIST loader and Conv model
builder.  A single imported asset bundle owns the seed-0 initialization
checkpoint and every cohort index.  Workers on different hosts may rebuild
loaders, but they must reproduce and verify those frozen indices before doing
any numerical work.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import sha256_file, sha256_json
from .io import atomic_write_json, read_json
from .lr_engine import (
    _unpack_batch,
    build_loader_bundle,
    build_model_runtime,
    canonical_parameter_name,
    parameter_tensor_digest,
)
from .perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    CONV3_PARAMETER_ORDER,
    GRADIENT_ZERO_EPSILON,
    K_BATCH_SIZE,
    K_BOUNDARY_SENTINEL,
    K_COSINE_MINIMUM,
    K_EXAMPLES,
    K_GRID,
    K_NORM_DELTA_MAXIMUM,
    K_REFERENCE,
    K_ZERO_FRACTION_DELTA_MAXIMUM,
    PerfectDiodeTKStudySpec,
    REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM,
    REFERENCE_Q90_ZERO_FRACTION_MAXIMUM,
    T_BATCH_SIZE,
    T_CORE_GRID,
    T_EXAMPLES,
    T_EXTENSION_GRID,
    T_RESIDUAL_THRESHOLD,
    select_k_measurements,
    select_t_measurements,
)


TK_ASSET_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-assets/v1"
TK_ASSET_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-assets-completion/v1"
)
TK_T_MEASUREMENTS_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-t-measurements/v1"
)
TK_K_MEASUREMENTS_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-k-measurements/v1"
)
TK_ROW_RESULT_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-row-result/v1"
TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-operating-point-audit/v1"
)
TK_SMOKE_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-smoke/v1"
DIAGNOSTIC_BASE_BATCH_SIZE = 16


class PerfectDiodeTKRuntimeError(RuntimeError):
    """Raised when runtime evidence cannot satisfy the fail-closed contract."""


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"Expected {path} to be a finite number. Provided value: {value!r}."
        )
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(
            f"Expected {path} to be a non-negative finite number. "
            f"Provided value: {value!r}."
        )
    return result


def _rms(tensor: Any) -> float:
    import torch

    values = tensor.detach().to(dtype=torch.float64)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise PerfectDiodeTKRuntimeError(
            "Expected a non-empty finite tensor for RMS calculation."
        )
    return float(torch.sqrt(torch.mean(values * values)).item())


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(_finite(value, "quantile sample") for value in values)
    if not ordered:
        raise ValueError(
            "Expected quantile samples to be non-empty. Provided value: []."
        )
    position = float(probability) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _stats(values: Sequence[float]) -> dict[str, float]:
    normalized = [_finite(value, "residual statistic sample", nonnegative=True) for value in values]
    if not normalized:
        raise ValueError(
            "Expected residual statistic samples to be non-empty. Provided value: []."
        )
    return {
        "mean": sum(normalized) / len(normalized),
        "median": _linear_quantile(normalized, 0.5),
        "p90": _linear_quantile(normalized, 0.9),
        "p99": _linear_quantile(normalized, 0.99),
        "max": max(normalized),
    }


def perfect_diode_projected_kkt_residual(
    state: Any,
    gradient: Any,
    *,
    epsilon: float = 1.0e-8,
) -> Any:
    """Return the sign-correct projected KKT residual for paired diode units."""

    import torch

    eps = _finite(epsilon, "epsilon")
    if eps <= 0.0:
        raise ValueError(
            f"Expected epsilon to be positive. Provided value: {epsilon!r}."
        )
    if tuple(state.shape) != tuple(gradient.shape):
        raise ValueError(
            "Expected state and gradient shapes to match. "
            f"Provided value: state={tuple(state.shape)!r}, "
            f"gradient={tuple(gradient.shape)!r}."
        )
    if state.ndim < 2 or int(state.shape[1]) % 2:
        raise ValueError(
            "Expected a paired hidden state with an even channel dimension. "
            f"Provided value: {tuple(state.shape)!r}."
        )
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Expected finite state and gradient tensors.")
    half = int(state.shape[1]) // 2
    result = torch.empty_like(gradient)
    excitation_state = state[:, :half]
    excitation_gradient = gradient[:, :half]
    result[:, :half] = torch.where(
        excitation_state > eps,
        excitation_gradient.abs(),
        torch.relu(-excitation_gradient),
    )
    inhibition_state = state[:, half:]
    inhibition_gradient = gradient[:, half:]
    result[:, half:] = torch.where(
        inhibition_state < -eps,
        inhibition_gradient.abs(),
        torch.relu(inhibition_gradient),
    )
    return result


def perfect_diode_clamped_occupancy(
    state: Any,
    *,
    epsilon: float = 1.0e-8,
) -> dict[str, int | float]:
    """Measure excitation/inhibition occupancy at the diode constraints."""

    import torch

    eps = _finite(epsilon, "epsilon")
    if state.ndim < 2 or int(state.shape[1]) % 2:
        raise ValueError(
            "Expected a paired hidden state with an even channel dimension. "
            f"Provided value: {tuple(state.shape)!r}."
        )
    half = int(state.shape[1]) // 2
    excitation = state[:, :half]
    inhibition = state[:, half:]
    excitation_clamped = int((excitation <= eps).sum().item())
    inhibition_clamped = int((inhibition >= -eps).sum().item())
    excitation_count = int(excitation.numel())
    inhibition_count = int(inhibition.numel())
    total = excitation_count + inhibition_count
    return {
        "clamped_count": excitation_clamped + inhibition_clamped,
        "unit_count": total,
        "fraction": (excitation_clamped + inhibition_clamped) / total,
        "excitation_clamped_count": excitation_clamped,
        "excitation_unit_count": excitation_count,
        "excitation_fraction": excitation_clamped / excitation_count,
        "inhibition_clamped_count": inhibition_clamped,
        "inhibition_unit_count": inhibition_count,
        "inhibition_fraction": inhibition_clamped / inhibition_count,
    }


def analytical_quadratic_layer_gradient(energy: Any, layer: Any) -> Any:
    """Evaluate the exact smooth Q-function gradient as ``2*a*z + b``.

    Perfect-diode constraints are handled separately by the projected-KKT
    residual.  The free-state energy itself is quadratic in each layer, and
    the minimizer already uses these same coefficient functions.  Using them
    here avoids ``ConvResistive.eval()``'s very large broadcast tensor while
    preserving the mathematical gradient exactly.
    """

    import torch

    a_factory = getattr(energy, "a_coef_fn", None)
    b_factory = getattr(energy, "b_coef_fn", None)
    if not callable(a_factory) or not callable(b_factory):
        raise PerfectDiodeTKRuntimeError(
            "Expected the T residual energy to expose callable quadratic "
            "a_coef_fn and b_coef_fn methods."
        )
    a_fn = a_factory(layer)
    b_fn = b_factory(layer)
    if not callable(a_fn) or not callable(b_fn):
        raise PerfectDiodeTKRuntimeError(
            "Expected callable layer-specific quadratic coefficient functions."
        )
    state = layer.state.detach()
    with torch.no_grad():
        a = a_fn()
        if not torch.is_tensor(a) or not bool(torch.isfinite(a).all()):
            raise PerfectDiodeTKRuntimeError(
                "Expected a finite tensor from the layer quadratic a coefficient."
            )
        try:
            gradient = 2.0 * a * state
        except RuntimeError as exc:
            raise PerfectDiodeTKRuntimeError(
                "Expected the layer quadratic a coefficient to broadcast to the "
                f"state shape. Provided value: a={tuple(a.shape)!r}, "
                f"state={tuple(state.shape)!r}."
            ) from exc
        del a
        b = b_fn()
        if not torch.is_tensor(b) or not bool(torch.isfinite(b).all()):
            raise PerfectDiodeTKRuntimeError(
                "Expected a finite tensor from the layer quadratic b coefficient."
            )
        try:
            gradient.add_(b)
        except RuntimeError as exc:
            raise PerfectDiodeTKRuntimeError(
                "Expected the layer quadratic b coefficient to broadcast to the "
                f"state shape. Provided value: b={tuple(b.shape)!r}, "
                f"state={tuple(state.shape)!r}."
            ) from exc
        del b
    if tuple(gradient.shape) != tuple(state.shape) or not bool(
        torch.isfinite(gradient).all()
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected the analytical quadratic gradient to be finite and "
            f"shape-matched. Provided value: gradient={tuple(gradient.shape)!r}, "
            f"state={tuple(state.shape)!r}."
        )
    return gradient.detach()


def _release_layer_residual_memory(device: Any) -> None:
    import torch

    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()


def _tensor_sequence_digest(tensors: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        dtype = str(value.dtype).encode("utf-8")
        shape = repr(tuple(value.shape)).encode("utf-8")
        raw = value.numpy().tobytes()
        for piece in (dtype, shape, raw):
            digest.update(len(piece).to_bytes(8, "big"))
            digest.update(piece)
    return digest.hexdigest()


def _artifact(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def parameters_in_contract_order(parameters: Iterable[Any]) -> tuple[Any, ...]:
    """Return Conv3 parameters in the frozen scientific name order.

    The model-native list legitimately groups all weights before all biases.
    Artifact identities instead use ``CONV3_PARAMETER_ORDER``.  Resolve that
    difference explicitly by name, while failing closed on duplicate, missing,
    or unexpected scientific parameters.
    """

    values = tuple(parameters)
    by_name: dict[str, Any] = {}
    duplicates: list[str] = []
    for parameter in values:
        name = canonical_parameter_name(parameter)
        if name in by_name:
            duplicates.append(name)
        else:
            by_name[name] = parameter
    expected = set(CONV3_PARAMETER_ORDER)
    observed = set(by_name)
    if duplicates or observed != expected or len(values) != len(expected):
        raise PerfectDiodeTKRuntimeError(
            "Expected exactly one scientific parameter for every frozen Conv3 "
            "parameter name. Provided value: "
            f"duplicates={sorted(duplicates)!r}, "
            f"missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}, "
            f"native_order={tuple(canonical_parameter_name(item) for item in values)!r}."
        )
    return tuple(by_name[name] for name in CONV3_PARAMETER_ORDER)


def _combine_batches(batches: Sequence[Any]) -> tuple[Any, Any, Any]:
    import torch

    unpacked = [_unpack_batch(batch) for batch in batches]
    return (
        torch.cat([item[0] for item in unpacked], dim=0),
        torch.cat([item[1] for item in unpacked], dim=0),
        torch.as_tensor(
            [index for item in unpacked for index in item[2]],
            dtype=torch.int64,
        ),
    )


def _materialize_base_batches(bundle: Any, *, count: int = 64) -> tuple[Any, ...]:
    bundle.reset_train_shuffle()
    iterator = iter(bundle.train_loader)
    batches: list[Any] = []
    for index in range(count):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise PerfectDiodeTKRuntimeError(
                f"Expected at least {count} deterministic 32-example batches. "
                f"Provided value: {index}."
            ) from exc
        images, _labels, source_indices = _unpack_batch(batch)
        if (
            int(images.shape[0]) != DIAGNOSTIC_BASE_BATCH_SIZE
            or len(source_indices) != DIAGNOSTIC_BASE_BATCH_SIZE
        ):
            raise PerfectDiodeTKRuntimeError(
                "Expected every frozen base batch to contain "
                f"{DIAGNOSTIC_BASE_BATCH_SIZE} indexed "
                f"examples. Provided value: images={int(images.shape[0])}, "
                f"indices={len(source_indices)}."
            )
        batches.append(batch)
    return tuple(batches)


def _cohort_indices(base_batches: Sequence[Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    k_indices = tuple(
        index
        for batch in base_batches[: K_EXAMPLES // DIAGNOSTIC_BASE_BATCH_SIZE]
        for index in _unpack_batch(batch)[2]
    )
    t_indices = tuple(
        index
        for batch in base_batches[: T_EXAMPLES // DIAGNOSTIC_BASE_BATCH_SIZE]
        for index in _unpack_batch(batch)[2]
    )
    if len(k_indices) != K_EXAMPLES or len(t_indices) != T_EXAMPLES:
        raise PerfectDiodeTKRuntimeError(
            "Expected exact T and K cohort sizes from frozen base batches."
        )
    return t_indices, k_indices


def _runtime_row(row: Mapping[str, Any], *, t: int, k: int) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    value["inference_iterations"] = int(t)
    value["training_iterations"] = int(k)
    value["reference_inference_iterations"] = K_REFERENCE
    value["reference_training_iterations"] = K_REFERENCE
    return value


def _atomic_runtime_save(runtime: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        runtime.save(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def prepare_shared_assets(
    spec: PerfectDiodeTKStudySpec,
    *,
    assets_root: str | Path,
    data_root: str | Path,
    device: str,
    download: bool = False,
) -> dict[str, Any]:
    """Create the one checkpoint/cohort bundle that every host must import."""

    root = Path(assets_root).expanduser().resolve() / spec.study_id
    root.mkdir(parents=True, exist_ok=True)
    assets_path = root / "assets.json"
    completion_path = root / "completion.json"
    checkpoint_path = root / "initialization.pt"
    if completion_path.exists():
        return validate_shared_assets(spec, root)
    if any(path.exists() for path in (assets_path, checkpoint_path)):
        raise PerfectDiodeTKRuntimeError(
            "Expected an incomplete shared asset directory to be empty before "
            f"retry. Provided value: {root}."
        )

    study = spec.data
    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    base_batches = _materialize_base_batches(bundle)
    t_indices, k_indices = _cohort_indices(base_batches)

    baseline = _runtime_row(spec.rows[0], t=T_CORE_GRID[0], k=K_GRID[0])
    runtime = build_model_runtime(
        study,
        baseline,
        device=device,
        learning_rate=1.0,
    )
    ordered_parameters = parameters_in_contract_order(runtime.parameters)
    parameter_names = tuple(
        canonical_parameter_name(parameter) for parameter in ordered_parameters
    )
    tensor_sha256 = parameter_tensor_digest(ordered_parameters)
    _atomic_runtime_save(runtime, checkpoint_path)
    del runtime

    assets = {
        "schema_version": TK_ASSET_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "architecture": "conv3",
        "model_seed": 0,
        "official_test_read": False,
        "initialization": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "parameter_tensor_sha256": tensor_sha256,
            "parameter_names": list(parameter_names),
            "shared_across_schemes": True,
        },
        "split": {
            "source": "mnist_train",
            "train_size": len(bundle.train_indices),
            "validation_size": len(bundle.validation_indices),
            "train_indices": list(bundle.train_indices),
            "validation_indices": list(bundle.validation_indices),
            "train_indices_sha256": bundle.train_indices_hash,
            "validation_indices_sha256": bundle.validation_indices_hash,
        },
        "cohorts": {
            "base_batch_size": DIAGNOSTIC_BASE_BATCH_SIZE,
            "t": {
                "examples": T_EXAMPLES,
                "batch_size": T_BATCH_SIZE,
                "source_indices": list(t_indices),
                "source_indices_sha256": sha256_json(list(t_indices)),
            },
            "k": {
                "examples": K_EXAMPLES,
                "batch_size": K_BATCH_SIZE,
                "source_indices": list(k_indices),
                "source_indices_sha256": sha256_json(list(k_indices)),
            },
        },
    }
    atomic_write_json(assets_path, assets, canonical=True)
    completion = {
        "schema_version": TK_ASSET_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "official_test_read": False,
        "outputs": [
            _artifact(checkpoint_path, base=root),
            _artifact(assets_path, base=root),
        ],
    }
    atomic_write_json(completion_path, completion, canonical=True)
    return validate_shared_assets(spec, root)


def validate_shared_assets(
    spec: PerfectDiodeTKStudySpec,
    asset_dir: str | Path,
) -> dict[str, Any]:
    """Validate the imported asset bundle without constructing a dataset."""

    root = Path(asset_dir).expanduser().resolve()
    assets_path = root / "assets.json"
    completion_path = root / "completion.json"
    if not assets_path.is_file() or not completion_path.is_file():
        raise PerfectDiodeTKRuntimeError(
            f"Expected a complete shared asset bundle. Provided value: {root}."
        )
    assets = read_json(assets_path)
    completion = read_json(completion_path)
    if (
        assets.get("schema_version") != TK_ASSET_SCHEMA_VERSION
        or assets.get("study_id") != spec.study_id
        or assets.get("config_sha256") != spec.config_sha256
        or assets.get("official_test_read") is not False
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected shared assets to match the exact T/K study and exclude "
            "the official test set."
        )
    if (
        completion.get("schema_version") != TK_ASSET_COMPLETION_SCHEMA_VERSION
        or completion.get("state") != "complete"
        or completion.get("study_id") != spec.study_id
        or completion.get("config_sha256") != spec.config_sha256
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected a matching terminal shared-asset completion marker."
        )
    for record in completion.get("outputs", []):
        path = root / str(record.get("path"))
        if (
            not path.is_file()
            or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")
        ):
            raise PerfectDiodeTKRuntimeError(
                f"Expected shared asset artifact hash and size to verify: {path}."
            )
    checkpoint = root / str(assets["initialization"]["path"])
    if sha256_file(checkpoint) != assets["initialization"]["sha256"]:
        raise PerfectDiodeTKRuntimeError(
            "Expected the imported initialization checkpoint SHA-256 to verify."
        )
    split = assets.get("split", {})
    if len(split.get("train_indices", [])) != 55_000 or len(
        split.get("validation_indices", [])
    ) != 5_000:
        raise PerfectDiodeTKRuntimeError(
            "Expected shared assets to freeze the 55,000/5,000 MNIST-train split."
        )
    cohorts = assets.get("cohorts", {})
    for key, count in (("t", T_EXAMPLES), ("k", K_EXAMPLES)):
        cohort = cohorts.get(key, {})
        indices = cohort.get("source_indices", [])
        if (
            len(indices) != count
            or sha256_json(indices) != cohort.get("source_indices_sha256")
        ):
            raise PerfectDiodeTKRuntimeError(
                f"Expected the imported {key.upper()} cohort indices to verify."
            )
    result = copy.deepcopy(assets)
    result["asset_dir"] = str(root)
    result["assets_sha256"] = sha256_file(assets_path)
    result["completion_sha256"] = sha256_file(completion_path)
    return result


@dataclass(frozen=True)
class LoadedTKAssets:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    parameter_tensor_sha256: str
    train_indices_sha256: str
    validation_indices_sha256: str
    t_indices_sha256: str
    k_indices_sha256: str
    t_batches: tuple[Any, ...]
    k_batches: tuple[Any, ...]


def load_shared_assets(
    spec: PerfectDiodeTKStudySpec,
    *,
    asset_dir: str | Path,
    data_root: str | Path,
    download: bool,
) -> LoadedTKAssets:
    """Rebuild ordinary-MNIST loaders and verify them against imported indices."""

    assets = validate_shared_assets(spec, asset_dir)
    root = Path(assets["asset_dir"])
    bundle = build_loader_bundle(
        spec.data,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    split = assets["split"]
    if (
        tuple(bundle.train_indices) != tuple(split["train_indices"])
        or tuple(bundle.validation_indices) != tuple(split["validation_indices"])
        or bundle.train_indices_hash != split["train_indices_sha256"]
        or bundle.validation_indices_hash != split["validation_indices_sha256"]
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected the host's regenerated MNIST-train split to match the "
            "imported frozen indices exactly."
        )
    base_batches = _materialize_base_batches(bundle)
    t_indices, k_indices = _cohort_indices(base_batches)
    if (
        tuple(t_indices) != tuple(assets["cohorts"]["t"]["source_indices"])
        or tuple(k_indices) != tuple(assets["cohorts"]["k"]["source_indices"])
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected the host's deterministic loader order to match the imported "
            "T and K cohort indices exactly."
        )
    t_batches = tuple(
        _combine_batches(base_batches[index : index + 4])
        for index in range(
            0, T_EXAMPLES // DIAGNOSTIC_BASE_BATCH_SIZE, 4
        )
    )
    k_batches = tuple(
        _combine_batches(base_batches[index : index + 2])
        for index in range(
            0, K_EXAMPLES // DIAGNOSTIC_BASE_BATCH_SIZE, 2
        )
    )
    return LoadedTKAssets(
        root=root,
        checkpoint=root / assets["initialization"]["path"],
        checkpoint_sha256=assets["initialization"]["sha256"],
        parameter_tensor_sha256=assets["initialization"][
            "parameter_tensor_sha256"
        ],
        train_indices_sha256=split["train_indices_sha256"],
        validation_indices_sha256=split["validation_indices_sha256"],
        t_indices_sha256=assets["cohorts"]["t"]["source_indices_sha256"],
        k_indices_sha256=assets["cohorts"]["k"]["source_indices_sha256"],
        t_batches=t_batches,
        k_batches=k_batches,
    )


def _require_runtime_identity(runtime: Any, assets: LoadedTKAssets) -> None:
    ordered_parameters = parameters_in_contract_order(runtime.parameters)
    names = tuple(
        canonical_parameter_name(parameter) for parameter in ordered_parameters
    )
    if names != CONV3_PARAMETER_ORDER:
        raise AssertionError("canonical Conv3 parameter reordering drifted")
    digest = parameter_tensor_digest(ordered_parameters)
    if digest != assets.parameter_tensor_sha256:
        raise PerfectDiodeTKRuntimeError(
            "Expected runtime parameters to load the exact shared initialization "
            f"checkpoint. Provided value: {digest!r}."
        )


def measure_t_candidate(
    spec: PerfectDiodeTKStudySpec,
    row: Mapping[str, Any],
    *,
    iteration_count: int,
    assets: LoadedTKAssets,
    device: str,
) -> dict[str, Any]:
    """Measure projected-KKT hidden and raw-output residuals for one T."""

    import torch

    t = int(iteration_count)
    runtime = build_model_runtime(
        spec.data,
        _runtime_row(row, t=t, k=K_REFERENCE),
        device=device,
        initialization_checkpoint=assets.checkpoint,
        learning_rate=1.0,
    )
    _require_runtime_identity(runtime, assets)
    if len(runtime.free_layers) != 4:
        raise PerfectDiodeTKRuntimeError(
            f"Expected three hidden layers and one output layer. "
            f"Provided value: {len(runtime.free_layers)}."
        )
    raw_values: list[list[float]] = [[] for _ in runtime.free_layers]
    selection_values: list[list[float]] = [[] for _ in runtime.free_layers]
    occupancy_totals = [
        {
            "clamped_count": 0,
            "unit_count": 0,
            "excitation_clamped_count": 0,
            "excitation_unit_count": 0,
            "inhibition_clamped_count": 0,
            "inhibition_unit_count": 0,
        }
        for _ in runtime.free_layers[:-1]
    ]
    seen = 0
    epsilon = float(spec.data["model"]["perfect_diode"]["clamp_epsilon"])
    for batch in assets.t_batches:
        images, _labels, _source_indices = _unpack_batch(batch)
        images = images.to(runtime.device)
        runtime.network.set_input(images, reset=True)
        runtime.minimizer_inference.compute_equilibrium()
        energy = getattr(runtime.minimizer_inference, "_fn", runtime.energy_fn)
        for layer_index, layer in enumerate(runtime.free_layers):
            gradient = analytical_quadratic_layer_gradient(energy, layer)
            if not bool(torch.isfinite(layer.state).all()) or not bool(
                torch.isfinite(gradient).all()
            ):
                raise PerfectDiodeTKRuntimeError(
                    f"Expected finite T={t} layer state and residual gradient."
                )
            raw_maxima = (
                gradient.abs()
                .reshape(gradient.shape[0], -1)
                .amax(dim=1)
                .detach()
                .cpu()
            )
            raw_values[layer_index].extend(
                raw_maxima.tolist()
            )
            if layer_index < 3:
                selection = perfect_diode_projected_kkt_residual(
                    layer.state.detach(), gradient, epsilon=epsilon
                )
                selection_maxima = (
                    selection.reshape(selection.shape[0], -1)
                    .amax(dim=1)
                    .detach()
                    .cpu()
                )
                selection_values[layer_index].extend(
                    selection_maxima.tolist()
                )
                occupancy = perfect_diode_clamped_occupancy(
                    layer.state.detach(), epsilon=epsilon
                )
                for key in occupancy_totals[layer_index]:
                    occupancy_totals[layer_index][key] += int(occupancy[key])
                del selection_maxima, selection
            else:
                selection_values[layer_index].extend(raw_maxima.tolist())
            del raw_maxima, gradient
            _release_layer_residual_memory(runtime.device)
        seen += int(images.shape[0])
    if seen != T_EXAMPLES:
        raise PerfectDiodeTKRuntimeError(
            f"Expected exactly {T_EXAMPLES} T examples. Provided value: {seen}."
        )
    layers = []
    roles = ("hidden_0", "hidden_1", "hidden_2", "output")
    for index, role in enumerate(roles):
        raw_stats = _stats(raw_values[index])
        selection_stats = _stats(selection_values[index])
        layer_record: dict[str, Any] = {
            "role": role,
            "name": str(getattr(runtime.free_layers[index], "name", role)),
            "selection_residual_mode": (
                "projected_kkt" if index < 3 else "raw"
            ),
            "selection_residual_p90": selection_stats["p90"],
            "selection_residual": selection_stats,
            "raw_residual": raw_stats,
        }
        if index < 3:
            counts = occupancy_totals[index]
            layer_record["clamped_occupancy"] = {
                **counts,
                "fraction": counts["clamped_count"] / counts["unit_count"],
                "excitation_fraction": counts["excitation_clamped_count"]
                / counts["excitation_unit_count"],
                "inhibition_fraction": counts["inhibition_clamped_count"]
                / counts["inhibition_unit_count"],
            }
        layers.append(layer_record)
    del runtime
    return {
        "iteration_count": t,
        "num_examples": seen,
        "batch_size": T_BATCH_SIZE,
        "official_test_read": False,
        "cohort_source_indices_sha256": assets.t_indices_sha256,
        "initialization_checkpoint_sha256": assets.checkpoint_sha256,
        "initialization_tensor_sha256": assets.parameter_tensor_sha256,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "smooth_gradient_method": "quadratic_coefficients_2az_plus_b",
        "threshold": T_RESIDUAL_THRESHOLD,
        "comparison": "strictly_less_than",
        "layers": layers,
    }


def _collect_gradients_for_k(
    spec: PerfectDiodeTKStudySpec,
    row: Mapping[str, Any],
    *,
    selected_t: int,
    training_iterations: int,
    assets: LoadedTKAssets,
    device: str,
) -> dict[str, Any]:
    import torch

    runtime = build_model_runtime(
        spec.data,
        _runtime_row(row, t=selected_t, k=training_iterations),
        device=device,
        initialization_checkpoint=assets.checkpoint,
        learning_rate=1.0,
    )
    _require_runtime_identity(runtime, assets)
    ordered_parameters = parameters_in_contract_order(runtime.parameters)
    parameters = {
        canonical_parameter_name(parameter): parameter
        for parameter in ordered_parameters
    }
    initial_weight_rms = {
        name: _rms(parameters[name].state) for name in CONV3_CONV_WEIGHTS
    }
    gradients: dict[str, list[Any]] = {
        name: [] for name in CONV3_CONV_WEIGHTS
    }
    equilibrium_digests: list[str] = []
    for batch in assets.k_batches:
        images, labels, _source_indices = _unpack_batch(batch)
        images = images.to(runtime.device)
        labels = labels.to(runtime.device)
        runtime.network.set_input(images, reset=True)
        runtime.minimizer_inference.compute_equilibrium()
        equilibrium_digests.append(
            _tensor_sequence_digest(layer.state for layer in runtime.free_layers)
        )
        runtime.cost_fn.set_target(labels)
        values = runtime.estimator.compute_gradient()
        if len(values) < len(runtime.parameters):
            raise PerfectDiodeTKRuntimeError(
                "Expected one BP gradient per scientific parameter."
            )
        for parameter, gradient in zip(runtime.parameters, values):
            name = canonical_parameter_name(parameter)
            if name not in gradients:
                continue
            if tuple(gradient.shape) != tuple(parameter.state.shape) or not bool(
                torch.isfinite(gradient).all()
            ):
                raise PerfectDiodeTKRuntimeError(
                    f"Expected finite shape-matched gradient for {name!r}."
                )
            gradients[name].append(
                gradient.detach().to(dtype=torch.float64, device="cpu").clone()
            )
    if any(len(values) != K_EXAMPLES // K_BATCH_SIZE for values in gradients.values()):
        raise PerfectDiodeTKRuntimeError(
            "Expected exactly eight gradient batches per ConvWeight parameter."
        )
    del runtime
    return {
        "training_iterations": int(training_iterations),
        "gradients": {name: tuple(values) for name, values in gradients.items()},
        "initial_weight_rms": initial_weight_rms,
        "free_equilibrium_sha256_by_batch": equilibrium_digests,
    }


def compare_k_gradients(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    candidate_k: int,
    reference_k: int,
) -> dict[str, Any]:
    """Compare candidate gradients and check high-K reference viability."""

    import torch

    if (
        candidate.get("free_equilibrium_sha256_by_batch")
        != reference.get("free_equilibrium_sha256_by_batch")
    ):
        raise PerfectDiodeTKRuntimeError(
            "Expected every K to reuse hash-identical free equilibria."
        )
    candidate_gradients = candidate.get("gradients", {})
    reference_gradients = reference.get("gradients", {})
    if set(candidate_gradients) != set(CONV3_CONV_WEIGHTS) or set(
        reference_gradients
    ) != set(CONV3_CONV_WEIGHTS):
        raise PerfectDiodeTKRuntimeError(
            "Expected K comparisons for exactly every Conv3 ConvWeight parameter."
        )
    diagnostics = []
    for name in CONV3_CONV_WEIGHTS:
        current_batches = tuple(candidate_gradients[name])
        reference_batches = tuple(reference_gradients[name])
        if len(current_batches) != 8 or len(reference_batches) != 8:
            raise PerfectDiodeTKRuntimeError(
                f"Expected eight matched gradient batches for {name!r}."
            )
        batch_records = []
        for batch_index, (current_value, reference_value) in enumerate(
            zip(current_batches, reference_batches)
        ):
            current = current_value.reshape(-1)
            target = reference_value.reshape(-1)
            if tuple(current.shape) != tuple(target.shape):
                raise PerfectDiodeTKRuntimeError(
                    f"Expected matched gradient-vector shape for {name!r} "
                    f"batch {batch_index}."
                )
            current_norm = float(torch.linalg.vector_norm(current).item())
            reference_norm = float(torch.linalg.vector_norm(target).item())
            norm_delta = abs(current_norm - reference_norm) / max(
                current_norm, reference_norm, 1.0e-30
            )
            difference_norm = float(
                torch.linalg.vector_norm(current - target).item()
            )
            vector_relative_error = difference_norm / max(
                reference_norm, 1.0e-30
            )
            current_zero = float(
                (current.abs() <= GRADIENT_ZERO_EPSILON)
                .to(torch.float64)
                .mean()
                .item()
            )
            reference_zero = float(
                (target.abs() <= GRADIENT_ZERO_EPSILON)
                .to(torch.float64)
                .mean()
                .item()
            )
            denominator = current_norm * reference_norm
            if denominator > 0.0:
                raw_cosine = float(
                    torch.dot(current, target).item() / denominator
                )
                if not math.isfinite(raw_cosine):
                    raise PerfectDiodeTKRuntimeError(
                        f"Expected a finite gradient-vector cosine for {name!r} "
                        f"batch {batch_index}. Provided value: {raw_cosine!r}."
                    )
                # Mathematically cosine is bounded by [-1, 1]. Identical
                # nonzero float64 vectors can round to 1+2e-16, so canonicalize
                # only this finite floating-point overshoot. A zero-vector
                # denominator remains undefined below.
                cosine = min(1.0, max(-1.0, raw_cosine))
            else:
                cosine = None
            batch_records.append(
                {
                    "batch_index": batch_index,
                    "gradient_l2": current_norm,
                    "reference_gradient_l2": reference_norm,
                    "relative_gradient_l2_norm_delta": norm_delta,
                    "gradient_vector_relative_error": vector_relative_error,
                    "gradient_vector_cosine": cosine,
                    "gradient_zero_fraction": current_zero,
                    "reference_gradient_zero_fraction": reference_zero,
                    "absolute_zero_fraction_delta": abs(
                        current_zero - reference_zero
                    ),
                    "reference_gradient_rms": _rms(reference_value),
                }
            )
        current_norm = sum(
            item["gradient_l2"] for item in batch_records
        ) / len(batch_records)
        reference_norm = sum(
            item["reference_gradient_l2"] for item in batch_records
        ) / len(batch_records)
        norm_delta = abs(current_norm - reference_norm) / max(
            current_norm, reference_norm, 1.0e-30
        )
        current_zero = sum(
            item["gradient_zero_fraction"] for item in batch_records
        ) / len(batch_records)
        reference_zero = sum(
            item["reference_gradient_zero_fraction"] for item in batch_records
        ) / len(batch_records)
        zero_delta = abs(current_zero - reference_zero)
        cosines = [item["gradient_vector_cosine"] for item in batch_records]
        cosine = (
            sum(float(value) for value in cosines) / len(cosines)
            if all(value is not None and math.isfinite(float(value)) for value in cosines)
            else None
        )
        vector_relative_error = sum(
            item["gradient_vector_relative_error"] for item in batch_records
        ) / len(batch_records)
        reference_batch_rms = [
            item["reference_gradient_rms"] for item in batch_records
        ]
        reference_batch_zero = [
            item["reference_gradient_zero_fraction"] for item in batch_records
        ]
        median_gradient_rms = _linear_quantile(reference_batch_rms, 0.5)
        q90_zero_fraction = _linear_quantile(reference_batch_zero, 0.9)
        initial_rms = _finite(
            reference["initial_weight_rms"][name],
            f"initial weight RMS for {name}",
        )
        nominal_update_unit = (
            median_gradient_rms / initial_rms if initial_rms > 0.0 else math.nan
        )
        reference_viable = (
            median_gradient_rms > REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM
            and q90_zero_fraction < REFERENCE_Q90_ZERO_FRACTION_MAXIMUM
            and math.isfinite(nominal_update_unit)
            and nominal_update_unit > 0.0
        )
        passed = (
            reference_viable
            and norm_delta <= K_NORM_DELTA_MAXIMUM
            and zero_delta <= K_ZERO_FRACTION_DELTA_MAXIMUM
            and cosine is not None
            and math.isfinite(cosine)
            and cosine >= K_COSINE_MINIMUM
        )
        diagnostics.append(
            {
                "parameter": name,
                "batch_count": len(batch_records),
                "batches": batch_records,
                "gradient_l2": current_norm,
                "reference_gradient_l2": reference_norm,
                "relative_gradient_l2_norm_delta": norm_delta,
                "gradient_vector_relative_error": vector_relative_error,
                "gradient_vector_cosine": cosine,
                "gradient_zero_fraction": current_zero,
                "reference_gradient_zero_fraction": reference_zero,
                "absolute_zero_fraction_delta": zero_delta,
                "reference_median_gradient_rms": median_gradient_rms,
                "reference_q90_zero_fraction": q90_zero_fraction,
                "initial_weight_rms": initial_rms,
                "reference_nominal_update_unit": nominal_update_unit,
                "reference_viable": reference_viable,
                "passed": passed,
            }
        )
    return {
        "candidate_k": int(candidate_k),
        "reference_k": int(reference_k),
        "batch_count": K_EXAMPLES // K_BATCH_SIZE,
        "reference_viable": all(
            item["reference_viable"] for item in diagnostics
        ),
        "passed": all(item["passed"] for item in diagnostics),
        "parameter_diagnostics": diagnostics,
        "free_equilibrium_sha256_by_batch": list(
            reference["free_equilibrium_sha256_by_batch"]
        ),
    }


def run_operating_point_audit(
    spec: PerfectDiodeTKStudySpec,
    row: Mapping[str, Any],
    *,
    selected_t: int,
    selected_k: int,
    assets: LoadedTKAssets,
    device: str,
    entry_id: str,
    execution_host: str,
    t_extension_used: bool,
    execution_environment_sha256: str,
) -> dict[str, Any]:
    """Freshly replay the selected operating point after grid selection."""

    t_measurement = measure_t_candidate(
        spec,
        row,
        iteration_count=selected_t,
        assets=assets,
        device=device,
    )
    selected_t_failures = [
        str(layer["role"])
        for layer in t_measurement["layers"]
        if not (
            isinstance(layer.get("selection_residual_p90"), (int, float))
            and not isinstance(layer.get("selection_residual_p90"), bool)
            and math.isfinite(float(layer["selection_residual_p90"]))
            and float(layer["selection_residual_p90"]) < T_RESIDUAL_THRESHOLD
        )
    ]
    t64_measurement = measure_t_candidate(
        spec,
        row,
        iteration_count=64,
        assets=assets,
        device=device,
    )
    t256_measurement = (
        measure_t_candidate(
            spec,
            row,
            iteration_count=256,
            assets=assets,
            device=device,
        )
        if t_extension_used
        else None
    )

    def measurement_passes(value: Mapping[str, Any]) -> bool:
        return all(
            isinstance(layer.get("selection_residual_p90"), (int, float))
            and not isinstance(layer.get("selection_residual_p90"), bool)
            and math.isfinite(float(layer["selection_residual_p90"]))
            and float(layer["selection_residual_p90"]) < T_RESIDUAL_THRESHOLD
            for layer in value["layers"]
        )

    t64_passed = measurement_passes(t64_measurement)
    t256_passed = (
        measurement_passes(t256_measurement)
        if t256_measurement is not None
        else None
    )
    required_sentinel_passed = (
        t256_passed is True if t_extension_used else t64_passed
    )
    selected_gradients = _collect_gradients_for_k(
        spec,
        row,
        selected_t=selected_t,
        training_iterations=selected_k,
        assets=assets,
        device=device,
    )
    audit_reference_k = (
        K_BOUNDARY_SENTINEL if selected_k == K_REFERENCE else K_REFERENCE
    )
    reference_gradients = _collect_gradients_for_k(
        spec,
        row,
        selected_t=selected_t,
        training_iterations=audit_reference_k,
        assets=assets,
        device=device,
    )
    k_measurement = compare_k_gradients(
        selected_gradients,
        reference_gradients,
        candidate_k=selected_k,
        reference_k=audit_reference_k,
    )
    k_measurement.update(
        {
            "cohort_examples": K_EXAMPLES,
            "batch_size": K_BATCH_SIZE,
            "official_test_read": False,
            "cohort_source_indices_sha256": assets.k_indices_sha256,
            "initialization_checkpoint_sha256": assets.checkpoint_sha256,
            "initialization_tensor_sha256": assets.parameter_tensor_sha256,
            "selected_t": selected_t,
            "fixed_step_minimization": True,
            "batch_state_policy": "reset_each_batch",
        }
    )
    passed = (
        not selected_t_failures
        and required_sentinel_passed
        and k_measurement["passed"] is True
    )
    return {
        "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": row["scheme"],
        "execution_backend": "tmux",
        "execution_host": execution_host,
        "execution_environment_sha256": execution_environment_sha256,
        "official_test_read": False,
        "selected_t": selected_t,
        "selected_k": selected_k,
        "k_audit_reference": audit_reference_k,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failure_reasons": (
            [f"t_layer_failed:{role}" for role in selected_t_failures]
            + (
                ["t256_extension_sentinel_failed"]
                if t_extension_used and t256_passed is not True
                else ["t64_core_sentinel_failed"]
                if not t_extension_used and not t64_passed
                else []
            )
            + ([] if k_measurement["passed"] else ["k_gradient_gate_failed"])
        ),
        "fresh_replay_after_selection": True,
        "t_measurement": t_measurement,
        "t64_sentinel_measurement": t64_measurement,
        "t64_sentinel_passed": t64_passed,
        "t256_extension_sentinel_measurement": t256_measurement,
        "t256_extension_sentinel_passed": t256_passed,
        "t_extension_used": t_extension_used,
        "k_measurement": k_measurement,
    }


def execute_tk_row(
    spec: PerfectDiodeTKStudySpec,
    *,
    row: Mapping[str, Any],
    assets: LoadedTKAssets,
    output_dir: str | Path,
    device: str,
    manifest_sha256: str,
    entry_id: str,
    execution_host: str,
    execution_environment_sha256: str,
) -> dict[str, Any]:
    """Execute one complete scheme row, including conditional sentinels."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    t_by_iteration = {
        count: measure_t_candidate(
            spec,
            row,
            iteration_count=count,
            assets=assets,
            device=device,
        )
        for count in T_CORE_GRID
    }
    t_selection = select_t_measurements(t_by_iteration)
    if t_selection["status"] == "needs_extension":
        for count in T_EXTENSION_GRID:
            t_by_iteration[count] = measure_t_candidate(
                spec,
                row,
                iteration_count=count,
                assets=assets,
                device=device,
            )
        t_selection = select_t_measurements(t_by_iteration)
    if t_selection["status"] == "needs_extension":
        raise AssertionError("T extension did not produce a terminal decision")
    t_artifact = {
        "schema_version": TK_T_MEASUREMENTS_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": row["scheme"],
        "official_test_read": False,
        "cohort_source_indices_sha256": assets.t_indices_sha256,
        "initialization_checkpoint_sha256": assets.checkpoint_sha256,
        "measurements": [
            t_by_iteration[count] for count in sorted(t_by_iteration)
        ],
        "selection": t_selection,
    }
    t_path = destination / "t_measurements.json"
    atomic_write_json(t_path, t_artifact, canonical=True)

    k_by_iteration: dict[int, dict[str, Any]] = {}
    k_sentinel: dict[str, Any] | None = None
    selected_t = t_selection.get("selected_t")
    if t_selection["status"] == "selected" and isinstance(selected_t, int):
        comparison_context = {
            "cohort_examples": K_EXAMPLES,
            "batch_size": K_BATCH_SIZE,
            "official_test_read": False,
            "cohort_source_indices_sha256": assets.k_indices_sha256,
            "initialization_checkpoint_sha256": assets.checkpoint_sha256,
            "initialization_tensor_sha256": assets.parameter_tensor_sha256,
            "selected_t": selected_t,
            "fixed_step_minimization": True,
            "batch_state_policy": "reset_each_batch",
        }
        reference = _collect_gradients_for_k(
            spec,
            row,
            selected_t=selected_t,
            training_iterations=K_REFERENCE,
            assets=assets,
            device=device,
        )
        for count in K_GRID:
            candidate = (
                reference
                if count == K_REFERENCE
                else _collect_gradients_for_k(
                    spec,
                    row,
                    selected_t=selected_t,
                    training_iterations=count,
                    assets=assets,
                    device=device,
                )
            )
            comparison = compare_k_gradients(
                candidate,
                reference,
                candidate_k=count,
                reference_k=K_REFERENCE,
            )
            comparison.update(comparison_context)
            k_by_iteration[count] = comparison
        k_selection = select_k_measurements(k_by_iteration)
        if k_selection["status"] == "needs_k128_sentinel":
            high_reference = _collect_gradients_for_k(
                spec,
                row,
                selected_t=selected_t,
                training_iterations=K_BOUNDARY_SENTINEL,
                assets=assets,
                device=device,
            )
            k_sentinel = compare_k_gradients(
                reference,
                high_reference,
                candidate_k=K_REFERENCE,
                reference_k=K_BOUNDARY_SENTINEL,
            )
            k_sentinel.update(comparison_context)
            k_selection = select_k_measurements(
                k_by_iteration, k128_sentinel=k_sentinel
            )
    else:
        k_selection = {
            "status": "not_run_unresolved_t",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": [],
        }
    k_artifact = {
        "schema_version": TK_K_MEASUREMENTS_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": row["scheme"],
        "official_test_read": False,
        "selected_t": selected_t,
        "cohort_source_indices_sha256": assets.k_indices_sha256,
        "initialization_checkpoint_sha256": assets.checkpoint_sha256,
        "measurements": [
            k_by_iteration[count] for count in sorted(k_by_iteration)
        ],
        "k128_sentinel": k_sentinel,
        "selection": k_selection,
    }
    k_path = destination / "k_measurements.json"
    atomic_write_json(k_path, k_artifact, canonical=True)

    diagnostic_status = (
        "selected"
        if t_selection["status"] == "selected"
        and k_selection["status"] == "selected"
        else (
            str(t_selection["status"])
            if t_selection["status"] != "selected"
            else str(k_selection["status"])
        )
    )
    if diagnostic_status == "selected":
        audit = run_operating_point_audit(
            spec,
            row,
            selected_t=int(t_selection["selected_t"]),
            selected_k=int(k_selection["selected_k"]),
            assets=assets,
            device=device,
            entry_id=entry_id,
            execution_host=execution_host,
            t_extension_used=bool(t_selection["extension_used"]),
            execution_environment_sha256=execution_environment_sha256,
        )
    else:
        audit = {
            "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "entry_id": entry_id,
            "row_id": row["row_id"],
            "architecture": "conv3",
            "scheme": row["scheme"],
            "execution_backend": "tmux",
            "execution_host": execution_host,
            "execution_environment_sha256": execution_environment_sha256,
            "official_test_read": False,
            "selected_t": None,
            "selected_k": None,
            "status": "not_run_unresolved_selection",
            "passed": False,
            "failure_reasons": [diagnostic_status],
            "fresh_replay_after_selection": False,
            "t_measurement": None,
            "k_measurement": None,
        }
    audit_path = destination / "operating_point_audit.json"
    atomic_write_json(audit_path, audit, canonical=True)
    audit_sha256 = sha256_file(audit_path)
    status = (
        "selected"
        if diagnostic_status == "selected" and audit["passed"] is True
        else (
            "unresolved_operating_point_audit"
            if diagnostic_status == "selected"
            else diagnostic_status
        )
    )
    result = {
        "schema_version": TK_ROW_RESULT_SCHEMA_VERSION,
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "manifest_sha256": manifest_sha256,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": row["scheme"],
        "execution_backend": "tmux",
        "execution_host": execution_host,
        "execution_environment_sha256": execution_environment_sha256,
        "run_name": row["run_name"],
        "voltage_amp": float(row["voltage_amp"]),
        "current_amp": float(row["current_amp"]),
        "input_gain": float(row["input_gain"]),
        "official_test_read": False,
        "status": status,
        "diagnostic_selection_status": diagnostic_status,
        "diagnostic_selected_t": t_selection.get("selected_t"),
        "diagnostic_selected_k": k_selection.get("selected_k"),
        "selected_t": selected_t if status == "selected" else None,
        "selected_k": (
            int(k_selection["selected_k"]) if status == "selected" else None
        ),
        "t_reference": 64,
        "k_reference": 64,
        "t_extension_used": bool(t_selection["extension_used"]),
        "k128_sentinel_used": bool(k_selection["k128_sentinel_used"]),
        "initialization_checkpoint_sha256": assets.checkpoint_sha256,
        "initialization_tensor_sha256": assets.parameter_tensor_sha256,
        "train_indices_sha256": assets.train_indices_sha256,
        "validation_indices_sha256": assets.validation_indices_sha256,
        "t_cohort_indices_sha256": assets.t_indices_sha256,
        "k_cohort_indices_sha256": assets.k_indices_sha256,
        "operating_point_audit_path": audit_path.name,
        "operating_point_audit_sha256": audit_sha256,
        "operating_point_audit_passed": audit["passed"] is True,
        "t_selection": t_selection,
        "k_selection": k_selection,
        "artifacts": [
            _artifact(t_path, base=destination),
            _artifact(k_path, base=destination),
            _artifact(audit_path, base=destination),
        ],
    }
    atomic_write_json(destination / "result.json", result, canonical=True)
    return result


def run_tk_smoke(
    spec: PerfectDiodeTKStudySpec,
    *,
    row: Mapping[str, Any],
    assets: LoadedTKAssets,
    output_dir: str | Path,
    device: str,
    manifest_sha256: str,
    entry_id: str,
    execution_host: str,
    execution_environment_sha256: str,
) -> dict[str, Any]:
    """Run one real T batch and one real K=4 gradient batch."""

    import torch

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    runtime = build_model_runtime(
        spec.data,
        _runtime_row(row, t=T_CORE_GRID[0], k=K_GRID[0]),
        device=device,
        initialization_checkpoint=assets.checkpoint,
        learning_rate=1.0,
    )
    _require_runtime_identity(runtime, assets)
    t_images, _t_labels, t_indices = _unpack_batch(assets.t_batches[0])
    runtime.network.set_input(t_images.to(runtime.device), reset=True)
    runtime.minimizer_inference.compute_equilibrium()
    epsilon = float(spec.data["model"]["perfect_diode"]["clamp_epsilon"])
    residual_p90 = {}
    for index, layer in enumerate(runtime.free_layers):
        energy = getattr(runtime.minimizer_inference, "_fn", runtime.energy_fn)
        gradient = analytical_quadratic_layer_gradient(energy, layer)
        if index < 3:
            residual = perfect_diode_projected_kkt_residual(
                layer.state.detach(), gradient, epsilon=epsilon
            )
            values = (
                residual.reshape(residual.shape[0], -1)
                .amax(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            del residual
        else:
            values = (
                gradient.abs()
                .reshape(gradient.shape[0], -1)
                .amax(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
        residual_p90[
            ("hidden_0", "hidden_1", "hidden_2", "output")[index]
        ] = _linear_quantile(values, 0.9)
        del values, gradient
        _release_layer_residual_memory(runtime.device)

    k_images, k_labels, k_indices = _unpack_batch(assets.k_batches[0])
    runtime.network.set_input(k_images.to(runtime.device), reset=True)
    runtime.minimizer_inference.compute_equilibrium()
    runtime.cost_fn.set_target(k_labels.to(runtime.device))
    gradients = runtime.estimator.compute_gradient()
    gradient_rms = {}
    for parameter, gradient in zip(runtime.parameters, gradients):
        name = canonical_parameter_name(parameter)
        if name in CONV3_CONV_WEIGHTS:
            if not bool(torch.isfinite(gradient).all()):
                raise PerfectDiodeTKRuntimeError(
                    f"Expected finite smoke gradient for {name!r}."
                )
            gradient_rms[name] = _rms(gradient)
    if tuple(gradient_rms) != CONV3_CONV_WEIGHTS:
        raise PerfectDiodeTKRuntimeError(
            "Expected smoke gradients for every Conv3 ConvWeight parameter."
        )
    result = {
        "schema_version": TK_SMOKE_SCHEMA_VERSION,
        "status": "passed",
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "manifest_sha256": manifest_sha256,
        "entry_id": entry_id,
        "row_id": row["row_id"],
        "scheme": row["scheme"],
        "execution_backend": "tmux",
        "execution_host": execution_host,
        "execution_environment_sha256": execution_environment_sha256,
        "device": str(runtime.device),
        "official_test_read": False,
        "initialization_checkpoint_sha256": assets.checkpoint_sha256,
        "t": {
            "iteration_count": T_CORE_GRID[0],
            "batch_size": len(t_indices),
            "smooth_gradient_method": "quadratic_coefficients_2az_plus_b",
            "source_indices_sha256": sha256_json(list(t_indices)),
            "selection_residual_p90_by_layer": residual_p90,
        },
        "k": {
            "iteration_count": K_GRID[0],
            "batch_size": len(k_indices),
            "source_indices_sha256": sha256_json(list(k_indices)),
            "gradient_rms_by_parameter": gradient_rms,
        },
    }
    result_path = destination / "result.json"
    atomic_write_json(result_path, result, canonical=True)
    completion = {
        "schema_version": "mnist-conv-perfectdiode-tk-smoke-completion/v1",
        "state": "complete",
        "study_id": spec.study_id,
        "manifest_sha256": manifest_sha256,
        "outputs": [
            _artifact(destination / "environment.json", base=destination),
            _artifact(result_path, base=destination),
        ],
    }
    atomic_write_json(destination / "completion.json", completion, canonical=True)
    return result


__all__ = [
    "LoadedTKAssets",
    "PerfectDiodeTKRuntimeError",
    "TK_ASSET_COMPLETION_SCHEMA_VERSION",
    "TK_ASSET_SCHEMA_VERSION",
    "TK_K_MEASUREMENTS_SCHEMA_VERSION",
    "TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION",
    "TK_ROW_RESULT_SCHEMA_VERSION",
    "TK_SMOKE_SCHEMA_VERSION",
    "TK_T_MEASUREMENTS_SCHEMA_VERSION",
    "compare_k_gradients",
    "analytical_quadratic_layer_gradient",
    "execute_tk_row",
    "load_shared_assets",
    "measure_t_candidate",
    "parameters_in_contract_order",
    "perfect_diode_clamped_occupancy",
    "perfect_diode_projected_kkt_residual",
    "prepare_shared_assets",
    "run_operating_point_audit",
    "run_tk_smoke",
    "validate_shared_assets",
]
