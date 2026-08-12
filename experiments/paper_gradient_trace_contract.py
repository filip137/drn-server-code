"""Frozen provenance contract for the 2026-07-31 gradient-trace study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_STUDY_ID = "perfectdiode-paper-gradient-trace-limited-20260731-v1"
FROZEN_SOURCE_COMMIT = "ade6248a5e9d80ec8157b8a5273f095e6b30715b"
FROZEN_SOURCE_ARCHIVE_SHA256 = (
    "f7f9d80b439ba8f0f4d1a3ed1e1dde4b63135f4b5e9fc5e028e9c76ee4e1df5a"
)
SOURCE_CONFIG_ROOT = (
    REPOSITORY_ROOT
    / "configs"
    / "conv"
    / "paper_medium_affine_perfectdiode_wide_seed0_20260729_v1"
)
ARM_PATTERN = re.compile(
    r"^(conv[123])_(baseline|ours|legacy)_(sgd|adam)_seed(\d+)$"
)
CONFIG_PREFIX = {
    ("baseline", "sgd"): "00",
    ("baseline", "adam"): "01",
    ("ours", "sgd"): "02",
    ("ours", "adam"): "03",
    ("legacy", "sgd"): "04",
    ("legacy", "adam"): "05",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_source_config_path(arm_id: str) -> Path:
    match = ARM_PATTERN.fullmatch(str(arm_id))
    if match is None:
        raise ValueError(f"Unexpected frozen gradient-trace arm id: {arm_id!r}.")
    architecture, scheme, optimizer, seed_text = match.groups()
    if int(seed_text) != 0:
        raise ValueError(f"Expected frozen seed zero; found {arm_id!r}.")
    prefix = CONFIG_PREFIX[(scheme, optimizer)]
    path = SOURCE_CONFIG_ROOT / architecture / (
        f"{prefix}_{scheme}_{optimizer}_seed0.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen source config for {arm_id}: {path}")
    return path


def validate_frozen_manifest(
    manifest: Mapping[str, Any],
    *,
    arm_id: str,
) -> dict[str, Any]:
    """Bind one run manifest to the immutable source/config study contract."""

    if manifest.get("study_id") != FROZEN_STUDY_ID:
        raise ValueError(
            f"Unexpected study id for {arm_id}: {manifest.get('study_id')!r}."
        )
    if bool(manifest.get("smoke")):
        raise ValueError(f"Refusing a smoke manifest for frozen arm {arm_id}.")

    git = manifest.get("git")
    if not isinstance(git, Mapping):
        raise ValueError(f"Missing frozen git provenance for {arm_id}.")
    if git.get("commit") != FROZEN_SOURCE_COMMIT:
        raise ValueError(
            f"Frozen source commit mismatch for {arm_id}: {git.get('commit')!r}."
        )
    if git.get("source_archive_sha256") != FROZEN_SOURCE_ARCHIVE_SHA256:
        raise ValueError(
            "Frozen source archive SHA-256 mismatch for "
            f"{arm_id}: {git.get('source_archive_sha256')!r}."
        )

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError(f"Missing manifest configuration for {arm_id}.")
    expected_path = expected_source_config_path(arm_id)
    expected_sha256 = _sha256_file(expected_path)
    if configuration.get("sha256") != expected_sha256:
        raise ValueError(
            f"Frozen source-config SHA-256 mismatch for {arm_id}: "
            f"expected={expected_sha256}, actual={configuration.get('sha256')!r}."
        )
    recorded_path = Path(str(configuration.get("path", "")))
    if recorded_path.name != expected_path.name:
        raise ValueError(
            f"Frozen source-config filename mismatch for {arm_id}: "
            f"expected={expected_path.name!r}, actual={recorded_path.name!r}."
        )
    expected_config = json.loads(expected_path.read_text(encoding="utf-8"))
    if configuration.get("resolved") != expected_config:
        raise ValueError(
            f"Resolved manifest configuration differs from frozen source for {arm_id}."
        )
    configured_epochs = int(expected_config["lab"]["epochs"])
    if int(configuration.get("configured_epochs", -1)) != configured_epochs:
        raise ValueError(f"Configured epoch budget mismatch for {arm_id}.")
    if int(configuration.get("epochs", -1)) != 10:
        raise ValueError(f"Expected ten executed diagnostic epochs for {arm_id}.")
    expected_diagnostics = {
        "gradient_trace_samples_per_epoch": 5,
        "checkpoint_every_epoch": True,
        "skip_terminal_official_test": True,
    }
    if configuration.get("diagnostic_overrides") != expected_diagnostics:
        raise ValueError(f"Diagnostic override mismatch for {arm_id}.")

    return {
        "status": "pass",
        "study_id": FROZEN_STUDY_ID,
        "arm_id": arm_id,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "source_archive_sha256": FROZEN_SOURCE_ARCHIVE_SHA256,
        "source_config_path": str(expected_path.resolve()),
        "source_config_sha256": expected_sha256,
        "configured_epochs": configured_epochs,
        "executed_epochs": 10,
        "diagnostic_overrides": expected_diagnostics,
    }
