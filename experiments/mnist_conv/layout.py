"""Global content-addressed result-store layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_RUN_ID = re.compile(r"run_[0-9a-f]{64}\Z")
_SWEEP_ID = re.compile(r"sweep_[0-9a-f]{64}\Z")
_LR_STUDY_ID = re.compile(r"lrstudy_[0-9a-f]{64}\Z")


def _validated_id(value: str, pattern: re.Pattern[str], kind: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(
            f"Expected {kind} id in canonical SHA-256 form. Provided value: {value!r}."
        )
    return value


def safe_label(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty result label. Provided value: {value!r}.")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    if not cleaned:
        raise ValueError(f"Expected a label containing path-safe characters. Provided value: {value!r}.")
    return cleaned


@dataclass(frozen=True)
class ResultLayout:
    """Paths beneath one global result root.

    A run is stored once even when several sweep manifests refer to it.  Failed
    and interrupted attempts remain outside the immutable published run.
    """

    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root).expanduser().resolve())

    @property
    def runs_root(self) -> Path:
        return self.root / "runs"

    @property
    def sweeps_root(self) -> Path:
        return self.root / "sweeps"

    @property
    def attempts_root(self) -> Path:
        return self.root / "attempts"

    @property
    def lr_studies_root(self) -> Path:
        return self.root / "lr_studies"

    def run_dir(self, label: str, run_id: str) -> Path:
        return self.runs_root / f"{safe_label(label)}--{_validated_id(run_id, _RUN_ID, 'run')}"

    def find_run_dir(self, run_id: str) -> Path | None:
        """Find a published run by identity without depending on its display label."""

        canonical_id = _validated_id(run_id, _RUN_ID, "run")
        matches = sorted(self.runs_root.glob(f"*--{canonical_id}")) if self.runs_root.exists() else []
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most one published directory for {canonical_id}. Provided value: {matches!r}."
            )
        return matches[0] if matches else None

    def sweep_dir(self, name: str, sweep_id: str) -> Path:
        return self.sweeps_root / f"{safe_label(name)}--{_validated_id(sweep_id, _SWEEP_ID, 'sweep')}"

    def find_sweep_dir(self, sweep_id: str) -> Path | None:
        canonical_id = _validated_id(sweep_id, _SWEEP_ID, "sweep")
        matches = sorted(self.sweeps_root.glob(f"*--{canonical_id}")) if self.sweeps_root.exists() else []
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most one sweep directory for {canonical_id}. Provided value: {matches!r}."
            )
        return matches[0] if matches else None

    def lr_study_dir(self, name: str, study_id: str) -> Path:
        return self.lr_studies_root / (
            f"{safe_label(name)}--{_validated_id(study_id, _LR_STUDY_ID, 'LR study')}"
        )

    def find_lr_study_dir(self, study_id: str) -> Path | None:
        canonical_id = _validated_id(study_id, _LR_STUDY_ID, "LR study")
        matches = (
            sorted(self.lr_studies_root.glob(f"*--{canonical_id}"))
            if self.lr_studies_root.exists()
            else []
        )
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected at most one published directory for {canonical_id}. "
                f"Provided value: {matches!r}."
            )
        return matches[0] if matches else None

    def attempt_root(self, run_id: str) -> Path:
        return self.attempts_root / _validated_id(run_id, _RUN_ID, "run")

    def attempt_dir(self, run_id: str, attempt_id: str) -> Path:
        if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id:
            raise ValueError(
                f"Expected attempt id to be one safe path component. Provided value: {attempt_id!r}."
            )
        return self.attempt_root(run_id) / attempt_id

    def claim_path(self, run_id: str) -> Path:
        return self.attempt_root(run_id) / ".claim.json"

    def manifest_path(self, name: str, sweep_id: str) -> Path:
        return self.sweep_dir(name, sweep_id) / "manifest.json"

    def ensure_roots(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.sweeps_root.mkdir(parents=True, exist_ok=True)
        self.attempts_root.mkdir(parents=True, exist_ok=True)
        self.lr_studies_root.mkdir(parents=True, exist_ok=True)
