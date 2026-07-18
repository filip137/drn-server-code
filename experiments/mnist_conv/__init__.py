"""Canonical v1 MNIST BP resistive-Conv experiment workflow."""

from .backend import ExecutionContext, TrainingBackend
from .collection import collect_sweep
from .identity import code_fingerprint, code_identity, code_provenance, run_fingerprint
from .layout import ResultLayout
from .manifest import build_manifest, load_manifest, publish_manifest
from .runner import ExecutionPolicy, RunExecution, execute_run, validate_bundle
from .specs import RunSpec, SpecValidationError, SweepSpec

__all__ = [
    "ExecutionContext", "ExecutionPolicy", "ResultLayout", "RunExecution",
    "RunSpec", "SpecValidationError", "SweepSpec", "TrainingBackend",
    "build_manifest", "code_fingerprint", "code_identity", "code_provenance",
    "collect_sweep", "execute_run", "load_manifest", "publish_manifest",
    "run_fingerprint", "validate_bundle",
]
