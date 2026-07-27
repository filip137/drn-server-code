from __future__ import annotations

from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPO_ROOT
    / "experiments"
    / "run_mnist_conv_perfectdiode_conv3_hparam_v2_local_20260727_v1.sh"
)
T8K8_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "run_mnist_conv_perfectdiode_conv3_hparam_t8k8_local_20260727_v4.sh"
)


def _array(text: str, name: str) -> list[str]:
    match = re.search(
        rf"(?ms)^{re.escape(name)}=\(\n(?P<body>.*?)^\)\n",
        text,
    )
    assert match is not None
    return [
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip()
    ]


def test_runner_has_strict_shell_contract_and_parses() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    assert "--preflight)" in text
    assert "--tk-gate)" in text
    assert "--smoke)" in text
    assert "--lane)" in text
    assert "--dispatch)" in text


def test_runner_freezes_surface_major_host_routing() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert _array(text, "main_surfaces") == [
        "conv3_baseline_v1_c1--sgd",
        "conv3_baseline_v1_c1--adam",
        "conv3_legacy_v4_c0p25--sgd",
        "conv3_legacy_v4_c0p25--adam",
    ]
    assert _array(text, "akib_surfaces") == [
        "conv3_ours_v4_c1--sgd",
        "conv3_ours_v4_c1--adam",
    ]
    assert "trex" not in text.lower()
    assert "jean" not in text.lower()


def test_runner_freezes_all_public_stages_in_order() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert _array(text, "stages") == [
        "audit",
        "import_tk",
        "optimizer_probe",
        "rho_canary_core",
        "rho_core_candidates",
        "select_core",
        "rho_canary_expansion",
        "rho_expansion_candidates",
        "select_expanded",
        "post_training_tk",
        "finalize_lr",
    ]
    assert 'for surface in "${main_surfaces[@]}"; do' in text
    assert 'for surface in "${akib_surfaces[@]}"; do' in text
    assert 'for stage in "${stages[@]}"; do' in text


def test_runner_uses_manifest_bound_worker_prefixes_and_host_smokes() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert text.count("local_worker_common=(") == 1
    assert text.count("remote_worker_common=(") == 1
    assert "--study \"${study}\"" in text
    assert "--surface-manifest \"${surface_manifest}\"" in text
    assert "--study \"${remote_study}\"" in text
    assert "--surface-manifest \"${remote_surface_manifest}\"" in text
    assert "conv3_baseline_v1_c1--adam" in text
    assert "conv3_ours_v4_c1--adam" in text
    assert text.count("--preflight-canary") == 2
    assert "PD_LR_TMUX_SESSION=akibscomputer" in text
    assert "PD_LR_TMUX_PANE_TOKEN=" in text


def test_dispatch_requires_receipt_tk_gate_smokes_and_idle_panes() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    dispatch = text.split("dispatch() {", 1)[1].split("\n}\n", 1)[0]
    assert "scheduled_receipt" in dispatch
    assert "receipt_runner_sha" in dispatch
    assert "preflight" in dispatch
    assert "representative_canary/main/completion.json" in dispatch
    assert (
        "representative_canary/akibscomputer/completion.json" in dispatch
    )
    assert "smoke_main" in dispatch
    assert "smoke_akibscomputer" in dispatch
    assert "require_idle_pane main:0.0" in dispatch
    assert "require_idle_pane akibscomputer:0.0" in dispatch
    assert 'tmux send-keys -t main:0.0' in dispatch
    assert 'tmux send-keys -t akibscomputer:0.0' in dispatch


def _function(text: str, name: str) -> str:
    return text.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]


def test_t8k8_runner_has_strict_versioned_shell_contract() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    subprocess.run(["bash", "-n", str(T8K8_RUNNER)], check=True)
    assert (
        "results/perfectdiode_conv3_lr_t8k8_20260727_v5"
        in text
    )
    assert (
        "lrstudy_294f233afd76a4075482d5e6d43c36f74739e63b88ecdf5cc8b6e3091fa323f6"
        in text
    )
    assert "/tmp/pd_conv3_lr_t8k8_20260727_source_v5" in text
    assert (
        "/home/filiposana/staged/pd_conv3_lr_t8k8_20260727_source_v5"
        in text
    )
    assert (
        "perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5.md"
        in text
    )


def test_t8k8_runner_preserves_surface_major_host_routing_and_stages() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    assert _array(text, "main_surfaces") == [
        "conv3_baseline_v1_c1--sgd",
        "conv3_baseline_v1_c1--adam",
        "conv3_legacy_v4_c0p25--sgd",
        "conv3_legacy_v4_c0p25--adam",
    ]
    assert _array(text, "akib_surfaces") == [
        "conv3_ours_v4_c1--sgd",
        "conv3_ours_v4_c1--adam",
    ]
    assert _array(text, "stages") == [
        "audit",
        "import_tk",
        "optimizer_probe",
        "rho_canary_core",
        "rho_core_candidates",
        "select_core",
        "rho_canary_expansion",
        "rho_expansion_candidates",
        "select_expanded",
        "post_training_tk",
        "finalize_lr",
    ]
    assert "trex" not in text.lower()
    assert "jean" not in text.lower()


def test_t8k8_runner_uses_three_fixed_worker_gates_not_tk_producer() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    main_gate = _function(text, "tk_gate_main")
    akib_gate = _function(text, "tk_gate_akibscomputer")
    assert "conv3_baseline_v1_c1--adam" in main_gate
    assert "conv3_legacy_v4_c0p25--adam" in main_gate
    assert main_gate.count("--preflight-tk-gate") == 2
    assert "conv3_ours_v4_c1--adam" in akib_gate
    assert akib_gate.count("--preflight-tk-gate") == 1
    assert "run_mnist_conv_perfectdiode_tk.py" not in text
    assert "run-entry" not in text
    assert "tk_gate_manifest" not in text
    assert "selected_t\") == 4" not in text
    assert "selected_k\") == 4" not in text


def test_t8k8_preflight_is_read_only_and_strictly_validates_receipts() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    preflight = _function(text, "preflight")
    assert "verify_scheduled_receipt" not in preflight
    assert "validate_fixed_tk_main" in preflight
    assert "validate_fixed_tk_akibscomputer" in preflight
    assert "validate_smoke_receipts" in preflight
    assert "fixed_tk_gate/main/conv3_baseline_v1_c1" in text
    assert "fixed_tk_gate/main/conv3_legacy_v4_c0p25" in text
    assert (
        "fixed_tk_gate/akibscomputer/conv3_ours_v4_c1"
        in text
    )
    assert "representative_canary/main/completion.json" in text
    assert (
        "representative_canary/akibscomputer/completion.json"
        in text
    )
    assert "mkdir " not in preflight
    assert "tee " not in preflight
    assert "tmux send-keys" not in preflight
    assert "--validate-preflight-tk-gate" in text
    assert "--validate-preflight-canary" in text
    assert "tk_gate_main >/dev/null" not in text
    assert "smoke_main >/dev/null" not in text


def test_t8k8_lanes_validate_gate_and_tracker_before_writes() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    main_lane = _function(text, "run_main_lane")
    akib_lane = _function(text, "run_akibscomputer_lane")
    assert main_lane.index("validate_fixed_tk_main") < main_lane.index(
        "mkdir -p"
    )
    assert main_lane.index("require_tracker_launch_ready") < main_lane.index(
        "mkdir -p"
    )
    assert akib_lane.index(
        "validate_fixed_tk_akibscomputer"
    ) < akib_lane.index("mkdir -p")
    assert akib_lane.index(
        "require_tracker_launch_ready"
    ) < akib_lane.index("mkdir -p")
    assert 'for surface in "${main_surfaces[@]}"; do' in main_lane
    assert 'for surface in "${akib_surfaces[@]}"; do' in akib_lane


def test_t8k8_dispatch_gates_first_side_effect_on_exact_tracker_entry() -> None:
    text = T8K8_RUNNER.read_text(encoding="utf-8")
    tracker_gate = _function(text, "require_tracker_launch_ready")
    dispatch = _function(text, "dispatch")
    receipt_check = _function(text, "verify_scheduled_receipt")
    assert "runner_sha256" in receipt_check
    assert "--require-experiment-id \"${experiment_id}\"" in tracker_gate
    assert "--require-launch-ready" in tracker_gate
    assert (
        "experiment_id=perfectdiode-conv3-lr-ordinary-mnist-t8k8-20260727-v5"
        in text
    )
    for required in (
        "preflight",
        "verify_scheduled_receipt",
        "validate_fixed_tk_main",
        "validate_fixed_tk_akibscomputer",
        "validate_smoke_receipts",
        "require_idle_pane main:0.0",
        "require_idle_pane akibscomputer:0.0",
        "require_tracker_launch_ready",
    ):
        assert required in dispatch
    first_send = dispatch.index("tmux send-keys")
    assert dispatch.index("preflight") < dispatch.index(
        "verify_scheduled_receipt"
    )
    assert dispatch.index("verify_scheduled_receipt") < first_send
    assert dispatch.index("require_tracker_launch_ready") < first_send
    assert 'tmux send-keys -t main:0.0' in dispatch
    assert 'tmux send-keys -t akibscomputer:0.0' in dispatch
