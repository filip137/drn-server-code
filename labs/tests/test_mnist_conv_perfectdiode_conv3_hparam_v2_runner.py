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
