"""Validate coverage, frozen controls and validation-only selections."""
import csv
import hashlib
import json
from pathlib import Path
import torch

WORK = Path(__file__).resolve().parents[3]
ROOT = WORK / "hwa-noise-sweep-20260908"

def read(path):
    return json.loads(path.read_text())

terminal = read(ROOT / "terminal.json")
assert terminal["status"] == "complete"
assert all(r["exit_code"] == 0 and r["status"] == "complete" for r in terminal["cases"])
assert hashlib.sha256(Path(terminal["source"]).read_bytes()).hexdigest() == terminal["source_sha256"]
csv_rows = list(csv.DictReader((ROOT / "analysis/metrics.csv").open()))
assert len(csv_rows) == 8
reports = {}
for arch in ("crossbar", "drn"):
    run = ROOT / "runs" / arch
    result = read(run / "result.json")
    manifest = read(run / "manifest.json")
    control = read(ROOT / "checks" / f"{arch}_unchanged_controls.json")
    assert result["status"] == "complete" and not result["smoke"]
    assert manifest["source_sha256"] == terminal["source_sha256"]
    assert manifest["epochs"] == 10 and manifest["actual_cuda_canary_passed"]
    assert not manifest["deployment_noise_scaled"] and not manifest["calibration_refit"]
    assert control["exact_original_PV_apparent_state"] and control["exact_original_PV_persistent_state"]
    for name in ("teacher", "original_hwa", "prepared"):
        source = manifest["architecture_inputs"].get(name)
        if source:
            assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    assert len(history) == 40
    for epoch in range(1, 11):
        rows = [r for r in history if r["epoch"] == epoch]
        assert sorted(r["multiplier"] for r in rows) == [0, 0.25, 0.5, 1]
        assert {r["training_examples"] for r in rows} == {55000}
        assert len({r["batch_order_sha256"] for r in rows}) == 1
        assert all(r["post_pv_validation"]["apparent"]["examples"] == 5000 for r in rows)
    for case in result["results"]:
        rows = [r for r in history if r["multiplier"] == case["multiplier"]]
        best = min(rows, key=lambda r: (r["post_pv_validation"]["apparent"]["kl_teacher_student"],
            -r["post_pv_validation"]["apparent"]["student_accuracy"], r["epoch"]))
        assert case["selected_epoch"] == best["epoch"]
        token = format(case["multiplier"], "g").replace(".", "p")
        selected_path = run / f"noise_{token}" / "selected.pt"
        saved = torch.load(selected_path, map_location="cpu", weights_only=False)
        epoch_saved = torch.load(selected_path.parent / f"epoch_{case['selected_epoch']:02d}.pt", map_location="cpu", weights_only=False)
        assert all(torch.equal(a, b) for a, b in zip(saved["master"], epoch_saved["master"], strict=True))
        assert all(torch.isfinite(t).all() and t.abs().max() <= 1 for t in saved["master"])
        assert (run / "selection.json").stat().st_mtime_ns < (selected_path.parent / "result.json").stat().st_mtime_ns
        csv_row = next(r for r in csv_rows if r["architecture"] == arch and float(r["noise_multiplier"]) == case["multiplier"])
        for prefix, metric in (("clean", case["clean_test"]), ("healthy_pv", case["healthy_pv_test"]["apparent"]),
                               ("faulted_pv", case["faulted_pv_test"]["apparent"])):
            assert metric["examples"] == 10000
            assert abs(float(csv_row[prefix+"_KL"]) - metric["kl_teacher_student"]) < 1e-12
            assert abs(float(csv_row[prefix+"_accuracy_percent"]) - 100*metric["student_accuracy"]) < 1e-10
    selected = min(result["results"], key=lambda r: (r["selected_validation"]["kl_teacher_student"],
        -r["selected_validation"]["student_accuracy"], r["multiplier"]))
    assert selected["multiplier"] == result["selected_multiplier"]
    reports[arch] = dict(epochs=10, cases=4, history_rows=40, train_examples_per_case=550000,
        validation_examples=5000, test_examples=10000, matched_batch_order=True,
        selected_multiplier=result["selected_multiplier"], selected_checkpoints_and_CSV_match=True,
        original_input_hashes_unchanged=True, exact_PV_control_replay=True)
for stem in ("hwa_noise_sweep_accuracy", "hwa_noise_sweep_kl"):
    for ext in ("png", "pdf", "svg"):
        a = ROOT / "analysis" / f"{stem}.{ext}"
        b = WORK / a.name
        assert a.read_bytes() == b.read_bytes()
verification = dict(status="passed", coverage=reports, figures_copied_exactly=True,
    selection_frozen_before_test=True, evidence_class="exploratory_noncanonical_model_based")
(ROOT / "analysis/verification.json").write_text(json.dumps(verification, indent=2)+"\n")
print(json.dumps(verification))
