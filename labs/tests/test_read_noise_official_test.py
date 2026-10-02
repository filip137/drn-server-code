"""No real dataset or accelerator access: test held-out access semantics."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import evaluate_read_noise_official_test as runner
from experiments.reporting import sha256_file, start_run, validate_run


def bundle(path):
    start_run(path, {"study_id": "unit", "run_id": path.name, "arm_id": "case"})


def test_reservation_blocks_duplicate_after_partial_failure(tmp_path):
    guard = runner.reserve_test(tmp_path, "a" * 64, {"partial": True})
    with pytest.raises(RuntimeError, match="already reserved"):
        runner.reserve_test(tmp_path, "a" * 64, {"partial": False})
    assert json.loads(guard.read_text()) == {"partial": True}


@pytest.mark.parametrize("capacity", [1, 63, 64, 65, 199, 10000])
def test_capacity_search_finds_integer_boundary(capacity):
    tried = []
    result = runner.largest_fitting_batch(lambda size: tried.append(size) or size <= capacity, 10000)
    assert result == capacity
    assert len(tried) <= 23


def runtime():
    return {"network": None, "free_layers": [], "cost_fn": None, "minimizer": None,
            "energy_fn": SimpleNamespace(_device="cpu"), "runtime_dtype": torch.float64}


@pytest.mark.parametrize("count,passes", [(10000, True), (9999, False), (10001, False)])
def test_official_count_requires_exactly_ten_thousand(monkeypatch, tmp_path, count, passes):
    bundle(tmp_path)
    def evaluate(**kwargs):
        assert kwargs["max_batches"] is None
        n = len(kwargs["loader"][0][1])
        return 0.25, 0.75, n
    monkeypatch.setattr(runner.trainer, "_evaluate_image_loader", evaluate)
    batches = [(torch.zeros(min(400, count - i), 1), torch.zeros(min(400, count - i)))
               for i in range(0, count, 400)]
    if passes:
        result = runner.evaluate_batches(runtime(), batches, tmp_path, "test", 10000)
        assert result["examples"] == 10000
        assert result["correct"] == 7500
    else:
        with pytest.raises(RuntimeError, match="examples|count"):
            runner.evaluate_batches(runtime(), batches, tmp_path, "test", 10000)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == len(batches)


def fixture_contract(tmp_path):
    source_config = Path(__file__).resolve().parents[2] / "configs/conv/eqprop_conv3_p95_read_noise_1em3_20260921_v1/conv3_baseline_p95_sigma0_seed0.json"
    config = json.loads(source_config.read_text())
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "best.pt").write_bytes(b"selected-best")
    (tmp_path / "final.pt").write_bytes(b"different-final")
    case = {"arm_id": "case", "seed": 0, "depth": 3, "epochs": 30, "test_target": "unit",
            "config_path": "config.json", "config_sha256": sha256_file(tmp_path / "config.json"),
            "selected_checkpoint": {"path": "best.pt", "sha256": sha256_file(tmp_path / "best.pt"),
                                    "epoch": 12, "validation_accuracy": 0.98},
            "reuse_audit": {"completion_criteria_met": True, "smoke": False,
                            "result_official_test_read": False, "configured_official_test_policy": "disabled",
                            "result_evaluation_split": "validation"},
            "prior_official_test": {"official_test_evaluations": 0, "official_test_examples": 0}}
    contract = {"schema_version": runner.SCHEMA, "study_id": "unit", "frozen": True,
                "checkpoint_selection": runner.SELECTION, "cases": [case],
                "source_files": {"experiments/evaluate_read_noise_official_test.py": sha256_file(Path(runner.__file__))}}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path, contract


def test_selected_checkpoint_identity_rejects_substituted_final(tmp_path):
    path, contract = fixture_contract(tmp_path)
    result = runner.load_case(path, sha256_file(path), "case", tmp_path)
    assert result[3] == tmp_path / "best.pt"
    (tmp_path / "best.pt").write_bytes((tmp_path / "final.pt").read_bytes())
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_changed_seal_or_prior_test_refused(tmp_path):
    path, contract = fixture_contract(tmp_path)
    digest = sha256_file(path)
    contract["cases"][0]["prior_official_test"]["official_test_evaluations"] = 1
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="contract SHA256"):
        runner.load_case(path, digest, "case", tmp_path)
    with pytest.raises(ValueError, match="prior official test"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_probe_mode_never_constructs_official_loader(monkeypatch, tmp_path):
    path, _ = fixture_contract(tmp_path)
    called = []
    monkeypatch.setattr(runner, "_build_runtime", lambda **kw: called.append(kw) or {})
    monkeypatch.setattr(runner, "check_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "probe", lambda *args: {"batch_size": 91, "official_test_read": False})
    monkeypatch.setattr(runner, "official_loader", lambda *args: pytest.fail("probe opened official test"))
    monkeypatch.setattr(runner, "reserve_test", lambda *args: pytest.fail("probe reserved official test"))
    output = tmp_path / "probe"
    assert runner.main(["--contract", str(path), "--contract-sha256", sha256_file(path),
                        "--case-id", "case", "--mode", "probe", "--dataset-root", str(tmp_path),
                        "--run-dir", str(output), "--target", "unit", "--device", "cpu"]) == 0
    assert called[0]["checkpoint_path"] == tmp_path / "best.pt"
    assert json.loads((output / "result.json").read_text())["completion"]["official_test_evaluations"] == 0
    assert validate_run(output) == []


def test_runtime_rejects_precision_loss_and_nonzero_bias(tmp_path):
    weight = torch.tensor([1.00000000000001], dtype=torch.float64)
    bias = torch.zeros(1, dtype=torch.float64)
    checkpoint = tmp_path / "model.pt"
    torch.save({"states": [weight, bias]}, checkpoint)
    parameters = [SimpleNamespace(name="Weight_0", state=weight.clone()),
                  SimpleNamespace(name="Bias_0", state=bias.clone())]
    value = {"runtime_dtype_name": "float64", "energy_fn": SimpleNamespace(params=lambda: parameters)}
    runner.check_runtime(value, checkpoint)
    parameters[0].state = weight.float().double()
    with pytest.raises(ValueError, match="exact selected"):
        runner.check_runtime(value, checkpoint)
    parameters[0].state = weight.clone()
    parameters[1].state.fill_(0.1)
    with pytest.raises(ValueError, match="zero biases"):
        runner.check_runtime(value, checkpoint)


def final_fixture(tmp_path):
    path, contract = fixture_contract(tmp_path)
    contract.update(schema_version=runner.FINAL_SCHEMA, checkpoint_selection=runner.FINAL_SELECTION,
                    final_checkpoint_authorized=True, prior_best_test_exposure_recorded=True,
                    source_commit="f" * 40)
    case = contract["cases"][0]
    case["selected_checkpoint"]["epoch"] = 30
    case["prior_best_test"] = {"contract_sha256": "c" * 64, "checkpoint_sha256": "b" * 64,
                               "result_sha256": "d" * 64, "official_test_evaluations": 1,
                               "official_test_examples": 10000}
    path.write_text(json.dumps(contract))
    return path, contract


def test_final_requires_complete_final_epoch_and_explicit_exposure(tmp_path):
    path, contract = final_fixture(tmp_path)
    runner.load_case(path, sha256_file(path), "case", tmp_path)
    case = contract["cases"][0]
    case["selected_checkpoint"]["epoch"] = 29
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="complete native epoch"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)
    case["selected_checkpoint"]["epoch"] = 30
    del case["prior_best_test"]
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="prior-best test exposure"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_final_cannot_retest_identical_best_checkpoint(tmp_path):
    path, contract = final_fixture(tmp_path)
    case = contract["cases"][0]
    case["prior_best_test"]["checkpoint_sha256"] = case["selected_checkpoint"]["sha256"]
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="reuse the prior result"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_final_declared_tensor_identical_reuse_cannot_launch(tmp_path):
    path, contract = final_fixture(tmp_path)
    contract["cases"][0]["reuse"] = {"tensor_identical": True, "serialization_hashes_differ": True}
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="declares result reuse"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_best_contract_does_not_accept_final_selection_or_previous_test(tmp_path):
    path, contract = fixture_contract(tmp_path)
    contract["checkpoint_selection"] = runner.FINAL_SELECTION
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="selection schema"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)
    contract["checkpoint_selection"] = runner.SELECTION
    contract["cases"][0]["prior_official_test"]["official_test_examples"] = 10000
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="prior official test"):
        runner.load_case(path, sha256_file(path), "case", tmp_path)


def test_capacity_carryforward_requires_receipt_and_unchanged_inference(tmp_path):
    path, contract = final_fixture(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    old_files = {"model/a.py": "a" * 64, "labs/datasets.py": "b" * 64,
                 "experiments/replay_conv_checkpoint_validation.py": "c" * 64}
    contract["source_files"].update(old_files)
    receipt = {"contract_sha256": "old-contract", "capacity_signature": runner.capacity_signature(config),
               "target": "unit", "official_test_read": False, "batch_size": 10000}
    record = {"receipt_sha256": "receipt", "contract_sha256": "old-contract",
              "capacity_signature": receipt["capacity_signature"], "target": "unit",
              "source_files": old_files.copy()}
    contract["capacity_carryforward"] = [record]
    runner.validate_capacity_receipt(contract, "new-contract", receipt, "receipt", config, "unit")
    with pytest.raises(ValueError, match="explicitly sealed"):
        runner.validate_capacity_receipt(contract, "new-contract", receipt, "tampered", config, "unit")
    record["source_files"]["model/a.py"] = "changed"
    with pytest.raises(ValueError, match="inference source differs"):
        runner.validate_capacity_receipt(contract, "new-contract", receipt, "receipt", config, "unit")
    record["source_files"]["model/a.py"] = old_files["model/a.py"]
    contract["schema_version"] = runner.SCHEMA
    with pytest.raises(ValueError, match="Best-checkpoint contract cannot"):
        runner.validate_capacity_receipt(contract, "new-contract", receipt, "receipt", config, "unit")


def test_missing_test_files_are_checked_without_opening_data(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("preflight read test bytes"))
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("preflight opened test data"))
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        runner.preflight_test_files(tmp_path)
    directory = tmp_path / "MNIST" / "raw"
    directory.mkdir(parents=True)
    # Builtin open is used only to manufacture fixture data, not by preflight.
    for name in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        with open(directory / name, "wb") as stream:
            stream.write(b"fixture")
    runner.preflight_test_files(tmp_path)


def test_final_uses_sealed_identity_without_git_checkout(tmp_path, monkeypatch):
    path, _ = final_fixture(tmp_path)
    monkeypatch.setattr(runner, "_git_state", lambda: pytest.fail("snapshot has no Git checkout"))
    monkeypatch.setattr(runner, "_build_runtime", lambda **kwargs: {})
    monkeypatch.setattr(runner, "check_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "probe", lambda *args: {"batch_size": 10000, "official_test_read": False})
    output = tmp_path / "final-probe"
    runner.main(["--contract", str(path), "--contract-sha256", sha256_file(path),
                 "--case-id", "case", "--mode", "probe", "--dataset-root", str(tmp_path),
                 "--run-dir", str(output), "--target", "unit", "--device", "cpu"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["git"]["commit"] == "f" * 40
    assert manifest["git"]["dirty"] is None
    assert manifest["pristine_holdout"] is False
    assert manifest["checkpoint_selection"] == "final_epoch"
    assert manifest["prior_best_test"]["official_test_examples"] == 10000
