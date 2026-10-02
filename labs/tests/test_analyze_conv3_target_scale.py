"""Fail-closed collection checks with tiny real saved checkpoint payloads."""
import itertools
import json

import pytest
import torch

from experiments import analyze_conv3_target_scale as analysis


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    # Bundle integrity has its own reporting tests; exercise scientific checks here.
    monkeypatch.setattr(analysis, "validate_run", lambda run: [])
    rates = [0.01, 0.02, 0.03, 0.04, 0., 0., 0.]
    payload = {"schema": [{"name": name} for name in
                           ["ConvWeight_0", "Bias_0", "Bias_1", "Bias_2"]],
               "states": [torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]}
    initializer = tmp_path / "source/init.pt"
    initializer.parent.mkdir()
    torch.save(payload, initializer)
    expected = dict(target_scale=1 / 16, lr=rates, init_checkpoint_path="init.pt",
                    initialization=dict(checkpoint_sha256=analysis.sha(initializer)))
    source = tmp_path / "source/config.json"
    write(source, expected)
    case = dict(index=0, scheme="baseline", target_scale=1 / 16, arm_id="baseline_16",
                path="config.json", sha256=analysis.sha(source))
    write(tmp_path / "paper_ready_results/bundles/table1_wide_bptt/conv3/baseline/seed0/config.json",
          dict(optimizer=dict(learning_rate=rates)))
    run = tmp_path / "local-canary/smoke/000_baseline"
    run.mkdir(parents=True)
    provenance = dict(train_batch_order_sha256=["order1", "order2", "order3"],
                      source_split="train", split_seed=0, shuffle_seed=0,
                      train_indices_sha256="a" * 64, validation_indices_sha256="b" * 64,
                      first_epoch_batch_order_sha256="c" * 64)
    resolved = dict(target_scale=1 / 16, training_algorithm="BP", seed=0,
                    init_checkpoint_path=str(initializer),
                    evaluation=dict(epoch_split="validation", checkpoint_selection="maximum_validation_accuracy",
                                    official_test_policy="disabled"),
                    training=dict(runtime_dtype="float32", epochs=3, batch_size=16,
                                  num_iterations_training=8, num_iterations_inference=8,
                                  batch_state_policy="reset_each_batch", optimizer_steps_applied=True,
                                  max_batches=2, max_test_batches=1),
                    dataset=dict(params=dict(batch_size=16, validation_batch_size=64, split_seed=0,
                                             shuffle_seed=0, affine_config=None, normalize=True,
                                             normalize_mean=0.1307, normalize_std=0.3081, normalize_scale=0.3),
                                 provenance=provenance),
                    architecture=dict(layer_shapes=[[2,28,28],[64,14,14],[128,7,7],[256,7,7],[20]],
                                      input_gain=360, non_linearity="perfect_diode",
                                      initial_parameter_state_sha256=analysis.model_state(initializer)),
                    bounds=dict(weight_min=0., weight_max=100.), amplification=dict(voltage_amp=1, current_amp=1),
                    minimizer=dict(adaptive_equilibrium=False),
                    optimizer=dict(name="Adam", lr_decay=1, learning_rate=rates),
                    diagnostics=dict(checkpoint_every_epoch=True))
    metrics = dict(best_epoch=2, best_test_accuracy=.8, best_validation_accuracy=.8,
                   final_test_accuracy=.7, final_test_loss=.001, final_train_loss=.002,
                   final_train_accuracy=.9, best_train_accuracy=.9,
                   official_test_evaluations=0, official_test_examples=0,
                   dataset_provenance=provenance, epoch_checkpoint_count=4, final_learning_rate=rates)
    write(run / "config.json", resolved)
    write(run / "metrics.json", metrics)
    write(run / "manifest.json", dict(configuration=dict(sha256=case["sha256"], resolved=expected),
                                      arm_id=case["arm_id"], smoke=True, dataset=dict(official_test_read=False)))
    write(run / "status.json", dict(state="complete"))
    write(run / "result.json", dict(completion=dict(criteria_met=True)))
    history = [dict(epoch=i, kind="epoch", evaluation_split="validation",
                    metrics=dict(train_loss=.002, train_accuracy=.9, validation_loss=.001,
                                 validation_accuracy=accuracy))
               for i, accuracy in enumerate([.6, .8, .7], 1)]
    (run / "metrics.jsonl").write_text("\n".join(map(json.dumps, history)))
    records = []
    for epoch in range(4):
        model = run / f"epoch_{epoch}_model.pt"
        optimizer = run / f"epoch_{epoch}_optimizer.pt"
        torch.save(payload, model)
        torch.save(dict(epoch=epoch, model_checkpoint_sha256=analysis.sha(model), continuation_state={},
                        optimizer_state_dict=dict(param_groups=[dict(lr=lr) for lr in rates])), optimizer)
        records.append(dict(epoch=epoch, model_path=model.name, optimizer_path=optimizer.name,
                            model_sha256=analysis.sha(model), optimizer_sha256=analysis.sha(optimizer)))
    (run / "epoch_checkpoint_index.jsonl").write_text("\n".join(map(json.dumps, records)))
    for name in ("best_model.pt", "final_model.pt"):
        torch.save(payload, run / name)
    return tmp_path, case, run


def test_smoke_validates_real_checkpoint_pairs(smoke):
    root, case, _ = smoke
    row, history, _ = analysis.validate_cell(root, case, 0, smoke=True)
    assert row["best_accuracy_percent"] == 80
    assert row["normalized_final_mse"] == .001 * 16 ** 2
    assert len(history) == 3


@pytest.mark.parametrize("section,key,value,message", [
    ("training", "batch_size", 8, "batch_size"),
    ("training", "epochs", 1, "epochs"),
    ("optimizer", "learning_rate", [.1] * 7, "rates"),
    ("architecture", "initial_parameter_state_sha256", "bad", "initial parameters"),
])
def test_rejects_actual_runtime_drift(smoke, section, key, value, message):
    root, case, run = smoke
    config = analysis.read(run / "config.json")
    config[section][key] = value
    write(run / "config.json", config)
    with pytest.raises(ValueError, match=message):
        analysis.validate_cell(root, case, 0, smoke=True)


def test_rejects_nonfinite_final_checkpoint(smoke):
    root, case, run = smoke
    path = run / "final_model.pt"
    payload = torch.load(path, weights_only=False)
    payload["states"][0][0] = float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.validate_cell(root, case, 0, smoke=True)


@pytest.mark.parametrize("filename,key,value,message", [
    ("config.json", "target_scale", 1., "target scale"),
    ("metrics.json", "official_test_evaluations", 1, "official-test access"),
])
def test_rejects_changed_target_or_test_access(smoke, filename, key, value, message):
    root, case, run = smoke
    payload = analysis.read(run / filename)
    payload[key] = value
    write(run / filename, payload)
    with pytest.raises(ValueError, match=message):
        analysis.validate_cell(root, case, 0, smoke=True)


def test_rejects_incomplete_epoch_checkpoints(smoke):
    root, case, run = smoke
    path = run / "epoch_checkpoint_index.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match="coverage"):
        analysis.validate_cell(root, case, 0, smoke=True)


def test_collect_rejects_cross_cell_mismatch_and_never_plots_smoke(tmp_path, monkeypatch):
    cases = [dict(index=i, scheme=scheme, target_scale=alpha, arm_id=str(i))
             for i, (scheme, alpha) in enumerate(itertools.product(analysis.SCHEMES, analysis.SCALES))]
    write(tmp_path / "preparation.json", dict(cases=cases))
    monkeypatch.setattr(analysis, "validate_cell", lambda root, case, index, smoke:
                        (dict(scheme=case["scheme"]), [], ("same" if index < 5 else "different",)))
    monkeypatch.setattr(analysis, "plot", lambda *args: pytest.fail("smoke must not plot"))
    summary = analysis.collect(tmp_path, smoke=True)
    assert not summary["complete"]
    assert summary["validated"] == 6
    assert "differs between cells" in summary["errors"][0]


def test_historical_ranking_reversal_is_labelled():
    rows = [dict(scheme=s, target_scale=a, best_accuracy_percent=(90 if s == "ours" else 80)
                 if a < 1 else (70 if s == "ours" else 80))
            for s, a in itertools.product(analysis.SCHEMES, (*analysis.SCALES, 1.))]
    assert all(change["uses_historical_reference"] for change in analysis.reversals(rows))
