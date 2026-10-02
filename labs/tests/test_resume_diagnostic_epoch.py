import json
from types import SimpleNamespace

import pytest
import torch

from experiments.epoch_continuation import EpochContinuation, continuation_contract
from experiments.resume_diagnostic_epoch import prepare_diagnostic_resume
from experiments.reporting import sha256_file
from labs.mnist_train import _save_epoch_diagnostic_checkpoint


def fixture_run(tmp_path, full_state=False):
    config = dict(lab=dict(epochs=30, dataset_key="mnist"),
        datasets=dict(mnist=dict(factory="labs.datasets.MnistTrainValidationDataset",
            params=dict(shuffle_seed=3, batch_size=16, num_workers=0))),
        batch_state_policy="reset_each_batch", parameter_order=["weight"], lr=[.01],
        learning_rates_by_parameter=dict(weight=.01),
        optimizer=dict(name="Adam", learning_rate=[.01], lr_decay=1., momentum=0.,
            weight_decay=0., betas=[.9, .999], eps=1e-8, amsgrad=False,
            foreach=False, fused=False, maximize=False, capturable=False, differentiable=False), max_batches=1,
        evaluation=dict(official_test=dict(policy="disabled")))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(dict(
        configuration=dict(sha256=sha256_file(path), resolved=config, epochs=30,
                           diagnostic_overrides={}), dataset=dict(official_test_read=False))))
    p = SimpleNamespace(name="weight", state=torch.nn.Parameter(torch.tensor([.7], dtype=torch.float64)))
    opt = torch.optim.Adam([dict(params=[p.state], lr=.01)], foreach=False, fused=False)

    class Energy:
        def save(self, path):
            torch.save(dict(format="drn.function.parameters", version=1,
                schema=[dict(name="weight", type="test.Weight", shape=[1], dtype="torch.float64")],
                states=[p.state.detach().clone()]), path)

    energy = Energy()
    loaders = SimpleNamespace(_train_sampler=SimpleNamespace(generator=torch.Generator().manual_seed(3)),
                              _worker_generator=torch.Generator().manual_seed(8))
    estimator = SimpleNamespace(_endpoint_read_noise_generators={"cpu": torch.Generator().manual_seed(19)},
                                endpoint_read_noise_draw_count=0)
    continuation = SimpleNamespace(loaders=loaders, estimator=estimator,
        contract=continuation_contract(config, epochs=30, max_batches=1,
            max_test_batches=None, device="cpu", gradient_trace_samples_per_epoch=0))
    for epoch in range(29):
        if epoch:
            step(p, opt, epoch)
            torch.randperm(17, generator=loaders._train_sampler.generator)
            torch.rand(3, generator=loaders._worker_generator)
            torch.randn(11, generator=estimator._endpoint_read_noise_generators["cpu"])
            estimator.endpoint_read_noise_draw_count += 11
        _save_epoch_diagnostic_checkpoint(run_dir=source, epoch=epoch,
            energy_fn=energy, optimizer=opt, parameters=[p],
            continuation=continuation if full_state else None)
    (source / "metrics.jsonl").write_text("".join(json.dumps(dict(
        kind="epoch", epoch=e, evaluation_split="validation",
        metrics=dict(train_loss=1/e, train_accuracy=.6, validation_loss=.4,
                     validation_accuracy=.9 if e == 7 else .8,
                     best_epoch=7 if e >= 7 else 1))) + "\n" for e in range(1,29)))
    return source, path, config, p, opt, energy


def step(p, opt, epoch):
    opt.zero_grad()
    p.state.grad = torch.full_like(p.state, 1 / epoch)
    opt.step()


def test_epoch28_restore_preserves_adam_and_finishes_original_horizon(tmp_path, monkeypatch):
    source, path, config, p, opt, energy = fixture_run(tmp_path)
    before = {str(f.relative_to(source)): sha256_file(f) for f in source.rglob("*") if f.is_file()}
    # Preparing recovery never constructs either train or official-test data.
    import labs.datasets
    monkeypatch.setattr(labs.datasets, "build_mnist_official_test_loader",
                        lambda **kwargs: pytest.fail("official test accessed"))
    target = tmp_path / "recovery"
    receipt = prepare_diagnostic_resume(source_run=source, run_dir=target,
        config_path=path, device="cpu")
    assert receipt["completed_epochs"] == 28 and receipt["remaining_epochs"] == 2
    assert receipt["continuation_quality"] == "approximate_stochastic_continuation"
    assert receipt["uninterrupted_trajectory"] is False
    assert sha256_file(target / "best_model.pt") == sha256_file(source / "checkpoints/epoch_007_model.pt")
    other = SimpleNamespace(name="weight", state=torch.nn.Parameter(torch.tensor([99.], dtype=torch.float64)))
    other_opt = torch.optim.Adam([dict(params=[other.state], lr=.01)])
    loaders = SimpleNamespace(_train_sampler=SimpleNamespace(generator=torch.Generator()),
                              _worker_generator=torch.Generator())
    cont = EpochContinuation(run_dir=target, config=config, epochs=30, chunk_epochs=30,
        parameters=[other], optimizer=other_opt, estimator=object(), loaders=loaders,
        recorder=None, contract=continuation_contract(config, epochs=30, max_batches=1,
            max_test_batches=None, device="cpu", gradient_trace_samples_per_epoch=0))
    saved = cont.restore()
    assert torch.equal(other.state, p.state)
    for key, value in opt.state[p.state].items():
        assert torch.equal(value, other_opt.state[other.state][key])
    assert len(saved["history"]["loss"]) == 28 and saved["best_epoch"] == 7
    assert list(range(cont.start + 1, cont.epochs + 1)) == [29, 30]
    for epoch in range(cont.start + 1, cont.epochs + 1):
        step(p, opt, epoch)
        step(other, other_opt, epoch)
        assert torch.equal(p.state, other.state)
        assert not cont.pause_after(epoch, history=saved["history"], best_accuracy=.9, best_epoch=7)
        _save_epoch_diagnostic_checkpoint(run_dir=target, epoch=epoch,
            energy_fn=energy, optimizer=other_opt, parameters=[other])
    assert other_opt.state[other.state]["step"].item() == 30
    index = [json.loads(line)["epoch"] for line in (target / "epoch_checkpoint_index.jsonl").read_text().splitlines()]
    assert index == list(range(31))
    assert before == {str(f.relative_to(source)): sha256_file(f) for f in source.rglob("*") if f.is_file()}


@pytest.mark.parametrize("damage", ["model_hash", "parameter_order", "adam_step", "official_test"])
def test_refuses_incompatible_or_damaged_recovery(tmp_path, damage):
    source, path, config, *_ = fixture_run(tmp_path)
    if damage == "model_hash":
        (source / "checkpoints/epoch_028_model.pt").write_bytes(b"damaged")
    elif damage == "official_test":
        config["evaluation"]["official_test"]["policy"] = "terminal"
        path.write_text(json.dumps(config))
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["configuration"].update(sha256=sha256_file(path), resolved=config)
        (source / "manifest.json").write_text(json.dumps(manifest))
    else:
        optimizer_path = source / "checkpoints/epoch_028_optimizer.pt"
        state = torch.load(optimizer_path, weights_only=False)
        if damage == "parameter_order":
            state["parameter_names"] = ["wrong"]
        else:
            state["optimizer_state_dict"]["state"][0]["step"].fill_(1)
        torch.save(state, optimizer_path)
        index = [json.loads(line) for line in (source / "epoch_checkpoint_index.jsonl").read_text().splitlines()]
        index[-1].update(optimizer_sha256=sha256_file(optimizer_path), optimizer_size_bytes=optimizer_path.stat().st_size)
        (source / "epoch_checkpoint_index.jsonl").write_text("".join(json.dumps(row)+"\n" for row in index))
    with pytest.raises(ValueError):
        prepare_diagnostic_resume(source_run=source, run_dir=tmp_path / "recovery",
            config_path=path, device="cpu", epoch=28)
    assert not (tmp_path / "recovery").exists()


def test_latest_safe_pair_ignores_incomplete_new_epoch(tmp_path):
    source, path, *_ = fixture_run(tmp_path)
    with (source / "epoch_checkpoint_index.jsonl").open("a") as handle:
        handle.write('{"epoch":29,')
    receipt = prepare_diagnostic_resume(source_run=source, run_dir=tmp_path / "recovery",
        config_path=path, device="cpu")
    assert receipt["completed_epochs"] == 28


@pytest.mark.parametrize("study", ["read_noise_batch128", "rescaled_target_noise"])
def test_strict_study_refuses_missing_private_rng(tmp_path, study):
    source, path, config, *_ = fixture_run(tmp_path, full_state=False)
    config[study] = {"checkpoint_recovery": "full_state_epoch_boundary_only"}
    path.write_text(json.dumps(config))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["configuration"].update(sha256=sha256_file(path), resolved=config)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="private RNG state is missing"):
        prepare_diagnostic_resume(source_run=source, run_dir=tmp_path / "recovery",
                                  config_path=path, device="cpu")


def test_full_epoch_state_restores_private_rng_and_identical_adam_transition(tmp_path):
    source, path, config, p, opt, _ = fixture_run(tmp_path, full_state=True)
    target = tmp_path / "recovery"
    receipt = prepare_diagnostic_resume(source_run=source, run_dir=target,
        config_path=path, device="cpu")
    assert receipt["continuation_quality"] == "exact_epoch_state_continuation"
    assert receipt["rng_missing"] == receipt["rng_reconstructed"] == []
    saved = torch.load(source / "checkpoints/epoch_028_optimizer.pt", weights_only=False)
    private = saved["continuation_state"]
    other = SimpleNamespace(name="weight", state=torch.nn.Parameter(torch.tensor([99.], dtype=torch.float64)))
    other_opt = torch.optim.Adam([dict(params=[other.state], lr=.01)])
    loaders = SimpleNamespace(_train_sampler=SimpleNamespace(generator=torch.Generator()),
                              _worker_generator=torch.Generator())
    class Noise:
        @property
        def endpoint_read_noise_draw_count(self):
            return self._endpoint_read_noise_draw_count
    estimator = Noise()
    cont = EpochContinuation(run_dir=target, config=config, epochs=30, chunk_epochs=30,
        parameters=[other], optimizer=other_opt, estimator=estimator, loaders=loaders,
        recorder=None, contract=private["contract"])
    restored = cont.restore()
    assert restored["epoch"] == 28
    for actual, key in ((loaders._train_sampler.generator, "sampler_rng"),
                        (loaders._worker_generator, "worker_rng")):
        expected = torch.Generator().set_state(private[key])
        assert torch.equal(torch.randperm(128, generator=actual), torch.randperm(128, generator=expected))
    expected_noise = torch.Generator().set_state(private["endpoint_noise"]["generators"]["cpu"])
    noise = estimator._endpoint_read_noise_generators["cpu"]
    assert estimator.endpoint_read_noise_draw_count == 28 * 11
    for epoch in (29, 30):
        expected_gradient = torch.randn(p.state.shape, generator=expected_noise, dtype=p.state.dtype)
        actual_gradient = torch.randn(other.state.shape, generator=noise, dtype=other.state.dtype)
        assert torch.equal(expected_gradient, actual_gradient)
        opt.zero_grad(); other_opt.zero_grad()
        p.state.grad = expected_gradient; other.state.grad = actual_gradient
        opt.step(); other_opt.step()
        assert torch.equal(p.state, other.state)
        for key, value in opt.state[p.state].items():
            assert torch.equal(value, other_opt.state[other.state][key])


def test_full_state_recovery_refuses_changed_runtime_contract(tmp_path, monkeypatch):
    source, path, *_ = fixture_run(tmp_path, full_state=True)
    monkeypatch.setenv("EXPERIMENT_SOURCE_ARCHIVE_SHA256", "changed")
    with pytest.raises(ValueError, match="scientific/runtime contract mismatch"):
        prepare_diagnostic_resume(source_run=source, run_dir=tmp_path / "recovery",
            config_path=path, device="cpu")
    assert not (tmp_path / "recovery").exists()


def test_exact_run_bootstraps_before_launch_without_epoch_override(tmp_path, monkeypatch):
    from experiments import exact_run
    source, path, *_ = fixture_run(tmp_path)
    invoked = []

    def launch(command, **kwargs):
        invoked.append(command)
        output = next((tmp_path / "output").glob("000_*"))
        state = torch.load(output / "continuation.pt", weights_only=False)
        assert state["epoch"] == 28 and state["contract"]["epochs"] == 30
        assert len(state["history"]["loss"]) == 28
        assert "--epochs" not in command
        assert "--checkpoint-every-epoch" in command
        assert command[command.index("--epoch-chunk-size")+1] == "30"
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["diagnostic_resume"]["uninterrupted_trajectory"] is False
        # Stop here: no numerical worker or dataset is launched by this unit test.
        return SimpleNamespace(returncode=9)

    monkeypatch.setattr(exact_run, "_git_state", lambda: {})
    monkeypatch.setattr(exact_run.subprocess, "run", launch)
    result = exact_run.run([path], output_root=tmp_path / "output", device="cpu",
        epoch_chunk_size=30, checkpoint_every_epoch=True, skip_terminal_official_test=True,
        resume_diagnostic_run=source, resume_diagnostic_epoch=28)
    assert len(invoked) == 1 and result["status"] == "failed"
