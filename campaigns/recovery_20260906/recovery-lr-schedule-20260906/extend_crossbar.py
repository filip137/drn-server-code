"""Apply the bounded recovery extension to the isolated September-5 source."""
from pathlib import Path

ROOT = Path(__file__).parent / "crossbar_code"

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)

p = ROOT / "experiments/mnist_analog_relu/staged_config.py"
s = p.read_text()
s = replace(s, "ADAM_DIAGNOSTIC_LEARNING_RATE_GRID = (\n", "ADAM_DIAGNOSTIC_LEARNING_RATE_GRID = (\n    3e-7,\n    1e-6,\n")
s = replace(s, "    hyperparameters: DiagnosticLiteralAdamHyperparameters\n", "    hyperparameters: DiagnosticLiteralAdamHyperparameters\n    learning_rate_schedule: Mapping[str, Any] | None = None\n")
start, end = s.index("def _parse_on_chip_adam_diagnostic_stage("), s.index("def _parse_fresh_apparent_diagnostic_stage(")
b = s[start:end]
b = replace(b, "    _keys(raw, path, required)\n", "    _keys(raw, path, required | ({'learning_rate_schedule'} if 'learning_rate_schedule' in raw else set()))\n")
b = replace(b, "    for name, required_value in expected.items():\n", "    kl_policy = 'best_held_apparent_validation_objective_then_accuracy_then_earlier_epoch_including_epoch0'\n    if raw['checkpoint_policy'] == kl_policy:\n        expected['checkpoint_policy'] = kl_policy\n    for name, required_value in expected.items():\n")
b = replace(b, "    if epochs != 3:\n        raise config_error(f\"{path}.epochs\", \"to equal 3\", raw[\"epochs\"])\n", "    if epochs > 60:\n        raise config_error(f'{path}.epochs', 'to be between 1 and 60', raw['epochs'])\n    schedule = raw.get('learning_rate_schedule')\n    if schedule is not None:\n        schedule = _object(schedule, f'{path}.learning_rate_schedule')\n        _keys(schedule, f'{path}.learning_rate_schedule', {'kind', 'final_factor', 'decay_epochs'})\n        if schedule['kind'] not in {'constant', 'exponential'}:\n            raise config_error(f'{path}.learning_rate_schedule.kind', 'to be constant or exponential', schedule['kind'])\n        factor = _number(schedule['final_factor'], f'{path}.learning_rate_schedule.final_factor', minimum=0.0)\n        horizon = _integer(schedule['decay_epochs'], f'{path}.learning_rate_schedule.decay_epochs', minimum=2)\n        if not 0 < factor <= 1 or (schedule['kind'] == 'constant' and factor != 1):\n            raise ValueError('Expected 0 < final_factor <= 1, with factor 1 for a constant schedule.')\n        schedule = dict(kind=schedule['kind'], final_factor=factor, decay_epochs=horizon)\n")
b = replace(b, "        epochs=3,\n", "        epochs=epochs,\n")
b = replace(b, "        hyperparameters=DiagnosticLiteralAdamHyperparameters(\n", "        learning_rate_schedule=schedule,\n        hyperparameters=DiagnosticLiteralAdamHyperparameters(\n")
s = s[:start]+b+s[end:]
p.write_text(s)

p = ROOT / "training/ibm_om_standard_crossbar.py"
s = p.read_text()
start, end = s.index("class PulseAdam:"), s.index("class ", s.index("class PulseAdam:")+10)
b = s[start:end]
b = replace(b, "        self.step_index = 0\n", "        self.step_index = 0\n        self.learning_rate_scale = 1.0\n")
b = replace(b, "        command = torch.where(self.enabled, command, torch.zeros_like(command))\n", "        if not math.isfinite(self.learning_rate_scale) or not 0 < self.learning_rate_scale <= 1:\n            raise ValueError('Expected a finite learning-rate scale in (0,1].')\n        command = command * self.learning_rate_scale\n        command = torch.where(self.enabled, command, torch.zeros_like(command))\n")
b = replace(b, "            \"step_index\": self.step_index,\n", "            \"step_index\": self.step_index,\n            \"learning_rate_scale\": self.learning_rate_scale,\n")
b = replace(b, "            or set(state) != expected\n", "            or set(state) not in (expected, expected | {'learning_rate_scale'})\n")
b = replace(b, "        step_index = nonnegative_integer(\"step_index\")\n", "        scale = float(state.get('learning_rate_scale', 1.0))\n        if not math.isfinite(scale) or not 0 < scale <= 1:\n            raise ValueError('Expected a saved learning-rate scale in (0,1].')\n        self.learning_rate_scale = scale\n        step_index = nonnegative_integer(\"step_index\")\n")
s = s[:start]+b+s[end:]
p.write_text(s)

p = ROOT / "experiments/mnist_analog_relu/staged_runtime.py"
s = p.read_text()
s = replace(s, "    *, student_accuracy: float, objective_value: float, epoch: int\n", "    *, student_accuracy: float, objective_value: float, epoch: int, kl_first: bool = False\n")
s = replace(s, "    return (-accuracy, loss, epoch)\n", "    return (loss, -accuracy, epoch) if kl_first else (-accuracy, loss, epoch)\n")
s = replace(s, '                "objective": stage.objective,\n', '                "objective": stage.objective,\n                "learning_rate_schedule": stage.learning_rate_schedule,\n')
s = replace(s, "            objective_value=candidates[epoch][1],\n            epoch=epoch,\n", "            objective_value=candidates[epoch][1],\n            epoch=epoch,\n            kl_first='validation_objective_then' in checkpoint_policy,\n")
start, end = s.index("def _run_on_chip_adam("), s.index("\ndef ", s.index("def _run_on_chip_adam(")+10)
b = s[start:end]
b = b.replace("_adam_selection_rank(", "selection_rank(")
b = replace(b, "    diagnostic = isinstance(stage, OnChipAdamDiagnosticStageSettings)\n", "    diagnostic = isinstance(stage, OnChipAdamDiagnosticStageSettings)\n    kl_first = 'validation_objective_then' in stage.checkpoint_policy\n    def selection_rank(**values):\n        return _adam_selection_rank(**values, kl_first=kl_first)\n    schedule = stage.learning_rate_schedule if diagnostic else None\n")
b = replace(b, "    origin_plant = _restore_current(origin, layout=layout, device=device)\n", "    if kl_first:\n        selection_metric = selection_objective_metric\n    origin_plant = _restore_current(origin, layout=layout, device=device)\n")
b = replace(b, "    for epoch in range(completed_epochs + 1, stage.epochs + 1):\n", "    for epoch in range(completed_epochs + 1, stage.epochs + 1):\n        if schedule and schedule['kind'] == 'exponential':\n            progress = min((epoch - 1) / (schedule['decay_epochs'] - 1), 1.0)\n            optimizer.learning_rate_scale = schedule['final_factor'] ** progress\n        else:\n            optimizer.learning_rate_scale = 1.0\n        effective_learning_rate = learning_rate * optimizer.learning_rate_scale\n")
b = replace(b, '            "train_objective": stage.objective,\n', '            "train_objective": stage.objective,\n            "effective_learning_rate": effective_learning_rate,\n            "learning_rate_schedule": schedule,\n')
b = replace(b, '        checkpoint = StagedDeviceState(\n', "        # Keep recoverable epoch 1, every fifth epoch, and the terminal state.\n        if stage.epochs > 3 and epoch != 1 and epoch % 5 and epoch != stage.epochs:\n            continue\n        checkpoint = StagedDeviceState(\n")
s = s[:start]+b+s[end:]
p.write_text(s)
print('Extended isolated crossbar runner: explicit decay, 1–60 epochs, KL-first selection, exact scale checkpointing.')
