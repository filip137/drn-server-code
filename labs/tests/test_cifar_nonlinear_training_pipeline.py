"""The pipeline must stop on incomplete child evidence and enforce its budget."""
import json
import subprocess
import sys
import time

import pytest

from experiments import reporting
from experiments.run_cifar_nonlinear_head_training import run_child


@pytest.mark.parametrize('state', ['complete', 'paused'])
def test_child_exit_zero_requires_complete_evidence(tmp_path, state):
    outer = tmp_path / 'outer'
    reporting.start_run(outer, dict(study_id='test', run_id='outer', arm_id='single'))
    child = tmp_path / 'child'
    child.mkdir()
    (child / 'status.json').write_text(json.dumps(dict(state=state)))
    command = [sys.executable, '-c', 'pass']
    if state == 'complete':
        run_child(command, child, outer, 'train', time.monotonic()+10)
    else:
        with pytest.raises(RuntimeError, match='without complete evidence'):
            run_child(command, child, outer, 'train', time.monotonic()+10)


def test_pipeline_timeout_terminates_only_owned_child(tmp_path, monkeypatch):
    import experiments.run_cifar_nonlinear_head_training as pipeline
    outer = tmp_path / 'outer'
    reporting.start_run(outer, dict(study_id='test', run_id='outer', arm_id='single'))
    children = []
    original = subprocess.Popen
    def tracked(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(pipeline.subprocess, 'Popen', tracked)
    with pytest.raises(TimeoutError, match='budget'):
        run_child([sys.executable, '-c', 'import time; time.sleep(20)'], tmp_path/'missing',
                  outer, 'train', time.monotonic()+.1)
    assert len(children) == 1 and children[0].poll() is not None
