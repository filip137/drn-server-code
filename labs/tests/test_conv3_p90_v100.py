"""V100/A100 sweep task selection and duplicate guards."""
import os
import sys
from pathlib import Path
import subprocess
import pytest
from experiments.exact_run import selected_configs
W=Path(__file__).resolve().parents[2]/'experiments/run_conv3_p90_read_noise_v100.sh'


@pytest.fixture(autouse=True, params=['v100', 'a100'])
def wrapper(request, monkeypatch):
    path = Path(__file__).resolve().parents[2] / f'experiments/run_conv3_p90_read_noise_{request.param}.sh'
    monkeypatch.setattr(sys.modules[__name__], 'W', path)


def invoke(tmp_path,task):
    c=tmp_path/'configs';c.mkdir(exist_ok=True)
    (c/'config-names.txt').write_text('\n'.join(str(i)+'.json' for i in range(18))+'\n')
    e=os.environ.copy();e.pop('SLURM_ARRAY_TASK_ID',None)
    if task is not None:e['SLURM_ARRAY_TASK_ID']=str(task)
    return subprocess.run(['bash',str(W),str(tmp_path/'source'),str(c),str(tmp_path/'out'),str(tmp_path/'data')],env=e,capture_output=True,text=True)


@pytest.mark.parametrize('task',[None,-1,18])
def test_invalid_task_stops_before_modules(tmp_path,task):
    assert invoke(tmp_path,task).returncode!=0
    assert not (tmp_path/'out').exists()


def test_duplicate_preserves_receipt(tmp_path):
    p=tmp_path/'out/task_0/worker_started';p.mkdir(parents=True);(p/'pid').write_text('original')
    assert invoke(tmp_path,0).returncode!=0
    assert (p/'pid').read_text()=='original'
    assert not (p/'finished_at').exists()


@pytest.mark.parametrize('task',range(18))
def test_preselected_single_config_ignores_inherited_index(tmp_path,task):
    c=tmp_path/'one.json';c.write_text('{}')
    assert selected_configs([c],0,environ={'SLURM_ARRAY_TASK_ID':str(task)})==[(0,c.resolve())]
    assert 'experiments.exact_run "$config" --index 0' in W.read_text()
