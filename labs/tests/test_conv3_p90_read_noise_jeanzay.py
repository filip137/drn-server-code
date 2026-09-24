"""Guard the packed follow-up against duplicate launches and double indexing."""
import os
from pathlib import Path
import subprocess

import pytest
from experiments.exact_run import selected_configs

WRAPPER=Path(__file__).resolve().parents[2]/'experiments/run_conv3_p90_read_noise_jeanzay.sh'


def invoke(tmp_path, task):
    configs=tmp_path/'configs';configs.mkdir(exist_ok=True)
    (configs/'config-names.txt').write_text('\n'.join(f'{i}.json' for i in range(15))+'\n')
    env=os.environ.copy();env.pop('SLURM_ARRAY_TASK_ID',None)
    if task is not None:env['SLURM_ARRAY_TASK_ID']=str(task)
    return subprocess.run(['bash',str(WRAPPER),str(tmp_path/'source'),str(configs),
                           str(tmp_path/'out'),str(tmp_path/'data')],env=env,capture_output=True,text=True)


@pytest.mark.parametrize('task',[None,8,-1])
def test_invalid_index_stops_before_allocation_use(tmp_path,task):
    r=invoke(tmp_path,task)
    assert r.returncode!=0
    assert not (tmp_path/'out').exists()


def test_duplicate_preserves_original_receipt(tmp_path):
    old=tmp_path/'out/task_0/worker_started';old.mkdir(parents=True)
    (old/'pid').write_text('original')
    r=invoke(tmp_path,0)
    assert r.returncode!=0 and 'File exists' in r.stderr
    assert (old/'pid').read_text()=='original'
    assert not (old/'finished_at').exists()


@pytest.mark.parametrize('task',range(8))
def test_each_preselected_config_ignores_inherited_index(tmp_path,task):
    config=tmp_path/'case.json';config.write_text('{}')
    assert selected_configs([config],0,environ={'SLURM_ARRAY_TASK_ID':str(task)})==[(0,config.resolve())]
    assert 'experiments.exact_run "$config" --index 0' in WRAPPER.read_text()


def test_pack_selection_covers_every_config_once():
    code=WRAPPER.read_text();loop=code[code.index('for index in "$task"'):code.index('  run_dir="$case_dir/case_$index"')]
    shell='configs=(); for i in {0..14}; do configs+=("$i"); done; config_dir=/configs; for task in {0..7}; do\n'+loop+'\nprintf "%s\\n" "$config"\ndone\ndone'
    result=subprocess.run(['bash','-c',shell],capture_output=True,text=True,check=True)
    assert sorted(int(Path(x).name) for x in result.stdout.splitlines())==list(range(15))
