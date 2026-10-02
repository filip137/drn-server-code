"""CPU-only checks of submission and collection safety; no Slurm or GPU calls."""
import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import cifar_beta_slurm_transport as t

class TransportTest(unittest.TestCase):
    def test_idempotent_existing_submission(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)),patch.object(t,'remote') as remote:
            t.atomic(Path(d)/'monitor/jobs.json',{'beta_3-1':{'job_id':'123'}})
            self.assertEqual(t.submit('1','beta_3'),'123');remote.assert_not_called()

    def test_lost_acknowledgement_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)),patch.object(t,'remote',side_effect=TimeoutError) as remote:
            t.atomic(Path(d)/'launch/identity.json',{})
            with self.assertRaises(TimeoutError):t.submit('1','beta_3')
            self.assertIsNone(t.read(Path(d)/'monitor/jobs.json')['beta_3-1']['job_id'])
            with self.assertRaisesRegex(RuntimeError,'Uncertain'):t.submit('1','beta_3')
            self.assertEqual(remote.call_count,1)

    def test_process_exit_requires_artifacts(self):
        row=dict(alive=False,queued=False,state='COMPLETED',exit_code='0:0',result_exists=False,rejected=False)
        self.assertFalse(t.completed(row));row['result_exists']=True;self.assertTrue(t.completed(row))
        row['exit_code']='1:0';self.assertFalse(t.completed(row))
        row.update(rejected=True,state='FAILED');self.assertTrue(t.completed(row))
        row['alive']=True;self.assertFalse(t.completed(row))

    def test_matched_runtime_including_cudnn(self):
        expected=dict(torch='2.5.0',cuda='12.2')
        t.verify_runtime(dict(expected,cudnn=8907),expected)
        for key,value in [('torch','2.5.1'),('cuda','12.1'),('cudnn',90100)]:
            actual=dict(expected,cudnn=8907);actual[key]=value
            with self.assertRaises(RuntimeError):t.verify_runtime(actual,expected)

    def test_changed_transport_blocks_before_submission(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,'ROOT',Path(d)),patch.object(t,'remote') as remote:
            p=Path(d)/'launch/script';p.parent.mkdir();p.write_text('changed')
            t.atomic(Path(d)/'launch/identity.json',{'launch/script':'oldhash'})
            with self.assertRaisesRegex(RuntimeError,'Changed frozen'):t.submit('1','native_bptt')
            remote.assert_not_called()

if __name__=='__main__':unittest.main()


def test_launch_failed_held_pending_is_operational_error():
    for reason in ('(launch failed requeued held)', 'launch_failed_requeued_held', '(JobHeldUser)'):
        assert t.operational_error(dict(state='PENDING', reason=reason))
    for reason in ('(Priority)', '(Resources)', '(QOSMaxGRESPerUser)', ''):
        assert not t.operational_error(dict(state='PENDING', reason=reason))


def test_completed_job_purged_from_live_queue_uses_accounting(tmp_path):
    jobs={'beta_3-1':dict(job_id='123',stage='1',case='beta_3')}
    cell=tmp_path/'cells/1/beta_3';cell.mkdir(parents=True)
    (cell/'result.json').write_text('{}')
    def output(argv, **kwargs):
        if argv[0]=='squeue':
            assert '--me' in argv and '-j' not in argv
            return '999|RUNNING|umg@v100|other-own-node'
        return '123|COMPLETED|0:0|fmu@v100|3413'
    with patch.object(t,'ROOT',tmp_path),patch.object(t,'command',side_effect=output):
        row=t.inspect(jobs)['beta_3-1']
    assert row['state']=='COMPLETED' and row['elapsed_seconds']==3413
    assert t.completed(row)


def test_missing_scheduler_and_accounting_records_remain_unknown(tmp_path):
    jobs={'beta_3-1':dict(job_id='123',stage='1',case='beta_3')}
    with patch.object(t,'ROOT',tmp_path),patch.object(t,'command',return_value=''):
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'No Slurm record'):
            t.inspect(jobs)


def test_failed_queue_query_keeps_stderr_and_cannot_mark_completed(tmp_path):
    jobs={'beta_3-1':dict(job_id='123',stage='1',case='beta_3')}
    failure=subprocess.CalledProcessError(1,['squeue'],stderr='Unable to contact slurm controller')
    with patch.object(t,'ROOT',tmp_path),patch.object(t.subprocess,'run',side_effect=failure):
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'Unable to contact slurm controller'):
            t.inspect(jobs)


def test_two_matched_initial_extensions_use_existing_idempotent_registry(tmp_path):
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'remote', side_effect=['991', '992']) as remote:
        t.atomic(tmp_path/'launch/identity.json', {})
        for case, job_id in [('beta_0p3', '991'), ('beta_1', '992')]:
            assert t.submit('1', case) == job_id
            assert t.submit('1', case) == job_id
            row = t.read(tmp_path/'monitor/jobs.json')[case+'-1']
            assert row['budget_seconds'] == 5400 and row['stage'] == '1'
        assert remote.call_count == 2
        assert [call.args[0][-1] for call in remote.call_args_list] == [
            t.REMOTE_ROOT+'/launch/beta_0p3-1.sbatch', t.REMOTE_ROOT+'/launch/beta_1-1.sbatch']


def test_extension_cannot_promote_or_admit_unrecognized_cases():
    for action in (t.submit, t.run):
        for stage, case in (('5', 'beta_0p3'), ('10', 'beta_1')):
            with unittest.TestCase().assertRaisesRegex(ValueError, 'requires review'):
                action(stage, case)
        with unittest.TestCase().assertRaisesRegex(ValueError, 'Undeclared'):
            action('1', 'beta_2')


def test_reviewed_beta1_stage5_checks_parent_and_submits_once(tmp_path):
    parent=tmp_path/'cells/1/beta_1/checkpoints/final_model.pt'
    parent.parent.mkdir(parents=True);parent.write_bytes(b'validated parent')
    t.atomic(tmp_path/'launch/identity.json', {})
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'remote',
            side_effect=[t.digest(parent)+'  parent', '994']) as remote:
        assert t.submit('5', 'beta_1') == '994'
        assert t.submit('5', 'beta_1') == '994'
        assert remote.call_count == 2
        assert remote.call_args_list[0].args[0][0] == 'sha256sum'
        assert remote.call_args_list[1].args[0][-1].endswith('/beta_1-5.sbatch')
    assert t.read(tmp_path/'monitor/jobs.json')['beta_1-5']['budget_seconds'] == 14400


def test_reviewed_beta1_stage5_mismatched_parent_cannot_submit(tmp_path):
    parent=tmp_path/'cells/1/beta_1/checkpoints/final_model.pt'
    parent.parent.mkdir(parents=True);parent.write_bytes(b'validated parent')
    t.atomic(tmp_path/'launch/identity.json', {})
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'remote',return_value='wrong  parent') as remote:
        with unittest.TestCase().assertRaisesRegex(RuntimeError, 'Remote parent differs'):
            t.submit('5', 'beta_1')
        assert remote.call_count == 1
    assert not (tmp_path/'monitor/jobs.json').exists()


def test_stage10_uses_measured_allocation_and_cannot_fall_back_to_old_script(tmp_path):
    parent=tmp_path/'cells/5/beta_0p1/checkpoints/final_model.pt'
    parent.parent.mkdir(parents=True);parent.write_bytes(b'validated epoch5')
    t.atomic(tmp_path/'launch/identity.json', {})
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'remote',return_value=t.digest(parent)+'  parent') as remote:
        with unittest.TestCase().assertRaisesRegex(RuntimeError, 'not prepared'):
            t.submit('10', 'beta_0p1')
        assert remote.call_count == 1
    assert not (tmp_path/'monitor/jobs.json').exists()
    (tmp_path/'launch/beta_0p1-10-timed.sbatch').write_text('prepared five-hour job')
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'remote',side_effect=[t.digest(parent)+'  parent','995']) as remote:
        assert t.submit('10','beta_0p1') == '995'
        assert t.submit('10','beta_0p1') == '995'
        assert remote.call_args_list[-1].args[0][-1].endswith('/beta_0p1-10-timed.sbatch')
    assert t.read(tmp_path/'monitor/jobs.json')['beta_0p1-10']['budget_seconds'] == 18000


def test_longer_stage_cannot_launch_when_measured_duration_exceeds_allowance(tmp_path):
    t.atomic(tmp_path/'analysis/beta_0p1-10-collection.json',
             dict(validated=True,rejected=False,elapsed_seconds=16000))
    with patch.object(t,'ROOT',tmp_path),patch.object(t,'remote') as remote:
        with unittest.TestCase().assertRaisesRegex(RuntimeError,'35200s.*28740s'):
            t.submit('20','beta_0p1')
        remote.assert_not_called()
    assert not (tmp_path/'monitor/jobs.json').exists()


def test_explicit_screen_collected_even_after_terminal_fallback_decision(tmp_path):
    from experiments import run_cifar_beta_stages as wrapper
    t.atomic(tmp_path/'monitor/jobs.json', {'beta_1-1': dict(job_id='993', stage='1', case='beta_1')})
    t.atomic(tmp_path/'analysis/decision.json', dict(state='search_exhausted', requests=[]))
    row=dict(job_id='993', state='COMPLETED', alive=False, queued=False, exit_code='0:0',
             status=dict(state='complete'), metrics_bytes=500, result_exists=True, rejected=False)
    def command(argv, **kwargs):
        return json.dumps({'beta_1-1': row}) if argv[0]=='ssh' else ''
    with patch.object(t, 'ROOT', tmp_path), patch.object(t, 'command', side_effect=command), \
         patch.object(wrapper, 'ROOT', tmp_path), patch.object(wrapper, 'submit'), \
         patch.object(wrapper, 'validate', return_value=dict(validated=True, environment=dict(torch='2.5.0', cuda='12.2'))) as validate, \
         patch.object(wrapper, 'advance') as advance:
        t.poll()
    assert validate.call_args.args == ('1', 'beta_1')
    assert t.read(tmp_path/'analysis/beta_1-1-collection.json')['job_id'] == '993'
    advance.assert_called_once()
