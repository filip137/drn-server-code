"""Read-only summary tests, using fake queue snapshots and small study records."""
import json

from experiments.summarize_cifar_eqprop_queue import study_summary, summarize, summarize_job


def terminal_study(state='complete'):
    return dict(root='/study', decision={'state': state}, receipts=[], errors=[])


def test_latest_attempt_progress_and_alerts_ignore_old_attempt_and_log_tail():
    row = dict(id='cifar-eqprop-case', state='running', reason='')
    current = dict(issued=200, progress_at=100, state='running', target='gpu-host', gpu_name='RTX5090',
        receipt=json.dumps({'receipt': {'started_at': 201}, 'log_tail': 'private and unnecessary',
                            'progress': {'progress': {'stage': 'training', 'cursor': {'total_updates': 420}}}}))
    old = dict(issued=10, progress_at=10, state='failed')
    job = summarize_job(row, {'job': row, 'attempts': [current, old]}, 2000)
    assert job['target'] == 'gpu-host' and job['progress']['cursor']['total_updates'] == 420
    assert job['progress_age_seconds'] == 1900 and 'private' not in json.dumps(job)
    report = summarize({'service': {'observed_at': 1900, 'alive': True}}, [job], [terminal_study()], 2000)
    assert not report['operational_complete']
    assert any(a['kind'] == 'progress_stale' for a in report['alerts'])


def test_completion_requires_terminal_studies_inactive_jobs_and_fresh_service():
    fresh = {'service': {'observed_at': 1990, 'alive': True}}
    assert summarize(fresh, [], [terminal_study('search_exhausted')], 2000)['operational_complete']
    assert not summarize(fresh, [], [terminal_study('pending')], 2000)['operational_complete']
    queued = summarize_job({'id': 'case', 'state': 'queued'}, {}, 2000)
    assert not summarize(fresh, [queued], [terminal_study()], 2000)['operational_complete']
    for bad in ({}, {'service': []}, {'service': {'observed_at': 'invalid'}},
                {'service': {'observed_at': float('nan')}},
                {'service': {'observed_at': 100, 'alive': True}}):
        report = summarize(bad, [], [terminal_study()], 2000)
        assert not report['operational_complete'] and report['alerts']
        json.dumps(report, allow_nan=False)


def test_study_receipts_keep_metrics_and_surface_malformed_inputs(tmp_path):
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    (analysis / 'decision.json').write_text(json.dumps({'state': 'quality_stop', 'records': {'large': 'ignored'}}))
    (analysis / 'case-5-collection.json').write_text(json.dumps({'case': 'case', 'stage': '5',
        'validated': True, 'rejected': False, 'validation': {'accuracy': .8, 'loss': .7, 'examples': 5000},
        'unused_payload': 'ignored'}))
    (analysis / 'broken-collection.json').write_text('{invalid')
    summary = study_summary(tmp_path)
    assert summary['decision'] == {'state': 'quality_stop'}
    assert summary['receipts'][0]['validation']['accuracy'] == .8
    assert 'unused_payload' not in summary['receipts'][0]
    assert len(summary['errors']) == 1


def test_slurm_owner_failure_and_stale_heartbeat_are_not_healthy(tmp_path):
    import time
    for name in ('analysis', 'monitor', 'launch'):
        (tmp_path/name).mkdir()
    (tmp_path/'analysis/decision.json').write_text('{"state":"pending"}')
    (tmp_path/'launch/transport.py').write_text('frozen transport marker')
    (tmp_path/'monitor/jobs.json').write_text('{}')
    (tmp_path/'monitor/status.json').write_text('{"jobs":{"case-1":{"job_id":"123","state":"RUNNING","metrics_bytes":99}},"errors":[]}')
    owner=tmp_path/'monitor/owner-status.json'
    for state,observed in [('blocked',time.time()),('healthy',time.time()-1000)]:
        owner.write_text(json.dumps(dict(state=state,observed_at=observed,error='SSH failed' if state=='blocked' else None)))
        row=study_summary(tmp_path)
        assert row['errors'] and row['slurm_jobs']['case-1']['metrics_bytes']==99
        report=summarize({'service':{'alive':True,'observed_at':time.time()}},[],[row],time.time())
        assert report['alerts'] and not report['operational_complete']
    owner.write_text(json.dumps(dict(state='healthy',observed_at=time.time(),error=None)))
    assert not study_summary(tmp_path)['errors']
