"""Run preparation + validation once from an authenticated Databricks terminal.

Usage: python tools/run_databricks_validation.py --repo-path /Users/me/repo
Writes a compact report; never exports a full source dataset or credentials.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run(repo_path, timeout=900):
    from databricks.sdk import WorkspaceClient
    client = WorkspaceClient()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    tasks = [
        {'task_key': 'prepare', 'notebook_task': {
            'notebook_path': repo_path.rstrip('/') + '/04_prepare_inputs.ipynb',
            'source': 'WORKSPACE'}, 'max_retries': 0},
        {'task_key': 'validate', 'depends_on': [{'task_key': 'prepare'}],
         'notebook_task': {
            'notebook_path': repo_path.rstrip('/') + '/04_validate_live.ipynb',
            'source': 'WORKSPACE'}, 'max_retries': 0},
    ]
    response = client.api_client.do('POST', '/api/2.2/jobs/runs/submit', body={
        'run_name': 'milly-ranking-validation', 'tasks': tasks,
        'timeout_seconds': timeout,
        'idempotency_token': hashlib.sha256((repo_path + commit).encode()).hexdigest(),
    })
    run_id = response['run_id']
    print('RUN_ID:', run_id, flush=True)
    Path('/tmp/milly_ranking_validation_run_id.json').write_text(json.dumps(response))
    deadline = time.monotonic() + timeout + 120
    while True:
        snapshot = client.jobs.get_run(run_id=run_id).as_dict()
        state = snapshot.get('state', {})
        if state.get('life_cycle_state') in {'TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'}:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f'Run {run_id} did not finish within the bounded wait')
        time.sleep(10)
    report = {
        'status': 'passed' if state.get('result_state') == 'SUCCESS' else 'failed',
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'code_commit': commit, 'run_id': run_id, 'run_state': state, 'tasks': [],
    }
    for task in snapshot.get('tasks', []):
        entry = {'task_key': task['task_key'], 'run_id': task.get('run_id'), 'state': task.get('state')}
        if task.get('run_id'):
            output = client.jobs.get_run_output(run_id=task['run_id']).as_dict()
            result = output.get('notebook_output', {}).get('result')
            if result:
                try:
                    entry['result'] = json.loads(result)
                except ValueError:
                    entry['result'] = result[:3000]
            if output.get('error'):
                entry['error'] = output['error'][:3000]
                entry['error_trace'] = output.get('error_trace', '')[-5000:]
        report['tasks'].append(entry)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-path', required=True)
    args = parser.parse_args()
    try:
        report = run(args.repo_path)
    except Exception as error:
        report = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)[:3000]}
    output = Path('reports/04_ranking_live_validation.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
