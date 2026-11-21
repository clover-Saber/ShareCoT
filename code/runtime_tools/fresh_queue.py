import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

PROC_ROOT = Path('/proc')


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    if os.name == 'posix':
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_jobs(jobs):
    ids = [j['id'] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate job IDs')
    for job in jobs:
        if job.get('kind') == 'api_prediction':
            raise ValueError('Main API prediction baselines are excluded by user instruction')
        if job.get('resource') not in ['cpu', 'gpu0', 'gpu1']:
            raise ValueError('Unknown execution resource')
        if set(job.get('depends', [])) - set(ids):
            raise ValueError('Unknown dependency')
    seen, stack = set(), set()
    by_id = {j['id']: j for j in jobs}
    def visit(jid):
        if jid in stack:
            raise ValueError('Cyclic dependency')
        if jid in seen:
            return
        stack.add(jid)
        for dep in by_id[jid].get('depends', []):
            visit(dep)
        stack.remove(jid); seen.add(jid)
    for jid in ids:
        visit(jid)


def verify_completed(job, old):
    outputs(job, unknown_exit=old.get('returncode') is None)


def ticks(pid):
    try:
        return (PROC_ROOT / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def cuda_devices(job):
    return '' if job['resource'] == 'cpu' else job['resource'][-1]


def live(pid, job, start_ticks=None, expected_argv=None):
    try:
        proc = PROC_ROOT / str(pid)
        parts = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if parts[0] in ['Z', 'X'] or (start_ticks is not None and parts[19] != str(start_ticks)):
            return False
        argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if expected_argv is None or argv != expected_argv or job['script'] not in argv:
            return False
        pos = argv.index(job['script'])
        environment = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if x)
        return argv[pos + 1:] == job.get('args', []) and environment.get('CUDA_VISIBLE_DEVICES') == cuda_devices(job)
    except (OSError, IndexError, UnicodeError, ValueError):
        return False


def field(value, name):
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError('Missing required JSON field: ' + name)
        value = value[key]
    return value


def matches(value, expected):
    for key, wanted in expected.items():
        actual = field(value, key)
        if isinstance(wanted, dict):
            matches(actual, wanted)
        elif type(actual) is not type(wanted) or actual != wanted:
            raise ValueError('JSON configuration differs: ' + key)


def reject_smoke(value):
    if isinstance(value, dict):
        if value.get('smoke_run') or value.get('status') == 'smoke_complete':
            raise ValueError('A smoke result cannot complete a production job')
        for nested in value.values():
            reject_smoke(nested)
    elif isinstance(value, list):
        for nested in value:
            reject_smoke(nested)


def outputs(job, unknown_exit=False):
    artifacts = job.get('artifacts', [])
    if not artifacts:
        raise ValueError('No required artifacts declared')
    missing = [p for p in artifacts if not Path(p).is_file() or Path(p).stat().st_size == 0]
    if missing:
        raise ValueError('Missing or empty artifacts: ' + str(missing))
    values = {}
    def read(path):
        if path not in artifacts:
            raise ValueError('Evidence must be a declared artifact: ' + path)
        if path not in values:
            values[path] = json.loads(Path(path).read_text(encoding='utf-8'),
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))
            if values[path] in [{}, [], None, '']:
                raise ValueError('Empty JSON artifact: ' + path)
            reject_smoke(values[path])
        return values[path]
    for path in artifacts:
        with Path(path).open('rb') as handle:
            prefix = handle.read(256).lstrip()
        if Path(path).suffix.lower() == '.json' or prefix.startswith((b'{', b'[')):
            read(path)
    if unknown_exit and not all(job.get(k) for k in ['terminal_contract', 'expected_json']):
        raise ValueError('Unknown exit requires terminal success and expected configuration')
    for path, expected in job.get('expected_json', {}).items():
        matches(read(path), expected)
    metrics = job.get('required_metrics')
    if job.get('kind') == 'training':
        if not metrics or not metrics.get('required_fields'):
            raise ValueError('Training requires explicit production metrics and fields')
        config = job.get('run_config')
        if not config or not job.get('expected_json', {}).get(config):
            raise ValueError('Training requires expected run configuration')
    if metrics:
        value = read(metrics['path'])
        for key in metrics.get('required_fields', []):
            if field(value, key) is None:
                raise ValueError('Null production metric: ' + key)
        if job.get('expected_steps') is not None and field(value, 'trained_steps') != job['expected_steps']:
            raise ValueError('Production training step count differs')
        if 'provenance' in value:
            if not job.get('run_config') or value['provenance'] != read(job['run_config']):
                raise ValueError('Production metrics provenance differs from run configuration')
    contract = job.get('terminal_contract')
    if contract:
        value = read(contract['path'])
        if value.get('status') != 'complete':
            raise ValueError('Terminal success status is absent')
        for key in contract.get('required_fields', []):
            if field(value, key) is None:
                raise ValueError('Null terminal evidence: ' + key)
        matches(value, contract.get('expected', {}))
        target = read(contract['metrics_path'])
        if contract.get('metrics_field'):
            if field(value, contract['metrics_field']) != target:
                raise ValueError('Terminal metrics differ from persisted metrics')
            for key in ['seed', 'method', 'trained_steps', 'checkpoint']:
                if key in target and (key not in value or type(value[key]) is not type(target[key]) or value[key] != target[key]):
                    raise ValueError('Terminal metadata differs from metrics: ' + key)
        elif contract['path'] != contract['metrics_path']:
            raise ValueError('Separate terminal file requires metrics_field')


def definitely_exited(pid):
    proc = PROC_ROOT / str(pid)
    try:
        parts = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        return parts[0] in ['Z', 'X']
    except FileNotFoundError:
        try:
            proc.stat()
        except FileNotFoundError:
            return True
        except OSError:
            pass
        return False
    except (OSError, IndexError):
        return False


def adopt_record(job, adopt):
    if not adopt.get('process_start_ticks') or not adopt.get('argv'):
        raise ValueError('Adoption requires expected process_start_ticks and full argv')
    identity_live = live(adopt['pid'], job, adopt['process_start_ticks'], adopt['argv'])
    if not identity_live and not definitely_exited(adopt['pid']):
        raise ValueError('Adoption process identity differs or is unreadable: ' + job['id'])
    record = {'id': job['id'], 'status': 'running', 'pid': adopt['pid'],
            'process_start_ticks': str(adopt['process_start_ticks']), 'launch_argv': adopt['argv'],
            'cuda_visible_devices': cuda_devices(job), 'started_at': adopt['started_at'],
            'log': adopt['log'], 'resource': job['resource'], 'adopted': True}
    if not identity_live:
        outputs(job, unknown_exit=True)
        record.update(status='complete', returncode=None, ended_at=stamp())
    return record


def matching_processes(job, old):
    matches_found = []
    for proc in PROC_ROOT.iterdir():
        if proc.name.isdigit():
            start = ticks(proc.name)
            if start is not None and live(proc.name, job, start, old.get('launch_argv')):
                matches_found.append((int(proc.name), start))
    return sorted(matches_found)


def reconcile_launch(job, old):
    found = matching_processes(job, old)
    if len(found) == 1:
        old.update(status='running', pid=found[0][0], process_start_ticks=found[0][1], adopted=True)
    elif found:
        old.update(status='failed', error='Multiple exact processes match launch intent', uncertain_processes=found)
    else:
        try:
            outputs(job, unknown_exit=True)
            old.update(status='complete', returncode=None, ended_at=stamp())
        except (ValueError, OSError, KeyError) as error:
            old.update(status='failed', unresolved_launch_intent=True,
                       error='Unresolved launch intent; refusing respawn: ' + str(error), ended_at=stamp())


def launch_job(job, root, python, state, children, save):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=cuda_devices(job), OMP_NUM_THREADS='8',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(root / 'code') + os.pathsep + env.get('PYTHONPATH', '')
    log = root / 'logs' / (job['id'] + '.log')
    log.parent.mkdir(exist_ok=True)
    argv = [python, '-u', job['script'], *job.get('args', [])]
    old = {'id': job['id'], 'status': 'launching', 'started_at': stamp(), 'log': str(log),
           'resource': job['resource'], 'launch_argv': argv,
           'cuda_visible_devices': cuda_devices(job)}
    state['runs'][job['id']] = old
    save()
    with log.open('a') as handle:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
    children[job['id']] = proc
    old.update(status='running', pid=proc.pid, process_start_ticks=ticks(proc.pid))
    save()
    print('START', job['id'], proc.pid, flush=True)


def propagate_blocked(jobs, state):
    changed = True
    while changed:
        changed = False
        for job in jobs:
            old = state['runs'].get(job['id'], {})
            if old.get('status') in ['complete', 'running', 'launching', 'failed', 'blocked_by_failed_dependency']:
                continue
            failed = [d for d in job.get('depends', []) if state['runs'].get(d, {}).get('status') in
                      ['failed', 'blocked_by_failed_dependency']]
            if failed:
                state['runs'][job['id']] = {'id': job['id'], 'status': 'blocked_by_failed_dependency', 'dependencies': failed}
                changed = True


def run(root, python):
    import fcntl
    root = Path(root)
    with (root / '.fresh_queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(root, python)


def _run_locked(root, python):
    state_path = root / 'execution_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'runs': {}, 'started_at': stamp()}
    state.update(stage='running', controller_pid=os.getpid(), accepted_main_rationales=946,
                 api_prediction_baselines_excluded=True, primary_seed=44, primary_weighting='adaptive')
    children = {}
    def save(jobs):
        state.update(updated_at=stamp(), total_jobs=len(jobs), completed_jobs=sum(
            state['runs'].get(j['id'], {}).get('status') == 'complete' for j in jobs),
            running_jobs=[jid for jid, value in state['runs'].items() if value.get('status') in ['running', 'launching']])
        atomic(state_path, state)
        atomic(root / 'status.json', state)
    while True:
        manifest = json.loads((root / 'execution_queue.json').read_text())
        jobs = manifest['jobs']
        validate_jobs(jobs)
        by_id = {j['id']: j for j in jobs}
        for job in jobs:
            jid = job['id']; old = state['runs'].get(jid, {})
            if old.get('status') == 'complete':
                verify_completed(job, old)
                continue
            if old.get('status') == 'launching':
                reconcile_launch(job, old)
                save(jobs)
                if old.get('uncertain_processes'):
                    raise ValueError('Multiple live processes require operator reconciliation: ' + jid)
                if old.get('unresolved_launch_intent'):
                    raise ValueError('Unresolved launch intent requires operator reconciliation: ' + jid)
                continue
            adopt = job.get('adopt')
            if not old and adopt:
                state['runs'][jid] = adopt_record(job, adopt)
                save(jobs)
        if any(v.get('uncertain_processes') for v in state['runs'].values()):
            raise ValueError('Uncertain live processes require operator reconciliation')
        if any(v.get('unresolved_launch_intent') for v in state['runs'].values()):
            raise ValueError('Unresolved launch intent requires operator reconciliation')
        for jid, old in list(state['runs'].items()):
            if old.get('status') != 'running':
                continue
            job = by_id[jid]
            process = children.get(jid)
            if process is not None:
                code = process.poll()
                if code is None:
                    continue
            else:
                if old.get('process_start_ticks') is None or not old.get('launch_argv'):
                    raise ValueError('Running job lacks durable process identity: ' + jid)
                if live(old['pid'], job, old['process_start_ticks'], old['launch_argv']):
                    continue
                if not definitely_exited(old['pid']):
                    raise ValueError('Running process identity requires operator reconciliation: ' + jid)
                code = None
            old.update(ended_at=stamp(), returncode=code)
            try:
                if code not in [0, None]:
                    raise ValueError('Child exit code ' + str(code))
                outputs(job, unknown_exit=code is None)
                old['status'] = 'complete'
                print('COMPLETE', jid, flush=True)
            except Exception as exc:
                old.update(status='failed', error=str(exc))
                print('FAILED', jid, str(exc), flush=True)
            children.pop(jid, None)
        propagate_blocked(jobs, state)
        occupied = {v['resource'] for v in state['runs'].values() if v.get('status') == 'running'}
        for job in sorted(jobs, key=lambda j: (j.get('priority', 100), j['id'])):
            jid = job['id']; old = state['runs'].get(jid, {})
            if old.get('status') in ['complete', 'running', 'launching', 'failed', 'blocked_by_failed_dependency']:
                continue
            deps = [state['runs'].get(d, {}).get('status') for d in job.get('depends', [])]
            if any(s != 'complete' for s in deps):
                continue
            needed = [job['script'], *job.get('code_dependencies', []), *job.get('inputs', [])]
            if not all(Path(p).is_file() for p in needed):
                state['runs'][jid] = {'id': jid, 'status': 'waiting_for_implementation_or_inputs',
                                      'missing': [p for p in needed if not Path(p).is_file()]}
                continue
            if job['resource'] in occupied:
                continue
            if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 2 * 1024**3:
                state['runs'][jid] = {'id': jid, 'status': 'waiting_for_disk_space'}
                continue
            launch_job(job, root, python, state, children, lambda: save(jobs))
            occupied.add(job['resource'])
        save(jobs)
        if all(state['runs'].get(j['id'], {}).get('status') == 'complete' for j in jobs):
            if manifest.get('implementation_complete'):
                state['stage'] = 'complete'; save(jobs); return 0
            state['stage'] = 'waiting_for_remaining_queue_implementation'; save(jobs)
        elif not occupied and all(state['runs'].get(j['id'], {}).get('status') in ['complete', 'failed', 'blocked_by_failed_dependency'] for j in jobs):
            state['stage'] = 'finished_with_failures'; save(jobs); return 1
        time.sleep(5)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.python))
