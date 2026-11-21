import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fresh_queue
from build_continuation_seed44_20261009 import VERSION, runtime_directory

EXPECTED_IDS = ('features_frozen_teacher44', 'main44_logistic_regression',
    'main44_random_forest', 'main44_logit_boost', 'main44_gradient_boosting',
    'main44_textcnn', 'main44_bert', 'main44_single_cot')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completion_ready(manifest, state, verify=fresh_queue.verify_completed):
    jobs = manifest['jobs']
    if tuple(j['id'] for j in jobs) != EXPECTED_IDS:
        raise ValueError('Expected exactly the frozen eight main-phase jobs')
    runs = state.get('runs', {})
    if set(runs) - set(EXPECTED_IDS):
        raise ValueError('Unknown job identities in main phase')
    if state.get('stage') in ('stopped_safety_gate', 'finished_with_failures', 'failed'):
        raise ValueError('Main phase failed or stopped at safety gate')
    for job in jobs:
        jid = job['id']
        old = runs.get(jid, {})
        if old.get('status') in ('failed', 'blocked_by_failed_dependency'):
            raise ValueError('Failed/blocked main phase: ' + jid)
        if old.get('uncertain_processes') or old.get('unresolved_launch_intent'):
            raise ValueError('Uncertain main process identity: ' + jid)
        if old.get('status') == 'complete':
            verify(job, old)
    ready = all(runs.get(jid, {}).get('status') == 'complete' for jid in EXPECTED_IDS)
    return ready and state.get('stage') == 'waiting_for_remaining_queue_implementation'


def process_record(pid):
    directory = Path('/proc') / str(pid)
    try:
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in ('Z', 'X'):
            return None
        return {'pid': int(pid), 'ticks': fields[19], 'ppid': int(fields[1]),
            'argv': [p.decode() for p in (directory / 'cmdline').read_bytes().split(b'\0') if p]}
    except FileNotFoundError:
        return None


def matches_identity(actual, expected):
    return actual is not None and actual['pid'] == expected['controller_pid'] and \
        actual['ticks'] == str(expected['process_start_ticks']) and actual['argv'] == expected['argv']


def physical_gpu_zero_uuid(gpu_inventory):
    inventory = [line.split(',') for line in gpu_inventory.splitlines() if line.strip()]
    zero = [parts[1].strip() for parts in inventory if len(parts) == 2 and parts[0].strip() == '0']
    if len(zero) != 1 or not zero[0].startswith('GPU-'):
        raise ValueError('Cannot establish physicalGPU0 identity')
    return zero[0]


def gpu_zero_idle(gpu_inventory, compute_inventory):
    zero = physical_gpu_zero_uuid(gpu_inventory)
    for line in compute_inventory.splitlines():
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) != 2:
            raise ValueError('Cannot parse live GPU process identity')
        if parts[1].strip() == zero:
            raise ValueError('PhysicalGPU0 occupied by live compute process: ' + parts[0].strip())
    return True


def require_gpu_zero_idle():
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    gpu_zero_idle(inventory, compute)
    return physical_gpu_zero_uuid(inventory)


def no_live_children(state, launch):
    for old in state['runs'].values():
        if old.get('pid') and process_record(old['pid']) is not None:
            raise ValueError('Main job PID still live or reused; refuse handoff')
    for directory in Path('/proc').iterdir():
        if directory.name.isdigit():
            record = process_record(int(directory.name))
            if record and record['ppid'] == launch['controller_pid']:
                raise ValueError('Controller still has a live child; refuse handoff')


def verify_snapshot(suite, expected_launch=None):
    runtime = runtime_directory(suite)
    manifest, launch = read(runtime / 'execution_queue.json'), read(runtime / 'launch.json')
    if manifest['version'] != VERSION or Path(manifest['suite']).resolve() != Path(suite).resolve():
        raise ValueError('Main manifest belongs to another suite/version')
    if expected_launch is not None and launch != expected_launch:
        raise ValueError('Main controller launch record changed')
    state = read(runtime / 'execution_status.json')
    return completion_ready(manifest, state), state, launch


def run(args):
    import fcntl
    suite = Path(args.suite).resolve()
    out = suite / 'experiments/03_efficiency'
    out.mkdir(parents=True, exist_ok=True)
    own = process_record(os.getpid())
    status = {'stage': 'waiting_for_main_phase', 'pid': os.getpid(),
        'process_start_ticks': own['ticks'], 'argv': own['argv'],
        'started_at': fresh_queue.stamp()}
    expected_launch = read(runtime_directory(suite) / 'launch.json')
    def save(**updates):
        status.update(updates, updated_at=fresh_queue.stamp())
        fresh_queue.atomic(out / 'controller_status.json', status)
    with (out / '.gate.lock').open('a') as gate_lock:
        fcntl.flock(gate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            save()
            while True:
                ready, state, launch = verify_snapshot(suite, expected_launch)
                if ready:
                    break
                if not matches_identity(process_record(launch['controller_pid']), launch):
                    raise ValueError('Main controller is dead or identity unknown before verified idle')
                time.sleep(args.poll_seconds)
            no_live_children(state, launch)
            actual = process_record(launch['controller_pid'])
            if actual is not None:
                if not matches_identity(actual, launch):
                    raise ValueError('Idle controller PID identity differs; refuse signal')
                descriptor = os.pidfd_open(launch['controller_pid'])
                try:
                    ready, state, launch = verify_snapshot(suite, expected_launch)
                    no_live_children(state, launch)
                    if not ready or not matches_identity(process_record(launch['controller_pid']), launch):
                        raise ValueError('Main controller no longer verified idle')
                    save(stage='handoff_verified_idle', stopped_controller=launch)
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                finally:
                    os.close(descriptor)
                deadline = time.monotonic() + 30
                while process_record(launch['controller_pid']) is not None:
                    if time.monotonic() > deadline:
                        raise ValueError('Idle controller did not exit; no benchmark launched')
                    time.sleep(.25)
            with (suite / '.fresh_queue.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                ready, state, launch = verify_snapshot(suite, expected_launch)
                no_live_children(state, launch)
                if not ready:
                    raise ValueError('Completion gate changed after lock acquisition')
                gpu_uuid = require_gpu_zero_idle()
                argv = [args.python, '-u', str(Path(__file__).resolve().parents[1] / 'evaluation_and_statistics/formal_efficiency.py'),
                    '--suite', str(suite)]
                save(stage='launching_efficiency', child_argv=argv, physical_gpu0_uuid=gpu_uuid,
                    child_cuda_visible_devices=gpu_uuid)
                with (out / 'benchmark.log').open('ab', buffering=0) as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, TOKENIZERS_PARALLELISM='false'))
                    save(stage='running_efficiency', child_pid=child.pid,
                         child_process_start_ticks=fresh_queue.ticks(child.pid))
                    code = child.wait()
                if code != 0:
                    raise ValueError('Efficiency process failed: exit ' + str(code))
                summary = read(out / 'summary.json')
                if summary.get('status') != 'complete' or len(summary.get('results', [])) != 12:
                    raise ValueError('Efficiency terminal summary incomplete')
                save(stage='complete', returncode=0)
        except BaseException as error:
            save(stage='stopped_safety_gate', error=repr(error))
            raise
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll-seconds must be between 1 and 60')
    raise SystemExit(run(args))
