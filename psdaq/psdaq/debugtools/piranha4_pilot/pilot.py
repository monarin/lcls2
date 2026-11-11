#!/usr/bin/env python3
"""TMO Piranha allocation pilot. Dry-run by default; see README.md.

No DAQ or hardware packages are imported by the parent. Only explicitly gated
child processes open hardware. Run this file by path using the deployed Python.
"""
import argparse
import datetime
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback

import inspection
from evidence import Capture, InspectionRecorder, endpoint, install_diagnostics

HOST = 'drp-srcf-cmp013'
DEVICE = '/dev/datadev_0'
RELEASE = '/sdf/group/lcls/ds/ana/sw/conda2/rel/tmo/lcls2_110326_disable000'
SUBMODULES = '/sdf/group/lcls/ds/ana/sw/conda2-v4/rel/lcls2_submodules_09092026'
ENV = '/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9'
CONFIG_SHA = '9efaf0c9b206e9823e6ba3751c7b0ee3cdc68b8fb3849889fabc5cf404adcca6'
ROOT_SHA = '26a2e0a397a52ecd01fc0612ba5919e4b8f8edbc689485ef975962924969ae20'
UART_SHA = '6a40cf5b8b1562ccfd4dbb1bf7d5e9b3b3b67a6eabb038a62d5a6874bb6fe459'
TX_SHA = '06b9327bef8d9da8631ef16a47a8212674c7179030bee2bcc66c034925a52a5d'
CAMERAS = {'fzp': (0, 'fzp_piranha'), 'atm': (1, 'atm_piranha_ip1')}


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')
    os.replace(str(temp), str(path))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_sources():
    return {name: sha(Path(__file__).with_name(name)) for name in
            ('pilot.py', 'inspection.py', 'control.py', 'evidence.py', 'inspect_hold.py')}


def check_environment():
    if socket.gethostname().split('.')[0] != HOST:
        raise RuntimeError('Hardware execution requires host ' + HOST)
    for name, expected in [('TESTRELDIR', RELEASE + '/install'), ('SUBMODULEDIR', SUBMODULES)]:
        if os.environ.get(name) != expected:
            raise RuntimeError('Source the pinned setup_env_daq.sh: wrong ' + name)
    if Path(sys.prefix).resolve() != Path(ENV).resolve():
        raise RuntimeError('Use Python from ' + ENV)


def preflight():
    """Metadata only. This does not establish exclusivity for invisible processes."""
    check_environment()
    device_stat = os.stat(DEVICE)
    if not stat.S_ISCHR(device_stat.st_mode):
        raise RuntimeError('Expected character device ' + DEVICE)
    holders, unreadable = [], 0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            fds = list((proc / 'fd').iterdir())
        except PermissionError:
            unreadable += 1
            continue
        except FileNotFoundError:
            continue
        for fd in fds:
            try:
                info = fd.stat()
                if stat.S_ISCHR(info.st_mode) and info.st_rdev == device_stat.st_rdev:
                    holders.append(str(fd))
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    if holders:
        raise RuntimeError('Board already open: ' + ', '.join(holders))
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(table).read_text().splitlines()[1:]:
            cols = line.split()
            if cols[3] == '0A' and int(cols[1].split(':')[1], 16) in (5561, 5562):
                raise RuntimeError('Barrier listener already present: ' + cols[1])
    return {'visible_device_holders': holders, 'inaccessible_process_fd_directories': unreadable,
            'limitation': 'Operator must establish exclusive board access, including invisible clients.'}


def instrument(pc, capture):
    original_rx = pc.MyUartPiranha4Rx
    tx = pc.clink.ClinkSerialTx
    original_send, original_escape = tx.sendString, tx.sendEscape

    class ObservedRx(original_rx):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            capture.receiver = self

        def _acceptFrame(self, frame):
            raw = bytearray(frame.getPayload())
            frame.read(raw, 0)
            capture.add('rx', bytes(raw), payload_bytes=len(raw))
            return super()._acceptFrame(frame)

        def _clear(self):
            capture.add('clear')
            return super()._clear()

        def _await(self, *args, **kwargs):
            index = capture.wait_index
            capture.wait_index += 1
            capture.phase = ('entry_wait', 'allocation_gcp_wait', 'vt_wait', 'vv_wait')[min(index, 3)]
            capture.add('await_enter')
            try:
                result = super()._await(*args, **kwargs)
            except BaseException:
                capture.failure_phase = capture.phase
                capture.add('await_failed')
                raise
            capture.add('await_return')
            if index == 1:
                capture.phase = 'CPA_check'
            return result

    def send(obj, text):
        capture.phase = ('startup_' if capture.initializing else 'allocation_') + text.split()[0].lower()
        capture.add('tx_string', text=text)
        return original_send(obj, text)

    def escape(obj):
        capture.add('tx_escape', b'\x1b')
        return original_escape(obj)

    pc.MyUartPiranha4Rx = ObservedRx
    tx.sendString, tx.sendEscape = send, escape


def load_real():
    check_environment()
    from psdaq.configdb import piranha4_config as pc
    import rogue
    files = {'config': pc.__file__, 'gateway': inspect.getfile(pc.cameralink_gateway.ClinkDevRoot),
             'uart': inspect.getfile(pc.clink.UartPiranha4),
             'tx': inspect.getfile(pc.clink.ClinkSerialTx)}
    hashes = {key: sha(path) for key, path in files.items()}
    for key, expected in [('config', CONFIG_SHA), ('gateway', ROOT_SHA), ('uart', UART_SHA), ('tx', TX_SHA)]:
        if hashes[key] != expected:
            raise RuntimeError('Baseline source changed: ' + key + ' ' + str(files[key]))
    return pc, {'files': files, 'sha256': hashes, 'rogue_version': str(rogue.Version.current()),
                'python': sys.version, 'executable': sys.executable,
                'environment': {key: os.environ.get(key) for key in
                                ('TESTRELDIR', 'SUBMODULEDIR', 'CONDA_PREFIX', 'PYTHONPATH', 'LANG', 'LC_ALL')}}


def on_signal(signum, frame):
    raise SystemExit('received signal %d' % signum)


def child_main(spec, loader=load_real):
    """One interpreter/root per camera. Parent supplies an identical Alloc body."""
    if inspection.enabled(spec):
        return inspection.run_child(spec, loader, sys.modules[__name__])
    if not (spec.get('execute') and spec.get('daq_stopped')):
        raise RuntimeError('Worker requires explicit execute and DAQ-stopped flags')
    directory = Path(spec['directory'])
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    root = pc = None
    capture = Capture()
    cleanup_errors = []
    failure = None
    current_trial = None
    try:
        pc, provenance = loader()
        instrument(pc, capture)
        write_json(directory / 'provenance.json', provenance)
        lane, detector = CAMERAS[spec['camera']]
        root = pc.piranha4_init(detector, dev=DEVICE, lanemask=1 << lane,
                               xpmpv=None, timebase='186M', verbosity=0)
        pc.piranha4_init_feb(None, None)
        capture.initializing = False
        ready = {'pid': os.getpid(), 'host': socket.gethostname(), 'device': DEVICE,
                 'camera': spec['camera'], 'lane': lane, 'ready_monotonic': time.monotonic()}
        write_json(directory / 'ready.json', ready)
        for current_trial in spec['trials']:
            command = directory.parent / ('alloc-%04d.json' % current_trial)
            deadline = time.monotonic() + spec['timeout'] + spec['settle_seconds'] + spec['interval'] + 10
            while not command.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('Parent did not supply allocation')
                time.sleep(0.05)
            capture.wait_index = 0
            capture.failure_phase = None
            capture.phase = 'connection_info'
            started = time.monotonic()
            capture.add('trial_start', trial=current_trial)
            info = pc.piranha4_connectionInfo(root, command.read_text())
            capture.phase = 'connection_info_return'
            dropped = capture.dump(directory / ('uart-%04d.json' % current_trial), drain=True)
            if dropped:
                raise RuntimeError('Capture overflow: %d events dropped' % dropped)
            capture.phase = 'dealloc'
            pc.piranha4_connectionShutdown()
            write_json(directory / ('result-%04d.json' % current_trial), {
                'ok': True, 'trial': current_trial, 'pid': os.getpid(), 'camera': spec['camera'],
                'started_monotonic': started, 'elapsed_seconds': time.monotonic() - started,
                'since_root_ready_seconds': started - ready['ready_monotonic'], 'connection_info': info})
            # Next warm trial starts a new segment; late RX remains in the ring.
    except BaseException as exc:
        # Preserve the failed segment before any potentially blocking cleanup.
        # A parent's termination request must not interrupt this bounded snapshot.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        failure = {'ok': False, 'trial': current_trial, 'pid': os.getpid(),
                   'phase': capture.failure_phase or capture.phase, 'exception': repr(exc),
                   'traceback': traceback.format_exc(),
                   'target_timeout': capture.failure_phase == 'allocation_gcp_wait'
                   and str(exc).startswith('*** await: prompt not seen:')}
        try:
            capture.dump(directory / 'uart-failure.json')
        except BaseException as snapshot_error:
            failure['capture_error'] = repr(snapshot_error)
        write_json(directory / 'failure.json', failure)
    finally:
        # Ignore repeat termination while bounded parent watchdog allows cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for name, action in [('connectionShutdown', pc.piranha4_connectionShutdown if pc else None),
                             ('root.stop', root.stop if root is not None else None)]:
            if action:
                try:
                    action()
                except BaseException as exc:
                    cleanup_errors.append(name + ': ' + repr(exc))
        try:
            capture.dump(directory / 'uart-final.json')
        except BaseException as exc:
            cleanup_errors.append('capture: ' + repr(exc))
        write_json(directory / 'exit.json', {'ok': failure is None and not cleanup_errors,
                                            'cleanup_errors': cleanup_errors, 'pid': os.getpid()})
    return 1 if failure or cleanup_errors else 0


def wait_files(children, name, timeout):
    deadline = time.monotonic() + timeout
    while True:
        found = []
        for proc, directory, _ in children:
            if (directory / 'failure.json').exists():
                raise RuntimeError('Child failure: ' + str(directory / 'failure.json'))
            path = directory / name
            if path.exists():
                found.append(read_json(path))
            elif proc.poll() is not None:
                raise RuntimeError('Child exited before ' + str(path))
        if len(found) == len(children):
            return found
        if time.monotonic() >= deadline:
            raise TimeoutError('Watchdog waiting for ' + name)
        time.sleep(0.05)


def stop_children(children, grace, terminate):
    """Bound both graceful exit and post-kill reaping, even for stuck devices."""
    status = {'forced_kill_pids': [], 'unreaped_pids': [], 'errors': []}
    if terminate:
        for proc, _, _ in children:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    status['errors'].append('terminate %d: %s' % (proc.pid, exc))
    deadline = time.monotonic() + grace
    while any(proc.poll() is None for proc, _, _ in children) and time.monotonic() < deadline:
        time.sleep(0.05)
    # Signal ALL survivors before waiting on any; one unkillable task must not
    # prevent its peer receiving SIGKILL or prevent a final failed summary.
    for proc, _, _ in children:
        if proc.poll() is None:
            status['forced_kill_pids'].append(proc.pid)
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                status['errors'].append('kill %d: %s' % (proc.pid, exc))
    deadline = time.monotonic() + min(grace, 2.)
    for proc, _, log in children:
        try:
            proc.wait(timeout=max(0., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            status['unreaped_pids'].append(proc.pid)
        log.close()
    status['exit_codes'] = [proc.poll() for proc, _, _ in children]
    return status


def pause_children(children, duration):
    deadline = time.monotonic() + duration
    while True:
        for proc, directory, _ in children:
            if proc.poll() is not None or (directory / 'failure.json').exists():
                raise RuntimeError('Child failed during settling: ' + str(directory))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def run_controller(config, output, worker_prefix=None):
    """Testable orchestration; CLI performs host/environment/exclusivity gates."""
    if inspection.enabled(config):
        return inspection.run_controller(config, output, worker_prefix, sys.modules[__name__])
    worker_prefix = worker_prefix or [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-spec']
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    summary = {'ok': False, 'config': config, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'script_sha256': sha(__file__), 'local_sources_sha256': local_sources(), 'trials': [], 'groups': []}
    write_json(output / 'manifest.json', summary)
    groups = [[i] for i in range(1, config['trials'] + 1)] if config['mode'] == 'cold' else [list(range(1, config['trials'] + 1))]
    children = []
    try:
        for group_index, trials in enumerate(groups, 1):
            print('Starting process group %d, trials %s' % (group_index, trials), flush=True)
            group = output / ('process-group-%04d' % group_index)
            group.mkdir()
            children = []
            for camera in config['cameras']:
                directory = group / camera
                directory.mkdir()
                spec = dict(config, camera=camera, directory=str(directory.resolve()), trials=trials)
                write_json(directory / 'spec.json', spec)
                log = (directory / 'process.log').open('w')
                try:
                    proc = subprocess.Popen(worker_prefix + [str((directory / 'spec.json').resolve())],
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                children.append((proc, directory, log))
            ready = wait_files(children, 'ready.json', config['timeout'])
            # Correct host/PID prevents a false supervisor election and barrier deadlock.
            for (proc, _, _), item in zip(children, ready):
                if item['pid'] != proc.pid or item['host'] != socket.gethostname():
                    raise RuntimeError('Worker PID/hostname does not match parent')
            peers = {item['camera']: {'active': True, 'proc_info': {
                'host': item['host'], 'pid': item['pid'], 'device': DEVICE}} for item in ready}
            pause_children(children, config['settle_seconds'])
            for trial in trials:
                print('Trial %d: calling connectionInfo for %s' % (trial, config['cameras']), flush=True)
                write_json(group / ('alloc-%04d.json' % trial), {'body': {'drp': peers}})
                results = wait_files(children, 'result-%04d.json' % trial, config['timeout'])
                summary['trials'].append({'trial': trial, 'results': results})
                write_json(output / 'summary.json', summary)
                if trial != trials[-1]:
                    pause_children(children, config['interval'])
            group_result = dict(group=group_index, **stop_children(children, config['cleanup_timeout'], terminate=False))
            summary['groups'].append(group_result)
            finished_children, children = children, []
            if (group_result['forced_kill_pids'] or group_result['unreaped_pids'] or group_result['errors']
                    or any(proc.returncode != 0 or not read_json(directory / 'exit.json')['ok']
                             for proc, directory, _ in finished_children)):
                raise RuntimeError('Worker exit or cleanup failed')
            if group_index != len(groups):
                time.sleep(config['interval'])
        summary['ok'] = True
    except BaseException as exc:
        summary['error'] = repr(exc)
        summary['traceback'] = traceback.format_exc()
    finally:
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if children:
                try:
                    summary['abort_cleanup'] = stop_children(children, config['cleanup_timeout'], terminate=True)
                except BaseException as exc:
                    summary['abort_cleanup_error'] = repr(exc)
                    summary['ok'] = False
            summary['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            write_json(output / 'summary.json', summary)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if summary['ok'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cold', 'warm'), default='cold')
    parser.add_argument('--cameras', choices=('fzp', 'atm', 'both'), default='fzp')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--settle-seconds', type=float, default=40., help='Delay after all roots initialize')
    parser.add_argument('--interval', type=float, default=2.)
    parser.add_argument('--timeout', type=float, default=60., help='Watchdog per init/allocation phase')
    parser.add_argument('--cleanup-timeout', type=float, default=10.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--daq-stopped', action='store_true', help='Operator confirms exclusive board access')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--hold-on-error', action='store_true', help='Retain both roots after allocation GCP timeout')
    modes.add_argument('--inspect-after-trial', type=int, help='Successful-trial rehearsal; release ends the campaign')
    parser.add_argument('--hold-seconds', type=float, help='Inspection duration (default 900; maximum 3600)')
    parser.add_argument('--release', type=Path, help='Release an existing hold in this campaign directory')
    parser.add_argument('--campaign-id', help='Exact campaign token from hold.json, required with --release')
    parser.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        return child_main(read_json(args.worker_spec))
    if args.release or args.campaign_id:
        if (not args.release or not args.campaign_id or args.execute or args.hold_on_error
                or args.inspect_after_trial is not None or args.hold_seconds is not None):
            parser.error('--release requires --campaign-id and cannot start another campaign')
        try:
            inspection.release(args.release, args.campaign_id, sys.modules[__name__])
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
        print('Release requested; check hold.json and summary.json for verified cleanup')
        return 0
    if not 1 <= args.trials <= 300:
        parser.error('--trials must be 1..300')
    if args.inspect_after_trial is not None and not 1 <= args.inspect_after_trial <= args.trials:
        parser.error('--inspect-after-trial must be within 1..trials')
    if args.hold_seconds is not None:
        if not inspection.enabled(vars(args)):
            parser.error('--hold-seconds requires an inspection mode')
        if not math.isfinite(args.hold_seconds) or not 0 < args.hold_seconds <= 3600:
            parser.error('--hold-seconds must be finite, positive and at most 3600')
    elif inspection.enabled(vars(args)):
        args.hold_seconds = 900.
    for name in ('settle_seconds', 'interval', 'timeout', 'cleanup_timeout'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name.endswith('timeout') and value == 0):
            parser.error(name + ' must be finite and positive (delays may be zero)')
    config = vars(args).copy()
    config.pop('worker_spec')
    config.pop('output')
    config.pop('release')
    config.pop('campaign_id')
    config['cameras'] = ['fzp', 'atm'] if args.cameras == 'both' else [args.cameras]
    if not args.execute:
        print(json.dumps({'dry_run': True, 'host': HOST, 'device': DEVICE, 'plan': config,
                          'release': RELEASE, 'hardware_opened': False}, indent=2))
        return 0
    if not args.daq_stopped or args.output is None:
        parser.error('--execute requires --daq-stopped and a new --output directory')
    # Parent keeps an advisory host lock for the entire campaign. This cannot
    # exclude existing DAQ/devGui clients; operator attestation remains required.
    check_environment()
    lock_fd = os.open('/tmp/tmo-piranha-datadev_0-pilot.lock',
                      os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config['preflight'] = preflight()
        print('Exclusive-access preflight:', json.dumps(config['preflight']), flush=True)
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        rc = run_controller(config, args.output)
        print(('Completed' if rc == 0 else 'STOPPED on failure') + ': ' + str(args.output / 'summary.json'), flush=True)
        return rc
    finally:
        os.close(lock_fd)


if __name__ == '__main__':
    sys.exit(main())
