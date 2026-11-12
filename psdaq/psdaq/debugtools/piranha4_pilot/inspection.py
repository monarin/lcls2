"""Opt-in coordinated inspection; the ordinary pilot remains a separate path."""
import datetime
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

from control import Channel, CleanupGuard


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def enabled(config):
    return bool(config.get('hold_on_error') or config.get('inspect_after_trial'))


def release(output, campaign, api):
    output = Path(output).resolve()
    record = api.read_json(output / 'hold.json')
    if (record.get('campaign') != campaign or record.get('directory') != str(output)
            or record.get('state') != 'held' or record['deadline_monotonic'] <= time.monotonic()
            or record.get('host') != socket.gethostname()):
        raise ValueError('No matching active hold on this host; release refused')
    api.write_json(output / 'release.json', {
        'campaign': campaign, 'directory': str(output), 'trial': record['trial'],
        'action': 'release', 'requested_utc': utc_now()})


def run_child(spec, loader, api):
    if not (spec.get('execute') and spec.get('daq_stopped')):
        raise RuntimeError('Worker requires explicit execute and DAQ-stopped flags')
    directory = Path(spec['directory'])
    channel = Channel(socket.socket(fileno=spec['control_fd']), spec['campaign'])
    guard = CleanupGuard()
    guard.arm(time.monotonic() + spec['timeout'] + spec['cleanup_timeout'])
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, api.on_signal)
    lane, detector = api.CAMERAS[spec['camera']]
    identity = dict(campaign=spec['campaign'], camera=spec['camera'], lane=lane,
                    pid=os.getpid(), host=socket.gethostname(), trial=0)
    root = pc = recorder = None
    capture = api.Capture()
    initialized = False
    failure = None
    cleanup_errors = []
    inspection_error = None
    cancelled = None
    status = dict(identity, state='initializing', updated_utc=utc_now(), endpoint=None,
                  target_timeout=False, deadline_monotonic=None, release_reason=None)

    def publish(state, **fields):
        status.update(fields, state=state, updated_utc=utc_now(), updated_monotonic=time.monotonic())
        api.write_json(directory / 'hold.json', status)

    def save_failure(exc):
        nonlocal failure
        if failure is not None:
            return
        detected = time.monotonic()
        failure = dict(identity, ok=False, phase=capture.failure_phase or capture.phase,
                       exception=repr(exc), traceback=traceback.format_exc(),
                       detected_monotonic=detected, detected_utc=utc_now(),
                       target_timeout=initialized and capture.failure_phase == 'allocation_gcp_wait'
                       and str(exc).startswith('*** await: prompt not seen:'))
        capture.add('failure_detected', detected_monotonic=detected)
        try:
            dropped = capture.dump(directory / 'uart-failure.json', drain=True)
            if dropped:
                failure['capture_error'] = 'Capture overflow: %d events dropped' % dropped
        except BaseException as exc:
            failure['capture_error'] = repr(exc)
        api.write_json(directory / 'failure.json', failure)

    try:
        publish('initializing')
        pc, provenance = loader()
        api.instrument(pc, capture)
        api.write_json(directory / 'provenance.json', provenance)
        restore = api.install_diagnostics(pc, capture)
        try:
            root = pc.piranha4_init(detector, dev=api.DEVICE, lanemask=1 << lane,
                                   xpmpv=None, timebase='186M', verbosity=0)
        finally:
            restore()
        pc.piranha4_init_feb(None, None)
        initialized = True
        capture.initializing = False
        endpoint = api.endpoint(root)
        ready = dict(identity, device=api.DEVICE, ready_monotonic=time.monotonic(),
                     endpoint=endpoint, watchdog_pid=guard.process.pid)
        publish('ready', endpoint=endpoint)
        api.write_json(directory / 'ready.json', ready)
        for trial in spec['trials']:
            identity['trial'] = status['trial'] = trial
            waiting_deadline = time.monotonic() + spec['timeout'] + spec['settle_seconds'] + spec['interval'] + 10
            guard.arm(waiting_deadline + spec['cleanup_timeout'])
            command = channel.wait(trial, {'allocate'}, waiting_deadline)
            allocation_deadline = command['deadline_monotonic']
            if not math.isfinite(allocation_deadline) or allocation_deadline <= time.monotonic():
                raise ValueError('Invalid allocation deadline')
            guard.arm(allocation_deadline + spec['cleanup_timeout'])
            capture.wait_index = 0
            capture.failure_phase = None
            capture.phase = 'connection_info'
            started = time.monotonic()
            publish('allocating')
            capture.add('trial_start', trial=trial)
            try:
                info = pc.piranha4_connectionInfo(root, api.json.dumps(command['allocation']))
            except BaseException as exc:
                save_failure(exc)
                if not (spec.get('hold_on_error') and failure['target_timeout']
                        and 'capture_error' not in failure):
                    raise
                result = dict(ok=False, target_timeout=True, trial=trial, pid=os.getpid(),
                              detected_monotonic=failure['detected_monotonic'],
                              failure_path=str(directory / 'failure.json'))
                kind = 'target_failure'
                decision_deadline = min(allocation_deadline, failure['detected_monotonic'] + spec['hold_seconds'])
            else:
                capture.phase = 'connection_info_return'
                dropped = capture.dump(directory / ('uart-%04d.json' % trial), drain=True)
                if dropped:
                    raise RuntimeError('Capture overflow: %d events dropped' % dropped)
                result = dict(ok=True, trial=trial, pid=os.getpid(), camera=spec['camera'],
                              started_monotonic=started, elapsed_seconds=time.monotonic() - started,
                              since_root_ready_seconds=started - ready['ready_monotonic'], connection_info=info)
                kind = 'allocation_done'
                decision_deadline = allocation_deadline
            publish(kind, target_timeout=bool(failure and failure['target_timeout']))
            channel.send(kind, trial, result=result)
            decision = channel.wait(trial, {'cleanup', 'hold', 'release'}, decision_deadline)
            if decision['kind'] == 'cleanup':
                if failure:
                    raise ValueError('Cleanup cannot resume a failed campaign')
                publish('deallocating')
                guard.arm(time.monotonic() + spec['cleanup_timeout'])
                capture.phase = 'dealloc'
                pc.piranha4_connectionShutdown()
                api.write_json(directory / ('result-%04d.json' % trial), result)
                continue
            if decision['kind'] == 'release':
                raise RuntimeError('Inspection cancelled before hold: ' + decision['reason'])
            hold_deadline = decision['deadline_monotonic']
            own_limit = failure['detected_monotonic'] + spec['hold_seconds'] if failure else time.monotonic() + spec['hold_seconds']
            if not math.isfinite(hold_deadline) or hold_deadline > own_limit or hold_deadline <= time.monotonic():
                raise ValueError('Invalid/expired inspection deadline')
            guard.arm(hold_deadline + spec['cleanup_timeout'])
            recorder = api.InspectionRecorder(capture, root, directory, dict(identity))
            capture.add('hold_enter', deadline_monotonic=hold_deadline)
            publish('held', deadline_monotonic=hold_deadline, reason=decision['reason'],
                    parked_monotonic=time.monotonic(), root_alive=True)
            recorder.snapshot(dict(status), force=True)
            channel.send('held', trial, status=dict(status))
            reason = 'timeout'
            while time.monotonic() < hold_deadline:
                recorder.snapshot(dict(status))
                try:
                    message = channel.receive(min(.05, hold_deadline - time.monotonic()))
                except EOFError:
                    reason = 'parent_lost'
                    inspection_error = 'Parent lost during inspection'
                    break
                if message is not None:
                    if message['trial'] != trial or message['kind'] != 'release':
                        raise ValueError('Unexpected/stale hold message')
                    reason = message['reason']
                    break
            publish('releasing', release_reason=reason)
            capture.add('hold_release', reason=reason)
            recorder.snapshot(dict(status), force=True)
            break
    except BaseException as exc:
        is_cancellation = isinstance(exc, SystemExit) and status['state'] in ('held', 'releasing')
        try:
            if is_cancellation:
                cancelled = repr(exc)
                publish('releasing', cancelled=cancelled, release_reason='signal')
            else:
                inspection_error = repr(exc)
                save_failure(exc)
                publish('inspection_unavailable', inspection_error=inspection_error,
                        release_reason='signal' if isinstance(exc, SystemExit) else 'error')
        finally:
            try:
                channel.send('cancelled' if is_cancellation else 'failed', identity['trial'], error=repr(exc))
            except (OSError, ValueError, TimeoutError):
                pass
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        guard.arm(time.monotonic() + spec['cleanup_timeout'])
        for name, action in [('connectionShutdown', pc.piranha4_connectionShutdown if pc else None),
                             ('root.stop', root.stop if root is not None else None)]:
            if action:
                try:
                    action()
                except BaseException as exc:
                    cleanup_errors.append(name + ': ' + repr(exc))
        try:
            capture.dump(directory / 'uart-final.json')
            publish('cleanup_failed' if cleanup_errors else 'released', root_alive=None if cleanup_errors else False)
            api.write_json(directory / 'exit.json', dict(identity, ok=not (failure or cleanup_errors or inspection_error or cancelled),
                cleanup_errors=cleanup_errors, inspection_error=inspection_error,
                release_reason=status['release_reason'], cancelled=cancelled))
        finally:
            channel.close()
            guard.close()
    return int(bool(failure or cleanup_errors or inspection_error or cancelled))


def cleanup_succeeded(children, result, target_timeout, api):
    clean = not (result['forced_kill_pids'] or result['unreaped_pids'] or result['errors'])
    for proc, directory, _ in children:
        path = directory / 'exit.json'
        record = api.read_json(path) if path.exists() else None
        clean = clean and bool(record is not None and not record['cleanup_errors']
            and not record['inspection_error'] and not record.get('cancelled')
            and proc.returncode in ((0, 1) if target_timeout else (0,)))
    return clean


def collect(children, channels, trial, kinds, deadline, on_message=None):
    messages = {}
    while len(messages) < len(children):
        for proc, directory, _ in children:
            if proc.poll() is not None:
                raise RuntimeError('Child exited before coordinated decision: ' + str(directory))
            if proc.pid in messages:
                continue
            message = channels[proc.pid].receive()
            if message is not None:
                if message['trial'] != trial or message['kind'] not in kinds:
                    raise RuntimeError('Unexpected child response: ' + repr(message))
                messages[proc.pid] = message
                if on_message:
                    on_message(message)
        if time.monotonic() >= (deadline() if callable(deadline) else deadline):
            raise TimeoutError('Watchdog waiting for coordinated ' + ','.join(sorted(kinds)))
        if len(messages) < len(children):
            time.sleep(.02)
    return [messages[proc.pid] for proc, _, _ in children]


def coordinated_trial(children, channels, trial, allocation, config, output, summary, api):
    allocation_deadline = time.monotonic() + config['timeout']
    outcome_deadline = [allocation_deadline]
    first_detection = [None]
    for channel in channels.values():
        channel.send('allocate', trial, allocation=allocation, deadline_monotonic=allocation_deadline)

    def note_failure(message):
        if message['kind'] == 'target_failure':
            summary['target_timeout'] = True
            summary['error'] = 'Allocation GCP timeout; no further trials'
            detected = message['result']['detected_monotonic']
            first_detection[0] = detected if first_detection[0] is None else min(first_detection[0], detected)
            inspection_deadline = first_detection[0] + config['hold_seconds']
            outcome_deadline[0] = min(allocation_deadline, inspection_deadline)
            api.write_json(output / 'hold.json', dict(campaign=config['campaign'], trial=trial,
                directory=str(output), state='awaiting_peers', host=socket.gethostname(),
                pid=os.getpid(), reason='target_timeout', target_timeout=True,
                deadline_monotonic=inspection_deadline, allocation_deadline_monotonic=allocation_deadline,
                detected_monotonic=first_detection[0],
                updated_utc=utc_now(), release_reason=None))
            api.write_json(output / 'summary.json', summary)

    outcomes = collect(children, channels, trial, {'allocation_done', 'target_failure'},
                       lambda: outcome_deadline[0], note_failure)
    failures = [item['result'] for item in outcomes if item['kind'] == 'target_failure']
    rehearsal = config.get('inspect_after_trial') == trial
    if not failures and not rehearsal:
        for channel in channels.values():
            channel.send('cleanup', trial)
        results = api.wait_files(children, 'result-%04d.json' % trial, config['cleanup_timeout'])
        return results, False
    detected = min(item['detected_monotonic'] for item in failures) if failures else time.monotonic()
    hold_deadline = detected + config['hold_seconds']
    reason = 'target_timeout' if failures else 'rehearsal'
    record = dict(campaign=config['campaign'], directory=str(output.resolve()), trial=trial,
                  host=socket.gethostname(), pid=os.getpid(), state='preparing', reason=reason,
                  target_timeout=bool(failures), detected_monotonic=detected,
                  deadline_monotonic=hold_deadline, created_utc=utc_now(), release_reason=None)
    api.write_json(output / 'hold.json', record)
    for channel in channels.values():
        channel.send('hold', trial, deadline_monotonic=hold_deadline, reason=reason)
    acknowledgments = collect(children, channels, trial, {'held'}, min(allocation_deadline, hold_deadline))
    record.update(state='held', held_utc=utc_now(), cameras=[item['status'] for item in acknowledgments],
                  release_command=shlex.join([sys.executable, str(Path(api.__file__).resolve()),
                      '--release', str(output.resolve()), '--campaign-id', config['campaign']]))
    api.write_json(output / 'hold.json', record)
    print('Inspection held: ' + str(output / 'hold.json'), flush=True)
    print('Release: ' + record['release_command'], flush=True)
    for item in record['cameras']:
        print('%s PID %s: %s' % (item['camera'], item['pid'], item['endpoint']), flush=True)
    release_reason = 'timeout'
    last_rejected = None
    while time.monotonic() < hold_deadline:
        path = output / 'release.json'
        if path.exists():
            try:
                request = api.read_json(path)
                valid = (isinstance(request, dict) and request.get('campaign') == config['campaign']
                         and request.get('directory') == str(output.resolve()) and request.get('trial') == trial
                         and request.get('action') == 'release')
                if valid:
                    release_reason = 'operator' if time.monotonic() < hold_deadline else 'timeout'
                    break
                rejection = 'Stale or foreign release request'
            except (ValueError, UnicodeError) as exc:
                rejection = repr(exc)
            if rejection != last_rejected:
                last_rejected = rejection
                record['rejected_release'] = rejection
                api.write_json(output / 'hold.json', record)
        for proc, _, _ in children:
            if proc.poll() is not None:
                # Workers enforce this same deadline independently. The parent
                # may resume an iteration after normal worker timeout/cleanup.
                # Exit records and return codes are still verified below by
                # run_controller; closure alone does not establish success.
                if time.monotonic() >= hold_deadline:
                    break
                raise RuntimeError('Child died during inspection')
            try:
                message = channels[proc.pid].receive()
            except EOFError:
                if time.monotonic() >= hold_deadline:
                    break
                raise
            if message is not None:
                raise RuntimeError('Child failed during inspection: ' + repr(message))
        time.sleep(min(.05, max(0., hold_deadline - time.monotonic())))
    record.update(state='releasing', release_reason=release_reason, released_utc=utc_now())
    api.write_json(output / 'hold.json', record)
    for channel in channels.values():
        try:
            channel.send('release', trial, reason=release_reason)
        except (BrokenPipeError, ConnectionResetError):
            if release_reason != 'timeout':
                raise
    summary['inspection'] = record
    return [item['result'] for item in outcomes], True


def run_controller(config, output, worker_prefix, api):
    worker_prefix = worker_prefix or [sys.executable, '-u', str(Path(api.__file__).resolve()), '--worker-spec']
    config = dict(config, campaign=uuid.uuid4().hex)
    output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    summary = dict(ok=False, config=config, started_utc=utc_now(), script_sha256=api.sha(api.__file__),
                   local_sources_sha256=api.local_sources(), trials=[], groups=[], target_timeout=False)
    api.write_json(output / 'manifest.json', summary)
    groups = [[trial] for trial in range(1, config['trials'] + 1)] if config['mode'] == 'cold' else [list(range(1, config['trials'] + 1))]
    children, channels = [], {}
    previous = {sig: signal.signal(sig, api.on_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    trial = 0
    signal_release = False
    try:
        for group_index, trials in enumerate(groups, 1):
            group = output / ('process-group-%04d' % group_index)
            group.mkdir()
            for camera in config['cameras']:
                directory = group / camera
                directory.mkdir()
                parent_socket, child_socket = socket.socketpair()
                spec = dict(config, camera=camera, directory=str(directory), trials=trials,
                            control_fd=child_socket.fileno())
                log = (directory / 'process.log').open('w')
                try:
                    api.write_json(directory / 'spec.json', spec)
                    proc = subprocess.Popen(worker_prefix + [str(directory / 'spec.json')],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(child_socket.fileno(),))
                except BaseException:
                    log.close()
                    parent_socket.close()
                    raise
                finally:
                    child_socket.close()
                children.append((proc, directory, log))
                channels[proc.pid] = Channel(parent_socket, config['campaign'])
            ready = api.wait_files(children, 'ready.json', config['timeout'])
            for (proc, _, _), item in zip(children, ready):
                if item['pid'] != proc.pid or item['host'] != socket.gethostname():
                    raise RuntimeError('Worker PID/hostname does not match parent')
            peers = {item['camera']: {'active': True, 'proc_info': {
                'host': item['host'], 'pid': item['pid'], 'device': api.DEVICE}} for item in ready}
            api.pause_children(children, config['settle_seconds'])
            held = False
            for trial in trials:
                allocation = {'body': {'drp': peers}}
                api.write_json(group / ('alloc-%04d.json' % trial), allocation)
                started = time.monotonic()
                results, held = coordinated_trial(children, channels, trial, allocation, config, output, summary, api)
                summary['trials'].append(dict(trial=trial, results=results, coordinated_seconds=time.monotonic() - started))
                api.write_json(output / 'summary.json', summary)
                if held:
                    break
                if trial != trials[-1]:
                    api.pause_children(children, config['interval'])
            group_result = dict(group=group_index, **api.stop_children(children, config['cleanup_timeout'], terminate=False))
            summary['groups'].append(group_result)
            cleanup_ok = cleanup_succeeded(children, group_result, summary['target_timeout'], api)
            children = []
            for channel in channels.values():
                channel.close()
            channels = {}
            if held:
                summary['inspection'].update(state='released' if cleanup_ok else 'cleanup_failed', cleanup_verified=cleanup_ok)
                api.write_json(output / 'hold.json', summary['inspection'])
            if not cleanup_ok:
                raise RuntimeError('Worker exit or cleanup failed')
            if held:
                break
            if group_index != len(groups):
                time.sleep(config['interval'])
        summary['ok'] = not summary['target_timeout']
    except BaseException as exc:
        summary.update(ok=False, error=repr(exc), traceback=traceback.format_exc())
        hold_path = output / 'hold.json'
        if hold_path.exists():
            record = api.read_json(hold_path)
            signal_release = isinstance(exc, SystemExit) and record['state'] == 'held'
            if signal_release:
                record.update(state='releasing', release_reason='parent_signal', released_utc=utc_now())
                summary['inspection'] = record
            elif record['state'] in ('released', 'cleanup_failed'):
                record['cleanup_error'] = repr(exc)
                summary['inspection'] = record
            else:
                record.update(state='inspection_unavailable',
                              release_reason=record.get('release_reason') or ('signal' if isinstance(exc, SystemExit) else 'error'))
            record.update(error=repr(exc), updated_utc=utc_now())
            api.write_json(hold_path, record)
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            for channel in channels.values():
                try:
                    channel.send('release', trial, reason='parent_signal' if signal_release else 'parent_shutdown')
                except (OSError, ValueError, TimeoutError):
                    pass
            if children:
                summary['abort_cleanup'] = api.stop_children(children, config['cleanup_timeout'], terminate=not signal_release)
                if signal_release:
                    clean = cleanup_succeeded(children, summary['abort_cleanup'], summary['target_timeout'], api)
                    summary['inspection'].update(state='released' if clean else 'cleanup_failed', cleanup_verified=clean)
                    api.write_json(output / 'hold.json', summary['inspection'])
                summary['failure_records'] = []
                for _, directory, _ in children:
                    path = directory / 'failure.json'
                    if path.exists():
                        try:
                            failure = api.read_json(path)
                            summary['target_timeout'] |= bool(failure.get('target_timeout'))
                            summary['failure_records'].append(str(path))
                        except (OSError, ValueError) as exc:
                            summary['failure_read_error'] = repr(exc)
            summary['finished_utc'] = utc_now()
            api.write_json(output / 'summary.json', summary)
        finally:
            for channel in channels.values():
                channel.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if summary['ok'] else 1
