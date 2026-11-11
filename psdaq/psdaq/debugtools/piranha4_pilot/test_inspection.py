"""Real fake subprocesses and clocks; this executable never constructs hardware."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import pilot
import inspection
from control import Channel
from test_pilot import fake_loader


def fake_worker(path):
    spec = pilot.read_json(path)
    original_capture = pilot.Capture

    class FaultCapture(original_capture):
        def dump(self, path, drain=False):
            if spec.get('test_snapshot_failure') and Path(path).name == 'uart-failure.json':
                raise OSError('injected original snapshot failure')
            return super().dump(path, drain)

    class Recorder:
        def __init__(self, capture, root, directory, identity):
            self.capture, self.directory = capture, directory
            self.sequence = 0
            self.last_snapshot = 0.

        def snapshot(self, state, force=False):
            if spec.get('test_recorder_failure') == spec['camera']:
                raise OSError('injected inspection disk failure')
            if spec.get('test_late_disk_failure') == spec['camera'] and self.sequence >= 2:
                raise OSError('injected late inspection disk failure')
            if spec.get('test_ack_hang') == spec['camera']:
                time.sleep(20.)
            if force or time.monotonic() - self.last_snapshot >= .1:
                self.capture.add('fake_late_rx', b'USER>')
                self.capture.dump(self.directory / ('late-%d.json' % (self.sequence % 4)))
                self.last_snapshot = time.monotonic()
                self.sequence += 1

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(pilot, 'Capture', FaultCapture))
        stack.enter_context(mock.patch.object(pilot, 'install_diagnostics', return_value=lambda: None))
        stack.enter_context(mock.patch.object(pilot, 'endpoint', return_value={
            'host': socket.gethostname(), 'base_port': 10000, 'ports': [10000, 10001, 10002],
            'backend': 'fake; no server/listeners or hardware'}))
        stack.enter_context(mock.patch.object(pilot, 'InspectionRecorder', Recorder))
        return pilot.child_main(spec, loader=lambda: fake_loader(spec))


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'run'
        self.config = dict(mode='warm', trials=3, cameras=['fzp', 'atm'], execute=True,
                           daq_stopped=True, timeout=2., cleanup_timeout=.6,
                           interval=0., settle_seconds=0., hold_on_error=True,
                           inspect_after_trial=None, hold_seconds=3.)
        self.worker = [sys.executable, str(Path(__file__).resolve()), '--fake-worker']

    def run_pilot(self, **changes):
        rc = pilot.run_controller(dict(self.config, **changes), self.output, self.worker)
        return rc, pilot.read_json(self.output / 'summary.json')

    def launch(self, **changes):
        output = self.output
        spec = Path(self.temp.name) / 'controller.json'
        pilot.write_json(spec, dict(config=dict(self.config, **changes), output=str(self.output)))
        log = (Path(self.temp.name) / 'controller.log').open('w')
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--fake-controller', str(spec)],
                                stdout=log, stderr=subprocess.STDOUT)

        def cleanup():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.)
            log.close()
            for ready_path in output.glob('**/ready.json'):
                for key in ('pid', 'watchdog_pid'):
                    pid = pilot.read_json(ready_path).get(key)
                    if pid and self.running(pid):
                        os.kill(pid, signal.SIGKILL)

        self.addCleanup(cleanup)
        return proc

    @staticmethod
    def running(pid):
        try:
            return Path('/proc/%d/stat' % pid).read_text().split(') ', 1)[1][0] != 'Z'
        except (FileNotFoundError, ProcessLookupError):
            return False

    def wait_for(self, predicate, timeout=6.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.02)
        self.fail('Timed out; controller log: ' + (Path(self.temp.name) / 'controller.log').read_text())

    def held(self):
        path = self.output / 'hold.json'
        if path.exists():
            record = pilot.read_json(path)
            if record['state'] == 'held':
                return record
        return None

    def test_enabled_success_preserves_warm_reuse(self):
        rc, summary = self.run_pilot()
        self.assertEqual(rc, 0)
        self.assertEqual(len(summary['trials']), 3)
        self.assertEqual(len({item['pid'] for trial in summary['trials'] for item in trial['results']}), 2)
        self.assertFalse((self.output / 'hold.json').exists())

    def test_pair_order_modes_and_both_failures(self):
        cases = [('fzp', 'cold', 1), ('atm', 'cold', 3), ('both', 'warm', 2), ('atm', 'warm', 3)]
        for camera, mode, trial in cases:
            with self.subTest(camera=camera, mode=mode, trial=trial):
                self.output = Path(self.temp.name) / ('run-' + camera + mode + str(trial))
                rc, summary = self.run_pilot(mode=mode, test_fault='gcp_timeout', test_fault_camera=camera,
                    test_fault_trial=trial, hold_seconds=.3)
                self.assertEqual(rc, 1)
                self.assertTrue(summary['target_timeout'])
                self.assertEqual(len(summary['trials']), trial)
                self.assertEqual(len(summary['inspection']['cameras']), 2)
                self.assertTrue(summary['inspection']['cleanup_verified'])
                self.assertEqual(summary['inspection']['release_reason'], 'timeout')

    def test_early_success_kept_past_ordinary_timeout_and_original_immutable(self):
        proc = self.launch(timeout=.8, hold_seconds=3., test_fault='gcp_timeout', test_fault_camera='atm',
                           test_allocation_delay={'atm': .15})
        record = self.wait_for(self.held)
        original = self.output / 'process-group-0001/atm/uart-failure.json'
        digest = pilot.sha(original)
        time.sleep(1.)
        for camera in ('fzp', 'atm'):
            directory = self.output / 'process-group-0001' / camera
            self.assertNotIn('shutdown', pilot.read_json(directory / 'fake_calls.json'))
            self.assertFalse((directory / 'exit.json').exists())
            self.assertTrue(self.running(pilot.read_json(directory / 'ready.json')['pid']))
        self.assertEqual(pilot.sha(original), digest)
        self.assertFalse((self.output / 'process-group-0001/alloc-0002.json').exists())
        inspection.release(self.output, record['campaign'], pilot)
        self.assertEqual(proc.wait(timeout=4.), 1)
        self.assertEqual(pilot.sha(original), digest)
        self.assertEqual(pilot.read_json(self.output / 'hold.json')['release_reason'], 'operator')

    def test_rehearsal_stops_without_fabricating_failure(self):
        rc, summary = self.run_pilot(hold_on_error=False, inspect_after_trial=2, hold_seconds=.2)
        self.assertEqual(rc, 0)
        self.assertEqual(len(summary['trials']), 2)
        self.assertEqual(summary['inspection']['reason'], 'rehearsal')
        self.assertFalse(list(self.output.glob('**/failure.json')))

    def test_rehearsal_does_not_hold_real_target(self):
        rc, summary = self.run_pilot(hold_on_error=False, inspect_after_trial=1,
                                    test_fault='gcp_timeout', test_fault_camera='fzp')
        self.assertEqual(rc, 1)
        self.assertNotIn('inspection', summary)
        self.assertTrue(pilot.read_json(self.output / 'process-group-0001/fzp/failure.json')['target_timeout'])

    def test_non_target_failures_and_missing_peer_never_ready_hold(self):
        for fault in ('init_failure', 'vt_timeout', 'process_exit', 'alloc_hang'):
            with self.subTest(fault=fault):
                self.output = Path(self.temp.name) / fault
                rc, summary = self.run_pilot(test_fault=fault, test_fault_camera='fzp', timeout=.8)
                self.assertEqual(rc, 1)
                self.assertNotIn('inspection', summary)
                self.assertTrue(all(code is not None for code in summary['abort_cleanup']['exit_codes']))

    def test_snapshot_and_ack_failures_abort_group(self):
        for extra in ({'test_snapshot_failure': True}, {'test_recorder_failure': 'fzp'}, {'test_ack_hang': 'fzp'}):
            with self.subTest(extra=extra):
                self.output = Path(self.temp.name) / next(iter(extra))
                rc, summary = self.run_pilot(test_fault='gcp_timeout', test_fault_camera='atm', timeout=.8, **extra)
                self.assertEqual(rc, 1)
                self.assertNotIn('inspection', summary)
                if (self.output / 'hold.json').exists():
                    self.assertNotEqual(pilot.read_json(self.output / 'hold.json')['state'], 'held')

    def test_target_deadline_includes_waiting_for_unfinished_peer(self):
        started = time.monotonic()
        rc, summary = self.run_pilot(test_fault='gcp_timeout', test_fault_camera='fzp',
            test_allocation_delay={'atm': 4.}, timeout=6., hold_seconds=.2)
        self.assertEqual(rc, 1)
        self.assertLess(time.monotonic() - started, 2.)
        self.assertTrue(summary['target_timeout'])
        self.assertEqual(pilot.read_json(self.output / 'hold.json')['state'], 'inspection_unavailable')

    def test_stale_and_malformed_release_do_not_end_hold(self):
        proc = self.launch(hold_seconds=4., test_fault='gcp_timeout', test_fault_camera='fzp')
        record = self.wait_for(self.held)
        with self.assertRaises(ValueError):
            inspection.release(self.output, 'wrong-token', pilot)
        for payload in ('not json', json.dumps(dict(campaign='stale', action='release', trial=1)),
                        json.dumps(dict(campaign=record['campaign'], directory=str(self.output),
                                        action='release', trial=2))):
            (self.output / 'release.json').write_text(payload)
            time.sleep(.15)
            self.assertIsNone(proc.poll())
            self.assertEqual(pilot.read_json(self.output / 'hold.json')['state'], 'held')
        inspection.release(self.output, record['campaign'], pilot)
        self.assertEqual(proc.wait(timeout=4.), 1)

    def test_late_snapshot_disk_error_ends_acknowledged_hold(self):
        proc = self.launch(test_fault='gcp_timeout', test_fault_camera='atm', test_late_disk_failure='fzp')
        self.wait_for(self.held)
        original = self.output / 'process-group-0001/atm/uart-failure.json'
        digest = pilot.sha(original)
        self.assertEqual(proc.wait(timeout=4.), 1)
        self.assertEqual(pilot.sha(original), digest)
        self.assertEqual(pilot.read_json(self.output / 'hold.json')['state'], 'inspection_unavailable')

    def test_parent_signals_release_and_cleanup(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                self.output = Path(self.temp.name) / str(sig)
                proc = self.launch(test_fault='gcp_timeout', test_fault_camera='atm')
                self.wait_for(self.held)
                proc.send_signal(sig)
                self.assertEqual(proc.wait(timeout=4.), 1)
                summary = pilot.read_json(self.output / 'summary.json')
                self.assertFalse(summary['ok'])
                self.assertTrue(all(code is not None for code in summary['abort_cleanup']['exit_codes']))

    def test_parent_sigkill_worker_cleanup_including_native_hang(self):
        for hang in (False, True, 'gil'):
            with self.subTest(hang=hang):
                self.output = Path(self.temp.name) / ('orphan-' + str(hang))
                proc = self.launch(test_fault='gcp_timeout', test_fault_camera='atm', test_cleanup_hang=hang)
                self.wait_for(self.held)
                ready = [pilot.read_json(path) for path in self.output.glob('**/ready.json')]
                proc.kill()
                proc.wait(timeout=2.)
                self.wait_for(lambda: all(not self.running(item['pid']) for item in ready), timeout=3.)
                for item in ready:
                    directory = self.output / 'process-group-0001' / item['camera']
                    self.assertEqual(pilot.read_json(directory / 'hold.json')['release_reason'], 'parent_lost')
                self.wait_for(lambda: all(not self.running(item['watchdog_pid']) for item in ready), timeout=3.)

    def test_enabled_cleanup_errors_and_hangs_remain_failures(self):
        for fault in ('shutdown_failure', 'stop_hang'):
            with self.subTest(fault=fault):
                self.output = Path(self.temp.name) / fault
                rc, summary = self.run_pilot(trials=1, test_fault=fault, test_fault_camera='atm')
                self.assertEqual(rc, 1)
                self.assertFalse(summary['ok'])

    def test_child_death_in_hold_aborts_peer(self):
        proc = self.launch(test_fault='gcp_timeout', test_fault_camera='fzp')
        record = self.wait_for(self.held)
        os.kill(record['cameras'][0]['pid'], signal.SIGKILL)
        self.assertEqual(proc.wait(timeout=4.), 1)
        self.assertEqual(pilot.read_json(self.output / 'hold.json')['state'], 'inspection_unavailable')

    def test_cli_controls_are_offline_and_validated(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(pilot.main(['--hold-on-error', '--cameras', 'both']), 0)
        result = json.loads(output.getvalue())
        self.assertFalse(result['hardware_opened'])
        self.assertEqual(result['plan']['hold_seconds'], 900.)
        invalid = [['--hold-seconds', '2'], ['--hold-on-error', '--hold-seconds', 'nan'],
                   ['--hold-on-error', '--hold-seconds', '0'], ['--hold-on-error', '--hold-seconds', '3601'],
                   ['--inspect-after-trial', '2'], ['--inspect-after-trial', '0'],
                   ['--inspect-after-trial', '1', '--hold-on-error']]
        for arguments in invalid:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pilot.main(arguments)

    def test_channel_rejects_stale_foreign_and_broken_connections(self):
        first, second = socket.socketpair()
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        sender, receiver = Channel(first, 'campaign'), Channel(second, 'campaign')
        sender.send('hold', 2)
        with self.assertRaises(ValueError):
            receiver.wait(1, {'hold'}, time.monotonic() + 1.)
        sender.campaign = 'foreign'
        sender.send('hold', 1)
        with self.assertRaises(ValueError):
            receiver.receive(.2)
        first.close()
        with self.assertRaises(EOFError):
            receiver.receive(.2)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-worker':
        sys.exit(fake_worker(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-controller':
        spec = pilot.read_json(sys.argv[2])
        sys.exit(pilot.run_controller(spec['config'], spec['output'],
            [sys.executable, str(Path(__file__).resolve()), '--fake-worker']))
    unittest.main()
