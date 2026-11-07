"""Offline tests: subprocess lifecycle, signal cleanup and capture. No hardware.

The fake loader is used only by this test executable, never by pilot.py CLI.
"""
import ast
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
import types
import unittest
from unittest import mock

import pilot


def fake_loader(spec):
    calls = []
    fault = spec.get('test_fault')
    fault_camera = spec.get('test_fault_camera', spec['camera'])
    bad = spec['camera'] == fault_camera
    pc = types.SimpleNamespace()

    class Tx:
        def sendString(self, text):
            calls.append(text)

        def sendEscape(self):
            calls.append('escape')

    class Rx:
        def __init__(self, *args, **kwargs):
            self._cur, self._resp = list('USER>'), ['CPA 0']

        def _acceptFrame(self, frame):
            data = bytearray(frame.getPayload())
            frame.read(data, 0)

        def _clear(self):
            self._cur, self._resp = [], []

        def _await(self):
            if bad and fault == 'gcp_timeout' and calls[-1] == 'gcp':
                raise Exception('*** await: prompt not seen: len 0, resp: []')

    class Root:
        def stop(self):
            calls.append('stop')
            if bad and (fault == 'stop_hang' or spec.get('test_cleanup_hang')):
                while True:
                    time.sleep(1)

    def init(*args, **kwargs):
        calls.append('init')
        if bad and fault == 'init_failure':
            raise RuntimeError('fake init failure')
        root = Root()
        root.rx = pc.MyUartPiranha4Rx('fake')
        class Frame:
            def getPayload(self):
                return 4

            def read(self, target, offset):
                target[:] = b'X\x00\x00\x00'
        root.rx._acceptFrame(Frame())
        root.tx = Tx()
        return root

    def connection(root, alloc):
        peers = list(json.loads(alloc)['body']['drp'].values())
        assert any(x['proc_info']['pid'] == os.getpid() for x in peers)
        assert all(x['proc_info']['host'] == socket.gethostname() for x in peers)
        calls.append('alloc')
        root.rx._await()
        root.rx._clear()
        root.tx.sendString('gcp')
        if bad and fault == 'alloc_hang':
            while True:
                time.sleep(1)
        root.rx._await()
        root.tx.sendString('vt')
        root.rx._await()
        root.tx.sendString('vv')
        root.rx._await()
        return {'model': '02', 'serno': 'fake', 'paddr': 0xff061508}

    def shutdown():
        calls.append('shutdown')
        pilot.write_json(Path(spec['directory']) / 'fake_calls.json', calls)
        if bad and fault == 'shutdown_failure':
            raise RuntimeError('fake shutdown failure')

    pc.MyUartPiranha4Rx = Rx
    pc.clink = types.SimpleNamespace(ClinkSerialTx=Tx)
    pc.piranha4_init = init
    pc.piranha4_init_feb = lambda *args: None
    pc.piranha4_connectionInfo = connection
    pc.piranha4_connectionShutdown = shutdown
    return pc, {'backend': 'fake; no hardware'}


def fake_worker(spec_path):
    spec = pilot.read_json(spec_path)
    return pilot.child_main(spec, loader=lambda: fake_loader(spec))


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'run'
        self.config = dict(mode='cold', trials=2, cameras=['fzp', 'atm'], execute=True,
                           daq_stopped=True, timeout=3., cleanup_timeout=.5,
                           interval=0., settle_seconds=0.)
        self.worker = [sys.executable, str(Path(__file__).resolve()), '--fake-worker']

    def run_pilot(self, **overrides):
        config = dict(self.config, **overrides)
        rc = pilot.run_controller(config, self.output, self.worker)
        return rc, pilot.read_json(self.output / 'summary.json')

    def test_cold_pair_has_new_pids_and_identical_alloc(self):
        rc, result = self.run_pilot()
        self.assertEqual(rc, 0)
        pids = [[x['pid'] for x in trial['results']] for trial in result['trials']]
        self.assertEqual(len(set(pids[0] + pids[1])), 4)
        for group in self.output.glob('process-group-*'):
            alloc = pilot.read_json(next(group.glob('alloc-*.json')))
            self.assertEqual(list(alloc['body']['drp']), ['fzp', 'atm'])
            for cam in self.config['cameras']:
                calls = pilot.read_json(group / cam / 'fake_calls.json')
                self.assertEqual(calls.count('init'), 1)
                self.assertEqual(calls.count('alloc'), 1)

    def test_warm_pair_reuses_pids_and_roots(self):
        rc, result = self.run_pilot(mode='warm', trials=3)
        self.assertEqual(rc, 0)
        first = [x['pid'] for x in result['trials'][0]['results']]
        self.assertTrue(all([x['pid'] for x in t['results']] == first for t in result['trials']))
        for cam in self.config['cameras']:
            calls = pilot.read_json(self.output / 'process-group-0001' / cam / 'fake_calls.json')
            self.assertEqual(calls.count('init'), 1)
            self.assertEqual(calls.count('alloc'), 3)
            self.assertGreaterEqual(calls.count('shutdown'), 3)

    def test_failure_stops_pair_before_next_trial(self):
        rc, result = self.run_pilot(mode='warm', trials=3, test_fault='gcp_timeout', test_fault_camera='atm')
        self.assertEqual(rc, 1)
        self.assertLessEqual(len(result['trials']), 1)
        group = self.output / 'process-group-0001'
        self.assertFalse((group / 'alloc-0002.json').exists())
        failure = pilot.read_json(group / 'atm/failure.json')
        self.assertTrue(failure['target_timeout'])
        self.assertEqual(failure['phase'], 'allocation_gcp_wait')
        self.assertTrue((group / 'atm/uart-final.json').exists())
        self.assertTrue(all(code is not None for code in result['abort_cleanup']['exit_codes']))

    def test_init_failure_creates_no_alloc(self):
        rc, result = self.run_pilot(test_fault='init_failure', test_fault_camera='fzp')
        self.assertEqual(rc, 1)
        self.assertFalse(list(self.output.glob('**/alloc-*.json')))

    def test_watchdog_terminates_and_reaps(self):
        rc, result = self.run_pilot(trials=1, timeout=.5, test_fault='alloc_hang', test_fault_camera='atm')
        self.assertEqual(rc, 1)
        self.assertIn('Watchdog', result['error'])
        self.assertTrue(all(code is not None for code in result['abort_cleanup']['exit_codes']))

    def test_stuck_cleanup_forces_kill_and_fails(self):
        rc, result = self.run_pilot(trials=1, test_fault='stop_hang', test_fault_camera='fzp')
        self.assertEqual(rc, 1)
        self.assertEqual(len(result['groups'][0]['forced_kill_pids']), 1)

    def test_shutdown_failure_is_not_success(self):
        rc, result = self.run_pilot(trials=1, test_fault='shutdown_failure', test_fault_camera='atm')
        self.assertEqual(rc, 1)
        self.assertFalse(result['ok'])

    def test_failure_capture_survives_hanging_cleanup(self):
        rc, result = self.run_pilot(trials=1, test_fault='gcp_timeout',
                                   test_fault_camera='atm', test_cleanup_hang=True)
        self.assertEqual(rc, 1)
        directory = self.output / 'process-group-0001/atm'
        snapshot = pilot.read_json(directory / 'uart-failure.json')
        self.assertTrue(any(e['kind'] == 'rx' and e['raw_hex'] == '58000000' for e in snapshot['events']))
        self.assertTrue(pilot.read_json(directory / 'failure.json')['target_timeout'])
        self.assertTrue(result['abort_cleanup']['forced_kill_pids'])

    def test_unkillable_peers_get_all_signals_and_bounded_summary(self):
        processes = []
        for pid in (90001, 90002):
            proc = mock.Mock(pid=pid, returncode=None)
            proc.poll.return_value = None
            proc.wait.side_effect = subprocess.TimeoutExpired('fake', .01)
            processes.append(proc)
        ready = [{'pid': p.pid, 'host': socket.gethostname(), 'camera': cam}
                 for p, cam in zip(processes, ['fzp', 'atm'])]
        config = dict(self.config, trials=1, cleanup_timeout=.01)
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        started = time.monotonic()
        with mock.patch.object(pilot.subprocess, 'Popen', side_effect=processes):
            with mock.patch.object(pilot, 'wait_files', side_effect=[ready, RuntimeError('fake allocation failed')]):
                rc = pilot.run_controller(config, self.output, self.worker)
        self.assertLess(time.monotonic() - started, 1.)
        self.assertEqual(rc, 1)
        summary = pilot.read_json(self.output / 'summary.json')
        self.assertEqual(summary['abort_cleanup']['unreaped_pids'], [90001, 90002])
        for proc in processes:
            proc.terminate.assert_called()
            proc.kill.assert_called()
            self.assertTrue(all('timeout' in call.kwargs for call in proc.wait.call_args_list))
        self.assertTrue(all(signal.getsignal(sig) == handler for sig, handler in old_handlers.items()))

    def test_default_dry_run_never_loads_or_preflights(self):
        with mock.patch.object(pilot, 'load_real', side_effect=AssertionError('imported hardware')):
            with mock.patch.object(pilot, 'preflight', side_effect=AssertionError('preflight')):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(pilot.main([]), 0)
        self.assertTrue(json.loads(output.getvalue())['dry_run'])

    def test_execute_needs_attestation_and_output(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                pilot.main(['--execute'])

    def test_wrong_host_refuses_before_hardware_import(self):
        with mock.patch.object(pilot.socket, 'gethostname', return_value='sdfiana025'):
            with self.assertRaisesRegex(RuntimeError, 'requires host'):
                pilot.main(['--execute', '--daq-stopped', '--output', str(self.output)])

    def test_worker_without_flags_refuses(self):
        with self.assertRaisesRegex(RuntimeError, 'explicit execute'):
            pilot.child_main({'execute': False})

    def test_output_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_pilot()

    def test_capture_bounds_and_drain(self):
        cap = pilot.Capture(max_events=2)
        for i in range(3):
            cap.add('rx', b'\xff\x00\x1b\r', index=i)
        path = Path(self.temp.name) / 'capture.json'
        self.assertEqual(cap.dump(path, drain=True), 1)
        data = pilot.read_json(path)
        self.assertEqual(len(data['events']), 2)
        self.assertEqual(data['events'][0]['raw_hex'], 'ff001b0d')
        cap.add('rx', b'new')
        self.assertEqual(cap.dump(path), 0)
        self.assertEqual(len(pilot.read_json(path)['events']), 1)

    def test_observer_preserves_actual_receiver_parsing(self):
        source = Path(__file__).resolve().parents[2] / 'configdb/piranha4_config.py'
        tree = ast.parse(source.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MyUartPiranha4Rx']

        class Base:
            def __init__(self, path):
                self._path = path

        class Frame:
            payload = b''.join(bytes([b, 0, 0, 0]) for b in b'CPA 0\r\nUSER>')

            def getPayload(self):
                return len(self.payload)

            def read(self, data, offset):
                data[:] = self.payload

        class Tx:
            def sendString(self, text):
                pass

            def sendEscape(self):
                pass

        namespace = {'clink': types.SimpleNamespace(ClinkSerialRx=Base)}
        exec(compile(tree, str(source), 'exec'), namespace)
        original = namespace['MyUartPiranha4Rx']('original')
        pc = types.SimpleNamespace(MyUartPiranha4Rx=namespace['MyUartPiranha4Rx'],
                                   clink=types.SimpleNamespace(ClinkSerialTx=Tx))
        cap = pilot.Capture()
        pilot.instrument(pc, cap)
        observed = pc.MyUartPiranha4Rx('observed')
        original._acceptFrame(Frame())
        observed._acceptFrame(Frame())
        self.assertEqual(observed._cur, original._cur)
        self.assertEqual(observed._resp, original._resp)
        self.assertTrue(observed._check())
        self.assertEqual(cap.events[0][1]['raw_hex'], Frame.payload.hex())

    def test_parent_interrupt_reaps_children(self):
        config = dict(self.config, mode='warm', trials=3, settle_seconds=10.)
        spec = Path(self.temp.name) / 'controller.json'
        pilot.write_json(spec, {'config': config, 'output': str(self.output)})
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--fake-controller', str(spec)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 5
            while len(list(self.output.glob('**/ready.json'))) != 2:
                self.assertIsNone(proc.poll())
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.05)
            proc.send_signal(signal.SIGINT)
            self.assertEqual(proc.wait(timeout=5), 1)
            summary = pilot.read_json(self.output / 'summary.json')
            self.assertFalse(summary['ok'])
            self.assertTrue(all(code is not None for code in summary['abort_cleanup']['exit_codes']))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-worker':
        sys.exit(fake_worker(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == '--fake-controller':
        args = pilot.read_json(sys.argv[2])
        sys.exit(pilot.run_controller(args['config'], args['output'],
                                      [sys.executable, str(Path(__file__).resolve()), '--fake-worker']))
    unittest.main()
