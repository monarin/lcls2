"""Offline evidence tests; the optional pinned integration opens only loopback.

Run with the pinned ENV/bin/python test_evidence.py -v. The real-root proof runs
in a deadline-bounded subprocess so a native library failure cannot hang the
suite. No gateway, detector, DMA, UART or hardware root is constructed.
"""
import contextlib
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import evidence
import inspect_hold


PINNED_ENV = Path('/sdf/group/lcls/ds/ana/sw/conda2/inst/envs/daq_20250402_r9')


class SoftwareNode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []
        self.cached = kwargs['value']

    def set(self, update, write=True):
        if write:
            raise AssertionError('Diagnostic publication must not call writeBlocks')
        self.cached = update
        self.updates.append(update)


class SoftwareRoot:
    def __init__(self, **kwargs):
        self.arguments = kwargs
        self.started = False
        self.start_count = 0
        self.zmqServer = types.SimpleNamespace(port=lambda: 43120, address='127.0.0.1:43120')

    def add(self, node):
        if self.started:
            raise AssertionError('Node added after root start')
        node.path = 'SoftwareOnly.' + node.name
        setattr(self, node.name, node)

    def start(self):
        self.start_count += 1
        self.started = True


def fake_install(capture):
    pc = types.SimpleNamespace(cameralink_gateway=types.SimpleNamespace(ClinkDevRoot=SoftwareRoot))
    with mock.patch.dict(sys.modules, {'pyrogue': types.SimpleNamespace(LocalVariable=SoftwareNode)}):
        restore = evidence.install_diagnostics(pc, capture)
        root = pc.cameralink_gateway.ClinkDevRoot()
    restore()
    return root


class CaptureTests(unittest.TestCase):
    def test_drop_in_attributes_and_rx_counters(self):
        capture = evidence.Capture(max_events=2)
        self.assertEqual((capture.phase, capture.wait_index, capture.failure_phase,
                          capture.receiver, capture.initializing), ('init', 0, None, None, True))
        capture.receiver = types.SimpleNamespace(_cur=list('USER>'), _resp=['CPA 0'])
        capture.phase = 'allocation_gcp_wait'
        capture.failure_phase = capture.phase
        capture.add('tx_string', text='gcp')
        capture.add('rx', b'first')
        capture.add('rx', b'last')
        snapshot = capture.snapshot()
        self.assertEqual((capture.rx_count, capture.rx_bytes), (2, 9))
        self.assertEqual(snapshot['last_rx'], snapshot['events'][-1]['monotonic'])
        self.assertEqual(snapshot['events'][-1]['raw_hex'], b'last'.hex())
        self.assertEqual(snapshot['cur'], list('USER>'))
        self.assertEqual(snapshot['resp'], ['CPA 0'])
        self.assertEqual(snapshot['failure_phase'], 'allocation_gcp_wait')
        self.assertFalse(snapshot['receiver_snapshot']['atomic'])
        self.assertEqual(snapshot['lost_intervals'][0]['first_sequence'], 1)
        self.assertEqual(snapshot['lost_intervals'][0]['reason'], 'overwritten')

    def test_receiver_and_fields_are_bounded(self):
        class NoRepr:
            def __repr__(self):
                raise AssertionError('Unbounded repr')

        capture = evidence.Capture()
        capture.receiver = types.SimpleNamespace(_cur=['\U0001f600' * 100000] * 10000,
                                                 _resp=['x' * 100000] * 10000)
        capture.add('rx', b'raw', text='y' * 100000, other=NoRepr())
        snapshot = capture.snapshot()
        self.assertTrue(snapshot['events'][0]['fields_truncated'])
        for name in ('cur', 'resp'):
            self.assertTrue(snapshot['receiver_snapshot']['buffers'][name]['truncated'])
            self.assertLessEqual(sum(map(len, snapshot[name])), evidence.BUFFER_CHARACTERS)
        self.assertLess(len(json.dumps(snapshot)), evidence.MAX_DIAGNOSTIC_BYTES)

    def test_snapshot_is_detached(self):
        capture = evidence.Capture(max_events=1)
        capture.add('rx', b'first', nested={'data': [1, 2]})
        snapshot = capture.snapshot()
        snapshot['events'][0]['nested']['data'].append(3)
        self.assertEqual(capture.snapshot()['events'][0]['nested']['data'], [1, 2])
        capture.add('rx', b'second')
        snapshot = capture.snapshot()
        capture.add('rx', b'third')
        self.assertEqual(snapshot['lost_intervals'][0]['count'], 1)

    def test_oversized_payload_is_not_hex_encoded(self):
        class NoHex(bytes):
            def hex(self):
                raise AssertionError('Oversized payload encoded')

        capture = evidence.Capture(max_bytes=1024)
        capture.add('rx', NoHex(b'x' * 2048))
        self.assertEqual((capture.rx_count, capture.rx_bytes, capture.dropped), (1, 2048, 1))
        self.assertIsNotNone(capture.last_rx)
        self.assertFalse(capture.events)
        self.assertEqual(capture.snapshot()['lost_intervals'][0]['reason'], 'oversized')

    def test_overflow_has_exact_recent_intervals_and_omitted_count(self):
        capture = evidence.Capture(max_bytes=1024, max_events=1)
        for iteration in range(300):
            capture.add('rx', b'ok')
            capture.add('rx', b'x' * 2048)
        snapshot = capture.snapshot()
        self.assertLessEqual(len(snapshot['lost_intervals']), evidence.LOSS_INTERVALS)
        self.assertGreater(snapshot['loss_history_omitted_events'], 0)
        self.assertEqual(snapshot['dropped_events'], 599)
        self.assertEqual(snapshot['dropped_events'], snapshot['loss_history_omitted_events']
                         + sum(interval['count'] for interval in snapshot['lost_intervals']))
        self.assertLessEqual(capture.size, capture.max_bytes)

    def test_callback_performs_no_disk_io(self):
        capture = evidence.Capture()
        with mock.patch('builtins.open', side_effect=AssertionError('RX wrote a file')):
            with mock.patch.object(Path, 'open', side_effect=AssertionError('RX wrote a file')):
                capture.add('rx', b'data')
        self.assertEqual(capture.rx_count, 1)

    def test_dump_drains_only_after_success(self):
        capture = evidence.Capture(max_events=1)
        capture.add('rx', b'first')
        capture.add('rx', b'second')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'failure.json'
            with mock.patch.object(evidence.os, 'replace', side_effect=OSError('disk failed')):
                with self.assertRaises(OSError):
                    capture.dump(path, drain=True)
            self.assertEqual((len(capture.events), capture.dropped), (1, 1))
            self.assertFalse(path.with_name('failure.json.tmp').exists())
            self.assertEqual(capture.dump(path, drain=True), 1)
            self.assertEqual((len(capture.events), capture.size, capture.dropped), (0, 0, 0))
            self.assertEqual(json.loads(path.read_text())['dropped_events'], 1)
            self.assertEqual((capture.rx_count, capture.rx_bytes), (2, 11))

    def test_dump_preserves_late_rx_and_concurrent_loss(self):
        capture = evidence.Capture(max_events=1)
        capture.add('rx', b'original')
        capture.add('rx', b'original2')
        writer = evidence._write_json

        def late_write(path, snapshot):
            self.assertTrue(capture.lock.acquire(blocking=False))
            capture.lock.release()
            capture.add('rx', b'late1')
            capture.add('rx', b'late2')
            writer(path, snapshot)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'original.json'
            with mock.patch.object(evidence, '_write_json', side_effect=late_write):
                self.assertEqual(capture.dump(path, drain=True), 1)
            self.assertEqual(json.loads(path.read_text())['rx_count'], 2)
        snapshot = capture.snapshot()
        self.assertEqual(snapshot['events'][0]['raw_hex'], b'late2'.hex())
        self.assertEqual(snapshot['dropped_events'], 2)
        self.assertEqual(snapshot['lost_intervals'][0]['first_sequence'], 2)
        self.assertEqual(snapshot['lost_intervals'][0]['last_sequence'], 3)
        self.assertEqual(snapshot['loss_history_omitted_events'], 0)

    def test_concurrent_ring_and_receiver_sampling(self):
        capture = evidence.Capture(max_events=10, max_bytes=4096)
        capture.receiver = types.SimpleNamespace(_cur=[], _resp=[])

        def receive():
            for iteration in range(2000):
                capture.receiver._cur = list(str(iteration))
                capture.receiver._resp = [str(iteration)]
                capture.add('rx', b'data')

        producer = threading.Thread(target=receive)
        producer.start()
        for iteration in range(100):
            snapshot = capture.snapshot()
            self.assertLessEqual(snapshot['retained_bytes'], capture.max_bytes)
            self.assertLessEqual(len(snapshot['events']), 10)
        producer.join(timeout=5)
        self.assertFalse(producer.is_alive())
        self.assertEqual(capture.rx_count, 2000)

    def test_invalid_limits(self):
        for limits in ({'max_bytes': 0}, {'max_events': 0}, {'max_events': -1}, {'max_bytes': 0.5}):
            with self.assertRaises(ValueError):
                evidence.Capture(**limits)


class DiagnosticsTests(unittest.TestCase):
    def test_factory_adds_before_start_and_restores(self):
        capture = evidence.Capture()
        pc = types.SimpleNamespace(cameralink_gateway=types.SimpleNamespace(ClinkDevRoot=SoftwareRoot))
        with mock.patch.dict(sys.modules, {'pyrogue': types.SimpleNamespace(LocalVariable=SoftwareNode)}):
            restore = evidence.install_diagnostics(pc, capture)
            try:
                root = pc.cameralink_gateway.ClinkDevRoot(unchanged='hardware arguments')
                self.assertFalse(root.started)
                self.assertEqual(root.arguments, {'unchanged': 'hardware arguments'})
                node = getattr(root, evidence.DIAGNOSTIC_NAME)
                self.assertEqual((node.mode, node.pollInterval, node.bulkOpEn), ('RO', 0, False))
                self.assertNotIn('localGet', node.__dict__)
                root.start()
                self.assertEqual(root.start_count, 1)
            finally:
                restore()
            restore()
        self.assertIs(pc.cameralink_gateway.ClinkDevRoot, SoftwareRoot)

    def test_factory_failure_restoration(self):
        factory = mock.Mock(side_effect=RuntimeError('init failed'))
        pc = types.SimpleNamespace(cameralink_gateway=types.SimpleNamespace(ClinkDevRoot=factory))
        with mock.patch.dict(sys.modules, {'pyrogue': types.SimpleNamespace(LocalVariable=SoftwareNode)}):
            restore = evidence.install_diagnostics(pc, evidence.Capture())
            try:
                with self.assertRaises(RuntimeError):
                    pc.cameralink_gateway.ClinkDevRoot()
            finally:
                restore()
        self.assertIs(pc.cameralink_gateway.ClinkDevRoot, factory)

    def test_actual_server_api_and_attach_commands(self):
        root = fake_install(evidence.Capture())
        details = evidence.endpoint(root)
        self.assertEqual(details['ports'], [43120, 43121, 43122])
        self.assertEqual(details['host'], '127.0.0.1')
        self.assertEqual(details['diagnostic_path'], 'SoftwareOnly.' + evidence.DIAGNOSTIC_NAME)
        command = shlex.split(details['client_command'])
        self.assertEqual(command[command.index('--port') + 1], '43120')
        self.assertEqual(command[command.index('--path') + 1], details['diagnostic_path'])
        root.zmqServer.port = lambda: 0
        with self.assertRaises(ValueError):
            evidence.endpoint(root)

    def test_rotation_force_throttle_and_original_immutable(self):
        capture = evidence.Capture(max_events=3)
        root = fake_install(capture)
        state = {'campaign': 'campaign', 'camera': 'fzp', 'trial': 3, 'lane': 0,
                 'pid': os.getpid(), 'phase': 'allocation_gcp_wait', 'state': 'holding',
                 'deadline': 999.0, 'release_reason': None}
        capture.failure_phase = state['phase']
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / 'uart-failure.json'
            capture.add('rx', b'original')
            capture.dump(original, drain=True)
            original_bytes = original.read_bytes()
            recorder = evidence.InspectionRecorder(capture, root, directory, {'camera': 'fzp'})
            with mock.patch.object(evidence.time, 'monotonic', return_value=100.0):
                initial = recorder.snapshot(state, force=True)
                self.assertEqual(initial['sequence'], 1)
                self.assertTrue(recorder.snapshot(state)['skipped'])
                self.assertEqual(len(recorder.node.updates), 1)
            for iteration in range(2, 11):
                capture.add('rx', b'late')
                with mock.patch.object(evidence.time, 'monotonic', return_value=100.0 + iteration):
                    metadata = recorder.snapshot(state)
                    self.assertFalse(metadata['skipped'])
            with mock.patch.object(evidence.time, 'monotonic', return_value=110.0):
                state['state'] = 'released'
                final = recorder.snapshot(state, force=True)
            self.assertEqual(final['sequence'], 11)
            self.assertEqual(final['overwritten_snapshot']['sequence'], 7)
            self.assertEqual(final['overwritten_snapshot_interval'], {'first': 1, 'last': 7, 'count': 7})
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertEqual(len(list((Path(directory) / 'inspection').iterdir())), 4)
            cached = json.loads(recorder.node.cached)
            self.assertEqual(cached['state']['state'], 'released')
            self.assertEqual(cached['failure_phase'], 'allocation_gcp_wait')
            self.assertGreater(cached['dropped_events'], 0)
            self.assertNotIn('events', cached)
            late = json.loads(Path(final['path']).read_text())['capture']['events']
            self.assertTrue(all(event['raw_hex'] == b'late'.hex() for event in late))
            self.assertLess(len(recorder.node.cached), evidence.MAX_DIAGNOSTIC_BYTES)

    def test_fixed_retention_even_with_forced_snapshots(self):
        capture = evidence.Capture()
        root = fake_install(capture)
        with tempfile.TemporaryDirectory() as directory:
            recorder = evidence.InspectionRecorder(capture, root, directory, {})
            for iteration in range(12):
                recorder.snapshot({'state': 'holding'}, force=True)
            self.assertEqual(len(list(recorder.directory.iterdir())), 4)
            with self.assertRaises(FileExistsError):
                evidence.InspectionRecorder(capture, root, directory, {})

    def test_io_failure_propagates_and_does_not_publish(self):
        capture = evidence.Capture()
        root = fake_install(capture)
        with tempfile.TemporaryDirectory() as directory:
            recorder = evidence.InspectionRecorder(capture, root, directory, {})
            with mock.patch.object(evidence, '_write_json', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    recorder.snapshot({'state': 'holding'}, force=True)
            self.assertEqual(recorder._sequence, 0)
            self.assertFalse(recorder.node.updates)

    def test_diagnostic_serialization_worst_case_is_bounded(self):
        capture = evidence.Capture(max_events=1, max_bytes=1024)
        capture.receiver = types.SimpleNamespace(_cur=['\U0001f600' * 100000],
                                                 _resp=['\U0001f600' * 100000])
        for iteration in range(100):
            capture.add('rx', b'ok')
            capture.add('rx', b'bad' * 1000)
        root = fake_install(capture)
        large = {'value': '\U0001f600' * 100000}
        with tempfile.TemporaryDirectory() as directory:
            recorder = evidence.InspectionRecorder(capture, root, directory, large)
            recorder.snapshot(large, force=True)
        self.assertLessEqual(len(recorder.node.cached), evidence.MAX_DIAGNOSTIC_BYTES)


class ClientTests(unittest.TestCase):
    def test_rejects_arbitrary_nodes_before_launch(self):
        with mock.patch.object(inspect_hold.subprocess, 'Popen', side_effect=AssertionError('launched')):
            for path in ('Root.ReadAll', 'Root.Device.Register', '__ROOT__',
                         'Root.' + evidence.DIAGNOSTIC_NAME + '.get'):
                with self.assertRaises(ValueError):
                    inspect_hold.read_snapshot('127.0.0.1', 12345, path)
            for timeout in (0, -1, float('nan'), float('inf'), 61):
                with self.assertRaises(ValueError):
                    inspect_hold.read_snapshot('127.0.0.1', 12345,
                                               'Root.' + evidence.DIAGNOSTIC_NAME, timeout)

    def test_client_calls_only_value_and_closes_pinned_resources(self):
        client = types.SimpleNamespace(_req=mock.Mock(), _ctx=mock.Mock(), _stop=mock.Mock(),
                                       value=mock.Mock(return_value='{"state": "holding"}'))
        interfaces = types.ModuleType('pyrogue.interfaces')
        interfaces.SimpleClient = mock.Mock(return_value=client)
        rogue = types.ModuleType('pyrogue')
        rogue.interfaces = interfaces
        zmq = types.SimpleNamespace(LINGER=1, SNDTIMEO=2, RCVTIMEO=3, MAXMSGSIZE=4)
        with mock.patch.dict(sys.modules, {'pyrogue': rogue, 'pyrogue.interfaces': interfaces, 'zmq': zmq}):
            path = 'Root.' + evidence.DIAGNOSTIC_NAME
            result = inspect_hold._cached_value('127.0.0.1', 12345, path, 1.0)
        self.assertEqual(result, {'state': 'holding'})
        interfaces.SimpleClient.assert_called_once_with(addr='127.0.0.1', port=12345)
        client.value.assert_called_once_with(path)
        client._req.close.assert_called_once_with(linger=0)
        client._ctx.term.assert_called_once_with()

    def test_import_or_rpc_stall_has_bounded_deadline_and_reaps(self):
        popen = subprocess.Popen
        children = []

        def stalled_worker(command, **kwargs):
            process = popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
            children.append(process)
            return process

        started = time.monotonic()
        with mock.patch.object(inspect_hold.subprocess, 'Popen', side_effect=stalled_worker):
            with self.assertRaises(subprocess.TimeoutExpired):
                inspect_hold.read_snapshot('127.0.0.1', 12345,
                                           'Root.' + evidence.DIAGNOSTIC_NAME, timeout=0.15)
        self.assertLess(time.monotonic() - started, 2)
        self.assertIsNotNone(children[0].poll())


def software_root_proof(directory):
    """Exercise the actual pinned server/client without a hardware import/open."""
    faulthandler.dump_traceback_later(15)
    import pyrogue as rogue
    import pyrogue.interfaces

    def free_loopback_triplet():
        for attempt in range(20):
            with contextlib.ExitStack() as sockets:
                first = sockets.enter_context(socket.socket())
                first.bind(('127.0.0.1', 0))
                base_port = first.getsockname()[1]
                if base_port > 65533:
                    continue
                try:
                    for port in (base_port + 1, base_port + 2):
                        sibling = sockets.enter_context(socket.socket())
                        sibling.bind(('127.0.0.1', port))
                except OSError:
                    continue
                return base_port
        raise RuntimeError('Cannot reserve three software-only loopback ports')

    def factory(**kwargs):
        root = rogue.Root(name='SoftwareOnlyPiranha', pollEn=False, initRead=False,
                          initWrite=False)
        root._doHeartbeat = False
        root.zmqServer = rogue.interfaces.ZmqServer(root=root, addr='127.0.0.1',
                                                   port=free_loopback_triplet())
        root.addInterface(root.zmqServer)
        return root

    capture = evidence.Capture(max_events=3)
    capture.receiver = types.SimpleNamespace(_cur=list('USER>'), _resp=['CPA 0'])
    pc = types.SimpleNamespace(cameralink_gateway=types.SimpleNamespace(ClinkDevRoot=factory))
    restore = evidence.install_diagnostics(pc, capture)
    try:
        root = pc.cameralink_gateway.ClinkDevRoot()
        assert not root.running
        assert getattr(root, evidence.DIAGNOSTIC_NAME).mode == 'RO'
        try:
            root.start()
        except BaseException:
            root.stop()
            raise
    finally:
        restore()
    original = Path(directory) / 'uart-failure.json'
    try:
        details = evidence.endpoint(root)
        assert details['base_port'] == root.zmqServer.port()
        assert details['host'] == '127.0.0.1'
        capture.failure_phase = 'allocation_gcp_wait'
        capture.add('rx', b'original')
        capture.dump(original, drain=True)
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        state = {'campaign': 'software-proof', 'trial': 1, 'camera': 'fzp', 'lane': 0,
                 'pid': os.getpid(), 'phase': 'allocation_gcp_wait', 'state': 'holding',
                 'deadline': time.monotonic() + 20, 'release_reason': None}
        with contextlib.ExitStack() as stack:
            for name in ('readBlocks', 'writeBlocks', 'verifyBlocks', 'checkBlocks'):
                stack.enter_context(mock.patch.object(root, name, side_effect=AssertionError(name)))
            recorder = evidence.InspectionRecorder(capture, root, directory, {'backend': 'software-only'})
            recorder.snapshot(state, force=True)
            cached = inspect_hold.read_snapshot(details['host'], details['base_port'],
                                                 details['diagnostic_path'], timeout=10.0)
            assert cached['state']['state'] == 'holding'
            assert cached['cur'] == list('USER>')
            assert cached['resp'] == ['CPA 0']
            assert cached['rx_count'] == 1
            for iteration in range(8):
                capture.add('rx', b'late')
                recorder.snapshot(state, force=True)
            state.update(state='released', release_reason='software-test')
            recorder.snapshot(state, force=True)
            cached = inspect_hold.read_snapshot(details['host'], details['base_port'],
                                                 details['diagnostic_path'], timeout=10.0)
            assert cached['state']['release_reason'] == 'software-test'
            assert cached['rx_count'] == 9
            assert cached['dropped_events'] == 5
            assert cached['failure_phase'] == 'allocation_gcp_wait'
            assert len(list(recorder.directory.iterdir())) == 4
            assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    finally:
        root.stop()
    closed = []
    for port in details['ports']:
        with socket.socket() as connection:
            connection.settimeout(0.5)
            closed.append(connection.connect_ex((details['host'], port)) != 0)
    assert all(closed), closed
    report = {'python': sys.version, 'executable': sys.executable, 'pyrogue': rogue.__file__,
              'endpoint': details, 'ports_closed_after_stop': closed,
              'original_sha256': digest, 'cached_value': cached,
              'scope': 'Software-only loopback root, no gateway or hardware, no GUI attach.'}
    (Path(directory) / 'software-proof.json').write_text(json.dumps(report, indent=2) + '\n')
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({'software_root_proof': 'passed', 'endpoint': details,
                      'ports_closed_after_stop': closed, 'original_sha256': digest}), flush=True)


class PinnedIntegrationTests(unittest.TestCase):
    def test_pinned_software_root_and_simple_client(self):
        python = PINNED_ENV / 'bin/python'
        if not python.exists():
            self.skipTest('Pinned ENV unavailable; run software-root proof at deployment site')
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        probe = subprocess.run([str(python), '-c', 'import pyrogue, pyrogue.interfaces'],
                               capture_output=True, text=True, timeout=15, env=environment)
        if probe.returncode:
            self.skipTest('Pinned PyRogue import unavailable: ' + probe.stderr[-1024:])
        with tempfile.TemporaryDirectory(prefix='piranha-evidence-test-') as directory:
            try:
                result = subprocess.run([str(python), '-u', str(Path(__file__).resolve()),
                                         '--software-root', directory], capture_output=True,
                                        text=True, timeout=40, env=environment)
            except subprocess.TimeoutExpired as error:
                self.fail('Software root timed out: %r %r' % (error.stdout, error.stderr))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(directory) / 'software-proof.json').read_text())
            self.assertTrue(all(report['ports_closed_after_stop']))
            self.assertEqual(report['cached_value']['rx_count'], 9)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--software-root':
        software_root_proof(sys.argv[2])
    else:
        unittest.main()
