"""Offline control-flow check only. Never import Rogue or open hardware/sockets.

Compile selected, unmodified AST definitions from the real source, preserving
line numbers. Hardware, gateway startup, transport and barriers are explicit
fakes. This cannot demonstrate occurrence of the real intermittent fault.
"""
import ast
import contextlib
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import socket
import time
import traceback
import types
import weakref

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'psdaq/psdaq/configdb/piranha4_config.py'
RESULTS = Path(__file__).with_name('offline_results.json')


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in names]
    assert {n.name for n in tree.body} == set(names)
    exec(compile(tree, str(path), 'exec'), namespace)


class Frame:
    def __init__(self, text):
        self.payload = b''.join(bytes([c, 0, 0, 0]) for c in text.encode('latin1'))

    def getPayload(self):
        return len(self.payload)

    def read(self, dest, offset):
        dest[:] = self.payload


class ReceiverBase:
    def __init__(self, path):
        self._path = path


class RegisterTree:
    def __init__(self, path, calls):
        self.path, self.calls = path, calls

    def __getattr__(self, name):
        return RegisterTree(self.path + '.' + name, self.calls)

    def __getitem__(self, index):
        return RegisterTree(self.path + '[%d]' % index, self.calls)

    def __call__(self, *args):
        self.calls.append((self.path, args))
        if self.path.endswith('TxClkFreq.get'):
            return 185.7
        if self.path.endswith('RxId.get'):
            return 0xff061508
        return 0


GCP_REPLY = 'Model P4_CM_02K10D_00_R\r\nSerial # 12345\r\nBiST Good\r\nCPA 0\r\nUSER>'


def run_case(mode, role):
    calls, frames = [], []

    class UART:
        def __init__(self):
            self._rx = ReceiverBase('fake-uart')

        def receive(self, text):
            frame = Frame(text)
            frames.append({'phase': calls[-1][0], 'hex': frame.payload.hex()})
            self._rx._acceptFrame(frame)

        def GCP(self):
            calls.append(('allocation_GCP', ()))
            if mode == 'success':
                self.receive(GCP_REPLY)
            elif mode == 'prompt_only':
                self.receive('USER>')
            elif mode == 'bad_prompt':
                self.receive(GCP_REPLY[:-5] + '\x00USER>')
            else:
                assert mode == 'no_reply'

        def VT(self):
            calls.append(('VT', ()))
            self.receive('Temperature 25\r\nUSER>')

        def VV(self):
            calls.append(('VV', ()))
            self.receive('Voltage 12\r\nUSER>')

    class Root:
        def __init__(self, **kwargs):
            calls.append(('construct_fake_root', kwargs))
            self.uart = UART()
            self.lane = next(iter(kwargs['laneConfig']))
            ch = types.SimpleNamespace(UartPiranha4=self.uart)
            top = types.SimpleNamespace()
            setattr(top, 'Ch[0]', ch)
            setattr(self, 'ClinkFeb[%d]' % self.lane, types.SimpleNamespace(ClinkTop=top))
            self.dmaStreams = [[None] * 4 for _ in range(4)]
            self.ClinkPcie = RegisterTree('ClinkPcie', calls)

        def start(self):
            # Explicit simulation of startup response, not real gateway code.
            calls.append(('fake_startup_reply', ()))
            self.uart.receive(GCP_REPLY)

        def StopRun(self):
            calls.append(('StopRun', ()))

        def stop(self):
            calls.append(('stop_fake_root', ()))

    class Barrier:
        def init(self, supervisor, nworker):
            self.supervisor, self.nworker = supervisor, nworker
            calls.append(('fake_barrier_init', [supervisor, nworker]))

        def wait(self):
            calls.append(('fake_barrier_wait', ()))

        def shutdown(self):
            calls.append(('fake_barrier_shutdown', ()))

    namespace = dict(clink=types.SimpleNamespace(ClinkSerialRx=ReceiverBase),
                     cameralink_gateway=types.SimpleNamespace(ClinkDevRoot=Root),
                     pr=types.SimpleNamespace(streamConnect=lambda *a: calls.append(('fake_streamConnect', ()))),
                     weakref=weakref, time=time, json=json, logging=logging,
                     os=os, socket=socket, args={}, xpmpv_global=None,
                     barrier_global=Barrier(), timTxId=lambda name: 0xfa000001)
    definitions(REPO / 'psdaq/psdaq/configdb/barrier.py', {'supervisor_info'}, namespace)
    definitions(SOURCE, {'MyUartPiranha4Rx', 'piranha4_init', 'piranha4_init_feb',
                         'piranha4_connectionInfo', 'piranha4_connectionShutdown',
                         'setup_timing', 'dumpTiming'}, namespace)
    # This literal is passed only to our fake Root, never opened.
    device = '/dev/datadev_0'
    me = {'active': True, 'proc_info': {'host': socket.gethostname(),
                                      'pid': os.getpid(), 'device': device}}
    other = {'active': True, 'proc_info': {'host': socket.gethostname(),
                                         'pid': os.getpid() + 1, 'device': device}}
    peers = {'self': me} if role == 'single' else {'other': other, 'self': me}
    result = {'mode': mode, 'role': role, 'hardware': 'all faked', 'barrier': 'faked; no sockets'}
    root = None
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            root = namespace['piranha4_init']('fzp_piranha', device, 1, None, '186M', 0)
            namespace['piranha4_init_feb'](None, None)
            result['connection_info'] = namespace['piranha4_connectionInfo'](
                root, json.dumps({'body': {'drp': peers}}))
        assert mode == 'success', 'Expected timeout was absent'
        assert result['connection_info']['model'] == '02'
        result['outcome'] = 'returned_success'
    except Exception as exc:
        if isinstance(exc, AssertionError):
            raise
        tb = traceback.extract_tb(exc.__traceback__)
        assert mode != 'success', traceback.format_exc()
        assert str(exc).startswith('*** await: prompt not seen:'), str(exc)
        assert any(f.name == 'piranha4_connectionInfo' and f.lineno == 286 for f in tb), tb
        result.update(outcome='expected_allocation_GCP_timeout', exception=str(exc),
                      traceback=[{'file': f.filename, 'line': f.lineno, 'function': f.name} for f in tb])
    finally:
        namespace['piranha4_connectionShutdown']()
        if root is not None:
            root.stop()
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    result['calls'] = list(calls)
    result['received_frames'] = frames
    result['final_cur'] = ''.join(root.uart._rx._cur)
    result['final_resp'] = root.uart._rx._resp
    result['supervisor'] = namespace['barrier_global'].supervisor
    result['nworker'] = namespace['barrier_global'].nworker
    assert result['supervisor'] == (role == 'single')
    assert result['nworker'] == (0 if role == 'single' else 1)
    assert [c[0] for c in calls].count('allocation_GCP') == 1
    return result


if __name__ == '__main__':
    # Silence repetitive warning output only; original parser/timeout unchanged.
    logging.disable(logging.CRITICAL)
    results = [run_case(mode, role) for mode, role in [
        ('success', 'single'), ('no_reply', 'single'),
        ('prompt_only', 'single'), ('bad_prompt', 'single'),
        ('success', 'worker'), ('no_reply', 'worker')]]
    output = {'source': str(SOURCE), 'sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'scope': 'Offline fault injection; not reproduction of hardware cause',
              'timeout': 'Real source default 5 seconds; real wall clock', 'results': results}
    RESULTS.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps([{'mode': r['mode'], 'role': r['role'], 'outcome': r['outcome'],
                       'seconds': r['elapsed_seconds']} for r in results], indent=2))
