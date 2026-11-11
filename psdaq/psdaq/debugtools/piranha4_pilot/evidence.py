"""Bounded, passive evidence for an already-owned Piranha root.

Install diagnostics around piranha4_init, restoring the factory in finally.
Only the owner's inspection loop samples receiver buffers and publishes JSON;
clients read that cached JSON, never the receiver or hardware. Receiver samples
are explicitly non-atomic and do not acquire the UART parser's locks.
"""
from collections import deque
import copy
from itertools import islice
import json
import math
import os
from pathlib import Path
import shlex
import sys
import threading
import time


DIAGNOSTIC_NAME = 'PiranhaHoldDiagnostics'
MAX_DIAGNOSTIC_BYTES = 256 * 1024
BUFFER_ITEMS = 256
BUFFER_CHARACTERS = 4096
LOSS_INTERVALS = 64
SNAPSHOT_SLOTS = 4
SNAPSHOT_INTERVAL = 1.0


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False)


def _bounded(value):
    """Copy only bounded built-in values, without calling arbitrary repr hooks."""
    budget = {'items': BUFFER_ITEMS, 'characters': BUFFER_CHARACTERS}
    truncated = False

    def visit(item, depth):
        nonlocal truncated
        if budget['items'] <= 0 or depth > 4:
            truncated = True
            return None
        budget['items'] -= 1
        if item is None or type(item) is bool:
            return item
        if type(item) is int and item.bit_length() <= 64:
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if type(item) is str:
            count = min(len(item), budget['characters'])
            budget['characters'] -= count
            truncated |= count < len(item)
            return item[:count]
        if type(item) in (list, tuple):
            result = []
            for child in item[:BUFFER_ITEMS]:
                if budget['items'] <= 0 or budget['characters'] <= 0:
                    break
                result.append(visit(child, depth + 1))
            truncated |= len(result) < len(item)
            return result
        if type(item) is dict:
            result = {}
            for key, child in islice(item.items(), BUFFER_ITEMS):
                if budget['items'] <= 0 or budget['characters'] <= 0:
                    break
                if type(key) is not str:
                    truncated = True
                    continue
                bounded_key = visit(key, depth + 1)
                result[bounded_key] = visit(child, depth + 1)
            truncated |= len(result) < len(item)
            return result
        truncated = True
        return None

    return visit(value, 0), truncated


def _receiver_snapshot(receiver):
    started = time.monotonic()
    result = {}
    limits = {}
    for name in ('cur', 'resp'):
        try:
            source = getattr(receiver, '_' + name) if receiver is not None else None
            before = len(source) if type(source) in (list, tuple, str) else None
            value, truncated = _bounded(source)
            after = len(source) if before is not None else None
            result[name] = value
            limits[name] = {'length_before': before, 'length_after': after,
                            'truncated': truncated}
        except (AttributeError, RuntimeError) as error:
            result[name] = None
            limits[name] = {'unavailable': type(error).__name__, 'truncated': True}
    result['receiver_snapshot'] = {
        'started_monotonic': started, 'finished_monotonic': time.monotonic(),
        'atomic': False, 'buffers': limits,
        'max_items_per_buffer': BUFFER_ITEMS, 'max_characters_per_buffer': BUFFER_CHARACTERS,
        'limitation': 'Unlocked bounded prefixes; RX may change buffers during sampling. '
                      'Lengths apply to sampled references, not necessarily current buffers.'}
    return result


def _write_json(path, value):
    """Replace one file atomically; propagate I/O errors to the owner loop."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w') as output:
            output.write(_json(value) + '\n')
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


class Capture:
    """Drop-in pilot Capture with bounded loss history and lifetime RX counters.

    Event sequences and RX/loss counters never reset. A successful draining dump
    acknowledges only the snapshotted prefix, preserving concurrent late RX.
    Failed writes do not drain. Ring copies use our lock; serialization, receiver
    sampling and disk I/O do not. Large frames are rejected before hex encoding.
    """
    def __init__(self, max_bytes=4 * 1024 * 1024, max_events=20000):
        if type(max_bytes) is not int or type(max_events) is not int:
            raise ValueError('Capture limits must be positive integers')
        if max_bytes <= 0 or max_events <= 0:
            raise ValueError('Capture limits must be positive integers')
        self.max_bytes, self.max_events = max_bytes, max_events
        self.events = deque()
        self.size = self.dropped = 0
        self.lock = threading.Lock()
        self.phase = 'init'
        self.wait_index = 0
        self.failure_phase = None
        self.receiver = None
        self.initializing = True
        self.rx_count = self.rx_bytes = 0
        self.last_rx = None
        self._sequence = 0
        self._loss_total = self._drained_losses = 0
        self._loss_counts = {'overwritten': 0, 'oversized': 0}
        self._losses = deque(maxlen=LOSS_INTERVALS)
        self._dump_lock = threading.Lock()

    def _lose(self, sequence, reason):
        self.dropped += 1
        self._loss_total += 1
        self._loss_counts[reason] += 1
        if (self._losses and self._losses[-1]['reason'] == reason
                and self._losses[-1]['last_sequence'] + 1 == sequence):
            interval = self._losses[-1]
            interval['last_sequence'] = sequence
            interval['loss_ordinal_last'] = self._loss_total
            interval['count'] += 1
        else:
            self._losses.append({'reason': reason, 'first_sequence': sequence,
                                 'last_sequence': sequence, 'count': 1,
                                 'loss_ordinal_first': self._loss_total,
                                 'loss_ordinal_last': self._loss_total})

    def add(self, kind, raw=b'', **fields):
        timestamp = time.monotonic()
        payload_bytes = memoryview(raw).nbytes
        event, truncated = _bounded(fields)
        event.update(kind=kind[:128], monotonic=timestamp, phase=self.phase[:256])
        event['fields_truncated'] = truncated
        too_large = payload_bytes > self.max_bytes // 2
        event['raw_hex'] = '' if too_large else raw.hex()
        cost = len(_json(event)) + 48
        with self.lock:
            self._sequence += 1
            event['sequence'] = self._sequence
            if kind == 'rx':
                self.rx_count += 1
                self.rx_bytes += payload_bytes
                self.last_rx = timestamp
            if too_large or cost > self.max_bytes:
                self._lose(self._sequence, 'oversized')
                return
            while self.events and (self.size + cost > self.max_bytes
                                   or len(self.events) >= self.max_events):
                old_cost, old_event = self.events.popleft()
                self.size -= old_cost
                self._lose(old_event['sequence'], 'overwritten')
            self.events.append((cost, event))
            self.size += cost

    def snapshot(self):
        """Return a detached ring snapshot plus bounded, non-atomic RX buffers."""
        with self.lock:
            events = [event for _, event in self.events]
            intervals = [dict(interval) for interval in self._losses]
            result = {
                'monotonic': time.monotonic(), 'through_sequence': self._sequence,
                'phase': self.phase[:256],
                'failure_phase': self.failure_phase[:256] if self.failure_phase else None,
                'rx_count': self.rx_count, 'rx_bytes': self.rx_bytes, 'last_rx': self.last_rx,
                'dropped_events': self.dropped, 'retained_bytes': self.size,
                'retained_events': len(events), 'max_bytes': self.max_bytes,
                'max_events': self.max_events, 'lost_intervals': intervals,
                'loss_history_omitted_events': self.dropped - sum(
                    interval['count'] for interval in intervals),
                'lifetime_loss_counters': dict(self._loss_counts, total=self._loss_total),
            }
        result['events'] = copy.deepcopy(events)
        result.update(_receiver_snapshot(self.receiver))
        return result

    def dump(self, path, drain=False):
        with self._dump_lock:
            snapshot = self.snapshot()
            _write_json(path, snapshot)
            if drain:
                with self.lock:
                    while self.events and self.events[0][1]['sequence'] <= snapshot['through_sequence']:
                        cost, _ = self.events.popleft()
                        self.size -= cost
                    self._drained_losses = snapshot['lifetime_loss_counters']['total']
                    self.dropped = self._loss_total - self._drained_losses
                    while self._losses and self._losses[0]['loss_ordinal_last'] <= self._drained_losses:
                        self._losses.popleft()
                    if self._losses and self._losses[0]['loss_ordinal_first'] <= self._drained_losses:
                        interval = self._losses[0]
                        acknowledged = self._drained_losses - interval['loss_ordinal_first'] + 1
                        interval['first_sequence'] += acknowledged
                        interval['count'] -= acknowledged
                        interval['loss_ordinal_first'] += acknowledged
            return snapshot['dropped_events']


def install_diagnostics(pc, capture):
    """Wrap only the root factory; return an idempotent restoration callable.

    Call after instrument() and before piranha4_init(), then restore in finally.
    The pinned init constructs ClinkDevRoot before start(), allowing this node to
    be added without changing any hardware initialization or startup commands.
    """
    import pyrogue as rogue

    original = pc.cameralink_gateway.ClinkDevRoot

    def factory(*args, **kwargs):
        root = original(*args, **kwargs)
        root.add(rogue.LocalVariable(
            name=DIAGNOSTIC_NAME, mode='RO', value=_json({
                'state': 'not_inspecting', 'phase': capture.phase[:256],
                'limitation': 'Owner has not published an inspection snapshot.'}),
            description='Cached passive Piranha hold evidence; no hardware getter.',
            pollInterval=0, bulkOpEn=False))
        return root

    def restore():
        if pc.cameralink_gateway.ClinkDevRoot is factory:
            pc.cameralink_gateway.ClinkDevRoot = original

    pc.cameralink_gateway.ClinkDevRoot = factory
    return restore


def endpoint(root):
    """Describe the existing pinned server, never create one or guess its port.

    PyRogue's address maps wildcard binding to loopback. Forward all three listed
    ports when tunneling. The generic GUI remains write-capable, not read-only.
    """
    server = root.zmqServer
    base_port = int(server.port())
    host, advertised_port = server.address.rsplit(':', 1)
    if not host or not 1 <= base_port <= 65533 or int(advertised_port) != base_port:
        raise ValueError('Existing PyRogue server has no valid bound endpoint')
    path = getattr(root, DIAGNOSTIC_NAME).path
    return {
        'host': host, 'base_port': base_port,
        'ports': [base_port, base_port + 1, base_port + 2],
        'diagnostic_path': path,
        'gui_command': shlex.join([sys.executable, '-m', 'pyrogue', 'gui',
                                   '--server=' + server.address]),
        'client_command': shlex.join([
            sys.executable, str(Path(__file__).resolve().with_name('inspect_hold.py')),
            '--host', host, '--port', str(base_port), '--path', path, '--timeout', '5']),
    }


class InspectionRecorder:
    """Four rotating late-RX files, created only when the owner enters inspection.

    snapshot accepts the parent's state dictionary and returns file/rotation
    metadata. Calls within one second return the previous metadata with skipped
    true. force bypasses that interval for entry/final records, not slot limits.
    The separate inspection directory must be new; original failure files are
    neither read nor replaced. All write/publication failures propagate.
    """
    def __init__(self, capture, root, directory, identity):
        self.capture = capture
        self.root = root
        self.directory = Path(directory) / 'inspection'
        self.identity, self.identity_truncated = _bounded(identity)
        self.node = getattr(root, DIAGNOSTIC_NAME)
        self._slots = [None] * SNAPSHOT_SLOTS
        self._sequence = 0
        self._last_time = None
        self._last_metadata = None
        self._lock = threading.Lock()
        self.directory.mkdir()

    def snapshot(self, state, force=False):
        with self._lock:
            now = time.monotonic()
            if not force and self._last_time is not None and now - self._last_time < SNAPSHOT_INTERVAL:
                return dict(copy.deepcopy(self._last_metadata), skipped=True)
            snapshot = self.capture.snapshot()
            bounded_state, state_truncated = _bounded(state)
            sequence = self._sequence + 1
            slot = (sequence - 1) % SNAPSHOT_SLOTS
            path = self.directory / ('rx-%d.json' % slot)
            previous = self._slots[slot]
            metadata = {
                'sequence': sequence, 'slot': slot, 'path': str(path), 'monotonic': now,
                'skipped': False, 'forced': bool(force), 'slots': SNAPSHOT_SLOTS,
                'through_sequence': snapshot['through_sequence'],
                'first_event_sequence': snapshot['events'][0]['sequence'] if snapshot['events'] else None,
                'last_event_sequence': snapshot['events'][-1]['sequence'] if snapshot['events'] else None,
                'overwritten_snapshot': copy.deepcopy(previous),
                'overwritten_snapshot_interval': {
                    'first': 1, 'last': sequence - SNAPSHOT_SLOTS,
                    'count': sequence - SNAPSHOT_SLOTS,
                } if sequence > SNAPSHOT_SLOTS else None,
            }
            diagnostics = {key: value for key, value in snapshot.items() if key != 'events'}
            diagnostics.update(identity=self.identity, identity_truncated=self.identity_truncated,
                               state=bounded_state, state_truncated=state_truncated,
                               snapshot=metadata)
            update = _json(diagnostics)
            if len(update) > MAX_DIAGNOSTIC_BYTES:
                raise ValueError('Diagnostic snapshot exceeds bounded response size')
            _write_json(path, {'capture': snapshot, 'diagnostics': diagnostics})
            self.node.set(update, write=False)
            self._sequence = sequence
            self._last_time = time.monotonic()
            self._last_metadata = copy.deepcopy(metadata)
            self._slots[slot] = {key: metadata[key] for key in (
                'sequence', 'monotonic', 'through_sequence', 'first_event_sequence',
                'last_event_sequence')}
            return metadata
