"""Focused negative wire-audit and repetition-schedule tests; no live data."""
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import requests

from audit_jungfrau import inspect_frame
from common import guard_http, wire
from plan_layout import schedule


def node(src, damage, kind, payload):
    return wire.XTC.pack(src, damage, kind, len(payload) + 12) + payload


def frame(segments=(0, 1), damage=0, dimensions=(1, 512, 1024)):
    parts, schemas = [], {}
    for segment in segments:
        key = 778 + segment
        schemas[key] = {'detname': 'jungfrau1M', 'alg': 'raw', 'segment': segment,
                        'arrays': 1, 'fields': [('raw', 1, 3)]}
        shapes = node(0, 0, 2, struct.pack('<5I', *dimensions, 0, 0))
        data = node(0, 0, 3, bytes(1024 * 1024))
        parts.append(node(key, damage, 1, shapes + data))
    root = node(3, 0, 0, b''.join(parts))
    return struct.pack('<QI', wire.from_ns(1_000_000_001), 12 << 24) + root, schemas


class AuditTests(unittest.TestCase):
    def check(self, data, schemas, timestamp=None):
        h = wire.HEADER.unpack_from(data)
        smd = (h[0] if timestamp is None else timestamp, h[1], 0, 0, 0, 64)
        with tempfile.TemporaryFile() as f:
            f.write(data)
            f.flush()
            return inspect_frame(f.fileno(), {'bd_offset': 0, 'bd_size': len(data)}, smd, schemas, 3)

    def test_valid_different_smd_src(self):
        self.assertEqual(self.check(*frame()), ([0, 1], 0))

    def test_missing_segment_is_reported(self):
        self.assertEqual(self.check(*frame(segments=(1,))), ([1], 0))

    def test_damage_is_preserved(self):
        self.assertEqual(self.check(*frame(damage=8)), ([0, 1], 8))

    def test_wrong_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            self.check(*frame(), timestamp=0)

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.check(*frame(dimensions=(1, 1024, 512)))

    def test_truncated_payload_rejected(self):
        data, schemas = frame()
        with self.assertRaises(ValueError):
            self.check(data[:-1], schemas)

    def test_wrong_names_id_rejected(self):
        data, schemas = frame()
        del schemas[778]
        with self.assertRaises(ValueError):
            self.check(data, schemas)


class ScheduleTests(unittest.TestCase):
    def index(self):
        services = [2, 4, 6, 8, 10, 12, 12, 10, 12, 10, 9, 7, 5]
        index = np.empty(len(services), dtype=[('service', '<u8'), ('timestamp', '<u8')])
        index['service'] = services
        index['timestamp'] = [wire.from_ns(999_990_000 + i*1000) for i in range(len(services))]
        return index

    @patch('plan_layout.SOURCE_EVENTS', 3)
    def test_two_repeats_and_prefix_have_one_step(self):
        index = self.index()
        records, times, cycles, period = schedule(index, 7)
        services = index['service'][records]
        self.assertEqual(np.count_nonzero(services == 6), 1)
        self.assertEqual(np.count_nonzero(services == 8), 1)
        self.assertEqual(cycles[services == 12].tolist(), [0, 0, 0, 1, 1, 1, 2])
        self.assertTrue(np.all(times[1:] > times[:-1]))
        self.assertGreater(int(times[-1]) >> 32, int(times[0]) >> 32)
        self.assertEqual(period, 7000)

    @patch('plan_layout.SOURCE_EVENTS', 3)
    def test_prefix_excludes_later_slowupdate(self):
        index = self.index()
        records, _, _, _ = schedule(index, 2)
        self.assertEqual(index['service'][records].tolist(), [2, 4, 6, 8, 10, 12, 12, 9, 7, 5])

    @patch('plan_layout.SOURCE_EVENTS', 3)
    def test_zero_events_rejected(self):
        with self.assertRaises(ValueError):
            schedule(self.index(), 0)

    def test_http_mutation_rejected(self):
        with patch.object(requests.sessions.Session, 'request', requests.sessions.Session.request):
            guard_http()
            with self.assertRaisesRegex(ValueError, 'forbids HTTP mutations'):
                requests.post('https://invalid.example/no-request-is-sent')


if __name__ == '__main__':
    unittest.main()
