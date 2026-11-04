"""Negative checks for acceptance assertions; no service or XTC access."""
import unittest
import numpy as np
from common import TIMES, expected_step, fingerprint, validate_events


class AcceptanceChecks(unittest.TestCase):
    def events(self):
        return [{'timestamp': t, 'step': expected_step(t)} for t in TIMES]

    def test_distributed_order_is_not_required(self):
        self.assertEqual(validate_events(self.events()[::-1]), [19, 20, 21])

    def test_duplicate_rejected(self):
        rows = self.events()
        rows[19] = rows[18]
        with self.assertRaises(RuntimeError):
            validate_events(rows)

    def test_missing_tail_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_events(self.events()[:-1])

    def test_misplaced_step_rejected(self):
        rows = self.events()
        rows[19]['step'] = 0
        with self.assertRaises(RuntimeError):
            validate_events(rows)

    def test_nan_payload_normalization_and_value_sensitivity(self):
        a = np.array([0x7fc00001, 0x3f800000], dtype=np.uint32).view(np.float32)
        b = np.array([0x7fc00002, 0x3f800000], dtype=np.uint32).view(np.float32)
        self.assertEqual(fingerprint(a), fingerprint(b))
        b[1] = 2
        self.assertNotEqual(fingerprint(a), fingerprint(b))
        self.assertNotEqual(fingerprint(a), fingerprint(a.astype(np.float64)))
        self.assertNotEqual(fingerprint(a), fingerprint(a.reshape(1, 2)))

    def test_only_known_missing_encoder_error_is_recorded(self):
        from serial_compare import read_field

        def missing(evt):
            raise TypeError("'NoneType' object is not subscriptable")

        known = 'MfxUsbEncoder01/raw/descriptions'
        record = read_field(known, missing, None, {'MfxUsbEncoder01/raw': None})
        self.assertTrue(record['missing_segments'])
        with self.assertRaises(RuntimeError):
            read_field('different/raw/value', missing, None, {})
        with self.assertRaises(RuntimeError):
            read_field(known, missing, None, {'MfxUsbEncoder01/raw': {'0': 0}})


if __name__ == '__main__':
    unittest.main()
