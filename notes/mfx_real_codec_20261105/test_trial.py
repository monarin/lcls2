"""Framing and error-check tests: reject truncated/mis-sized codec output."""
import unittest

import numpy as np

from trial import comparison, validate_blob


class TrialChecks(unittest.TestCase):
    def test_blob_framing(self):
        # A full uncompressed chunk followed by a four-byte tail chunk.
        blob = np.array([16388], '<i8').tobytes() + np.array([16384, 4], '<u2').tobytes() + bytes(16388)
        validate_blob(blob, 16388)
        invalid = [blob[:-1], blob + b'x', blob[:10],
                   np.array([16384], '<i8').tobytes() + blob[8:],
                   blob[:8] + np.array([0, 16388], '<u2').tobytes() + blob[12:]]
        for candidate in invalid:
            with self.assertRaises(RuntimeError):
                validate_blob(candidate, 16388)

    def test_error_and_nonfinite_detection(self):
        ref = np.array([1, -2, np.inf, -np.inf, np.nan], np.float32)
        self.assertTrue(comparison(ref, ref.copy())['pass'])
        changed = ref.copy()
        changed[0] += 4
        self.assertFalse(comparison(ref, changed)['pass'])
        changed = ref.copy()
        changed[2] = -np.inf
        self.assertFalse(comparison(ref, changed)['pass'])
        changed = ref.copy()
        changed[0] = np.nan
        self.assertFalse(comparison(ref, changed)['pass'])
        self.assertEqual(comparison(ref, changed)['nonfinite_mismatches'], 1)
        changed = ref.copy()
        changed[2] = 1
        self.assertEqual(comparison(ref, changed)['nonfinite_mismatches'], 1)
        self.assertTrue(np.isfinite(comparison(ref, changed)['max_abs_error']))


if __name__ == '__main__':
    unittest.main()
