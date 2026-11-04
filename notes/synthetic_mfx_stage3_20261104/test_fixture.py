"""Focused failure/rollover checks; the real pilot gets a separate full validator."""
import struct
import unittest

from xtc_fixture import HEADER, children, from_ns, insert_times, to_ns
from verify_pilot import compare_allowed


class FixtureTests(unittest.TestCase):
    def test_second_rollover(self):
        left = from_ns(7_999_999_998)
        right = from_ns(8_000_000_021)
        times = insert_times(left, right, 4)
        self.assertEqual([to_ns(t) for t in times],
                         [8_000_000_003, 8_000_000_008, 8_000_000_013, 8_000_000_018])
        self.assertTrue(all(t >> 32 == 8 and t & 0xffffffff < 1_000_000_000 for t in times))

    def test_tight_gap(self):
        self.assertEqual(insert_times(100, 1914, 4), [553, 1006, 1459, 1912])

    def test_insufficient_gap(self):
        with self.assertRaisesRegex(ValueError, "No space"):
            insert_times(100, 104, 4)

    def test_unnormalized_timestamp(self):
        with self.assertRaisesRegex(ValueError, "Unnormalized"):
            to_ns((7 << 32) | 1_000_000_000)

    def test_truncated_child(self):
        data = HEADER.pack(1, 12 << 24, 0, 0, 0, 24) + struct.pack("<IHHI", 0, 0, 3, 16)
        with self.assertRaisesRegex(ValueError, "Bad child extent"):
            list(children(data))

    def test_reject_control_change(self):
        original = HEADER.pack(1, (9 << 24) | 3, 5, 1, 0, 12)
        changed = HEADER.pack(2, (9 << 24) | 4, 5, 1, 0, 12)
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            compare_allowed(original, changed, 9, 0, {}, {}, 255)

    def test_preserve_damage_with_new_timestamp(self):
        original = HEADER.pack(1, (9 << 24) | 3, 5, 1, 0, 12)
        changed = HEADER.pack(2, (9 << 24) | 3, 5, 1, 0, 12)
        compare_allowed(original, changed, 9, 0, {}, {}, 255)


if __name__ == "__main__":
    unittest.main()
