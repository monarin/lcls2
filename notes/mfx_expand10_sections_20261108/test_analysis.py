import unittest
from analyze import partition, union_ns


class Intervals(unittest.TestCase):
    def test_union_overlap_and_touch(self):
        self.assertEqual(union_ns([(1,5),(2,3),(4,8),(8,10),(12,14)]),11)

    def test_nested_exclusive_and_gaps(self):
        got = partition([(2,9,'outer'),(3,5,'inner'),(3,4,'leaf')],0,10)
        self.assertEqual(got, {'outside_instrumented_sections':3,'outer':5,'inner':1,'leaf':1})

    def test_clip_window(self):
        self.assertEqual(partition([(0,10,'outer'),(4,9,'inner')],3,7), {'outer':1,'inner':3})

    def test_non_nested_rejected(self):
        with self.assertRaises(ValueError):
            partition([(0,5,'a'),(3,7,'b')],0,10)

    def test_touching_ranges(self):
        self.assertEqual(partition([(0,5,'a'),(5,10,'b')],0,10), {'a':5,'b':5})


if __name__ == '__main__':
    unittest.main()
