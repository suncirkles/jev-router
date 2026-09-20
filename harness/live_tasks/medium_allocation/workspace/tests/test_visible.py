import unittest

from allocation import allocate


class AllocationTests(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(allocate(10, {"a": 1, "b": 1, "c": 1}), {"a": 4, "b": 3, "c": 3})

    def test_zero_total(self):
        self.assertEqual(allocate(0, {"a": 2, "b": 1}), {"a": 0, "b": 0})


if __name__ == "__main__":
    unittest.main()
