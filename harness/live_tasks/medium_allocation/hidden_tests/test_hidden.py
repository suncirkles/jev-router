import unittest

from allocation import allocate


class HiddenAllocationTests(unittest.TestCase):
    def test_decimal_strings_and_order(self):
        result = allocate(7, {"z": "0.1", "a": "0.2"})
        self.assertEqual(result, {"a": 5, "z": 2})
        self.assertEqual(list(result), ["a", "z"])

    def test_exact_sum_and_tie_break(self):
        result = allocate(101, {"beta": 1, "alpha": 1, "gamma": 1})
        self.assertEqual(sum(result.values()), 101)
        self.assertEqual(result, {"alpha": 34, "beta": 34, "gamma": 33})

    def test_invalid_total(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                allocate(value, {"a": 1})

    def test_invalid_weights(self):
        cases = ({"a": -1}, {"a": "NaN"}, {"a": "Infinity"}, {"a": 0}, {"": 1})
        for weights in cases:
            with self.subTest(weights=weights), self.assertRaises((TypeError, ValueError)):
                allocate(1, weights)


if __name__ == "__main__":
    unittest.main()
