import unittest

from intervals import collapse


class HiddenCollapseTests(unittest.TestCase):
    def test_negative_and_duplicates(self):
        self.assertEqual(collapse([-1, -3, -2, 2, 1, -2, 4]), "-3--1,1-2,4")

    def test_generator(self):
        self.assertEqual(collapse(x for x in [4, 3, 2]), "2-4")

    def test_input_not_mutated(self):
        values = [3, 1, 2]
        collapse(values)
        self.assertEqual(values, [3, 1, 2])

    def test_invalid_values(self):
        for values in ([True], [1.0], ["1"]):
            with self.subTest(values=values), self.assertRaises(TypeError):
                collapse(values)


if __name__ == "__main__":
    unittest.main()
