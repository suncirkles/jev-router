import unittest

from intervals import collapse


class CollapseTests(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(collapse([1, 2, 3, 5, 8, 7]), "1-3,5,7-8")

    def test_empty(self):
        self.assertEqual(collapse([]), "")


if __name__ == "__main__":
    unittest.main()
