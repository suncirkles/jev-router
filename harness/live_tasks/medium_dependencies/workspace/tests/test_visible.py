import unittest

from dependencies import DependencyError, build_order


class DependencyTests(unittest.TestCase):
    def test_deterministic_order(self):
        graph = {"deploy": ["build", "test"], "test": ["build"], "build": [], "docs": []}
        self.assertEqual(build_order(graph), ["build", "docs", "test", "deploy"])

    def test_cycle(self):
        with self.assertRaises(DependencyError):
            build_order({"a": ["b"], "b": ["a"]})


if __name__ == "__main__":
    unittest.main()
