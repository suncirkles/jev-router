import unittest

from dependencies import DependencyError, build_order


class HiddenDependencyTests(unittest.TestCase):
    def test_duplicates_and_generator(self):
        graph = {"c": (x for x in ["a", "a", "b"]), "a": [], "b": ["a"]}
        self.assertEqual(build_order(graph), ["a", "b", "c"])

    def test_missing_and_self(self):
        for graph in ({"a": ["missing"]}, {"a": ["a"]}):
            with self.subTest(graph=graph), self.assertRaises(DependencyError):
                build_order(graph)

    def test_string_iterable_and_bad_names(self):
        for graph in ({"a": "b", "": []}, {1: []}):
            with self.subTest(graph=graph), self.assertRaises(DependencyError):
                build_order(graph)

    def test_input_not_mutated(self):
        graph = {"b": ["a"], "a": []}
        build_order(graph)
        self.assertEqual(graph, {"b": ["a"], "a": []})


if __name__ == "__main__":
    unittest.main()
