import unittest
from pathlib import Path

from safe_paths import UnsafePathError, build_extraction_plan


class SafePathTests(unittest.TestCase):
    def test_normalizes_separators(self):
        self.assertEqual(build_extraction_plan(["a//b.txt", "c\\d.py"], Path("out")),
                         [("a//b.txt", Path("out/a/b.txt")), ("c\\d.py", Path("out/c/d.py"))])

    def test_parent_rejected(self):
        with self.assertRaises(UnsafePathError):
            build_extraction_plan(["a/../secret"], Path("out"))


if __name__ == "__main__":
    unittest.main()
