import unittest
from pathlib import Path

from safe_paths import UnsafePathError, build_extraction_plan


class HiddenSafePathTests(unittest.TestCase):
    def test_absolute_drive_unc_and_trailing_rejected(self):
        names = ["/etc/passwd", "\\rooted", "C:/boot.ini", "c:relative", "//server/share", "folder/"]
        for name in names:
            with self.subTest(name=name), self.assertRaises(UnsafePathError):
                build_extraction_plan([name], Path("out"))

    def test_invalid_values(self):
        for name in ("", "a\x00b", None, 7):
            with self.subTest(name=name), self.assertRaises(UnsafePathError):
                build_extraction_plan([name], Path("out"))

    def test_dot_and_repeated_separators(self):
        self.assertEqual(build_extraction_plan(["./a///./b"], Path("root")),
                         [("./a///./b", Path("root/a/b"))])

    def test_casefolded_collision(self):
        with self.assertRaises(UnsafePathError):
            build_extraction_plan(["A/File.txt", "a/file.TXT"], Path("out"))

    def test_order_preserved(self):
        plan = build_extraction_plan(["z.txt", "a.txt"], Path("dest"))
        self.assertEqual([name for name, _ in plan], ["z.txt", "a.txt"])


if __name__ == "__main__":
    unittest.main()
