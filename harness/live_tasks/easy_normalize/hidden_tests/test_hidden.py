import unittest

from key_normalizer import normalize_key


class HiddenNormalizeKeyTests(unittest.TestCase):
    def test_unicode_nfkc_and_casefold(self):
        self.assertEqual(normalize_key("Ｓｔｒａße № １２"), "strasse-no-12")

    def test_non_ascii_letters_are_preserved(self):
        self.assertEqual(normalize_key("  Καλημέρα__κόσμε "), "καλημέρα-κόσμε")

    def test_non_string_rejected(self):
        for value in (None, 7, b"x"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_key(value)

    def test_runs_collapse(self):
        self.assertEqual(normalize_key("a...___///b"), "a-b")


if __name__ == "__main__":
    unittest.main()
