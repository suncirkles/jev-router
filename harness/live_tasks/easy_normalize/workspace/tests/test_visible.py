import unittest

from key_normalizer import normalize_key


class NormalizeKeyTests(unittest.TestCase):
    def test_ascii_and_punctuation(self):
        self.assertEqual(normalize_key("  Hello,  World!  "), "hello-world")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            normalize_key(" -- ")


if __name__ == "__main__":
    unittest.main()
