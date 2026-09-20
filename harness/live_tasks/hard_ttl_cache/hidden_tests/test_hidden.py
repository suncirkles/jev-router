import threading
import time
import unittest

from ttl_cache import TTLCache


class HiddenTTLCacheTests(unittest.TestCase):
    def test_none_is_cached_and_clear_invalidates(self):
        cache = TTLCache(10)
        calls = []
        self.assertIsNone(cache.get_or_compute("x", lambda: calls.append(1)))
        self.assertIsNone(cache.get_or_compute("x", lambda: calls.append(2)))
        cache.clear()
        cache.get_or_compute("x", lambda: calls.append(3))
        self.assertEqual(calls, [1, 3])

    def test_exception_shared_then_retry(self):
        cache = TTLCache(10)
        barrier = threading.Barrier(3)
        calls = []
        errors = []

        def failing():
            calls.append(1)
            time.sleep(0.05)
            raise RuntimeError("boom")

        def worker():
            barrier.wait()
            try:
                cache.get_or_compute("x", failing)
            except RuntimeError as error:
                errors.append(str(error))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(calls, [1])
        self.assertEqual(errors, ["boom"] * 3)
        self.assertEqual(cache.get_or_compute("x", lambda: 12), 12)

    def test_different_keys_compute_concurrently(self):
        cache = TTLCache(10)
        entered = threading.Barrier(2)

        def factory(value):
            entered.wait(timeout=1)
            return value

        results = []
        threads = [threading.Thread(target=lambda v=v: results.append(cache.get_or_compute(v, lambda: factory(v)))) for v in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertCountEqual(results, [1, 2])

    def test_validation_and_zero_ttl(self):
        for ttl in (-1, float("inf"), float("nan")):
            with self.subTest(ttl=ttl), self.assertRaises(ValueError):
                TTLCache(ttl)
        cache = TTLCache(0)
        calls = []
        cache.get_or_compute("x", lambda: calls.append(1) or 1)
        cache.get_or_compute("x", lambda: calls.append(2) or 2)
        self.assertEqual(calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
