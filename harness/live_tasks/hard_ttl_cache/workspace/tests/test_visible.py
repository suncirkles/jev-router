import threading
import time
import unittest

from ttl_cache import TTLCache


class TTLCacheTests(unittest.TestCase):
    def test_cached_until_expired(self):
        now = [10.0]
        cache = TTLCache(5, clock=lambda: now[0])
        calls = []
        self.assertEqual(cache.get_or_compute("x", lambda: calls.append(1) or 7), 7)
        self.assertEqual(cache.get_or_compute("x", lambda: 8), 7)
        now[0] = 16
        self.assertEqual(cache.get_or_compute("x", lambda: 8), 8)
        self.assertEqual(len(calls), 1)

    def test_single_flight(self):
        cache = TTLCache(10)
        barrier = threading.Barrier(4)
        calls = []
        results = []

        def factory():
            calls.append(1)
            time.sleep(0.05)
            return 9

        def worker():
            barrier.wait()
            results.append(cache.get_or_compute("x", factory))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(results, [9] * 4)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
