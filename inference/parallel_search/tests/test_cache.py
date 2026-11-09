from __future__ import annotations

import threading
import time
import unittest

from inference.parallel_search.cache import SingleFlightLRU


class SingleFlightLRUTest(unittest.TestCase):
    def test_concurrent_callers_share_one_computation(self):
        cache = SingleFlightLRU[str](1024, len)
        barrier = threading.Barrier(8)
        calls = 0
        calls_lock = threading.Lock()
        results: list[tuple[str, str]] = []

        def producer() -> tuple[str, bool]:
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.05)
            return "result", True

        def worker() -> None:
            barrier.wait()
            results.append(cache.get_or_compute("key", producer))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(calls, 1)
        self.assertEqual([value for value, _ in results], ["result"] * 8)
        self.assertEqual(sum(status == "miss" for _, status in results), 1)
        self.assertEqual(sum(status == "shared" for _, status in results), 7)

    def test_weighted_lru_evicts_oldest_entry(self):
        cache = SingleFlightLRU[str](6, len)
        cache.get_or_compute("a", lambda: ("aaa", True))
        cache.get_or_compute("b", lambda: ("bbb", True))
        cache.get_or_compute("a", lambda: ("unused", True))
        cache.get_or_compute("c", lambda: ("ccc", True))

        value, status = cache.get_or_compute("b", lambda: ("new", True))
        self.assertEqual((value, status), ("new", "miss"))
        self.assertEqual(cache.info()["evictions"], 2)

    def test_uncacheable_value_is_still_shared_inflight(self):
        cache = SingleFlightLRU[str](1024, len)
        calls = 0

        def producer() -> tuple[str, bool]:
            nonlocal calls
            calls += 1
            return "transient", False

        self.assertEqual(cache.get_or_compute("key", producer), ("transient", "miss"))
        self.assertEqual(cache.get_or_compute("key", producer), ("transient", "miss"))
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
