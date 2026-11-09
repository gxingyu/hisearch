from __future__ import annotations

import threading
import unittest
from queue import Full

from retrieval.batching import RetrievalBatcher


class RetrievalBatchingTests(unittest.TestCase):
    def test_concurrent_queries_are_batched_and_keep_their_order_and_topk(self):
        calls = []
        threads = []

        def retrieve(queries, topk):
            calls.append((queries, topk))
            threads.append(threading.get_ident())
            return [[{"query": query, "rank": rank} for rank in range(topk)] for query in queries]

        batcher = RetrievalBatcher(retrieve, batch_size=4, wait_ms=1000)
        try:
            futures = [batcher.submit(f"query {index}", index % 3 + 1) for index in range(8)]
            results = [future.result(timeout=3) for future in futures]
        finally:
            batcher.close()
        self.assertEqual([len(queries) for queries, topk in calls], [4, 4])
        self.assertEqual([topk for queries, topk in calls], [3, 3])
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        for index, documents in enumerate(results):
            self.assertEqual(len(documents), index % 3 + 1)
            self.assertTrue(all(document["query"] == f"query {index}" for document in documents))

    def test_partial_batch_flushes_without_waiting_for_more_queries(self):
        batcher = RetrievalBatcher(lambda queries, topk: [[{"query": query}] for query in queries], wait_ms=5)
        try:
            self.assertEqual(batcher.submit("alone", 3).result(timeout=1), [{"query": "alone"}])
        finally:
            batcher.close()

    def test_batch_failure_reaches_all_callers_and_worker_recovers(self):
        def retrieve(queries, topk):
            if "fail" in queries:
                raise ValueError("retrieval failed")
            return [[{"query": query}] for query in queries]

        batcher = RetrievalBatcher(retrieve, batch_size=2, wait_ms=20)
        try:
            futures = [batcher.submit(query, 3) for query in ("fail", "other")]
            for future in futures:
                with self.assertRaisesRegex(ValueError, "retrieval failed"):
                    future.result(timeout=1)
            self.assertEqual(batcher.submit("recovered", 3).result(timeout=1), [{"query": "recovered"}])
        finally:
            batcher.close()

    def test_bounded_queue_and_cancellation(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def retrieve(queries, topk):
            calls.append(queries)
            started.set()
            release.wait(timeout=3)
            return [[{"query": query}] for query in queries]

        batcher = RetrievalBatcher(retrieve, batch_size=1, wait_ms=0, queue_size=1)
        try:
            first = batcher.submit("first", 3)
            self.assertTrue(started.wait(timeout=1))
            cancelled = batcher.submit("cancelled", 3)
            self.assertTrue(cancelled.cancel())
            with self.assertRaises(Full):
                batcher.submit("overflow", 3)
            release.set()
            first.result(timeout=1)
        finally:
            release.set()
            batcher.close()
        self.assertEqual(calls, [["first"]])

    def test_close_drains_pending_queries_and_rejects_new_work(self):
        batcher = RetrievalBatcher(lambda queries, topk: [[{"query": query}] for query in queries], wait_ms=1000)
        future = batcher.submit("pending", 3)
        batcher.close()
        batcher.close()
        self.assertEqual(future.result(), [{"query": "pending"}])
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            batcher.submit("late", 3)


if __name__ == "__main__":
    unittest.main()
