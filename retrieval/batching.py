from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Callable


@dataclass
class PendingQuery:
    query: str
    topk: int
    future: Future


class RetrievalBatcher:
    def __init__(
        self,
        retrieve: Callable[[list[str], int], list[list[dict]]],
        batch_size: int = 32,
        wait_ms: float = 10.0,
        queue_size: int = 4096,
    ):
        if batch_size < 1 or wait_ms < 0 or queue_size < 1:
            raise ValueError("Invalid retrieval batch or queue settings")
        self.retrieve = retrieve
        self.batch_size = batch_size
        self.wait_seconds = wait_ms / 1000.0
        self.queue = Queue(maxsize=queue_size)
        self.lock = Lock()
        self.closed = False
        self.thread = Thread(target=self._run, name="dense-retrieval", daemon=True)
        self.thread.start()

    def submit(self, query: str, topk: int) -> Future:
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError("Retrieval service is stopping")
            self.queue.put_nowait(PendingQuery(query, topk, future))
        return future

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.queue.put(None)
        self.thread.join()

    def _run(self):
        while True:
            first = self.queue.get()
            if first is None:
                return
            pending = [first]
            deadline = monotonic() + self.wait_seconds
            stopping = False
            while len(pending) < self.batch_size:
                remaining = max(0.0, deadline - monotonic())
                try:
                    following = self.queue.get(timeout=remaining)
                except Empty:
                    break
                if following is None:
                    stopping = True
                    break
                pending.append(following)
            active = [item for item in pending if item.future.set_running_or_notify_cancel()]
            if active:
                try:
                    results = self.retrieve([item.query for item in active], max(item.topk for item in active))
                    if len(results) != len(active):
                        raise RuntimeError("Retrieval returned an incorrect query count")
                except Exception as exc:
                    for item in active:
                        item.future.set_exception(exc)
                else:
                    for item, documents in zip(active, results):
                        item.future.set_result(documents[: item.topk])
            if stopping:
                return
