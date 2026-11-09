"""Thread-safe weighted LRU cache with per-key request coalescing."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar


T = TypeVar("T")


@dataclass
class _Flight(Generic[T]):
    event: threading.Event
    value: T | None = None
    error: BaseException | None = None


class SingleFlightLRU(Generic[T]):
    """Bounded LRU cache that computes each missing key only once concurrently."""

    def __init__(self, max_weight: int, weight_fn: Callable[[T], int]) -> None:
        if max_weight < 0:
            raise ValueError("max_weight must be >= 0")
        self._max_weight = max_weight
        self._weight_fn = weight_fn
        self._entries: OrderedDict[Hashable, tuple[T, int]] = OrderedDict()
        self._inflight: dict[Hashable, _Flight[T]] = {}
        self._weight = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.shared = 0
        self.evictions = 0

    def get_or_compute(
        self,
        key: Hashable,
        producer: Callable[[], tuple[T, bool]],
    ) -> tuple[T, str]:
        """Return ``(value, status)`` where status is hit, shared, or miss.

        ``producer`` returns ``(value, cacheable)``. Concurrent callers always
        share the produced value even when it is not admitted to the LRU.
        """
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return cached[0], "hit"

            flight = self._inflight.get(key)
            if flight is None:
                flight = _Flight(event=threading.Event())
                self._inflight[key] = flight
                self.misses += 1
                leader = True
            else:
                self.shared += 1
                leader = False

        if not leader:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value, "shared"  # type: ignore[return-value]

        try:
            value, cacheable = producer()
        except BaseException as exc:
            with self._lock:
                flight.error = exc
                self._inflight.pop(key, None)
                flight.event.set()
            raise

        with self._lock:
            if cacheable:
                self._admit(key, value)
            flight.value = value
            self._inflight.pop(key, None)
            flight.event.set()
        return value, "miss"

    def _admit(self, key: Hashable, value: T) -> None:
        weight = max(0, self._weight_fn(value))
        if self._max_weight == 0 or weight > self._max_weight:
            return
        old = self._entries.pop(key, None)
        if old is not None:
            self._weight -= old[1]
        self._entries[key] = (value, weight)
        self._weight += weight
        while self._weight > self._max_weight and self._entries:
            _, (_, evicted_weight) = self._entries.popitem(last=False)
            self._weight -= evicted_weight
            self.evictions += 1

    def info(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "weight": self._weight,
                "max_weight": self._max_weight,
                "inflight": len(self._inflight),
                "hits": self.hits,
                "misses": self.misses,
                "shared": self.shared,
                "evictions": self.evictions,
            }
