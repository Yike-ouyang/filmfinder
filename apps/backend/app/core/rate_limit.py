from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_limited(self, key: str, limit: int, window_seconds: int) -> bool:
        now = self._clock()
        cutoff = now - float(window_seconds)
        with self._lock:
            events = self._events.get(key)
            if events is None:
                return False
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)
                return False
            return len(events) >= int(limit)

    def record(self, key: str, window_seconds: int) -> None:
        now = self._clock()
        cutoff = now - float(window_seconds)
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            events.append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
