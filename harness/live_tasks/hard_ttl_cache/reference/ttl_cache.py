import math
import threading
import time


class _Flight:
    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None


class TTLCache:
    def __init__(self, ttl_seconds, clock=time.monotonic):
        if isinstance(ttl_seconds, bool):
            raise ValueError("ttl must be finite and non-negative")
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError):
            raise ValueError("ttl must be finite and non-negative") from None
        if not math.isfinite(ttl) or ttl < 0:
            raise ValueError("ttl must be finite and non-negative")
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._values = {}
        self._flights = {}

    def get_or_compute(self, key, factory):
        with self._lock:
            cached = self._values.get(key)
            if cached is not None and self._clock() < cached[1]:
                return cached[0]
            flight = self._flights.get(key)
            owner = flight is None
            if owner:
                flight = _Flight()
                self._flights[key] = flight
        if not owner:
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value
        try:
            value = factory()
        except BaseException as error:
            with self._lock:
                flight.error = error
                del self._flights[key]
                flight.event.set()
            raise
        with self._lock:
            flight.value = value
            self._values[key] = (value, self._clock() + self._ttl)
            del self._flights[key]
            flight.event.set()
        return value

    def clear(self):
        with self._lock:
            self._values.clear()
