import time


class TTLCache:
    def __init__(self, ttl_seconds, clock=time.monotonic):
        raise NotImplementedError

    def get_or_compute(self, key, factory):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
