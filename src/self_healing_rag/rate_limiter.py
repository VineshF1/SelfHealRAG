"""Token-bucket rate limiter."""
import threading
import time


class RateLimiter:
    def __init__(self, rpm: int = 60):
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self._rpm = rpm
        self._tokens = float(rpm)
        self._max_tokens = float(rpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return
        wait = (1.0 - self._tokens) / self._rpm * 60.0
        time.sleep(max(wait, 0.01))
        self._refill()
        self._tokens -= 1.0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc_info):
        pass

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        gained = elapsed / 60.0 * self._rpm
        self._tokens = min(self._max_tokens, self._tokens + gained)
        self._last_refill = now
