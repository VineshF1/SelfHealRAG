"""Token-bucket rate limiter for Gemini API calls.

Usage:
    limiter = RateLimiter(rpm=60)   # 60 requests per minute
    async with limiter:
        response = model.generate_content(prompt)
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter.

    Allows up to *rpm* (requests-per-minute) calls.  If the bucket is empty
    the caller blocks until a token becomes available.
    """

    def __init__(self, rpm: int = 60):
        if rpm < 1:
            raise ValueError("rpm must be >= 1")
        self._rpm = rpm
        self._tokens = float(rpm)
        self._max_tokens = float(rpm)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────

    def acquire(self) -> None:
        """Block until a token is available."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return

        # Sleep until at least one token is available
        wait = (1.0 - self._tokens) / self._rpm * 60.0
        logger.debug("Rate limit hit — sleeping %.2fs", wait)
        time.sleep(max(wait, 0.01))
        self._refill()
        self._tokens -= 1.0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc_info):
        pass

    # ── Internals ───────────────────────────────────────────

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        gained = elapsed / 60.0 * self._rpm
        self._tokens = min(self._max_tokens, self._tokens + gained)
        self._last_refill = now
