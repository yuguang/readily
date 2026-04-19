"""
Minimal async token-bucket rate limiter for the compliance LLM.

The compliance pipeline runs many per-section LLM requests in parallel.  A
plain ``asyncio.Semaphore`` caps in-flight concurrency but does nothing about
**request-per-minute** limits imposed by the provider (Gemini, OpenAI, etc.).

``AsyncTokenBucket`` enforces a global request-per-minute ceiling across all
workers in the same event loop without adding external dependencies.  Tokens
refill continuously at ``rate_per_sec = rpm / 60``; ``acquire()`` awaits
until one token is available.

The bucket is lazily constructed per event loop so tests that create fresh
loops get fresh state automatically.  ``configure_global_bucket(rpm)`` can
be used to install a specific bucket for the duration of a test or a
production run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class AsyncTokenBucket:
    """An asyncio-friendly token bucket rate limiter.

    Args:
        rate_per_sec: Sustained fill rate in tokens/second.
        capacity:     Maximum burst size.  Defaults to ``rate_per_sec`` so
                      short bursts up to one second of headroom are allowed.

    A ``rate_per_sec`` of ``0`` disables rate limiting entirely
    (``acquire`` returns immediately).
    """

    def __init__(self, rate_per_sec: float, capacity: float | None = None) -> None:
        self.rate_per_sec = max(0.0, float(rate_per_sec))
        self.capacity = float(capacity) if capacity is not None else max(1.0, self.rate_per_sec)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate_per_sec > 0.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0.0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_sec)
        self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* worth of budget is available.

        No-op when the bucket is disabled (``rate_per_sec == 0``).
        """
        if not self.enabled:
            return
        if tokens > self.capacity:
            # Clamp: callers shouldn't request more than one bucket-worth in a
            # single acquire, but be forgiving — just wait the full sustained
            # interval instead of blocking forever.
            tokens = self.capacity
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait_for = deficit / self.rate_per_sec
            # Release the lock while sleeping so other coroutines can make
            # progress (they'll re-check the token count on wake-up).
            await asyncio.sleep(wait_for)


# ---------------------------------------------------------------------------
# Process-wide default bucket
# ---------------------------------------------------------------------------


_global_bucket: Optional[AsyncTokenBucket] = None


def configure_global_bucket(rpm: int) -> AsyncTokenBucket:
    """Install a process-wide bucket sized for *rpm* requests per minute."""
    global _global_bucket
    _global_bucket = AsyncTokenBucket(rate_per_sec=rpm / 60.0)
    return _global_bucket


def get_global_bucket() -> AsyncTokenBucket | None:
    """Return the current global bucket, or ``None`` if none has been configured."""
    return _global_bucket


async def acquire_global(tokens: float = 1.0) -> None:
    """Acquire *tokens* from the global bucket.  No-op if unconfigured."""
    bucket = _global_bucket
    if bucket is None or not bucket.enabled:
        return
    await bucket.acquire(tokens)
