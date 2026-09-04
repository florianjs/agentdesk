"""Retrying upstream calls.

Two rules decide everything here:

1. **Retry only what a retry can fix.** Rate limits, timeouts and 5xx are transient. A 400 or a
   401 is a bug in our request — retrying it burns latency and money to get the same answer.
2. **Never retry in lockstep.** Without jitter, every worker that failed at the same instant
   retries at the same instant, and the recovering provider is hit by the whole fleet at once.

`sleep` and `rng` are injected so the backoff schedule can be asserted in tests without waiting
for real seconds or hoping for a particular random draw.
"""

import asyncio
import random
from collections.abc import Awaitable, Callable

import openai


class TransientUpstreamError(RuntimeError):
    """A provider failure that arrived dressed as a success.

    A router can return HTTP 200 carrying `finish_reason: "error"`, no content and zero tokens.
    Nothing raised, nothing charged, nothing usable — and the SDK has no exception for it, so we
    raise our own rather than letting it look like a model that chose not to answer.
    """


# Transient by nature: the same request may well succeed a moment later.
RETRYABLE: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.APIConnectionError,
    TransientUpstreamError,
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_S = 1.0


def backoff_delay(attempt: int, base: float, draw: float) -> float:
    """Exponential backoff with full jitter: delay ∈ [0, base * 2^(attempt-1)].

    Full jitter, rather than a fixed delay plus noise, is what actually spreads a retrying
    fleet across the recovery window.
    """
    return base * (2.0 ** (attempt - 1)) * draw


async def with_retry[T](
    call: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay_s: float = DEFAULT_BASE_DELAY_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `call`, retrying only transient upstream failures.

    Three attempts is the ceiling on purpose: beyond that a retry stops being resilience and
    starts adding latency to an incident that is already happening.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except RETRYABLE:
            if attempt == max_attempts:
                raise
            await sleep(backoff_delay(attempt, base_delay_s, rng()))

    raise AssertionError("unreachable: the loop either returns or raises")
