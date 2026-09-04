"""OpenRouter client and model routing.

One provider endpoint, every model: switching models is a string change, which is what makes
routing decisions cheap to test and cheap to reverse.

The API key never leaves the backend. Everything upstream goes through this module so that
timeouts, model choice and usage accounting live in exactly one place.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

# Imported eagerly and never used directly: the OpenAI SDK imports httpx lazily while processing
# a response, and two concurrent responses can catch the module half-initialised —
# "partially initialized module 'httpx' has no attribute 'Response'". Importing it here, at
# module load, means it is fully built before any request runs.
import httpx  # noqa: F401
from openai import AsyncOpenAI

from agentdesk.config import settings

BASE_URL = "https://openrouter.ai/api/v1"

type ModelTier = Literal["cheap", "smart", "fallback", "judge"]

# A request without a timeout is a zombie connection waiting to happen.
DEFAULT_TIMEOUT_S = 30.0


def resolve_model(tier: ModelTier) -> str:
    """Map a stable tier name to the configured model id.

    Callers name a tier, never a model. Swapping the model behind a tier is then a config
    change instead of a code change.
    """
    match tier:
        case "cheap":
            return settings.model_cheap
        case "smart":
            return settings.model_smart
        case "fallback":
            return settings.model_fallback
        case "judge":
            return settings.model_judge


class MissingCredentials(RuntimeError):
    """No API key configured.

    Raised in place of the SDK's own error, which tells the reader to set `OPENAI_API_KEY` —
    a variable this service does not use, sending them to the wrong place.
    """


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """The shared async client. Cached: connection pooling matters under load."""
    if not settings.openrouter_api_key:
        raise MissingCredentials(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=settings.openrouter_api_key,
        timeout=DEFAULT_TIMEOUT_S,
        default_headers={
            "HTTP-Referer": "https://github.com/florianjs/agentdesk",
            "X-Title": "AgentDesk",
        },
    )


async def close_client() -> None:
    """Release the connection pool.

    Async HTTP connections outlive the event loop unless they are closed on the way out, which
    surfaces as noisy generator-shutdown errors. Scripts call this before exiting; the API calls
    it from its lifespan handler.
    """
    if get_client.cache_info().currsize:
        await get_client().close()
        get_client.cache_clear()


def with_cost_accounting(extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask the provider to report the credit cost alongside token counts.

    Without this, `usage` carries token counts only and every cost figure is an estimate
    derived from a price list that drifts.
    """
    body = dict(extra_body or {})
    body["usage"] = {"include": True}
    return body


@dataclass(frozen=True, slots=True)
class Usage:
    """What one call consumed. The unit of every cost and budget decision."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None = None
    cached_tokens: int = 0
    # Billed as output, never shown to the customer. This is the line item that makes a model
    # listed at 5x cost 25x per ticket.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def per_thousand_usd(self) -> float | None:
        """Cost of 1,000 calls of this shape — the number a customer actually asks about."""
        return None if self.cost_usd is None else self.cost_usd * 1000


def read_usage(response: object) -> Usage | None:
    """Extract usage from a chat completion, tolerating provider-specific extra fields.

    The SDK types `usage` loosely because providers differ; this narrows it once so callers
    never poke at raw attributes.
    """
    model = getattr(response, "model", "") or ""
    raw = getattr(response, "usage", None)
    if raw is None:
        return None

    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", 0) or 0
    reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0
    cost = getattr(raw, "cost", None)

    return Usage(
        model=model,
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        cost_usd=float(cost) if cost is not None else None,
        cached_tokens=int(cached),
        reasoning_tokens=int(reasoning),
    )
