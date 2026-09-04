"""The real model call, wired for the loop.

The loop takes a callable so it can be tested against a script. This is the production one:
retry on transient upstream failures, and a fallback model when the primary is down for longer
than a retry can cover.

The fallback is a *different vendor*, not a smaller model from the same one. An outage usually
takes a provider with it, so a fallback that shares the primary's infrastructure is a fallback
that fails at the same moment.
"""

from typing import Any, cast

from openai import AsyncOpenAI

from agentdesk.agent.loop import ModelCall
from agentdesk.config import settings
from agentdesk.llm.client import ModelTier, resolve_model, with_cost_accounting
from agentdesk.llm.retry import RETRYABLE, with_retry


def model_call(client: AsyncOpenAI, *, tier: ModelTier = "smart", model: str = "") -> ModelCall:
    """Build the callable the loop drives.

    Temperature 0: the agent's job is to follow a policy, and a policy applied differently to the
    same facts is a bug even when both answers read well.

    `model` names one directly, bypassing the tier. Callers in the service always name a tier;
    only the model comparison names a model, because that is the one place the point is to run
    something the configuration does not select.
    """
    primary = model or resolve_model(tier)

    async def call(messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> Any:
        async def attempt(model: str) -> Any:
            return await client.chat.completions.create(
                model=model,
                # The SDK types these as large closed unions of TypedDicts. The transcript is
                # plain data by design — it round-trips through Postgres — so it is cast once
                # here rather than reshaped into SDK objects at every step of the loop.
                messages=cast(Any, messages),
                tools=cast(Any, schemas),
                temperature=0,
                extra_body=with_cost_accounting(),
            )

        try:
            return await with_retry(
                lambda: attempt(primary),
                max_attempts=settings.retry_max_attempts,
                base_delay_s=settings.retry_base_delay_s,
            )
        except RETRYABLE:
            # Retries exhausted: the primary is not coming back within this request. One attempt
            # on the fallback, un-retried — a second wait would spend the caller's patience on a
            # provider that has already been given three chances.
            return await attempt(settings.model_fallback)

    return call
