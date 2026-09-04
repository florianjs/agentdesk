"""Dependency injection."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from agentdesk.db import get_sessionmaker
from agentdesk.llm.client import get_client

API_KEY_PREFIX = "ad_"


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def get_llm() -> AsyncOpenAI:
    return get_client()


async def get_api_key(
    x_api_key: Annotated[str | None, Header(description="Per-customer API key")] = None,
) -> str:
    if x_api_key is None or not x_api_key.startswith(API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid api key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


Session = Annotated[AsyncSession, Depends(get_session)]
LLM = Annotated[AsyncOpenAI, Depends(get_llm)]
ApiKey = Annotated[str, Depends(get_api_key)]
