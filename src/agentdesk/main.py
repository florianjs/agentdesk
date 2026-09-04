"""Application entry point.

uv run uvicorn agentdesk.main:app --reload --port 8002
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentdesk.config import settings
from agentdesk.db import create_schema, dispose_engine
from agentdesk.graph import close_checkpointer
from agentdesk.llm.client import MissingCredentials, close_client
from agentdesk.routes import runs

DESCRIPTION = """
A support agent that diagnoses a customer's problem, proposes an action, and stops.

Anything that moves money waits for a human: `POST /v1/runs` returns `awaiting_approval` with
the proposed action attached, and the run resumes only once someone approves or rejects it.

Every run is bounded by four budgets — iterations, tokens, spend and wall clock — and records
the tools it called, in order.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fail at startup, not per request. Without a key the agent cannot take a single step, and
    # discovering that on the first request means answering 500 to a caller whose request was
    # merely invalid — dependencies resolve before parameters are validated.
    if not settings.openrouter_api_key:
        raise MissingCredentials(
            "OPENROUTER_API_KEY is not set; refusing to start. See .env.example."
        )

    await create_schema()
    yield
    # Connection pools outlive the process unless closed here.
    await close_client()
    await close_checkpointer()
    await dispose_engine()


app = FastAPI(title="AgentDesk", version="0.1.0", description=DESCRIPTION, lifespan=lifespan)
app.include_router(runs.router, prefix="/v1")


@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict[str, str]:
    """Unauthenticated on purpose: load balancers do not carry API keys."""
    return {"status": "ok"}
