"""Pydantic models for AgentDesk.

Budgets and run status are the agent's two structural guardrails: a Pydantic cap and a state
machine cannot be talked around by a prompt.
"""

from typing import Literal

from pydantic import BaseModel, Field

type RunStatus = Literal[
    "running",
    "awaiting_approval",
    "answered",
    "escalated",
    "budget_exceeded",
    "failed",
]


class Budgets(BaseModel):
    """Four counters, checked BEFORE every iteration.

    On overrun the agent does not raise: it wraps up cleanly and the status is recorded.
    """

    max_iterations: int = 10
    max_tokens_total: int = 50_000
    max_cost_eur: float = 0.50
    max_wall_seconds: float = 120.0


class ProposeRefund(BaseModel):
    """Propose a refund. Does NOT execute it: creates a request awaiting human approval."""

    order_id: str
    amount_eur: float = Field(le=500, gt=0, description="Hard cap: 500 EUR")
    reason: str


class Run(BaseModel):
    """An agent run, persisted at every iteration (no persistence, no human-in-the-loop)."""

    id: str
    status: RunStatus = "running"
    iterations: int = 0
    cost_eur: float = 0.0
