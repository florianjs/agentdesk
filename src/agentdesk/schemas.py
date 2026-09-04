"""The shapes that cross a boundary: the API's requests and responses, and the run itself.

Two of them are guardrails rather than data. `Budgets` bounds what a run may spend, and
`RunStatus` is a closed set — an agent that can only ever be in one of six states is an agent
whose behaviour can be enumerated, which is what makes a trajectory eval possible at all.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentdesk.config import settings

type RunStatus = Literal[
    "running",
    "awaiting_approval",
    "answered",
    "escalated",
    "budget_exceeded",
    "failed",
]

# Statuses from which nothing further happens on its own. `awaiting_approval` is deliberately
# absent: a suspended run is not finished, it is waiting for a person.
TERMINAL: frozenset[str] = frozenset({"answered", "escalated", "budget_exceeded", "failed"})


class Budgets(BaseModel):
    """Four ceilings, checked before every iteration.

    Defaults come from configuration so a deployment can tighten them without a release, and a
    caller may override them per run — a batch job and a live chat do not deserve the same
    patience.
    """

    max_iterations: int = Field(default_factory=lambda: settings.max_iterations, gt=0)
    max_tokens_total: int = Field(default_factory=lambda: settings.max_tokens_total, gt=0)
    max_cost_usd: float = Field(default_factory=lambda: settings.max_cost_usd, gt=0)
    max_wall_seconds: float = Field(default_factory=lambda: settings.max_wall_seconds, gt=0)


class RunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    customer_email: str = Field(default="", max_length=200)
    budgets: Budgets = Field(default_factory=Budgets)


class Approval(BaseModel):
    """A human's verdict on a suspended run."""

    note: str = Field(default="", max_length=500)


class RunView(BaseModel):
    """A run as the API returns it.

    `pending_action` is the whole point of the shape: when a run is suspended, the caller is
    shown exactly what it would be approving, not a run id and a status to go look up.
    """

    id: str
    status: RunStatus
    answer: str | None = None
    pending_action: dict[str, Any] | None = None
    stop_reason: str = ""
    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    cost_is_partial: bool = False
