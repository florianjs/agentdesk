"""Four budgets, checked before every iteration.

An agent loop's failure mode is not a crash — it is a loop that keeps going. Each budget here
stops a different runaway, and they are not interchangeable:

- **iterations** — a model that calls a tool, reads the result, and calls it again forever;
- **tokens** — a conversation that grows without bound, because the whole transcript is resent
  on every turn (iteration 10 costs far more than iteration 1);
- **cost** — the bill, which token counts alone do not give you once models differ in price;
- **wall clock** — a request that hangs upstream, which none of the other three notice.

Spend is tracked in USD, not EUR: that is the currency the provider reports. Converting it at a
hard-coded rate would turn a measured number into a guessed one. Refunds, which are customer
money, stay in EUR — the two currencies belong to two different systems.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from agentdesk.llm.client import Usage
from agentdesk.schemas import Budgets


@dataclass
class BudgetLedger:
    """What the run has spent so far, and whether it may continue.

    `now` is injected so the wall-clock budget can be tested without waiting for real seconds.
    """

    budgets: Budgets
    now: Callable[[], float] = time.monotonic
    started_at: float = field(default=0.0)

    iterations: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    # Set when a provider omits `cost` from its usage block. The cost budget cannot be enforced
    # on a run where that happened, and silently reporting 0.0 would read as "this was free".
    cost_is_partial: bool = False

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = self.now()

    def record(self, usage: Usage | None) -> None:
        """Charge one model call to the run."""
        self.iterations += 1
        if usage is None:
            # No usage block at all: the call happened, so the iteration counts. Nothing else can.
            self.cost_is_partial = True
            return
        self.tokens += usage.total_tokens
        if usage.cost_usd is None:
            self.cost_is_partial = True
        else:
            self.cost_usd += usage.cost_usd

    def elapsed_s(self) -> float:
        return self.now() - self.started_at

    def exceeded(self) -> str | None:
        """The name of the first budget that is spent, or None.

        Checked *before* an iteration rather than after: the point is to not make the call, and a
        check that runs afterwards has already paid for what it was meant to prevent.
        """
        if self.iterations >= self.budgets.max_iterations:
            return f"max_iterations ({self.budgets.max_iterations})"
        if self.tokens >= self.budgets.max_tokens_total:
            return f"max_tokens_total ({self.budgets.max_tokens_total})"
        if self.cost_usd >= self.budgets.max_cost_usd:
            return f"max_cost_usd ({self.budgets.max_cost_usd})"
        if self.elapsed_s() >= self.budgets.max_wall_seconds:
            return f"max_wall_seconds ({self.budgets.max_wall_seconds})"
        return None
