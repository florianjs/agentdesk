"""The whole of a run, in one object that survives the process.

An agent that keeps its conversation in a local variable cannot be paused, and an agent that
cannot be paused cannot ask a human anything: the request would have to block for as long as the
person takes to answer. Persisting the state at every iteration is what makes human-in-the-loop
a state transition instead of a held connection — and it is the same property that lets a run
resume after a deploy, and lets a failed run be inspected after the fact.

The transcript is stored verbatim, in the provider's message shape. Storing a summary instead
would be cheaper and would make the run unresumable: the next call needs the exact tool-call ids
it answered, not a description of them.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from agentdesk.agent.budgets import BudgetLedger
from agentdesk.schemas import Budgets, RunStatus, RunView


@dataclass
class AgentState:
    """One run: what was said, what it has spent, and what it is waiting for."""

    run_id: str
    messages: list[dict[str, Any]]
    ledger: BudgetLedger
    customer_email: str = ""
    status: RunStatus = "running"
    answer: str | None = None
    # The proposed action a human must approve, kept whole. Re-deriving it from the transcript at
    # approval time would mean parsing tool calls again in a second place, with a second chance
    # of reading them differently from the loop that produced them.
    pending_action: dict[str, Any] | None = None
    stop_reason: str = ""
    # Every tool the run has invoked, in order. The trajectory evals score this, not the answer:
    # an agent that refunds first and looks up the order afterwards got the right answer wrong.
    trajectory: list[str] = field(default_factory=list)

    def view(self) -> RunView:
        return RunView(
            id=self.run_id,
            status=self.status,
            answer=self.answer,
            pending_action=self.pending_action,
            stop_reason=self.stop_reason,
            iterations=self.ledger.iterations,
            tokens=self.ledger.tokens,
            cost_usd=round(self.ledger.cost_usd, 6),
            cost_is_partial=self.ledger.cost_is_partial,
        )

    def to_row(self) -> dict[str, Any]:
        """The state as database columns.

        Counters are columns rather than JSON fields: "which runs blew their budget this week"
        is a question worth being able to ask in SQL.
        """
        return {
            "id": self.run_id,
            "status": self.status,
            "customer_email": self.customer_email,
            "messages": json.dumps(self.messages),
            "answer": self.answer,
            "pending_action": json.dumps(self.pending_action) if self.pending_action else None,
            "stop_reason": self.stop_reason,
            "trajectory": json.dumps(self.trajectory),
            "iterations": self.ledger.iterations,
            "tokens": self.ledger.tokens,
            "cost_usd": self.ledger.cost_usd,
            "cost_is_partial": self.ledger.cost_is_partial,
            "budgets": self.ledger.budgets.model_dump_json(),
            "elapsed_s": self.ledger.elapsed_s(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AgentState":
        """Rebuild a run from storage.

        The wall-clock budget restarts here. That is intentional: a run suspended overnight for
        an approval has spent no compute waiting, and charging it the elapsed hours would kill it
        on resume for something it did not do. The three cumulative budgets do carry over.
        """
        ledger = BudgetLedger(budgets=Budgets.model_validate_json(row["budgets"]))
        ledger.iterations = row["iterations"]
        ledger.tokens = row["tokens"]
        ledger.cost_usd = row["cost_usd"]
        ledger.cost_is_partial = row["cost_is_partial"]

        pending = row["pending_action"]
        trajectory = row["trajectory"]
        return cls(
            run_id=row["id"],
            messages=json.loads(row["messages"]),
            ledger=ledger,
            customer_email=row["customer_email"],
            status=row["status"],
            answer=row["answer"],
            pending_action=json.loads(pending) if pending else None,
            stop_reason=row["stop_reason"],
            trajectory=json.loads(trajectory) if trajectory else [],
        )
