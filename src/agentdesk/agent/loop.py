"""The agent loop.

An agent is a while loop over a model that can call tools. That is the whole idea, and it fits
below — the framework versions of this are the same twenty lines with a graph API around them.

What is worth writing carefully is not the loop but its edges:

- **The budget is checked before the call, never after.** A check that runs afterwards has
  already paid for the thing it was meant to prevent.
- **A blown budget wraps up; it does not raise.** The customer gets a handover message and the
  run is recorded as `budget_exceeded`. What it must not do is spend one more model call to
  write a polite goodbye — that is a request made *after* the ceiling was declared reached.
- **Every tool call gets a result message, including on the turn the run suspends.** Providers
  reject a transcript with an unanswered tool call, so a run that stopped mid-batch would be
  unresumable.
- **A tool failure is data.** It comes back as a result the model can read and route around.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from agentdesk.agent.budgets import BudgetLedger
from agentdesk.agent.prompts import CURRENT
from agentdesk.agent.state import AgentState
from agentdesk.agent.tools import (
    REQUIRES_APPROVAL,
    ToolContext,
    execute_tool,
    proposed_total,
    tool_schemas,
)
from agentdesk.llm.client import read_usage
from agentdesk.llm.tools import raise_if_upstream_error
from agentdesk.schemas import Budgets

log = logging.getLogger("agentdesk.agent")

# Called with the running transcript, returns a chat completion. Injected rather than imported
# so the loop can be tested against a scripted model — every branch below is reachable in a unit
# test without a network call, which is why they are all actually tested.
type ModelCall = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[Any]]

# Called after every iteration, with the state as it stands. In the API this writes to Postgres.
type Checkpoint = Callable[[AgentState], Awaitable[None]]

HANDOVER = (
    "I'm not able to finish this myself right now, so I'm passing it to a colleague who will "
    "pick it up. Sorry for the delay."
)


async def _no_checkpoint(state: AgentState) -> None:
    """Default persistence: none. Scripts and tests run without a database."""


def new_run(
    message: str, *, customer_email: str = "", budgets: Budgets | None = None
) -> AgentState:
    """A fresh run, with the opening transcript and an empty ledger."""
    return AgentState(
        run_id=str(uuid.uuid4()),
        messages=[
            {"role": "system", "content": CURRENT},
            {"role": "user", "content": message},
        ],
        ledger=BudgetLedger(budgets=budgets or Budgets()),
        customer_email=customer_email,
    )


def _assistant_message(message: Any) -> dict[str, Any]:
    """The model's turn, as a plain dict.

    Converted out of the SDK's objects immediately: the transcript is written to a database and
    read back by a different process, so it must be data all the way down.
    """
    calls = message.tool_calls or []
    turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if calls:
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in calls
        ]
    return turn


async def run_loop(
    state: AgentState,
    *,
    call_model: ModelCall,
    checkpoint: Checkpoint = _no_checkpoint,
) -> AgentState:
    """Advance a run until it answers, suspends, escalates, fails, or runs out of budget.

    Returns the same state object, mutated. Callers persist it through `checkpoint`, which runs
    after every iteration — a crash between two iterations then loses one step, not the run.
    """
    schemas = tool_schemas()
    context = ToolContext(run_id=state.run_id, customer_email=state.customer_email)

    while state.status == "running":
        spent = state.ledger.exceeded()
        if spent:
            state.status = "budget_exceeded"
            state.stop_reason = spent
            state.answer = HANDOVER
            log.warning("run %s stopped: %s", state.run_id, spent)
            break

        try:
            response = await call_model(state.messages, schemas)
            raise_if_upstream_error(response, "agent")
        except Exception as error:
            # The run is over, but it is over in a recorded way: a failed run keeps its
            # transcript and its counters, which is what makes it debuggable afterwards.
            log.exception("run %s failed", state.run_id)
            state.status = "failed"
            state.stop_reason = f"{type(error).__name__}: {error}"
            state.answer = HANDOVER
            break

        state.ledger.record(read_usage(response))
        message = response.choices[0].message
        state.messages.append(_assistant_message(message))

        calls = message.tool_calls or []
        if not calls:
            state.status = "answered"
            state.answer = message.content or ""
            break

        # Every call in the batch is answered before any of them can end the turn: a transcript
        # holding a tool call without its result is rejected by the provider on the next request,
        # so suspending mid-batch would leave a run that can never be resumed.
        suspend_on: dict[str, Any] | None = None
        escalated = False
        # Re-read from the transcript before each batch: a resumed run must not get a fresh
        # refund allowance because a human approved the last one.
        context.proposed_eur = proposed_total(state.messages)

        for call in calls:
            name = call.function.name
            state.trajectory.append(name)
            result = await execute_tool(name, call.function.arguments, context)
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            if name in REQUIRES_APPROVAL and "error" not in result and suspend_on is None:
                suspend_on = {"tool": name, **result}
            if name == "escalate" and "error" not in result:
                escalated = True

        if suspend_on is not None:
            # Approval wins over escalation: a proposed refund is a decision a human must take,
            # and dropping it into a queue as a plain escalation loses the amount and the reason.
            state.status = "awaiting_approval"
            state.pending_action = suspend_on
        elif escalated:
            state.status = "escalated"
            state.answer = (
                "I've passed this to a colleague who can help further — they'll follow up with you."
            )

        await checkpoint(state)

    await checkpoint(state)
    return state


async def resume_after_decision(
    state: AgentState,
    *,
    approved: bool,
    note: str = "",
    call_model: ModelCall,
    checkpoint: Checkpoint = _no_checkpoint,
) -> AgentState:
    """Continue a suspended run once a human has decided.

    The verdict enters the transcript as a system message, not as a tool result: the tool already
    answered ("awaiting_approval") on the turn it was called, and rewriting that answer after the
    fact would make the stored transcript a version of events that never happened.
    """
    if state.status != "awaiting_approval":
        raise ValueError(f"run {state.run_id} is {state.status}, not awaiting_approval")

    # Worded against the transcript, not in a vacuum. The tool result above still reads
    # "awaiting_approval" — that is what was true when it ran — and a mild "the reviewer
    # approved it" loses to it: the first version of this message produced an answer telling an
    # approved customer their refund still needed review. The instruction now contradicts the
    # stale state explicitly.
    verdict = (
        "APPROVED. The action has been carried out and is no longer pending. The earlier tool "
        "result saying 'awaiting_approval' is out of date"
        if approved
        else "REJECTED. The action will not be carried out"
    )
    state.messages.append(
        {
            "role": "system",
            "content": (
                f"A human reviewer has reviewed the pending action: {verdict}"
                + (f'. Reviewer note: "{note}"' if note else "")
                + ". Tell the customer what was decided, as settled fact, in your own words. Do "
                "not say it is awaiting review and do not propose the same action again."
            ),
        }
    )
    state.pending_action = None
    state.status = "running"
    return await run_loop(state, call_model=call_model, checkpoint=checkpoint)
