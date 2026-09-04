"""Persistence is what makes human-in-the-loop possible, so the round trip is tested."""

from fakes import ScriptedModel, completion, tool_call

from agentdesk.agent.loop import new_run, run_loop
from agentdesk.agent.state import AgentState
from agentdesk.schemas import Budgets


async def test_a_suspended_run_survives_a_round_trip_through_storage() -> None:
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call(
                    "propose_refund",
                    {"order_id": "A-1003", "amount_eur": 89.5, "reason": "lost"},
                )
            ],
            cost=0.002,
        ),
        completion(content="Approved — you'll see it on your card."),
    )
    original = await run_loop(
        new_run("where is my parcel", customer_email="a@b.c"), call_model=model
    )

    restored = AgentState.from_row(original.to_row())

    assert restored.status == "awaiting_approval"
    assert restored.pending_action == original.pending_action
    assert restored.messages == original.messages
    assert restored.trajectory == ["propose_refund"]
    assert restored.customer_email == "a@b.c"
    # The cumulative counters carry over: a resumed run does not get a fresh allowance.
    assert (restored.ledger.iterations, restored.ledger.tokens) == (1, 120)
    assert restored.ledger.cost_usd == 0.002
    assert restored.ledger.budgets == original.ledger.budgets


async def test_the_wall_clock_restarts_on_resume() -> None:
    """A run suspended overnight spent no compute waiting; the hours are not its to pay."""
    state = new_run("hello", budgets=Budgets(max_wall_seconds=30.0))
    state.ledger.started_at = state.ledger.now() - 3600

    restored = AgentState.from_row(state.to_row())

    assert restored.ledger.elapsed_s() < 1.0
    assert restored.ledger.exceeded() is None


async def test_the_transcript_is_stored_verbatim() -> None:
    """Tool-call ids must survive: the next request answers them by id, not by description."""
    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"}, call_id="c9")]),
        completion(content="delivered"),
    )
    state = await run_loop(new_run("A-1001?"), call_model=model)

    restored = AgentState.from_row(state.to_row())
    assistant = next(m for m in restored.messages if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "c9"
    assert any(m.get("tool_call_id") == "c9" for m in restored.messages)
