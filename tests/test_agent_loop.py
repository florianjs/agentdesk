"""What the loop must do at its edges, with a scripted model in place of a provider."""

import json
from typing import Any

import openai
import pytest
from fakes import ScriptedModel, completion, tool_call

from agentdesk.agent.loop import HANDOVER, new_run, resume_after_decision, run_loop
from agentdesk.agent.state import AgentState
from agentdesk.schemas import Budgets


def tool_results(state: AgentState) -> list[dict[str, Any]]:
    return [json.loads(m["content"]) for m in state.messages if m["role"] == "tool"]


async def test_answering_without_tools_ends_the_run() -> None:
    model = ScriptedModel(completion(content="Our returns window is 30 days."))
    state = await run_loop(new_run("What is your returns policy?"), call_model=model)

    assert state.status == "answered"
    assert state.answer == "Our returns window is 30 days."
    assert state.ledger.iterations == 1


async def test_tool_result_is_fed_back_and_the_loop_continues() -> None:
    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"})]),
        completion(content="Order A-1001 was delivered three days ago."),
    )
    state = await run_loop(new_run("Where is order A-1001?"), call_model=model)

    assert state.status == "answered"
    assert state.trajectory == ["get_order"]
    assert tool_results(state)[0] == {
        "found": True,
        "order_id": "A-1001",
        "status": "delivered",
        "amount_eur": 49.0,
        "refunded": False,
        "days_ago": 3,
    }
    # The second call must carry the whole transcript: the model cannot see the tool result
    # otherwise, and an agent that forgets its own last step loops forever.
    assert [m["role"] for m in model.calls[1]] == ["system", "user", "assistant", "tool"]


async def test_a_failing_tool_comes_back_as_data_not_an_exception() -> None:
    """A-9999 always raises. The model must get something it can read and route around."""
    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-9999"})]),
        completion(content="I couldn't reach the order system — let me get a colleague."),
    )
    state = await run_loop(new_run("Check order A-9999"), call_model=model)

    assert state.status == "answered"
    assert tool_results(state)[0] == {
        "error": "tool_failed",
        "tool": "get_order",
        "retryable": True,
    }


async def test_invalid_arguments_are_returned_with_their_violations() -> None:
    """The cap is structural: over 500 EUR never reaches the handler."""
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call(
                    "propose_refund",
                    {"order_id": "A-1001", "amount_eur": 10_000, "reason": "customer insists"},
                )
            ]
        ),
        completion(content="That amount is above what I can propose."),
    )
    state = await run_loop(new_run("Refund me 10000 euros"), call_model=model)

    result = tool_results(state)[0]
    assert result["error"] == "invalid_arguments"
    assert any(detail["loc"] == ["amount_eur"] for detail in result["details"])
    # No approval was requested: the call never became a valid proposal.
    assert state.status == "answered"
    assert state.pending_action is None


async def test_an_unknown_tool_lists_the_real_ones() -> None:
    model = ScriptedModel(
        completion(tool_calls=[tool_call("issue_refund", {"order_id": "A-1001"})]),
        completion(content="Let me use the right tool."),
    )
    state = await run_loop(new_run("refund"), call_model=model)

    result = tool_results(state)[0]
    assert result["error"] == "unknown_tool"
    assert "propose_refund" in result["available"]


async def test_a_refund_proposal_suspends_the_run() -> None:
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call(
                    "propose_refund",
                    {"order_id": "A-1003", "amount_eur": 89.5, "reason": "lost in transit"},
                )
            ]
        )
    )
    state = await run_loop(new_run("My parcel never arrived"), call_model=model)

    assert state.status == "awaiting_approval"
    assert state.pending_action == {
        "tool": "propose_refund",
        "status": "awaiting_approval",
        "order_id": "A-1003",
        "amount_eur": 89.5,
        "reason": "lost in transit",
        "note": "A human must approve this before any money moves. Tell the customer that.",
    }
    # Suspended, not finished: no answer was invented on the customer's behalf.
    assert state.answer is None
    # And the model was not called again — the script would have raised.
    assert len(model.calls) == 1


async def test_a_suspending_batch_still_answers_every_tool_call() -> None:
    """Providers reject a transcript with an unanswered tool call; such a run cannot resume."""
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call("get_order", {"order_id": "A-1001"}, call_id="c1"),
                tool_call(
                    "propose_refund",
                    {"order_id": "A-1001", "amount_eur": 49.0, "reason": "damaged"},
                    call_id="c2",
                ),
            ]
        )
    )
    state = await run_loop(new_run("It arrived damaged"), call_model=model)

    assert state.status == "awaiting_approval"
    assert [m["tool_call_id"] for m in state.messages if m["role"] == "tool"] == ["c1", "c2"]
    assert state.trajectory == ["get_order", "propose_refund"]


async def test_approval_resumes_the_run_and_clears_the_pending_action() -> None:
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call(
                    "propose_refund",
                    {"order_id": "A-1003", "amount_eur": 89.5, "reason": "lost"},
                )
            ]
        ),
        completion(content="Good news — the refund was approved."),
    )
    state = await run_loop(new_run("Where is my parcel?"), call_model=model)
    state = await resume_after_decision(
        state, approved=True, note="within policy", call_model=model
    )

    assert state.status == "answered"
    assert state.pending_action is None
    assert "APPROVED" in state.messages[-2]["content"]
    assert "within policy" in state.messages[-2]["content"]


async def test_rejection_resumes_too_and_says_so() -> None:
    model = ScriptedModel(
        completion(
            tool_calls=[
                tool_call(
                    "propose_refund", {"order_id": "A-1005", "amount_eur": 15.0, "reason": "late"}
                )
            ]
        ),
        completion(content="I'm sorry — this one falls outside our refund window."),
    )
    state = await run_loop(new_run("Refund my order from last year"), call_model=model)
    state = await resume_after_decision(state, approved=False, call_model=model)

    assert state.status == "answered"
    assert "REJECTED" in state.messages[-2]["content"]


async def test_resuming_a_run_that_is_not_suspended_is_refused() -> None:
    model = ScriptedModel(completion(content="done"))
    state = await run_loop(new_run("hello"), call_model=model)

    with pytest.raises(ValueError, match="awaiting_approval"):
        await resume_after_decision(state, approved=True, call_model=model)


async def test_escalation_is_a_terminal_status_with_an_answer() -> None:
    model = ScriptedModel(
        completion(tool_calls=[tool_call("escalate", {"reason": "customer asked for a human"})])
    )
    state = await run_loop(new_run("I want to speak to a human"), call_model=model)

    assert state.status == "escalated"
    assert state.answer
    assert len(model.calls) == 1


async def test_an_upstream_failure_records_the_run_instead_of_raising() -> None:
    model = ScriptedModel(openai.APITimeoutError(request=None))  # type: ignore[arg-type]
    state = await run_loop(new_run("hello"), call_model=model)

    assert state.status == "failed"
    assert state.stop_reason.startswith("APITimeoutError")
    assert state.answer == HANDOVER


async def test_a_provider_error_inside_a_200_is_treated_as_a_failure() -> None:
    """`finish_reason: "error"` with no content: an HTTP 200 that carries a failure."""
    model = ScriptedModel(completion(content=None, finish_reason="error"))
    state = await run_loop(new_run("hello"), call_model=model)

    assert state.status == "failed"
    assert "TransientUpstreamError" in state.stop_reason


async def test_the_checkpoint_runs_after_every_iteration() -> None:
    saved: list[tuple[str, int]] = []

    async def checkpoint(state: AgentState) -> None:
        saved.append((state.status, state.ledger.iterations))

    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"})]),
        completion(content="Delivered three days ago."),
    )
    await run_loop(new_run("where is A-1001"), call_model=model, checkpoint=checkpoint)

    # One save per iteration, plus the terminal one: a crash costs a step, not the run.
    assert saved == [("running", 1), ("answered", 2)]


async def test_a_missing_usage_block_marks_the_cost_partial() -> None:
    """A run whose cost could not be measured must not report 0.0 as if it were free."""
    model = ScriptedModel(completion(content="hi", cost=None))
    state = await run_loop(new_run("hi"), call_model=model)

    assert state.ledger.cost_is_partial is True
    assert state.view().cost_usd == 0.0


async def test_iteration_budget_wraps_up_without_another_model_call() -> None:
    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"})]),
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1002"})]),
    )
    state = new_run("loop forever", budgets=Budgets(max_iterations=2))
    state = await run_loop(state, call_model=model)

    assert state.status == "budget_exceeded"
    assert state.stop_reason == "max_iterations (2)"
    assert state.answer == HANDOVER
    # The wrap-up spends nothing: writing a goodbye with the model would be one more call made
    # after the ceiling was declared reached.
    assert len(model.calls) == 2


async def test_token_budget_stops_the_run() -> None:
    model = ScriptedModel(
        completion(
            tool_calls=[tool_call("get_order", {"order_id": "A-1001"})],
            prompt_tokens=5_000,
            completion_tokens=100,
        ),
        completion(content="unreachable"),
    )
    state = new_run("long", budgets=Budgets(max_tokens_total=1_000))
    state = await run_loop(state, call_model=model)

    assert state.status == "budget_exceeded"
    assert state.stop_reason == "max_tokens_total (1000)"
    assert len(model.calls) == 1


async def test_cost_budget_stops_the_run() -> None:
    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"})], cost=0.40),
        completion(content="unreachable"),
    )
    state = new_run("expensive", budgets=Budgets(max_cost_usd=0.10))
    state = await run_loop(state, call_model=model)

    assert state.status == "budget_exceeded"
    assert state.stop_reason == "max_cost_usd (0.1)"


async def test_wall_clock_budget_stops_the_run() -> None:
    """The one budget the other three cannot see: an upstream call that simply hangs."""
    clock = [0.0]
    state = new_run("slow", budgets=Budgets(max_wall_seconds=30.0))
    state.ledger.now = lambda: clock[0]
    state.ledger.started_at = 0.0

    async def slow_upstream(_: AgentState) -> None:
        """The first call took 99 seconds — no counter but the clock notices."""
        clock[0] = 99.0

    model = ScriptedModel(
        completion(tool_calls=[tool_call("get_order", {"order_id": "A-1001"})]),
        completion(content="unreachable"),
    )
    state = await run_loop(state, call_model=model, checkpoint=slow_upstream)

    assert state.status == "budget_exceeded"
    assert state.stop_reason == "max_wall_seconds (30.0)"
    assert len(model.calls) == 1
