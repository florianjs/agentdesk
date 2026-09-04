"""The graph engine's own seams, tested without a model, a database or a network.

What is worth pinning here is everything that translates between the framework's world and this
codebase's: the message shapes, the error shapes, and the counters. A migration goes wrong in
the translation, not in the graph.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from agentdesk.agent.tools import REGISTRY, ProposeRefund, ToolContext
from agentdesk.graph import (
    State,
    build_tools,
    ledger_of,
    psycopg_url,
    structured_error,
    to_agent_state,
    to_plain,
)
from agentdesk.schemas import Budgets


def test_validation_errors_keep_the_native_engine_s_shape() -> None:
    """Both engines must hand the model the same thing, or one of them is measuring the other."""
    try:
        ProposeRefund(order_id="A-1", amount_eur=10_000, reason="x")
    except ValidationError as error:
        payload = json.loads(structured_error(error))
    assert payload["error"] == "invalid_arguments"
    assert any(detail["loc"] == ["amount_eur"] for detail in payload["details"])


def test_a_failing_tool_is_reported_as_retryable() -> None:
    payload = json.loads(structured_error(ConnectionError("order service unavailable")))
    assert payload == {"error": "tool_failed", "retryable": True}


def test_the_tools_reuse_the_same_schemas_so_the_cap_cannot_drift() -> None:
    tools = {tool.name: tool for tool in build_tools(ToolContext(run_id="r"))}
    assert set(tools) == set(REGISTRY)
    assert tools["propose_refund"].args_schema is ProposeRefund


def test_an_assistant_turn_survives_the_translation_with_its_call_ids() -> None:
    """The next request answers tool calls by id; losing them makes the run unresumable."""
    message = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "get_order", "args": {"order_id": "A-1001"}}],
    )
    plain = to_plain(message)
    assert plain["role"] == "assistant"
    assert plain["tool_calls"][0]["id"] == "c1"
    assert json.loads(plain["tool_calls"][0]["function"]["arguments"]) == {"order_id": "A-1001"}


def test_the_other_message_kinds_map_to_the_transport_roles() -> None:
    assert to_plain(SystemMessage("rules"))["role"] == "system"
    assert to_plain(HumanMessage("hello"))["role"] == "user"
    tool = to_plain(ToolMessage(content="{}", tool_call_id="c9"))
    assert (tool["role"], tool["tool_call_id"]) == ("tool", "c9")


def test_the_counters_rebuild_into_the_same_ledger_both_engines_use() -> None:
    ledger = ledger_of(
        State(
            budgets=Budgets(max_iterations=3).model_dump(),
            started_at=0.0,
            iterations=3,
            tokens=120,
            cost_usd=0.02,
            cost_is_partial=True,
        )
    )
    assert ledger.exceeded() == "max_iterations (3)"
    assert (ledger.tokens, ledger.cost_usd, ledger.cost_is_partial) == (120, 0.02, True)


def test_an_interrupted_graph_reads_as_awaiting_approval() -> None:
    """LangGraph reports the suspension out of band; the API's status must still say it."""
    values = {
        "messages": [SystemMessage("rules"), HumanMessage("refund please")],
        "budgets": Budgets().model_dump(),
        "started_at": 0.0,
        "iterations": 2,
        "tokens": 50,
        "cost_usd": 0.01,
        "status": "running",
        "pending_action": {"tool": "propose_refund", "amount_eur": 30.0},
        "trajectory": ["get_order", "propose_refund"],
    }
    state = to_agent_state("r1", values, interrupted=True)
    assert state.status == "awaiting_approval"
    assert state.view().pending_action == {"tool": "propose_refund", "amount_eur": 30.0}
    assert state.trajectory == ["get_order", "propose_refund"]


def test_the_checkpointer_gets_a_url_its_own_driver_understands() -> None:
    """The checkpointer speaks psycopg; the rest of the service speaks asyncpg."""
    assert psycopg_url().startswith("postgresql://")
    assert "+asyncpg" not in psycopg_url()
