"""The same agent, as a LangGraph state machine.

Everything the hand-written loop does has a name here, and that is the point of building it by
hand first — the mapping is one to one, so the framework took an afternoon instead of a month:

| `agent/loop.py`                        | LangGraph                                   |
| -------------------------------------- | ------------------------------------------- |
| `AgentState` written to `runs`         | `State` TypedDict + a checkpointer          |
| `execute_tool` with structured errors  | `ToolNode(handle_tool_errors=...)`          |
| the budget check before each call      | a conditional edge into a `wrap_up` node    |
| `status = "awaiting_approval"`         | `interrupt()` inside an `approval` node     |
| `runs.id`                              | `config.configurable.thread_id`             |
| `resume_after_decision`                | `Command(resume=...)`                       |

The graph keeps the hand-written engine's semantics exactly — tools all execute, the refund
result comes back as `awaiting_approval`, and only then does the run suspend — so the same
thirty scenarios score both engines. A migration that changes behaviour and passes its evals has
only proved the evals were loose.

Both engines stay in the repository. A comparison with no artefact is a comparison the next
person has to redo.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer, Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ValidationError

from agentdesk.agent.budgets import BudgetLedger
from agentdesk.agent.loop import HANDOVER
from agentdesk.agent.prompts import CURRENT
from agentdesk.agent.state import AgentState
from agentdesk.agent.tools import REGISTRY, REQUIRES_APPROVAL, ToolContext
from agentdesk.config import settings
from agentdesk.llm.client import BASE_URL, DEFAULT_TIMEOUT_S
from agentdesk.schemas import Budgets

log = logging.getLogger("agentdesk.graph")


class State(TypedDict, total=False):
    """The shared state. `add_messages` is a reducer: nodes return what they add, not the whole
    list, which is what stops two nodes from overwriting each other's turn."""

    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int
    tokens: int
    cost_usd: float
    cost_is_partial: bool
    started_at: float
    budgets: dict[str, Any]
    status: str
    stop_reason: str
    answer: str | None
    pending_action: dict[str, Any] | None
    trajectory: list[str]


def structured_error(error: Exception) -> str:
    """The hand-written engine's error shapes, kept identical.

    `ToolNode` catches the exception and hands it here; without this the model would receive a
    stack-trace string, which it cannot act on the way it acts on a list of field violations.
    """
    if isinstance(error, ValidationError):
        return json.dumps({"error": "invalid_arguments", "details": json.loads(error.json())})
    return json.dumps({"error": "tool_failed", "retryable": True})


def build_tools(context: ToolContext) -> list[BaseTool]:
    """The same registry, wrapped as LangChain tools.

    The Pydantic models are reused as `args_schema`, so the 500 EUR cap is enforced by the same
    class in both engines. A second definition of the cap would be a second thing to keep true.
    """
    tools: list[BaseTool] = []
    for name, (model, handler) in REGISTRY.items():

        def make(model: type[BaseModel] = model, handler: Any = handler) -> Any:
            async def call(**kwargs: Any) -> str:
                return json.dumps(await handler(model(**kwargs), context), ensure_ascii=False)

            return call

        tools.append(
            StructuredTool.from_function(
                coroutine=make(),
                name=name,
                description=(model.__doc__ or "").strip(),
                args_schema=model,
            )
        )
    return tools


def build_model(tools: list[BaseTool]) -> Any:
    """OpenRouter through LangChain, asked for the same cost accounting as the native client."""
    chat = ChatOpenAI(
        model=settings.model_smart,
        base_url=BASE_URL,
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        temperature=0,
        timeout=DEFAULT_TIMEOUT_S,
        max_retries=settings.retry_max_attempts,
        extra_body={"usage": {"include": True}},
    )
    return chat.bind_tools(tools)


def read_usage(message: BaseMessage) -> tuple[int, float | None]:
    """Tokens and credit cost off an AI message, tolerating what the integration does not map."""
    usage = getattr(message, "usage_metadata", None) or {}
    tokens = int(usage.get("total_tokens", 0) or 0)
    metadata = getattr(message, "response_metadata", None) or {}
    raw = metadata.get("usage") or metadata.get("token_usage") or {}
    cost = raw.get("cost") if isinstance(raw, dict) else None
    return tokens, float(cost) if cost is not None else None


def ledger_of(state: State) -> BudgetLedger:
    """The same ledger class both engines use, rebuilt from the graph's flat counters."""
    ledger = BudgetLedger(
        budgets=Budgets.model_validate(state["budgets"]), started_at=state["started_at"]
    )
    ledger.iterations = state.get("iterations", 0)
    ledger.tokens = state.get("tokens", 0)
    ledger.cost_usd = state.get("cost_usd", 0.0)
    ledger.cost_is_partial = state.get("cost_is_partial", False)
    return ledger


def build_graph(
    context: ToolContext, checkpointer: Checkpointer = None
) -> CompiledStateGraph[State]:
    """Four nodes: think, act, ask a human, give up cleanly."""
    tools = build_tools(context)
    model = build_model(tools)

    async def agent(state: State) -> dict[str, Any]:
        try:
            reply = await model.ainvoke(state["messages"])
        except Exception as error:
            # Without this the exception escapes the graph and the caller gets a traceback
            # instead of a run. The hand-written engine records `failed` and keeps the
            # transcript; matching that here is not a nicety, it is what makes a failed run
            # inspectable afterwards. Found the same way as the ToolNode default: in production
            # traffic, from a provider that returned 402 mid-suite.
            log.exception("graph run failed")
            return {
                "status": "failed",
                "stop_reason": f"{type(error).__name__}: {error}",
                "answer": HANDOVER,
            }

        tokens, cost = read_usage(reply)
        return {
            "messages": [reply],
            "iterations": state.get("iterations", 0) + 1,
            "tokens": state.get("tokens", 0) + tokens,
            "cost_usd": state.get("cost_usd", 0.0) + (cost or 0.0),
            "cost_is_partial": state.get("cost_is_partial", False) or cost is None,
            "answer": None if reply.tool_calls else str(reply.content),
            "status": "running" if reply.tool_calls else "answered",
            "trajectory": state.get("trajectory", []) + [call["name"] for call in reply.tool_calls],
        }

    async def approval(state: State) -> dict[str, Any]:
        """Suspend for a human.

        `interrupt` stops the graph here and stores everything; the resume value arrives as this
        function's return value on the next invocation, so the code reads as if the human had
        answered inline. That is the piece worth taking from the framework — the hand-written
        engine needed a second entry point to do it.
        """
        pending = state.get("pending_action")
        decision = interrupt(pending)

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        note = decision.get("note", "") if isinstance(decision, dict) else ""
        verdict = (
            "APPROVED. The action has been carried out and is no longer pending. The earlier "
            "tool result saying 'awaiting_approval' is out of date"
            if approved
            else "REJECTED. The action will not be carried out"
        )
        return {
            "status": "running",
            "pending_action": None,
            "messages": [
                SystemMessage(
                    f"A human reviewer has reviewed the pending action: {verdict}"
                    + (f'. Reviewer note: "{note}"' if note else "")
                    + ". Tell the customer what was decided, as settled fact, in your own words. "
                    "Do not say it is awaiting review and do not propose the same action again."
                )
            ],
        }

    async def wrap_up(state: State) -> dict[str, Any]:
        """The clean exit. It writes a fixed message rather than asking the model for a polite
        goodbye, which would be one more call made after the ceiling was declared reached."""
        return {
            "status": "budget_exceeded",
            "stop_reason": ledger_of(state).exceeded() or "budget",
            "answer": HANDOVER,
        }

    async def act(state: State) -> dict[str, Any]:
        """`ToolNode`, plus the one thing it does not know: which results need a human.

        Every tool in the batch runs before anything suspends — a transcript holding a tool call
        without its result is rejected by the provider on the next request.
        """
        # `handle_tool_errors` is not optional here: the default re-raises, which ends the run
        # on any tool failure. The hand-written engine cannot have that bug — its errors are
        # caught inside `execute_tool` by construction — while here correct behaviour is a
        # constructor keyword that is easy to leave out. It was, on the first run.
        result = await ToolNode(tools, handle_tool_errors=structured_error).ainvoke(state)
        messages = result["messages"]

        pending = None
        escalated = False
        last = state["messages"][-1]
        names = {call["id"]: call["name"] for call in getattr(last, "tool_calls", [])}
        for message in messages:
            name = names.get(message.tool_call_id, "")
            payload = json.loads(message.content) if message.content else {}
            if not isinstance(payload, dict) or "error" in payload:
                continue
            if name in REQUIRES_APPROVAL and pending is None:
                pending = {"tool": name, **payload}
            if name == "escalate":
                escalated = True

        update: dict[str, Any] = {"messages": messages}
        if pending is not None:
            # Approval wins over escalation: a proposed refund is a decision a human must take,
            # and filing it as a plain escalation loses the amount and the reason.
            update |= {"status": "awaiting_approval", "pending_action": pending}
        elif escalated:
            update |= {
                "status": "escalated",
                "answer": (
                    "I've passed this to a colleague who can help further — they'll follow up "
                    "with you."
                ),
            }
        return update

    def after_agent(state: State) -> str:
        return "act" if state["status"] == "running" else END

    def after_act(state: State) -> str:
        if state["status"] == "awaiting_approval":
            return "approval"
        if state["status"] == "escalated":
            return END
        return "wrap_up" if ledger_of(state).exceeded() else "agent"

    def after_approval(state: State) -> str:
        return "wrap_up" if ledger_of(state).exceeded() else "agent"

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("act", act)
    graph.add_node("approval", approval)
    graph.add_node("wrap_up", wrap_up)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["act", END])
    graph.add_conditional_edges("act", after_act, ["agent", "approval", "wrap_up", END])
    graph.add_conditional_edges("approval", after_approval, ["agent", "wrap_up"])
    graph.add_edge("wrap_up", END)

    return graph.compile(checkpointer=checkpointer)


def initial_state(message: str, budgets: Budgets, prompt: str = CURRENT) -> State:
    return State(
        messages=[SystemMessage(prompt), HumanMessage(message)],
        iterations=0,
        tokens=0,
        cost_usd=0.0,
        cost_is_partial=False,
        started_at=time.monotonic(),
        budgets=budgets.model_dump(),
        status="running",
        stop_reason="",
        answer=None,
        pending_action=None,
        trajectory=[],
    )


def to_plain(message: AnyMessage) -> dict[str, Any]:
    """A LangChain message as the transport dict the rest of the codebase speaks."""
    if isinstance(message, AIMessage):
        turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            turn["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                for call in message.tool_calls
            ]
        return turn
    role = {"system": "system", "human": "user", "tool": "tool"}.get(message.type, message.type)
    turn = {"role": role, "content": message.content}
    if role == "tool":
        turn["tool_call_id"] = getattr(message, "tool_call_id", "")
    return turn


def to_agent_state(run_id: str, values: dict[str, Any], interrupted: bool) -> AgentState:
    """The graph's result, in the shape the scorer and the API already understand.

    Without this the two engines could not be measured by the same suite, and a migration
    validated by a different eval is a migration validated by nothing.
    """
    ledger = ledger_of(cast(State, values))
    status = values.get("status", "running")
    if interrupted:
        status = "awaiting_approval"
    return AgentState(
        run_id=run_id,
        messages=[to_plain(message) for message in values["messages"]],
        ledger=ledger,
        status=status,
        answer=values.get("answer"),
        pending_action=values.get("pending_action"),
        stop_reason=values.get("stop_reason", ""),
        trajectory=list(values.get("trajectory", [])),
    )


async def start(
    message: str,
    *,
    run_id: str,
    budgets: Budgets | None = None,
    customer_email: str = "",
    checkpointer: Checkpointer = None,
    prompt: str = CURRENT,
) -> AgentState:
    """Run the graph until it answers, suspends, escalates, or gives up."""
    context = ToolContext(run_id=run_id, customer_email=customer_email)
    app = build_graph(context, checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = await app.ainvoke(initial_state(message, budgets or Budgets(), prompt), config)
    return to_agent_state(run_id, result, interrupted="__interrupt__" in result)


async def stream(
    message: str,
    *,
    run_id: str,
    checkpointer: Checkpointer,
    budgets: Budgets | None = None,
    customer_email: str = "",
) -> AsyncIterator[dict[str, Any]]:
    """The same run, reported step by step as it happens.

    This is the framework's clearest win over the hand-written engine. There, streaming meant
    surfacing tokens; the steps between them — "looking up the order", "searching the docs" —
    were invisible until the run finished, because nothing published them. `astream_events`
    publishes every node and every tool for free, so the customer sees progress during the eight
    seconds an agent takes rather than a spinner.

    A checkpointer is required: the final state is read back from it once the stream ends, which
    is also what makes the run resumable if it stopped for an approval.
    """
    context = ToolContext(run_id=run_id, customer_email=customer_email)
    app = build_graph(context, checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    async for event in app.astream_events(initial_state(message, budgets or Budgets()), config):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if text := str(chunk.content):
                yield {"type": "token", "text": text}
        elif kind == "on_tool_start":
            yield {"type": "tool_start", "tool": event["name"], "args": event["data"].get("input")}
        elif kind == "on_tool_end":
            yield {"type": "tool_end", "tool": event["name"]}

    snapshot = await app.aget_state(config)
    state = to_agent_state(run_id, dict(snapshot.values), interrupted=bool(snapshot.interrupts))
    yield {"type": "done", "run": state.view().model_dump()}


async def state_of(run_id: str, checkpointer: Checkpointer) -> AgentState:
    """Read a run back out of its checkpoint, without advancing it."""
    app = build_graph(ToolContext(run_id=run_id), checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    snapshot = await app.aget_state(config)
    return to_agent_state(run_id, dict(snapshot.values), interrupted=bool(snapshot.interrupts))


async def resume(
    *,
    run_id: str,
    approved: bool,
    note: str = "",
    customer_email: str = "",
    checkpointer: Checkpointer,
) -> AgentState:
    """Continue a suspended graph with the human's verdict.

    Nothing about the run is passed back in: the checkpointer holds the transcript, the counters
    and the position in the graph. That is the framework's strongest single feature — the
    hand-written engine needed a table, an upsert and a serialiser to reach the same place.
    """
    context = ToolContext(run_id=run_id, customer_email=customer_email)
    app = build_graph(context, checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    result = await app.ainvoke(Command(resume={"approved": approved, "note": note}), config)
    return to_agent_state(run_id, result, interrupted="__interrupt__" in result)


# The checkpointer, and its own connection pool. LangGraph's Postgres saver speaks psycopg; the
# rest of the service speaks asyncpg through SQLAlchemy. Two Postgres drivers in one process is
# a real cost of the framework, and it is here rather than hidden behind a helper so that it is
# visible when the trade-off is being weighed.
_pool: "AsyncConnectionPool[Any] | None" = None
_saver: AsyncPostgresSaver | None = None


def psycopg_url() -> str:
    """The same database, spelled the way psycopg expects."""
    return settings.database_url.replace("postgresql+asyncpg://", "postgresql://")


async def get_checkpointer() -> AsyncPostgresSaver:
    global _pool, _saver
    if _saver is None:
        _pool = AsyncConnectionPool(
            conninfo=psycopg_url(),
            max_size=5,
            open=False,
            # The saver writes outside an explicit transaction and expects dict rows; without
            # both, `setup()` reports a schema that is already there and then fails to use it.
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        await _pool.open()
        _saver = AsyncPostgresSaver(_pool)
        await _saver.setup()
    return _saver


async def close_checkpointer() -> None:
    global _pool, _saver
    if _pool is not None:
        await _pool.close()
    _pool, _saver = None, None


def diagram() -> str:
    """The graph as mermaid. One of the framework's quieter wins: the picture cannot drift from
    the code, because it is generated from it."""
    return build_graph(ToolContext(run_id="diagram")).get_graph().draw_mermaid()
