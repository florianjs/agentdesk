"""The run API: start one, look at it, decide on it.

Four endpoints, and their shape is the point. Starting a run and approving its action are
separate requests because the human in the middle takes minutes or hours to answer, and a design
that holds a connection open for that is a design that cannot survive a deploy.

`POST /runs` therefore returns whatever the run reached — an answer, or a suspension with the
proposed action attached — and never waits for a person.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from agentdesk import graph
from agentdesk.agent.loop import Checkpoint, new_run, resume_after_decision, run_loop
from agentdesk.agent.model import model_call
from agentdesk.agent.state import AgentState
from agentdesk.db import load_run, save_run
from agentdesk.deps import LLM, ApiKey, Session
from agentdesk.graph import get_checkpointer
from agentdesk.schemas import Approval, RunRequest, RunView

router = APIRouter(tags=["runs"])
log = logging.getLogger("agentdesk.api")


def _checkpoint(session: Session) -> Checkpoint:
    async def save(state: AgentState) -> None:
        await save_run(session, state)

    return save


def _log(state: AgentState) -> None:
    """One line per run, carrying the trajectory.

    The tools an agent called, in order, is the field that makes a bad run diagnosable: an answer
    that looks right after skipping the lookup is the failure mode worth being able to see.
    """
    log.info(
        "run %s %s after %d iterations, %d tokens, $%.4f (%s)%s",
        state.run_id,
        state.status,
        state.ledger.iterations,
        state.ledger.tokens,
        state.ledger.cost_usd,
        " → ".join(state.trajectory) or "no tools",
        f" — {state.stop_reason}" if state.stop_reason else "",
    )


@router.post("/runs", response_model=RunView, summary="Start a run")
async def start(
    request: RunRequest,
    session: Session,
    llm: LLM,
    _: ApiKey,
    engine: Annotated[
        Literal["native", "graph"],
        Query(description="The hand-written loop, or the LangGraph one. Same behaviour."),
    ] = "native",
) -> RunView:
    """Run the agent until it answers, suspends for approval, escalates, or runs out of budget.

    A suspended run is a 200, not an error: it is the system working as designed. The caller
    tells the two apart by `status`, and sees exactly what it would be approving in
    `pending_action`.

    Both engines write a row to `runs` whatever their persistence. The LangGraph checkpoint is
    what makes a run resumable, but it is opaque blobs keyed by thread id — "which runs are
    waiting for a human" is a business question, and it needs a table you can query.
    """
    if engine == "graph":
        state = await graph.start(
            request.message,
            run_id=str(uuid.uuid4()),
            budgets=request.budgets,
            customer_email=request.customer_email,
            checkpointer=await get_checkpointer(),
        )
        state.engine = "graph"
        state.customer_email = request.customer_email
        await save_run(session, state)
    else:
        state = new_run(
            request.message, customer_email=request.customer_email, budgets=request.budgets
        )
        state = await run_loop(state, call_model=model_call(llm), checkpoint=_checkpoint(session))
    _log(state)
    return state.view()


@router.post("/runs/stream", summary="Start a run, streamed step by step")
async def start_streaming(request: RunRequest, session: Session, _: ApiKey) -> StreamingResponse:
    """Server-sent events: one per token, plus one when a tool starts and when it ends.

    The tool events are the reason this endpoint exists. Streaming tokens alone leaves the
    customer watching nothing for the seconds an agent spends looking things up — the useful
    signal during that time is "searching the documentation", not an empty buffer.

    Graph engine only: the events come from the framework's own event stream.
    """
    run_id = str(uuid.uuid4())
    checkpointer = await get_checkpointer()

    async def events() -> AsyncIterator[str]:
        final: dict[str, Any] | None = None
        async for event in graph.stream(
            request.message,
            run_id=run_id,
            checkpointer=checkpointer,
            budgets=request.budgets,
            customer_email=request.customer_email,
        ):
            if event["type"] == "done":
                final = event["run"]
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        # The row goes in after the stream, not during: a half-finished run in the approval
        # queue is a run a human could open and find nothing to decide on.
        if final is not None:
            state = await graph.state_of(run_id, checkpointer)
            state.engine = "graph"
            state.customer_email = request.customer_email
            await save_run(session, state)
            log.info("run %s %s (streamed)", run_id, state.status)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/runs/{run_id}", response_model=RunView, summary="Look at a run")
async def read(run_id: str, session: Session, _: ApiKey) -> RunView:
    return (await _fetch(session, run_id)).view()


@router.post("/runs/{run_id}/approve", response_model=RunView, summary="Approve the pending action")
async def approve(
    run_id: str, decision: Approval, session: Session, llm: LLM, _: ApiKey
) -> RunView:
    return await _decide(run_id, decision, session, llm, approved=True)


@router.post("/runs/{run_id}/reject", response_model=RunView, summary="Reject the pending action")
async def reject(run_id: str, decision: Approval, session: Session, llm: LLM, _: ApiKey) -> RunView:
    """Rejecting is not cancelling: the run resumes and tells the customer what was decided.

    A rejected proposal that ends in silence leaves the person who asked for a refund with no
    answer at all, which is a worse outcome than the refusal itself.
    """
    return await _decide(run_id, decision, session, llm, approved=False)


async def _fetch(session: Session, run_id: str) -> AgentState:
    state = await load_run(session, run_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such run")
    return state


async def _decide(
    run_id: str, decision: Approval, session: Session, llm: AsyncOpenAI, *, approved: bool
) -> RunView:
    state = await _fetch(session, run_id)
    if state.status != "awaiting_approval":
        # 409, not 400: the request is well-formed and was valid a moment ago. This is what a
        # second reviewer clicking approve on an already-decided run gets, and the status tells
        # them which way it went.
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"run is {state.status}, nothing is awaiting approval"
        )

    log.info("run %s %s by a human", run_id, "approved" if approved else "rejected")
    if state.engine == "graph":
        # Nothing about the run is passed back in: the checkpointer holds the transcript, the
        # counters and the position in the graph.
        resumed = await graph.resume(
            run_id=run_id,
            approved=approved,
            note=decision.note,
            customer_email=state.customer_email,
            checkpointer=await get_checkpointer(),
        )
        resumed.engine = "graph"
        resumed.customer_email = state.customer_email
        state = resumed
        await save_run(session, state)
    else:
        state = await resume_after_decision(
            state,
            approved=approved,
            note=decision.note,
            call_model=model_call(llm),
            checkpoint=_checkpoint(session),
        )
    _log(state)
    return state.view()
