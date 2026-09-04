"""What the agent is allowed to do.

Each tool is a Pydantic model plus an async handler. The model is the schema the model sees, the
validation the arguments must pass, and — for anything that touches money — the enforcement
point. `amount_eur: float = Field(le=500)` is not a hint to the model: a refund above the cap
cannot be constructed, so no prompt can talk its way past it.

Two rules shape every tool here:

- **The model proposes, the code disposes.** `propose_refund` creates a request awaiting human
  approval. Nothing in this registry moves money.
- **A tool never raises into the loop.** Failures come back as structured results the model can
  read, because an error it can see is an error it can work around; an exception it cannot see
  ends the conversation.

Docstrings are prompt, not documentation: they are shipped to the model as the tool description
and decide whether it picks the right one.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from agentdesk.config import settings

log = logging.getLogger("agentdesk.tools")

type Handler = Callable[[BaseModel, "ToolContext"], Awaitable[dict[str, Any]]]


class ToolContext(BaseModel):
    """What a handler is allowed to know about the run it serves.

    `proposed_eur` is mutable state on purpose: it is the running total of refunds this run has
    already proposed, and it is what makes the cap a limit on the *run* rather than on one call.
    """

    run_id: str
    customer_email: str = ""
    proposed_eur: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class SearchDocs(BaseModel):
    """Search the product documentation. Use this before answering any question about how the
    product works, what a feature does, or what a policy says. Ask a standalone question — the
    search cannot see this conversation."""

    query: str = Field(min_length=3, max_length=300, description="A standalone question")


class GetOrder(BaseModel):
    """Look up one order: its status, amount, and whether it has already been refunded. Use this
    before discussing any specific order."""

    order_id: str = Field(min_length=1, max_length=64)


class ProposeRefund(BaseModel):
    """Propose a refund for an order. This does NOT refund anything: it creates a request that a
    human must approve. Say so to the customer — do not promise the money is on its way."""

    order_id: str = Field(min_length=1, max_length=64)
    # The per-call cap is enforced here, in the type: a model cannot construct a larger refund.
    # It is not sufficient on its own — see `handle_propose_refund`, where a weaker model was
    # measured proposing 3 x 400 EUR on a 49 EUR order, each call legal, the total absurd.
    amount_eur: float = Field(gt=0, le=500, description="Hard cap: 500 EUR per call")
    reason: str = Field(min_length=3, max_length=500)


class Escalate(BaseModel):
    """Hand the conversation to a human agent. Use this when the customer's request is outside
    what these tools can do, when they ask for a human, or when you have tried and failed."""

    reason: str = Field(min_length=3, max_length=500)


async def handle_search_docs(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    """Ask DocPilot. The suite answers its own questions."""
    assert isinstance(args, SearchDocs)
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f"{settings.docpilot_url}/collections/{settings.docpilot_collection}/search",
            params={"q": args.query, "limit": 3},
            headers={"X-API-Key": settings.docpilot_api_key},
        )
        response.raise_for_status()

    hits = response.json()["hits"]
    # Compact on purpose: every character returned here is spent from the agent's context on
    # every subsequent turn, because the whole transcript is resent each time.
    return {
        "results": [
            {
                "heading": hit["heading"],
                "excerpt": hit["content"][:400],
                "source": hit["source_url"],
            }
            for hit in hits
        ]
    }


async def handle_get_order(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    assert isinstance(args, GetOrder)
    from agentdesk.orders import get_order

    order = await get_order(args.order_id)
    if order is None:
        # Not an error: "no such order" is information the agent should act on, by asking the
        # customer to check the reference rather than by giving up.
        return {"found": False, "order_id": args.order_id}
    return {"found": True, **order}


MAX_RUN_REFUND_EUR = 500.0


def proposed_total(messages: list[dict[str, Any]]) -> float:
    """What this run has already had accepted as a refund proposal, read from the transcript.

    Derived rather than stored: the transcript is what survives a restart and an approval, so a
    counter kept anywhere else would reset exactly when the limit matters most — on the resumed
    run, after a human has already approved one payment.
    """
    total = 0.0
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") == "awaiting_approval":
            total += float(payload.get("amount_eur", 0.0))
    return total


async def handle_propose_refund(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    """Record a refund request. Approval, and execution, happen elsewhere.

    Two checks the schema cannot make, because both depend on state the type system never sees:

    - **the order's own value.** Refunding more than the customer paid is wrong whatever the cap
      says, and the amount lives in the order, not in the arguments;
    - **the running total for this run.** A per-call ceiling is defeated by asking three times.
      This was not hypothetical: a cheaper model, measured on the `adv-split-refund` scenario,
      proposed 3 x 400 EUR against a 49 EUR order — every call valid, the total 24x the price.

    Both come back as structured errors naming the real limit, so a model that hit them by
    mistake can correct, and one that hit them on purpose gets nowhere.
    """
    assert isinstance(args, ProposeRefund)

    from agentdesk.orders import get_order

    order = await get_order(args.order_id)
    if order is None:
        return {"error": "unknown_order", "order_id": args.order_id}

    paid = float(order["amount_eur"])
    if args.amount_eur > paid:
        return {
            "error": "exceeds_order_amount",
            "order_id": args.order_id,
            "order_amount_eur": paid,
            "requested_eur": args.amount_eur,
        }

    running = context.proposed_eur + args.amount_eur
    if running > min(paid, MAX_RUN_REFUND_EUR):
        return {
            "error": "exceeds_run_total",
            "already_proposed_eur": context.proposed_eur,
            "requested_eur": args.amount_eur,
            "limit_eur": min(paid, MAX_RUN_REFUND_EUR),
        }

    context.proposed_eur = running
    return {
        "status": "awaiting_approval",
        "order_id": args.order_id,
        "amount_eur": args.amount_eur,
        "reason": args.reason,
        "note": "A human must approve this before any money moves. Tell the customer that.",
    }


async def handle_escalate(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    assert isinstance(args, Escalate)
    return {"status": "escalated", "reason": args.reason}


REGISTRY: dict[str, tuple[type[BaseModel], Handler]] = {
    "search_docs": (SearchDocs, handle_search_docs),
    "get_order": (GetOrder, handle_get_order),
    "propose_refund": (ProposeRefund, handle_propose_refund),
    "escalate": (Escalate, handle_escalate),
}

# Tools whose result suspends the run for a human decision. Kept as data rather than a check
# inside a handler: which actions need approval is a policy, and policy belongs where it can be
# read at a glance.
REQUIRES_APPROVAL = frozenset({"propose_refund"})


def tool_schemas() -> list[dict[str, Any]]:
    """The registry, in the shape the model expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (model.__doc__ or "").strip(),
                "parameters": model.model_json_schema(),
            },
        }
        for name, (model, _) in REGISTRY.items()
    ]


async def execute_tool(name: str, raw_arguments: str, context: ToolContext) -> dict[str, Any]:
    """Run a tool and always return a result the model can read.

    Every failure mode becomes data: an unknown tool, arguments that do not validate, a handler
    that raises. The model sees what went wrong and can correct, work around it, or escalate —
    behaviour that costs nothing beyond treating errors as information.
    """
    entry = REGISTRY.get(name)
    if entry is None:
        return {"error": "unknown_tool", "available": sorted(REGISTRY)}

    model, handler = entry
    try:
        args = model.model_validate_json(raw_arguments)
    except ValidationError as error:
        # The violations go back verbatim: "amount_eur must be <= 500" is something the model can
        # act on, where "invalid arguments" leaves it guessing.
        return {"error": "invalid_arguments", "details": json.loads(error.json())}

    try:
        return await handler(args, context)
    except Exception:
        log.exception("tool %s failed", name)
        return {"error": "tool_failed", "tool": name, "retryable": True}
