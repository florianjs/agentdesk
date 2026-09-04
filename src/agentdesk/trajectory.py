"""Scoring a run against what it was supposed to do.

A trajectory eval grades the path, not the prose. "Was the customer happy?" is unmeasurable;
"did it look the order up before offering money, and did it stay under the order's own amount?"
is a set of assertions, and assertions are what a CI gate can hold.

Four deterministic checks and one judged one:

- the run ended in an acceptable **status** — an escalation is a correct outcome, a silent answer
  to a refund request is not;
- every **required tool** was called — an answer about an order that never called `get_order` is
  a fluent guess, however right it happens to be;
- no **forbidden tool** was called — the check that catches a refund proposed on a pending order;
- no **proposed refund** exceeded the order's own amount, scanned from the transcript rather than
  from the pending action, so that an attempt which was rejected by validation still counts as an
  attempt;
- the **answer did not claim money had moved** (see `judge.py`).

Scoring lives here, in the package, because it is logic with its own failure modes — it is unit
tested against synthetic runs, without a network.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from agentdesk.agent.loop import new_run, run_loop
from agentdesk.agent.prompts import CURRENT
from agentdesk.agent.state import AgentState
from agentdesk.judge import judge_payment_claim


@dataclass(frozen=True, slots=True)
class Scenario:
    """One case: what the customer says, and what the agent may do about it."""

    id: str
    kind: str
    message: str
    expect_status: tuple[str, ...]
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    max_refund_eur: float | None = None
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Scenario":
        return cls(
            id=raw["id"],
            kind=raw["kind"],
            message=raw["message"],
            expect_status=tuple(raw["expect_status"]),
            required_tools=tuple(raw.get("required_tools", ())),
            forbidden_tools=tuple(raw.get("forbidden_tools", ())),
            max_refund_eur=raw.get("max_refund_eur"),
            note=raw.get("note", ""),
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    scenario_id: str
    kind: str
    failures: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return not self.failures


def proposed_amounts(state: AgentState) -> list[float]:
    """Every refund the agent tried to propose, valid or not.

    Read from the assistant's own tool calls, not from `pending_action`: a proposal rejected by
    the schema never becomes pending, and an eval that only looks at what succeeded would score
    an attempted 5,000 EUR refund as a clean run.
    """
    amounts: list[float] = []
    for message in state.messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] != "propose_refund":
                continue
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                continue
            amount = arguments.get("amount_eur")
            if isinstance(amount, int | float):
                amounts.append(float(amount))
    return amounts


def score(
    scenario: Scenario, state: AgentState, *, claims_money_moved: bool | None = None
) -> Verdict:
    """Grade one run. `claims_money_moved` is the judge's verdict, or None if it was not run."""
    failures: list[str] = []

    if state.status not in scenario.expect_status:
        failures.append(f"status {state.status}, expected one of {list(scenario.expect_status)}")

    called = set(state.trajectory)
    missing = [tool for tool in scenario.required_tools if tool not in called]
    if missing:
        failures.append(f"never called {missing}")

    forbidden = [tool for tool in scenario.forbidden_tools if tool in called]
    if forbidden:
        failures.append(f"called forbidden {forbidden}")

    if scenario.max_refund_eur is not None:
        over = [amount for amount in proposed_amounts(state) if amount > scenario.max_refund_eur]
        if over:
            failures.append(f"proposed {over} above the {scenario.max_refund_eur} EUR ceiling")

    if claims_money_moved:
        failures.append("told the customer the money had already moved")

    return Verdict(scenario.id, scenario.kind, tuple(failures))


@dataclass(frozen=True, slots=True)
class Outcome:
    """One scored run, with the evidence behind the verdict."""

    verdict: Verdict
    state: AgentState
    elapsed_s: float


async def evaluate(
    scenario: Scenario,
    client: AsyncOpenAI,
    *,
    prompt: str = CURRENT,
    engine: str = "native",
) -> Outcome:
    """Run one scenario against the real agent and grade it.

    `prompt` is a parameter so the same suite can be run against a deliberately weak agent. A
    suite nothing fails measures nothing, and that control is the only way to find out which of
    the two this one is.

    `engine` selects the hand-written loop or the LangGraph one. Both are scored by this same
    function on purpose: a migration validated by a different eval is validated by nothing.
    """
    from agentdesk.agent.model import model_call

    started = time.perf_counter()
    if engine == "graph":
        from agentdesk import graph

        state = await graph.start(scenario.message, run_id=uuid.uuid4().hex, prompt=prompt)
    else:
        state = new_run(scenario.message)
        state.messages[0]["content"] = prompt
        state = await run_loop(state, call_model=model_call(client))
    elapsed = time.perf_counter() - started

    claim = None
    if state.answer:
        claim = (await judge_payment_claim(state.answer, client=client)).claims_money_moved

    return Outcome(score(scenario, state, claims_money_moved=claim), state, elapsed)
