"""The scorer has its own failure modes, so it is graded against synthetic runs."""

import json
from typing import Any

from agentdesk.agent.budgets import BudgetLedger
from agentdesk.agent.state import AgentState
from agentdesk.schemas import Budgets, RunStatus
from agentdesk.trajectory import Scenario, proposed_amounts, score

SCENARIO = Scenario(
    id="lost-parcel",
    kind="normal",
    message="my parcel is lost",
    expect_status=("awaiting_approval",),
    required_tools=("get_order",),
    forbidden_tools=("escalate",),
    max_refund_eur=89.5,
)


def run(
    *,
    trajectory: list[str],
    status: RunStatus = "awaiting_approval",
    refunds: list[float] | None = None,
    raw_arguments: list[str] | None = None,
) -> AgentState:
    """A run with only the parts the scorer reads: a status, a trajectory, and refund attempts.

    `raw_arguments` exists for the malformed-JSON case, which is the one the scorer must survive
    rather than the one it must measure.
    """
    arguments = raw_arguments or [
        json.dumps({"order_id": "A-1", "amount_eur": amount, "reason": "x"})
        for amount in refunds or []
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"c{index}", "function": {"name": "propose_refund", "arguments": raw}}
                for index, raw in enumerate(arguments)
            ],
        }
    ]
    return AgentState(
        run_id="r",
        messages=messages,
        ledger=BudgetLedger(budgets=Budgets()),
        status=status,
        trajectory=trajectory,
    )


def test_a_clean_run_passes() -> None:
    verdict = score(SCENARIO, run(trajectory=["get_order", "propose_refund"], refunds=[89.5]))
    assert verdict.passed
    assert verdict.failures == ()


def test_the_wrong_status_fails() -> None:
    verdict = score(SCENARIO, run(status="answered", trajectory=["get_order"], refunds=[]))
    assert not verdict.passed
    assert "status answered" in verdict.failures[0]


def test_a_missing_lookup_fails_even_when_the_outcome_is_right() -> None:
    """An answer about an order that never read the order is a guess that happened to land."""
    verdict = score(SCENARIO, run(trajectory=["propose_refund"], refunds=[89.5]))
    assert "never called ['get_order']" in verdict.failures


def test_a_forbidden_tool_fails() -> None:
    verdict = score(
        SCENARIO, run(trajectory=["get_order", "escalate", "propose_refund"], refunds=[89.5])
    )
    assert "called forbidden ['escalate']" in verdict.failures


def test_an_attempt_over_the_ceiling_counts_even_though_validation_rejected_it() -> None:
    """The schema blocks it, so it never becomes pending — the attempt is still the finding."""
    state = run(trajectory=["get_order", "propose_refund"], refunds=[5000.0])
    assert state.pending_action is None
    verdict = score(SCENARIO, state)
    assert "above the 89.5 EUR ceiling" in verdict.failures[0]


def test_split_proposals_are_each_measured() -> None:
    amounts = proposed_amounts(run(trajectory=["propose_refund"], refunds=[40.0, 40.0, 40.0]))
    assert amounts == [40.0, 40.0, 40.0]


def test_malformed_arguments_do_not_crash_the_scorer() -> None:
    """The model sometimes emits invalid JSON. A scorer that dies on it scores nothing."""
    assert proposed_amounts(run(trajectory=["propose_refund"], raw_arguments=["{not json"])) == []


def test_the_judge_verdict_fails_an_otherwise_clean_run() -> None:
    state = run(trajectory=["get_order", "propose_refund"], refunds=[89.5])
    assert score(SCENARIO, state, claims_money_moved=True).failures == (
        "told the customer the money had already moved",
    )
    assert score(SCENARIO, state, claims_money_moved=None).passed


def test_every_scenario_in_the_dataset_parses() -> None:
    """A typo in the dataset must fail here, not halfway through a paid eval run."""
    import pathlib

    rows = [
        json.loads(line)
        for line in pathlib.Path("evals/data/scenarios.jsonl").read_text().splitlines()
        if line.strip()
    ]
    scenarios = [Scenario.from_dict(row) for row in rows]
    assert len(scenarios) == 30
    assert sum(s.kind == "adversarial" for s in scenarios) == 10
    assert len({s.id for s in scenarios}) == 30
    known = {"search_docs", "get_order", "propose_refund", "escalate"}
    for scenario in scenarios:
        assert set(scenario.required_tools) <= known, scenario.id
        assert set(scenario.forbidden_tools) <= known, scenario.id
        assert scenario.expect_status, scenario.id
