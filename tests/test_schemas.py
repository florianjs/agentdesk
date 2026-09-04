import pytest
from pydantic import ValidationError

from agentdesk.agent.tools import ProposeRefund
from agentdesk.schemas import TERMINAL, Budgets, RunView


def test_refund_cap_is_structural() -> None:
    """The cap lives in the tool schema, so no prompt can talk around it."""
    with pytest.raises(ValidationError):
        ProposeRefund(order_id="A-1", amount_eur=10_000, reason="angry customer")


def test_refund_rejects_non_positive_amount() -> None:
    with pytest.raises(ValidationError):
        ProposeRefund(order_id="A-1", amount_eur=0, reason="test")


def test_refund_accepted_under_cap() -> None:
    assert ProposeRefund(order_id="A-1", amount_eur=30, reason="lost parcel").amount_eur == 30


def test_budgets_default_from_configuration() -> None:
    from agentdesk.config import settings

    budgets = Budgets()
    assert budgets.max_iterations == settings.max_iterations
    assert budgets.max_cost_usd == settings.max_cost_usd


def test_budgets_reject_a_zero_ceiling() -> None:
    """A budget of zero would stop every run before its first call — a config typo, not a policy."""
    with pytest.raises(ValidationError):
        Budgets(max_iterations=0)


def test_run_status_is_a_closed_set() -> None:
    assert RunView(id="r1", status="running").status == "running"
    with pytest.raises(ValidationError):
        RunView(id="r1", status="whatever")  # type: ignore[arg-type]


def test_awaiting_approval_is_not_terminal() -> None:
    """A suspended run is not finished; it is waiting for a person."""
    assert "awaiting_approval" not in TERMINAL
    assert "answered" in TERMINAL
