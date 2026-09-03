import pytest
from pydantic import ValidationError

from agentdesk.schemas import Budgets, ProposeRefund, Run


def test_refund_cap_is_structural() -> None:
    """The cap lives in the schema: no prompt can talk around it."""
    with pytest.raises(ValidationError):
        ProposeRefund(order_id="A-1", amount_eur=10_000, reason="angry customer")


def test_refund_rejects_non_positive_amount() -> None:
    with pytest.raises(ValidationError):
        ProposeRefund(order_id="A-1", amount_eur=0, reason="test")


def test_refund_accepted_under_cap() -> None:
    r = ProposeRefund(order_id="A-1", amount_eur=30, reason="lost parcel")
    assert r.amount_eur == 30


def test_budgets_defaults() -> None:
    b = Budgets()
    assert (b.max_iterations, b.max_tokens_total, b.max_cost_eur, b.max_wall_seconds) == (
        10,
        50_000,
        0.50,
        120.0,
    )


def test_run_status_is_closed_set() -> None:
    assert Run(id="r1").status == "running"
    with pytest.raises(ValidationError):
        Run(id="r1", status="whatever")  # type: ignore[arg-type]
