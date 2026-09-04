"""The HTTP surface, without a database and without a provider.

Both are replaced by doubles that refuse on *use* rather than on creation. FastAPI resolves
dependencies before it validates parameters, so a dependency that raises when it is built turns
every 422 into a 500 — and the test would then be asserting the framework's resolution order
instead of the endpoint's behaviour.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentdesk.agent.budgets import BudgetLedger
from agentdesk.agent.state import AgentState
from agentdesk.deps import get_llm, get_session
from agentdesk.main import app
from agentdesk.schemas import Budgets

KEY = {"X-API-Key": "ad_dev"}


class RefusingSession:
    """A session that fails the moment anything is executed on it."""

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("this request must not reach the database")


class Row:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    def mappings(self) -> "Row":
        return self

    def first(self) -> dict[str, Any] | None:
        return self.row


class StubSession:
    """Returns one prepared row, and accepts writes without doing anything."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    async def execute(self, *args: Any, **kwargs: Any) -> Row:
        return Row(self.row)

    async def commit(self) -> None:
        return None


class RefusingClient:
    """Stands in for the provider. Touching it at all is the failure."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"this request must not reach the model (touched .{name})")


@contextmanager
def client_with(session: Any) -> Iterator[TestClient]:
    """A client whose database is `session` and whose provider refuses to be touched at all."""
    app.dependency_overrides[get_llm] = lambda: RefusingClient()
    app.dependency_overrides[get_session] = lambda: session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with client_with(RefusingSession()) as test_client:
        yield test_client


def answered_run() -> dict[str, Any]:
    state = AgentState(
        run_id="r1",
        messages=[{"role": "user", "content": "hi"}],
        ledger=BudgetLedger(budgets=Budgets()),
        status="answered",
        answer="all done",
    )
    return state.to_row()


def test_health_needs_no_key() -> None:
    """Load balancers do not carry API keys."""
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_a_run_without_a_key_is_rejected(client: TestClient) -> None:
    assert client.post("/v1/runs", json={"message": "hello"}).status_code == 401


def test_a_key_of_the_wrong_shape_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/runs", json={"message": "hi"}, headers={"X-API-Key": "dp_dev"})
    assert response.status_code == 401


def test_an_empty_message_is_a_422_not_a_500(client: TestClient) -> None:
    """The doubles enforce it: a rejected request must cost neither a query nor a token."""
    assert client.post("/v1/runs", json={"message": ""}, headers=KEY).status_code == 422


def test_an_oversized_message_is_rejected_before_it_is_paid_for(client: TestClient) -> None:
    response = client.post("/v1/runs", json={"message": "x" * 5000}, headers=KEY)
    assert response.status_code == 422


def test_a_budget_of_zero_is_refused(client: TestClient) -> None:
    """A caller-supplied budget is still validated: zero would stop the run before it began."""
    response = client.post(
        "/v1/runs", json={"message": "hi", "budgets": {"max_iterations": 0}}, headers=KEY
    )
    assert response.status_code == 422


def test_an_unknown_run_is_a_404() -> None:
    with client_with(StubSession(None)) as client:
        assert client.get("/v1/runs/nope", headers=KEY).status_code == 404


def test_reading_a_run_returns_its_counters() -> None:
    with client_with(StubSession(answered_run())) as client:
        body = client.get("/v1/runs/r1", headers=KEY).json()
    assert body["status"] == "answered"
    assert body["answer"] == "all done"
    assert body["pending_action"] is None


def test_approving_a_finished_run_is_a_409_not_a_second_refund() -> None:
    """What a second reviewer gets when they click approve on a run someone already decided."""
    with client_with(StubSession(answered_run())) as client:
        response = client.post("/v1/runs/r1/approve", json={}, headers=KEY)
    assert response.status_code == 409
    assert "answered" in response.json()["detail"]
