"""The MCP surface: what it exposes, and what it refuses to expose.

Both halves matter. A server that grows a tool nobody described is a server a client will guess
with; a server that grows an `approve_run` tool has handed the rubber stamp to the agent.
"""

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentdesk.config import settings
from agentdesk.mcp_server import RequireApiKey, http_app, server

EXPECTED_TOOLS = {"classify_ticket", "search_product_docs", "start_support_run"}


async def test_the_surface_is_three_tools_and_stays_three() -> None:
    """Few tools, well described. Twenty vague ones are worse than three clear ones."""
    tools = await server.list_tools()
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


async def test_nothing_here_can_approve_a_refund() -> None:
    """The security property of the whole server: approval is a human action on a human endpoint.

    A client that could approve its own proposals is an agent holding its own rubber stamp, and
    the client's guardrails are not ours to trust.
    """
    names = {tool.name for tool in await server.list_tools()}
    assert not any("approve" in name or "refund" in name for name in names)


async def test_every_tool_carries_a_description_the_client_can_choose_on() -> None:
    """The description is the interface: it is what the client's model reads before choosing."""
    for tool in await server.list_tools():
        assert tool.description and len(tool.description) > 80, tool.name


async def test_the_run_resource_is_addressable_by_id() -> None:
    templates = await server.list_resource_templates()
    assert [str(template.uri_template) for template in templates] == ["supportly://runs/{run_id}"]


def guarded_app(key: str) -> TestClient:
    inner = Starlette(routes=[Route("/mcp", lambda request: PlainTextResponse("ok"))])
    return TestClient(RequireApiKey(inner, key))


def test_the_remote_transport_refuses_an_unauthenticated_request() -> None:
    """A remote MCP server is a public API, whatever the protocol calls it."""
    assert guarded_app("ad_secret").get("/mcp").status_code == 401


def test_the_remote_transport_refuses_the_wrong_key() -> None:
    response = guarded_app("ad_secret").get("/mcp", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


def test_the_remote_transport_lets_the_right_key_through() -> None:
    response = guarded_app("ad_secret").get("/mcp", headers={"X-API-Key": "ad_secret"})
    assert (response.status_code, response.text) == (200, "ok")


def test_serving_over_http_without_a_key_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing here beats discovering an open server in a log."""
    monkeypatch.setattr(settings, "mcp_api_key", "")
    with pytest.raises(RuntimeError, match="MCP_API_KEY"):
        http_app(8003)
