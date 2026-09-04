"""Reading a forced tool call off a completion.

Every function here exists because a provider returned something the SDK's types say is
impossible. The router serves the same model from several vendors and they disagree on the
edges, so the shapes are normalised in one place instead of at every call site:

- `tool_calls` is typed as a union whose other arm has no `.function` at all;
- `finish_reason` can be `"error"`, which the SDK's closed set does not include;
- tool arguments sometimes arrive wrapped in a spurious key instead of at the top level.
"""

import json
from collections.abc import Collection
from typing import Any, cast

from agentdesk.llm.retry import TransientUpstreamError


class NoToolCall(RuntimeError):
    """The model answered without calling the tool it was required to call."""


def raise_if_upstream_error(response: Any, model: str) -> None:
    """Turn a provider failure dressed as a success into a real, retryable error.

    The SDK types `finish_reason` as a closed set that does not include `"error"` — but the
    router returns exactly that, with no content and zero tokens, inside an HTTP 200. The cast
    states that mismatch once instead of scattering it.
    """
    if cast(str, response.choices[0].finish_reason) == "error":
        raise TransientUpstreamError(f"provider returned an error finish_reason for {model}")


def tool_arguments(response: Any) -> str:
    """Return the raw JSON arguments of the first function call in a completion."""
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        raise NoToolCall("model did not call the tool")

    function = getattr(calls[0], "function", None)
    if function is None:
        raise NoToolCall(f"unexpected tool call type: {type(calls[0]).__name__}")

    arguments: str = function.arguments
    return arguments


def parse_tool_payload(raw: str, fields: Collection[str]) -> dict[str, Any]:
    """Parse tool arguments, unwrapping the spurious nesting some providers add.

    A model asked for `{"accurate": true}` occasionally answers
    `{"parameter name": {"accurate": true}}`. The payload is right, the envelope is not, and
    validating it straight fails with "field required" for every field at once — a confusing
    error for what is a transport quirk.

    Only a single unknown wrapper whose contents do match the schema is unwrapped; anything
    else is passed through so a genuinely malformed response still fails loudly.
    """
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        return cast(dict[str, Any], payload)

    known = set(fields)

    # Shape 1: a single unknown wrapper key around the real payload.
    if not known & payload.keys() and len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, dict) and known & inner.keys():
            return cast(dict[str, Any], inner)

    # Shape 2: double-encoded — a field's value is a JSON *string* holding the whole object.
    # Seen from a provider returning
    # `{"answers_the_question": "{\"answers_the_question\": true, \"reasoning\": \"...\"}"}`.
    # Validating that directly fails on every field at once, which reads like a broken schema
    # rather than a transport quirk.
    for value in payload.values():
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            # Recognised by shape, not by completeness: the decoded object must name at least
            # one field of the schema and nothing outside it. Requiring *every* field would miss
            # the common case where the model omits one that has a default.
            if isinstance(decoded, dict) and decoded.keys() & known and decoded.keys() <= known:
                return cast(dict[str, Any], decoded)

    return cast(dict[str, Any], payload)
