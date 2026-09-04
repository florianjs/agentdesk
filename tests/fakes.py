"""A scripted model, so every branch of the loop is testable without a network call.

The loop takes its model call as an argument for exactly this reason. A test that has to stub
the OpenAI SDK ends up asserting things about the SDK; this one asserts things about the loop.
"""

import json
from types import SimpleNamespace
from typing import Any


def tool_call(name: str, arguments: dict[str, Any] | str, call_id: str = "call_1") -> Any:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(
        id=call_id, type="function", function=SimpleNamespace(name=name, arguments=raw)
    )


def completion(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cost: float | None = 0.001,
    finish_reason: str = "stop",
) -> Any:
    return SimpleNamespace(
        model="fake/model",
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
        ),
    )


class ScriptedModel:
    """Returns the next scripted response; raises if the loop asks for one too many.

    Running out of script is a failure, not a fallback: a loop that made an extra call after it
    should have stopped is the bug these tests exist to catch.
    """

    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[dict[str, Any]], schemas: list[dict[str, Any]]) -> Any:
        # Copied: the loop mutates the transcript in place, so keeping the reference would make
        # every recorded call look identical to the last one.
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("the loop called the model more times than the script allows")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
