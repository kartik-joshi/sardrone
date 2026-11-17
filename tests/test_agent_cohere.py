"""The Cohere back-end, checked against the installed SDK's own types.

No network call is made. A stand-in client records what the adapter would
send and returns a response built with the SDK's own response model, so:

* every message the agent sends during a mission must validate as one of
  Cohere's v2 chat message types, and every tool as a ToolV2;
* a response in Cohere's shape must come back as the right tool call,
  arguments and token counts.

What this cannot show: that the live API accepts the request, that a trial
key allows tool use, or how well a Cohere model does the missions.
"""

import json

import pytest

cohere = pytest.importorskip("cohere", reason="cohere SDK not installed (pip install -e '.[agent]')")

from cohere import (  # noqa: E402
    AssistantChatMessageV2,
    SystemChatMessageV2,
    ToolCallV2,
    ToolChatMessageV2,
    ToolV2,
    UserChatMessageV2,
    V2ChatResponse,  # what ClientV2.chat returns (checked: its return annotation)
)

from sardrone.agent.agent import run_agent  # noqa: E402
from sardrone.agent.models import CohereModel  # noqa: E402
from sardrone.agent.sim import MissionSim, Person, Scenario  # noqa: E402
from sardrone.agent.tools import TOOLS  # noqa: E402

MESSAGE_TYPES = {
    "system": SystemChatMessageV2,
    "user": UserChatMessageV2,
    "assistant": AssistantChatMessageV2,
    "tool": ToolChatMessageV2,
}


def cohere_response(tool_name=None, arguments=None, text="", tokens=(120, 15)):
    """A chat response as the SDK would return it."""
    message = {"role": "assistant", "tool_plan": text or None}
    if tool_name:
        message["tool_calls"] = [
            {"id": f"call_{tool_name}", "type": "function",
             "function": {"name": tool_name, "arguments": json.dumps(arguments or {})}}
        ]  # fmt: skip
    else:
        message["content"] = [{"type": "text", "text": text}]
    return V2ChatResponse.model_validate(
        {
            "id": "resp-1",
            "finish_reason": "TOOL_CALL" if tool_name else "COMPLETE",
            "message": message,
            "usage": {"tokens": {"input_tokens": tokens[0], "output_tokens": tokens[1]}},
        }
    )


class FakeClient:
    """Stands in for cohere.ClientV2: records requests, replays responses."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def chat(self, **request):
        self.requests.append(json.loads(json.dumps(request)))  # what would go on the wire
        return self.responses.pop(0)


def adapter(responses):
    model = CohereModel.__new__(CohereModel)  # skip __init__: no key, no network
    model.model, model.name, model.temperature, model.seed = "test-model", "cohere:test-model", 0.0, 51
    model.min_gap_s, model._last_call, model.calls = 0.0, 0.0, 0
    model._client = FakeClient(responses)
    return model


def validate_request(request):
    for message in request["messages"]:
        MESSAGE_TYPES[message["role"]].model_validate(message, strict=False)
        # The SDK's models are lenient; check the required fields directly too.
        if message["role"] == "tool":
            assert isinstance(message["tool_call_id"], str) and isinstance(message["content"], str)
        if message["role"] == "assistant" and "tool_calls" in message:
            for call in message["tool_calls"]:
                ToolCallV2.model_validate(call)
                json.loads(call["function"]["arguments"])  # arguments must be a JSON string
    for tool in request["tools"]:
        parsed = ToolV2.model_validate(tool)
        assert parsed.type == "function" and parsed.function.name in {t["name"] for t in TOOLS}
        assert parsed.function.parameters["type"] == "object"


def test_requests_during_a_whole_mission_match_the_sdk_types():
    script = [
        ("set_search_area", {"area": "north field"}), ("start_search", {"altitude_m": 25}),
        ("get_status", {}), ("get_status", {}), ("confirm_target", {"target_id": "target_1"}),
        ("report_person", {"target_id": "target_1"}), ("return_home", {}),
    ]  # fmt: skip
    model = adapter([cohere_response(name, args, text="plan") for name, args in script])
    sim = MissionSim(Scenario(people=(Person("p1", "north field", "adult in a red jacket", 40),)))
    transcript = run_agent(model, "Search the north field and report anyone.", sim)

    assert transcript.ended == "returned_home" and sim.reported_people() == ["p1"]
    requests = model._client.requests
    assert len(requests) == len(script)
    for request in requests:
        validate_request(request)
        assert request["model"] == "test-model" and request["temperature"] == 0.0 and request["seed"] == 51
    # The last request carries the whole conversation: system, user, then a
    # tool call and its result for each of the six earlier steps.
    roles = [m["role"] for m in requests[-1]["messages"]]
    assert roles == ["system", "user"] + ["assistant", "tool"] * 6
    first_result = requests[-1]["messages"][3]
    assert first_result["tool_call_id"] == "call_set_search_area"
    assert json.loads(first_result["content"])["ok"] is True


def test_response_parsing_tool_call_arguments_and_tokens():
    model = adapter([cohere_response("start_search", {"altitude_m": 15}, text="go low", tokens=(321, 45))])
    reply = model.reply([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], TOOLS)
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_start_search", "start_search", {"altitude_m": 15})
    assert (reply.input_tokens, reply.output_tokens) == (321, 45)
    assert reply.text == "go low"


def test_a_text_answer_without_a_tool_call():
    model = adapter([cohere_response(text="The drone is idle on the ground.", tokens=(50, 9))])
    reply = model.reply([{"role": "user", "content": "status?"}], TOOLS)
    assert reply.tool_calls == () and reply.text == "The drone is idle on the ground."
    assert reply.output_tokens == 9


def test_the_reminder_after_a_text_answer_is_a_valid_conversation():
    # Text first (the loop then adds a reminder), then a tool call.
    model = adapter([cohere_response(text="Thinking."), cohere_response("return_home")])
    run_agent(model, "Do not launch.", MissionSim(Scenario()))
    second = model._client.requests[1]
    validate_request(second)
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]


@pytest.mark.parametrize(
    "model_type, data",
    [
        (ToolChatMessageV2, {"role": "tool", "content": "x"}),  # no tool_call_id
        (ToolV2, {"type": "nope", "function": {"name": "a", "parameters": {}}}),
        (ToolCallV2, {"id": "1", "type": "function", "function": {"name": "a", "arguments": {"x": 1}}}),
        (AssistantChatMessageV2, {"role": "robot", "tool_plan": "p"}),
    ],
)
def test_the_sdk_types_really_reject_malformed_messages(model_type, data):
    """The checks above only mean something if the SDK's models are strict
    enough to refuse a wrong message. They are, for these mistakes."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        model_type.model_validate(data)
