"""Model back-ends for the agent. All speak one small interface:

    reply = model.reply(messages, tools)      # -> Reply(tool_calls, text, tokens)

`messages` uses the Cohere v2 chat format (role "system" / "user" /
"assistant" / "tool"), so the Cohere back-end passes it through unchanged.

* CohereModel     the real thing (needs COHERE_API_KEY; calls cost money)
* RuleBasedModel  no language model at all: keyword rules. A baseline that
                  shows which missions are easy and gives the evaluation
                  something to measure before any API call is made.
* ScriptedModel   replays a fixed list of tool calls, for tests.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from sardrone.agent.sim import ALTITUDES_M, AREAS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    tool_calls: tuple[ToolCall, ...] = ()
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def reply(self, messages: list[dict], tools: list[dict]) -> Reply: ...


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> None:
    """Read KEY=value lines from .env into the environment (the file is
    git-ignored; the key never goes into the repository)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class MonthlyLimitReached(RuntimeError):
    """The API key has no calls left this month."""


class CohereModel:
    """Cohere chat with tool use, through the official SDK's v2 client.

    Checked against the installed cohere 7.3.1: `ClientV2.chat(model=,
    messages=, tools=, temperature=, seed=)` returns a response whose
    `message.tool_calls` items have `.id`, `.function.name` and
    `.function.arguments` (a JSON string), and whose `usage.tokens` has
    `input_tokens` and `output_tokens`.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.0, seed: int = 51,
                 min_seconds_between_calls: float = 3.5, retry_wait_s: float = 20.0,
                 max_retries: int = 6):  # fmt: skip
        import cohere

        # A free trial key allows 20 chat requests per minute (Cohere's
        # rate-limit page, checked 2026-11-16). 3.5 s between calls stays
        # under that; 3.2 s did not leave enough margin on 2026-11-22 (one
        # 429 mid-run). Set to 0 with a production key. A 429 is retried
        # with the identical request after a wait, so it cannot change
        # what the model decides; retries are counted in `rate_limited`.
        self.min_gap_s = min_seconds_between_calls
        self.retry_wait_s, self.max_retries, self.rate_limited = retry_wait_s, max_retries, 0
        self._last_call = 0.0
        self.calls = 0

        load_env_file()
        key = os.environ.get("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set (put it in .env, which is git-ignored)")
        self.model = model or os.environ.get("COHERE_MODEL")
        if not self.model:
            raise RuntimeError("no Cohere model named: pass --cohere-model or set COHERE_MODEL in .env")
        self.name = f"cohere:{self.model}"
        self.temperature, self.seed = temperature, seed
        self._client = cohere.ClientV2(api_key=key)

    def reply(self, messages: list[dict], tools: list[dict]) -> Reply:
        import time

        wait = self.min_gap_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        from cohere.errors import TooManyRequestsError

        self.calls += 1
        for attempt in range(self.max_retries + 1):
            self._last_call = time.monotonic()
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": tool} for tool in tools],
                    temperature=self.temperature,
                    seed=self.seed,
                )
                break
            except TooManyRequestsError as error:
                # Two different limits answer 429: calls per minute (wait and
                # retry) and calls per month (retrying cannot help: stop).
                if "month" in str(getattr(error, "body", "") or error).lower():
                    raise MonthlyLimitReached(
                        "the Cohere trial key's monthly call limit is used up; it resets next month"
                    ) from error
                if attempt == self.max_retries:
                    raise
                self.rate_limited += 1
                time.sleep(self.retry_wait_s * (attempt + 1))
        message = response.message
        calls = tuple(
            ToolCall(call.id, call.function.name, json.loads(call.function.arguments or "{}"))
            for call in (message.tool_calls or [])
        )
        text = " ".join(getattr(item, "text", "") for item in (message.content or []))
        tokens = response.usage.tokens if response.usage and response.usage.tokens else None
        return Reply(
            calls, text or (message.tool_plan or ""),
            int(tokens.input_tokens or 0) if tokens else 0, int(tokens.output_tokens or 0) if tokens else 0,
        )  # fmt: skip


COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "grey", "brown")


@dataclass
class RuleBasedModel:
    """Keyword rules, no language understanding.

    It picks the first area name that appears in the mission, an altitude
    if one is written as "15 m", a battery limit if a percentage appears,
    and a clothing colour if one is named. It then follows one fixed
    routine. It cannot handle order ("first ..., then ..."), negation
    ("do not report ..."), or anything not on its keyword lists, and the
    evaluation shows exactly that.
    """

    name: str = "rule-based"
    _skipped: set = field(default_factory=set)

    def reply(self, messages: list[dict], tools: list[dict]) -> Reply:
        mission = next(m["content"] for m in messages if m["role"] == "user").lower()
        calls = [c for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else {}
        names = [c["function"]["name"] for c in calls]

        def call(name, **arguments):
            return Reply((ToolCall(f"rule_{len(calls) + 1}", name, arguments),))

        if "set_search_area" not in names:
            self._skipped.clear()
            return call("set_search_area", area=next((a for a in AREAS if a in mission), "whole area"))
        if "start_search" not in names:
            match = re.search(r"\b(15|25|40)\s*m\b", mission)
            return call("start_search", altitude_m=int(match.group(1)) if match else 25)

        limit = re.search(r"(\d+)\s*%", mission)
        colour = next((c for c in COLOURS if c in mission), None)
        if names[-1] == "confirm_target" and last.get("ok"):
            target = json.loads(calls[-1]["function"]["arguments"])["target_id"]
            if colour is None or colour in last.get("description", ""):
                return call("report_person", target_id=target)
            self._skipped.add(target)
        if names[-1] != "get_status":
            return call("get_status")
        if limit and last.get("battery_percent", 100) <= int(limit.group(1)):
            return call("return_home")
        if last.get("unconfirmed_sightings"):
            return call("confirm_target", target_id=last["unconfirmed_sightings"][0])
        waiting = [t for t in last.get("confirmed_not_reported", []) if t not in self._skipped]
        if waiting and colour is None:
            return call("report_person", target_id=waiting[0])
        if last.get("state") == "SEARCH_COMPLETE":
            return call("return_home")
        return call("get_status")


@dataclass
class ScriptedModel:
    """Replays a fixed list of (tool name, arguments). For tests."""

    script: list
    name: str = "scripted"
    _index: int = 0

    def reply(self, messages: list[dict], tools: list[dict]) -> Reply:
        if self._index >= len(self.script):
            return Reply(text="nothing more to do")
        name, arguments = self.script[self._index]
        self._index += 1
        return Reply((ToolCall(f"s{self._index}", name, arguments),), input_tokens=100, output_tokens=10)
