"""The agent's six tools, described as JSON schemas.

A language model cannot fly anything. It can only produce text. "Tool use"
means: we tell the model which functions exist and what arguments they
take (the schemas below); the model answers with the NAME of a function
and its arguments as JSON; our code runs the function and hands the result
back as the next message. The model never touches the drone directly, and
everything it may do is listed here.
"""

import json

from sardrone.agent.sim import ALTITUDES_M, AREAS

TOOLS = [
    {
        "name": "set_search_area",
        "description": "Choose where to search. Must be called before start_search.",
        "parameters": {
            "type": "object",
            "properties": {"area": {"type": "string", "enum": list(AREAS), "description": "Name of the area."}},
            "required": ["area"],
        },
    },
    {
        "name": "start_search",
        "description": "Take off and start the search pattern over the chosen area.",
        "parameters": {
            "type": "object",
            "properties": {
                "altitude_m": {
                    "type": "integer", "enum": list(ALTITUDES_M),
                    "description": "Flight altitude. Lower sees people larger but covers ground more slowly.",
                }
            },  # fmt: skip
            "required": ["altitude_m"],
        },
    },
    {
        "name": "get_status",
        "description": "Battery, search progress, and any sightings. Call repeatedly while searching; the search advances between calls.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "confirm_target",
        "description": "Fly to a sighting and look closely. Returns a description of the person (clothing, posture). Costs battery.",
        "parameters": {
            "type": "object",
            "properties": {"target_id": {"type": "string", "description": "A sighting id from get_status."}},
            "required": ["target_id"],
        },
    },
    {
        "name": "report_person",
        "description": "Report a CONFIRMED person's position to the rescue team.",
        "parameters": {
            "type": "object",
            "properties": {"target_id": {"type": "string", "description": "A confirmed target id."}},
            "required": ["target_id"],
        },
    },
    {
        "name": "return_home",
        "description": "Fly back to the launch point and land. This ends the mission.",
        "parameters": {"type": "object", "properties": {}},
    },
]

TOOL_NAMES = tuple(tool["name"] for tool in TOOLS)

SYSTEM_PROMPT = (
    "You operate a search-and-rescue drone through the tools provided. Follow the operator's mission "
    "exactly. Work in this order: set the search area, start the search, then call get_status "
    "repeatedly. Confirm every sighting before deciding what to do with it. Report a person only if they "
    "match what the operator asked for; if the operator did not restrict who to look for, report "
    "everyone you confirm. Respect any battery limit the operator gives: once the battery is at or below "
    "it, return home at once. When the search is complete and nothing is left to confirm or report, "
    "return home. Always end the mission with return_home. Use one tool call at a time."
)


def call_tool(sim, name: str, arguments: dict) -> dict:
    """Run one tool call on the simulator. Unknown tools and bad arguments
    come back as errors the model can read, never as exceptions."""
    if name not in TOOL_NAMES:
        sim.log.append((name, dict(arguments or {}), False))
        sim.errors += 1
        return {"ok": False, "error": f"unknown tool {name!r}; available: {list(TOOL_NAMES)}"}
    try:
        return getattr(sim, name)(**(arguments or {}))
    except TypeError as error:
        sim.log.append((name, dict(arguments or {}), False))
        sim.errors += 1
        return {"ok": False, "error": f"bad arguments for {name}: {error}"}


def as_text(result: dict) -> str:
    return json.dumps(result, sort_keys=True)
