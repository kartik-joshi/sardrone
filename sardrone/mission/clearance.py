"""Clearance: never fly lower than the tallest thing in the world plus a margin.

There is no obstacle avoidance. The missions stay safe only by flying above
everything, so every altitude a mission will fly (the search, the low
second pass, the confirm orbit, an altitude the agent chooses) is checked
against the tallest object in the world: trees, vehicles, rocks and other
clutter. People are left out; they are far lower than any of these.

Heights are tops above the ground, in metres, from each model's own bounding
box and scale (the same numbers `sardrone.worlds` used to place them).
"""

from __future__ import annotations

CLEARANCE_M = 3.0  # the same margin the confirm orbit has always kept above the trees


def object_tops(spec: dict) -> list[tuple[str, float]]:
    """(name, height of its top above the ground in metres) for every tree and
    clutter object in a world spec."""
    from sardrone.worlds.assets import ROCK_MODEL, TREES, VEHICLE_MODELS

    tops = [(f"tree {t['kind']}", TREES[t["kind"]].hi[2] * t["scale"]) for t in spec.get("trees", [])]
    for item in spec.get("clutter", []):
        kind, z, size = item["kind"], float(item["z"]), item["size"]
        if kind.startswith("vehicle:"):  # a mesh: size holds its scale
            top = z + VEHICLE_MODELS[kind.split(":", 1)[1]].hi[2] * size[2]
        elif kind == "rock_mesh":
            top = z + ROCK_MODEL.hi[2] * size[2]
        else:  # a primitive: size is in metres; z + full height is an upper bound
            top = z + size[2]
        tops.append((kind, top))
    return tops


def tallest_object_m(spec: dict) -> float:
    return max((top for _, top in object_tops(spec)), default=0.0)


def lowest_safe_altitude_m(spec: dict, margin_m: float = CLEARANCE_M) -> float:
    return tallest_object_m(spec) + margin_m


def clearance_problems(spec: dict, altitudes_m: dict[str, float | None], margin_m: float = CLEARANCE_M) -> list[str]:
    """One message per altitude that leaves less than `margin_m` above the
    tallest object; empty if all are safe. None altitudes are not flown."""
    tops = object_tops(spec)
    if not tops:
        return []
    name, tallest = max(tops, key=lambda item: item[1])
    return [
        f"{what} altitude {altitude:g} m leaves {altitude - tallest:.1f} m above the tallest object "
        f"({name}, {tallest:.1f} m); at least {margin_m:g} m is needed"
        for what, altitude in altitudes_m.items()
        if altitude is not None and altitude - tallest < margin_m
    ]


def require_clearance(spec: dict, altitudes_m: dict[str, float | None], margin_m: float = CLEARANCE_M) -> None:
    """Refuse a mission that would fly too low: raise before takeoff."""
    problems = clearance_problems(spec, altitudes_m, margin_m)
    if problems:
        raise ValueError(f"{spec.get('id', 'world')}: " + "; ".join(problems))
