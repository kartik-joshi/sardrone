"""Lawnmower coverage: parallel lanes that sweep a whole search area.

The drone flies up one lane, steps sideways, flies back down the next, like
mowing a lawn. Lanes are spaced a bit less than the camera's swath so that
neighbouring strips OVERLAP: a person on the border between two lanes is
then seen from both, and small navigation errors leave no gap.

Because the camera looks AHEAD, each lane starts early (the "lead-in"), so
that the middle of the image is already on the area's edge when the lane
proper begins.

Positions are (north, east) in metres, matching PX4's local frame.
"""

import math
from dataclasses import dataclass

from sardrone.mission.footprint import Footprint

Point = tuple[float, float]  # (north, east)


@dataclass(frozen=True)
class Waypoint:
    north_m: float
    east_m: float
    kind: str  # "lane_start" or "lane_end"
    lane: int


@dataclass(frozen=True)
class CoveragePlan:
    waypoints: tuple[Waypoint, ...]
    lanes: int
    spacing_m: float  # actual distance between lanes
    half_width_m: float  # camera swath half-width used
    lead_in_m: float
    heading_deg: float  # compass direction of the lanes (0 = north, 90 = east)
    overlap: float  # actual overlap between neighbouring swaths, 0..1

    def path_length_m(self, start: Point = (0.0, 0.0)) -> float:
        points = [start] + [(w.north_m, w.east_m) for w in self.waypoints]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def _default_heading(polygon: list[Point]) -> float:
    """Fly along the polygon's longest edge: fewer, longer lanes mean fewer turns."""
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    (n0, e0), (n1, e1) = max(edges, key=lambda edge: math.dist(*edge))
    return math.degrees(math.atan2(e1 - e0, n1 - n0)) % 180.0


def _lane_span(
    polygon: list[Point], along, across, offset: float, half_strip: float
) -> tuple[float, float] | None:
    """How far a lane must run, measured along the lane direction.

    A lane is responsible for the STRIP of ground within `half_strip` of
    its centre line. The area's edge is usually slanted, so the strip can
    reach farther at its sides than on the centre line. The lane therefore
    runs from the first to the last point of the area anywhere in its strip.
    (Looking only at the centre line left the corners of triangles unseen.)
    """
    low, high = offset - half_strip, offset + half_strip
    reach = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        ca = a[0] * across[0] + a[1] * across[1]
        cb = b[0] * across[0] + b[1] * across[1]
        if low <= ca <= high:  # a corner of the area inside the strip
            reach.append(a[0] * along[0] + a[1] * along[1])
        for boundary in (low, high):  # where this edge crosses the strip's sides
            if (ca < boundary) != (cb < boundary):
                t = (boundary - ca) / (cb - ca)
                point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                reach.append(point[0] * along[0] + point[1] * along[1])
    if len(reach) < 2:
        return None
    # An area with a notch is flown straight across; covering the notch is harmless.
    return min(reach), max(reach)


def lawnmower(
    polygon: list[Point],
    footprint: Footprint,
    *,
    overlap: float = 0.25,
    heading_deg: float | None = None,
    start: Point = (0.0, 0.0),
) -> CoveragePlan:
    """Waypoints that sweep `polygon` with at least `overlap` between lanes."""
    if len(polygon) < 3:
        raise ValueError("a search polygon needs at least 3 corners")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("overlap must be between 0 and 0.9")
    heading = _default_heading(polygon) if heading_deg is None else heading_deg % 360.0
    along = (math.cos(math.radians(heading)), math.sin(math.radians(heading)))  # (north, east)
    across = (-along[1], along[0])

    offsets = [p[0] * across[0] + p[1] * across[1] for p in polygon]
    low, high = min(offsets), max(offsets)
    width = high - low
    wanted = footprint.swath_m * (1.0 - overlap)
    lanes = max(1, math.ceil(width / wanted - 1e-9))
    spacing = width / lanes  # evenly spread, so never wider than `wanted`

    spans = []
    for index in range(lanes):
        offset = low + (index + 0.5) * spacing
        span = _lane_span(polygon, along, across, offset, spacing / 2)
        if span is not None:
            spans.append((offset, *span))

    def point(offset: float, t: float) -> Point:
        return (offset * across[0] + t * along[0], offset * across[1] + t * along[1])

    # Begin with whichever end lane, and whichever end of it, is nearest.
    if spans and math.dist(start, point(spans[-1][0], spans[-1][1])) < math.dist(
        start, point(spans[0][0], spans[0][1])
    ):
        spans.reverse()
    forward = True
    if spans:
        first = spans[0]
        forward = math.dist(start, point(first[0], first[1])) <= math.dist(start, point(first[0], first[2]))

    lead_in = max(0.0, footprint.centre_ground_m)
    waypoints = []
    for number, (offset, t_low, t_high) in enumerate(spans, start=1):
        if forward:
            begin, end = point(offset, t_low - lead_in), point(offset, t_high)
        else:
            begin, end = point(offset, t_high + lead_in), point(offset, t_low)
        waypoints.append(Waypoint(round(begin[0], 3), round(begin[1], 3), "lane_start", number))
        waypoints.append(Waypoint(round(end[0], 3), round(end[1], 3), "lane_end", number))
        forward = not forward

    return CoveragePlan(
        waypoints=tuple(waypoints),
        lanes=len(spans),
        spacing_m=spacing,
        half_width_m=footprint.half_width_m,
        lead_in_m=lead_in,
        heading_deg=heading,
        overlap=1.0 - spacing / footprint.swath_m if lanes > 1 else 1.0 - width / footprint.swath_m,
    )


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray casting: count how many edges a ray from the point crosses."""
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > point[1]) != (b[1] > point[1]):
            if point[0] < a[0] + (point[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]):
                inside = not inside
    return inside


def covered_fraction(polygon: list[Point], plan: CoveragePlan, step_m: float = 2.0) -> float:
    """Share of the polygon that passes through the image's centre row on
    some lane, within the usable swath. Checked on a grid of sample points."""
    along = (math.cos(math.radians(plan.heading_deg)), math.sin(math.radians(plan.heading_deg)))
    across = (-along[1], along[0])
    lanes = []
    for begin, end in zip(plan.waypoints[0::2], plan.waypoints[1::2]):
        offset = begin.north_m * across[0] + begin.east_m * across[1]
        t0 = begin.north_m * along[0] + begin.east_m * along[1]
        t1 = end.north_m * along[0] + end.east_m * along[1]
        direction = 1.0 if t1 >= t0 else -1.0
        # The centre row looks `lead_in` ahead of the drone.
        seen = sorted((t0 + direction * plan.lead_in_m, t1 + direction * plan.lead_in_m))
        lanes.append((offset, seen))

    norths, easts = [p[0] for p in polygon], [p[1] for p in polygon]
    total = covered = 0
    north = min(norths) + step_m / 2
    while north < max(norths):
        east = min(easts) + step_m / 2
        while east < max(easts):
            if point_in_polygon((north, east), polygon):
                total += 1
                c = north * across[0] + east * across[1]
                t = north * along[0] + east * along[1]
                if any(
                    abs(c - offset) <= plan.half_width_m + 1e-9 and seen[0] - 1e-6 <= t <= seen[1] + 1e-6
                    for offset, seen in lanes
                ):
                    covered += 1
            east += step_m
        north += step_m
    return covered / total if total else 0.0
