"""Mission metrics: how well did the flight go, as numbers.

    CTE  cross-track error: how far the drone strayed sideways from the
         line it was meant to fly. Small CTE = the planned coverage is the
         coverage that actually happened.
    ADE  average displacement error: the mean distance between where
         something was predicted to be and where it really was, over time.
         Here: a tracker's predicted box centres against the true ones.

Both are averages of distances, so both are in the unit of their inputs
(metres for CTE; pixels for ADE in the image).
"""

import math

Point = tuple[float, float]


def distance_to_segment(point: Point, a: Point, b: Point) -> float:
    """Shortest distance from `point` to the line SEGMENT from a to b.

    The point is dropped perpendicularly onto the line through a and b.
    If the foot of that perpendicular lies beyond an end of the segment,
    the distance to that end is used instead.
    """
    ax, ay = a
    bx, by = b
    length_squared = (bx - ax) ** 2 + (by - ay) ** 2
    if length_squared == 0.0:
        return math.dist(point, a)
    t = ((point[0] - ax) * (bx - ax) + (point[1] - ay) * (by - ay)) / length_squared
    t = min(1.0, max(0.0, t))
    return math.dist(point, (ax + t * (bx - ax), ay + t * (by - ay)))


def cross_track_errors(positions: list[Point], path: list[Point]) -> list[float]:
    """For each flown position, its distance to the nearest part of the planned path."""
    if len(path) < 2:
        raise ValueError("a path needs at least two points")
    segments = list(zip(path, path[1:]))
    return [min(distance_to_segment(p, a, b) for a, b in segments) for p in positions]


def cte_summary(positions: list[Point], path: list[Point]) -> dict:
    errors = sorted(cross_track_errors(positions, path))
    if not errors:
        return {"samples": 0, "mean_m": 0.0, "p95_m": 0.0, "max_m": 0.0}
    return {
        "samples": len(errors),
        "mean_m": round(sum(errors) / len(errors), 3),
        "p95_m": round(errors[min(len(errors) - 1, int(0.95 * len(errors)))], 3),
        "max_m": round(errors[-1], 3),
    }


def average_displacement_error(predicted: list[Point], actual: list[Point]) -> float:
    """Mean distance between matching predicted and true positions."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have the same length")
    if not predicted:
        return 0.0
    return sum(math.dist(p, a) for p, a in zip(predicted, actual)) / len(predicted)


def coverage_percent(covered_fraction: float) -> float:
    return round(100.0 * covered_fraction, 1)
