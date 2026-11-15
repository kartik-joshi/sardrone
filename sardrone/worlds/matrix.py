"""The variation matrix: which worlds to build.

A YAML file lists episodes. Each one is a `VariationRow`: the handful of
choices (world type, number of people, time of day, seed, ...) from which a
whole world is generated. Validation happens here, so a typo in the YAML
fails immediately instead of producing a strange world.

    defaults: {altitude_m: 25, area_m: 120}
    episodes:
      - {id: pilot-01, world: meadow, seed: 101, people: 3, time_of_day: noon}
"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

WORLDS = ("meadow", "forest_edge", "park")
POSES = ("standing", "sitting", "lying")
OCCLUSIONS = ("none", "partial")
TIMES_OF_DAY = ("noon", "golden_hour", "dusk")
# Only weather that is really simulated may be listed here. Fog was tried in
# Phase 2A and Gazebo Harmonic does not render it (see docs/DEVLOG.md).
WEATHERS = ("clear",)
ALTITUDES_M = (15.0, 25.0, 40.0)


@dataclass(frozen=True)
class VariationRow:
    id: str
    world: str
    seed: int
    people: int = 3
    poses: tuple[str, ...] = ()  # empty = chosen by the seed
    occlusion: str = "none"
    time_of_day: str = "noon"
    weather: str = "clear"
    altitude_m: float = 25.0
    camera_pitch_deg: float = 45.0
    area_m: float = 120.0  # side of the square search area, centred on takeoff

    def __post_init__(self):
        def check(value, allowed, what):
            if value not in allowed:
                raise ValueError(f"{self.id}: {what} {value!r} is not one of {allowed}")

        check(self.world, WORLDS, "world")
        check(self.occlusion, OCCLUSIONS, "occlusion")
        check(self.time_of_day, TIMES_OF_DAY, "time_of_day")
        check(self.weather, WEATHERS, "weather")
        check(float(self.altitude_m), ALTITUDES_M, "altitude_m")
        for pose in self.poses:
            check(pose, POSES, "pose")
        if not 1 <= self.people <= 5:
            raise ValueError(f"{self.id}: people must be 1..5, got {self.people}")
        if self.poses and len(self.poses) != self.people:
            raise ValueError(
                f"{self.id}: {len(self.poses)} poses listed for {self.people} people"
            )
        if not 0.0 <= self.camera_pitch_deg <= 90.0:
            raise ValueError(f"{self.id}: camera_pitch_deg must be 0..90")
        if not 40.0 <= self.area_m <= 400.0:
            raise ValueError(f"{self.id}: area_m must be 40..400")


def row_from_dict(data: dict) -> VariationRow:
    known = {f.name for f in fields(VariationRow)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown matrix keys {sorted(unknown)}; allowed: {sorted(known)}")
    data = dict(data)
    if "poses" in data:
        data["poses"] = tuple(data["poses"])
    return VariationRow(**data)


def sample_rows(block: dict, defaults: dict | None = None) -> list[VariationRow]:
    """Expand a `sample` block into seeded rows.

        sample:
          prefix: pilot          # ids become pilot-001, pilot-002, ...
          count: 10
          seed: 7                # seed of the SAMPLING; each row gets its own world seed
          world: [meadow, forest_edge, park]
          people: [1, 2, 3, 4, 5]
          altitude_m: [15, 25]

    Every key holding a list is varied. Values are dealt like cards: the
    list is shuffled and handed out in turn, then reshuffled when it runs
    out. So with 3 worlds and 10 rows each world appears 3 or 4 times,
    instead of the lopsided counts that independent random picks can give.
    """
    import random

    block = dict(block)
    prefix, count, seed = block.pop("prefix"), int(block.pop("count")), int(block.pop("seed"))
    fixed = {**(defaults or {}), **{k: v for k, v in block.items() if not isinstance(v, list)}}
    varied = {key: value for key, value in block.items() if isinstance(value, list)}

    decks = {}
    for key in sorted(varied):
        rng = random.Random(f"{seed}:{key}")
        deck = []
        while len(deck) < count:
            hand = list(varied[key])
            rng.shuffle(hand)
            deck.extend(hand)
        decks[key] = deck[:count]

    rows = []
    for index in range(count):
        choices = {key: decks[key][index] for key in decks}
        rows.append(
            row_from_dict(
                {**fixed, **choices, "id": f"{prefix}-{index + 1:03d}", "seed": seed * 1000 + index + 1}
            )
        )
    return rows


def load_matrix(path: Path | str) -> list[VariationRow]:
    """Read a matrix YAML: explicit `episodes`, a `sample` block, a list of
    blocks under `samples`, or any mix. `defaults` are applied to every row.

    Several blocks in one file are how a dataset keeps its training and
    test episodes apart: each block has its own prefix and its own seed, so
    no test world can be the same as a training world.
    """
    document = yaml.safe_load(Path(path).read_text())
    defaults = document.get("defaults") or {}
    rows = [row_from_dict({**defaults, **episode}) for episode in document.get("episodes") or []]
    blocks = ([document["sample"]] if document.get("sample") else []) + (document.get("samples") or [])
    for block in blocks:
        rows += sample_rows(block, defaults)
    seeds = [row.seed for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: two episodes share a world seed; give each sample block its own seed")
    if not rows:
        raise ValueError(f"{path}: no episodes and no sample block")
    ids = [row.id for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate episode ids: {duplicates}")
    return rows


def find_row(rows: list[VariationRow], row_id: str) -> VariationRow:
    for row in rows:
        if row.id == row_id:
            return row
    raise KeyError(f"no episode {row_id!r}; available: {[r.id for r in rows]}")
