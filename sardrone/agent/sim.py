"""A small mission simulator for testing the agent, with no drone and no PX4.

Evaluating an agent on real flights would take five minutes per try and
would mix up two questions: "did the agent decide well?" and "did the drone
fly well?". This simulator answers only the first. It keeps a clock, a
battery, named search areas and a list of people; each tool call changes
that state in a simple, fully predictable way. Because it is deterministic,
the same decisions always give the same score.

It models what matters for decisions: searching takes time and battery,
sightings turn up as the search proceeds, a sighting has to be confirmed
before anything is known about the person, and the battery sets a limit.
"""

from dataclasses import dataclass, field

AREAS = ("north field", "south field", "east meadow", "west woods", "whole area")
ALTITUDES_M = (15, 25, 40)

# Per call. Chosen so that a full search of one area costs about 30 %.
SEARCH_STEP_COVERAGE = 20.0  # percent of the area covered per get_status
SEARCH_STEP_BATTERY = 6.0
CONFIRM_BATTERY = 4.0
START_BATTERY = 3.0
HOME_BATTERY = 5.0


@dataclass(frozen=True)
class Person:
    id: str
    area: str
    description: str  # what a confirmation reveals, e.g. "adult in a red jacket, sitting"
    appears_at: float  # coverage percent of the search at which they are sighted


@dataclass
class Scenario:
    people: tuple[Person, ...] = ()
    battery: float = 100.0


@dataclass
class MissionSim:
    scenario: Scenario
    battery: float = 0.0
    area: str | None = None
    altitude_m: int | None = None
    state: str = "IDLE"  # IDLE | SEARCHING | SEARCH_COMPLETE | HOME | CRASHED
    coverage: float = 0.0
    sighted: list = field(default_factory=list)  # target ids, in order of sighting
    confirmed: dict = field(default_factory=dict)  # target id -> description
    reported: list = field(default_factory=list)
    log: list = field(default_factory=list)  # every call: (tool, arguments, ok)
    errors: int = 0
    battery_when_home: float | None = None
    lowest_battery_while_away: float = 100.0

    def __post_init__(self):
        self.battery = self.scenario.battery
        self._targets = {}  # target id -> Person

    # ---- helpers -------------------------------------------------------------

    def _result(self, tool: str, arguments: dict, ok: bool, **payload) -> dict:
        self.log.append((tool, dict(arguments), ok))
        if not ok:
            self.errors += 1
        return {"ok": ok, **payload}

    def _spend(self, amount: float) -> None:
        self.battery = max(0.0, round(self.battery - amount, 1))
        if self.state not in ("HOME", "IDLE"):
            self.lowest_battery_while_away = min(self.lowest_battery_while_away, self.battery)
        if self.battery <= 0.0 and self.state != "HOME":
            self.state = "CRASHED"

    def _ended(self) -> str | None:
        if self.state == "HOME":
            return "the drone is already home; the mission is over"
        if self.state == "CRASHED":
            return "the battery is empty; the drone is down"
        return None

    # ---- the six tools -------------------------------------------------------

    def set_search_area(self, area: str) -> dict:
        arguments = {"area": area}
        if (ended := self._ended()) or self.state == "SEARCHING":
            return self._result("set_search_area", arguments, False, error=ended or "cannot change area while searching")
        if area not in AREAS:
            return self._result("set_search_area", arguments, False, error=f"unknown area; choose one of {list(AREAS)}")
        self.area, self.coverage, self.state = area, 0.0, "IDLE"
        return self._result("set_search_area", arguments, True, area=area)

    def start_search(self, altitude_m: int = 25) -> dict:
        arguments = {"altitude_m": altitude_m}
        if ended := self._ended():
            return self._result("start_search", arguments, False, error=ended)
        if self.area is None:
            return self._result("start_search", arguments, False, error="set a search area first")
        if altitude_m not in ALTITUDES_M:
            return self._result("start_search", arguments, False, error=f"altitude must be one of {list(ALTITUDES_M)}")
        if self.state == "SEARCHING":
            return self._result("start_search", arguments, False, error="already searching")
        self.altitude_m, self.state = altitude_m, "SEARCHING"
        self._spend(START_BATTERY)
        return self._result("start_search", arguments, True, state=self.state, battery_percent=self.battery)

    def get_status(self) -> dict:
        """Report the state. While searching, this also lets time pass."""
        if self.state == "SEARCHING":
            self.coverage = min(100.0, self.coverage + SEARCH_STEP_COVERAGE)
            self._spend(SEARCH_STEP_BATTERY)
            for person in self.scenario.people:
                in_area = self.area == "whole area" or person.area == self.area
                if in_area and person.appears_at <= self.coverage and person.id not in [p.id for p in self._targets.values()]:
                    target = f"target_{len(self._targets) + 1}"
                    self._targets[target] = person
                    self.sighted.append(target)
            if self.coverage >= 100.0 and self.state == "SEARCHING":
                self.state = "SEARCH_COMPLETE"
        pending = [t for t in self.sighted if t not in self.confirmed]
        unreported = [t for t in self.confirmed if t not in self.reported]
        return self._result(
            "get_status", {}, True, state=self.state, area=self.area, battery_percent=self.battery,
            coverage_percent=self.coverage, unconfirmed_sightings=pending, confirmed_not_reported=unreported,
            reported=list(self.reported),
        )  # fmt: skip

    def confirm_target(self, target_id: str) -> dict:
        arguments = {"target_id": target_id}
        if ended := self._ended():
            return self._result("confirm_target", arguments, False, error=ended)
        if target_id not in self._targets:
            return self._result("confirm_target", arguments, False, error="no such sighting")
        if target_id in self.confirmed:
            return self._result("confirm_target", arguments, False, error="already confirmed")
        self._spend(CONFIRM_BATTERY)
        self.confirmed[target_id] = self._targets[target_id].description
        return self._result(
            "confirm_target", arguments, True, description=self.confirmed[target_id], battery_percent=self.battery
        )

    def report_person(self, target_id: str) -> dict:
        arguments = {"target_id": target_id}
        if target_id not in self.confirmed:
            return self._result("report_person", arguments, False, error="confirm the target before reporting it")
        if target_id in self.reported:
            return self._result("report_person", arguments, False, error="already reported")
        self.reported.append(target_id)
        return self._result("report_person", arguments, True, reported=target_id)

    def return_home(self) -> dict:
        if ended := self._ended():
            return self._result("return_home", {}, False, error=ended)
        self._spend(HOME_BATTERY)
        if self.state != "CRASHED":
            self.battery_when_home = self.battery
            self.state = "HOME"
        return self._result("return_home", {}, self.state == "HOME", state=self.state, battery_percent=self.battery)

    # ---- for scoring ---------------------------------------------------------

    def reported_people(self) -> list[str]:
        return [self._targets[target].id for target in self.reported]
