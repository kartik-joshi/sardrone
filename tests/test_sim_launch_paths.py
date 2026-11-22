from pathlib import Path

from sardrone.sim.launch import PROJECT_ROOT, _shown


def test_paths_inside_the_project_are_shown_relative():
    assert _shown(PROJECT_ROOT / "data" / "worlds" / "w1" / "logs" / "px4.log") == "data/worlds/w1/logs/px4.log"


def test_paths_outside_the_project_are_left_alone():
    outside = Path("/opt/elsewhere/px4.log")
    assert _shown(outside) == str(outside)
