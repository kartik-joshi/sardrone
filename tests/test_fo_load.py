"""Step 7. Skipped unless FiftyOne is installed (`make view` installs it)."""

import json
import uuid
from datetime import datetime

import pytest

from sardrone.sim.synthetic import write_synthetic_episode
from tests.conftest import START_NS

pytest.importorskip("fiftyone", reason="fiftyone not installed - run `make view` once")

import fiftyone as fo  # noqa: E402

from sardrone.fo.load import build_dataset, sidecar_to_fields  # noqa: E402


def test_sidecar_to_fields_is_flat_and_typed():
    fields = sidecar_to_fields(
        {
            "episode_id": "flight-0001",
            "world": "forest",
            "weather": "fog",
            "time_of_day": "dusk",
            "mission_prompt": "find the hiker",
            "people_placed": 2,
            "people_found": 1,
            "success": False,
            "duration_s": 24,
            "start_time_ns": START_NS + 500_000_000,
            "message_counts": {"/camera/front": 242, "/gps": 242},
            "event_counts": {"takeoff": 1, "land": 2},
            "mcap_file": "flight-0001.mcap",
        }
    )
    assert fields["weather"] == "fog"
    assert fields["people_placed"] == 2 and fields["people_found"] == 1
    assert fields["success"] is False
    assert fields["duration_s"] == 24.0 and isinstance(fields["duration_s"], float)
    assert fields["start_time"].isoformat() == "2026-11-14T00:00:00.500000+00:00"
    assert (fields["num_images"], fields["num_messages"], fields["num_events"]) == (242, 484, 3)
    assert "mcap_file" not in fields and "message_counts" not in fields


@pytest.fixture
def dataset_name():
    name = f"sar-episodes-test-{uuid.uuid4().hex[:8]}"
    yield name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


def test_build_dataset_one_multimodal_sample_per_mcap(tmp_path, dataset_name):
    write_synthetic_episode(tmp_path / "flight-0001.mcap", start_wall_ns=START_NS)
    write_synthetic_episode(tmp_path / "flight-0002.mcap", start_wall_ns=START_NS)
    (tmp_path / "flight-0002.json").unlink()  # an episode that lost its sidecar

    dataset = build_dataset(tmp_path, dataset_name)
    assert dataset.persistent is True
    assert dataset.media_type == "multimodal"
    assert len(dataset) == 2

    first, second = dataset.sort_by("filepath")
    assert first.filepath == str((tmp_path / "flight-0001.mcap").resolve())
    assert first.media_type == "multimodal"
    sidecar = json.loads((tmp_path / "flight-0001.json").read_text())
    for key in ("world", "weather", "time_of_day", "mission_prompt", "people_placed",
                "people_found", "success"):
        assert first[key] == sidecar[key]
    assert first["duration_s"] == 10.0
    assert first["num_images"] == 101
    assert first["start_time"] == datetime(2026, 11, 14, 0, 0, 0)
    assert second.tags == ["no_sidecar"]


def test_build_dataset_is_idempotent_and_tracks_the_folder(tmp_path, dataset_name):
    write_synthetic_episode(tmp_path / "flight-0001.mcap", start_wall_ns=START_NS)
    dataset = build_dataset(tmp_path, dataset_name)
    sample_id = dataset.first().id

    # Running again must not duplicate, and must pick up sidecar edits.
    sidecar_path = tmp_path / "flight-0001.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["weather"] = "fog"
    sidecar_path.write_text(json.dumps(sidecar))
    dataset = build_dataset(tmp_path, dataset_name)
    assert len(dataset) == 1
    assert dataset.first().id == sample_id
    assert dataset.first()["weather"] == "fog"

    # A new file is added; a deleted file is removed.
    write_synthetic_episode(tmp_path / "flight-0002.mcap", start_wall_ns=START_NS)
    (tmp_path / "flight-0001.mcap").unlink()
    dataset = build_dataset(tmp_path, dataset_name)
    assert [s["episode_id"] for s in dataset] == ["flight-0002"]
