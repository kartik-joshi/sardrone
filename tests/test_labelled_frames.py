"""Phase 2D: reading labelled frames out of an episode, and building the
FiftyOne image dataset with ground-truth boxes."""

import json
import uuid

import pytest
from PIL import Image

from sardrone.mcap.episode_writer import EVENT_STATE, EpisodeMeta, EpisodeWriter
from sardrone.mcap.frames import Quat
from sardrone.mcap.reader import FrameBox, has_labels, read_calibration, read_labelled_frames
from sardrone.sim.camera import SyntheticCamera, pinhole_intrinsics
from tests.conftest import START_NS

TENTH = 100_000_000  # 0.1 s in nanoseconds
PEOPLE = [
    {"id": "person_1", "label": 1, "pose": "standing", "clothing": "red", "occlusion": "none",
     "box_centre": [0.0, 15.0, 0.9], "box_size": [0.55, 0.36, 1.8], "box_quat_wxyz": [1, 0, 0, 0]},
    {"id": "person_2", "label": 2, "pose": "lying", "clothing": "blue", "occlusion": "partial",
     "box_centre": [0.0, -40.0, 0.2], "box_size": [1.36, 0.36, 1.82], "box_quat_wxyz": [1, 0, 0, 0]},
]  # fmt: skip


def write_labelled_episode(path, seconds=3.0, height_m=15.0, climb_after_s=0.0):
    """A 10 Hz episode: 640x480 pictures, one or two boxes on each, facing north."""
    camera = SyntheticCamera(pitch_deg=45.0)
    meta = EpisodeMeta(world="meadow", weather="clear", time_of_day="noon", mission_prompt="p")
    extra = {"world_id": "w-1", "seed": 9, "altitude_m": 15, "occlusion": "none", "people": PEOPLE}
    fx, fy, cx, cy = pinhole_intrinsics(640, 480, 80.0)
    # Facing north in NED: yaw 0, level.
    level = Quat(1.0, 0.0, 0.0, 0.0)
    with EpisodeWriter(path, meta, camera_pitch_deg=45.0, extra_sidecar=extra) as episode:
        episode.log_calibration(START_NS, 640, 480, fx, fy, cx, cy)
        for tick in range(int(seconds * 10) + 1):
            t = START_NS + tick * TENTH
            height = height_m if tick / 10 >= climb_after_s else 0.5
            episode.log_pose_ned(t, 0.0, 0.0, -height, level)
            if tick == 0:
                episode.log_event(t, EVENT_STATE, "START -> TAKEOFF")
            if tick == 10:
                episode.log_event(t, EVENT_STATE, "TAKEOFF -> SEARCH (2 lanes, 30 m apart)")
            episode.log_image(t, camera.read())
            if tick == 12:
                continue  # one picture with NO label message at all
            boxes = [{"id": "person_1", "x0": 300 + tick, "y0": 200, "x1": 320 + tick, "y1": 250}]
            if tick >= 20:
                boxes.append({"id": "person_2", "x0": 0, "y0": 100, "x1": 3, "y1": 102})  # a tiny sliver
            episode.log_ground_truth_boxes(t, boxes)
    return path


# ---- reader ------------------------------------------------------------------


def test_frames_are_thinned_to_two_per_second(tmp_path):
    path = write_labelled_episode(tmp_path / "ep.mcap")
    frames = list(read_labelled_frames(path))
    # 3.0 s at 2 Hz: 0.0, 0.5, ... 3.0 = 7 frames.
    assert [f.time_s for f in frames] == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert [f.time_ns for f in frames] == [START_NS + k * 5 * TENTH for k in range(7)]
    assert all(f.jpeg.startswith(b"\xff\xd8") for f in frames)


def test_every_labelled_frame_at_full_rate_and_unlabelled_ones_are_skipped(tmp_path):
    path = write_labelled_episode(tmp_path / "ep.mcap")
    frames = list(read_labelled_frames(path, rate_hz=None))
    assert len(frames) == 30  # 31 pictures, one of them without labels
    assert START_NS + 12 * TENTH not in [f.time_ns for f in frames]


def test_frame_carries_boxes_pose_and_mission_state(tmp_path):
    path = write_labelled_episode(tmp_path / "ep.mcap")
    frames = list(read_labelled_frames(path))
    assert frames[0].boxes == (FrameBox("person_1", 300.0, 200.0, 320.0, 250.0),)
    assert frames[1].boxes[0].x0 == 305.0  # the box recorded at tick 5, not a neighbour's
    assert [b.id for b in frames[4].boxes] == ["person_1", "person_2"]
    assert frames[4].boxes[0].width == 20.0 and frames[4].boxes[0].height == 50.0
    assert [f.mission_state for f in frames] == ["TAKEOFF", "TAKEOFF", "SEARCH", "SEARCH", "SEARCH", "SEARCH", "SEARCH"]
    assert frames[3].height_m == pytest.approx(15.0)
    assert (frames[3].position.x, frames[3].position.y) == pytest.approx((0.0, 0.0))


def test_has_labels_and_calibration(tmp_path):
    path = write_labelled_episode(tmp_path / "ep.mcap")
    assert has_labels(path) is True
    calibration = read_calibration(path)
    assert (calibration["width"], calibration["height"]) == (640, 480)
    assert calibration["cx"] == 320.0 and calibration["fx"] == pytest.approx(381.4, abs=0.1)

    from sardrone.sim.synthetic import write_synthetic_episode

    assert has_labels(write_synthetic_episode(tmp_path / "plain.mcap", start_wall_ns=START_NS).path) is False


# ---- FiftyOne dataset --------------------------------------------------------

fo = pytest.importorskip("fiftyone", reason="fiftyone not installed - run `make view` once")

from sardrone.fo import frames as frames_module  # noqa: E402
from sardrone.fo.frames import (  # noqa: E402
    build_frames_dataset,
    count_agreement,
    is_truncated,
    normalise_box,
)
from sardrone.fo.overlay import box_pixels, save_overlays  # noqa: E402


@pytest.fixture
def dataset_name():
    name = f"test-frames-{uuid.uuid4().hex[:8]}"
    yield name
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


def test_normalise_box_numbers():
    box = FrameBox("p", 320.0, 240.0, 384.0, 336.0)  # 64 x 96 px in a 1280 x 960 image
    assert normalise_box(box, 1280, 960) == pytest.approx([0.25, 0.25, 0.05, 0.1])
    assert normalise_box(FrameBox("p", 0, 0, 1280, 960), 1280, 960) == [0.0, 0.0, 1.0, 1.0]


def test_truncated_means_touching_the_border():
    assert is_truncated(FrameBox("p", 0.0, 100.0, 30.0, 150.0), 1280, 960) is True
    assert is_truncated(FrameBox("p", 1250.0, 100.0, 1280.0, 150.0), 1280, 960) is True
    assert is_truncated(FrameBox("p", 600.0, 400.0, 640.0, 480.0), 1280, 960) is False


def test_frames_dataset_has_images_links_and_boxes(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap")
    messages = []
    dataset = build_frames_dataset(
        tmp_path / "episodes", tmp_path / "frames", dataset_name, log=messages.append
    )
    assert len(dataset) == 7 and dataset.media_type == "image"
    assert "7 frames" in messages[0]

    samples = list(dataset.sort_by("time_s"))
    first = samples[0]
    # The picture on disk is the camera's own JPEG, unchanged.
    assert Image.open(first.filepath).size == (640, 480)
    assert first.filepath.endswith("ep-one/ep-one_0000000.jpg")
    assert samples[1].filepath.endswith("ep-one_0000500.jpg")
    # Link back to the episode and the exact moment.
    assert first.episode_id == "ep-one" and first.timestamp_ns == START_NS
    assert samples[3].timestamp_ns == START_NS + 15 * TENTH and samples[3].time_s == 1.5
    assert first.world == "meadow" and first.world_id == "w-1" and first.seed == 9
    assert first.altitude_m == 15.0 and first.time_of_day == "noon"
    assert samples[2].mission_state == "SEARCH" and first.height_m == pytest.approx(15.0)

    detection = first.ground_truth.detections[0]
    assert detection.label == "person" and detection.person_id == "person_1"
    assert detection.pose == "standing" and detection.clothing == "red"
    # Relative box -> back to the pixels that were recorded.
    assert box_pixels(detection, 640, 480) == pytest.approx((300.0, 200.0, 320.0, 250.0))
    assert detection.bounding_box == pytest.approx([300 / 640, 200 / 480, 20 / 640, 50 / 480])
    assert (detection.width_px, detection.height_px, detection.truncated) == (20.0, 50.0, False)


def test_box_counts_and_tiny_boxes(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap")
    messages = []
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=messages.append)
    # person_2's 3 x 2 px sliver appears from 2.0 s on (3 frames) and is dropped.
    assert "3 tiny boxes dropped" in messages[0]
    assert dataset.values("num_people") == [1] * 7
    assert dataset.count("ground_truth.detections") == 7
    for sample in dataset:
        assert sample.num_people == len(sample.ground_truth.detections)
        for detection in sample.ground_truth.detections:
            x, y, w, h = detection.bounding_box
            assert 0 <= x and 0 <= y and x + w <= 1 and y + h <= 1 and w > 0 and h > 0
    # Geometry agrees: facing north from 15 m, person_1 (15 m north) is in
    # view and person_2 (40 m south, behind the drone) is not.
    assert dataset.values("expected_by_geometry") == [1] * 7
    assert count_agreement(dataset) == {
        "frames": 7, "same": 7, "within_one": 7, "labels_fewer": 0, "labels_more": 0,
    }  # fmt: skip


def test_ground_frames_are_skipped(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap", climb_after_s=1.0)
    messages = []
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=messages.append)
    assert "2 ground frames skipped" in messages[0]  # t = 0.0 and 0.5 s at 0.5 m height
    assert min(dataset.values("height_m")) == pytest.approx(15.0)
    assert len(dataset) == 5


def test_extracting_again_replaces_instead_of_duplicating(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap")
    write_labelled_episode(tmp_path / "episodes" / "ep-two.mcap", seconds=1.0)
    for _ in range(2):
        dataset = build_frames_dataset(
            tmp_path / "episodes", tmp_path / "frames", dataset_name, log=lambda m: None, force=True
        )
    assert dataset.count_values("episode_id") == {"ep-one": 7, "ep-two": 3}
    assert len(list((tmp_path / "frames" / "ep-one").glob("*.jpg"))) == 7


def test_unlabelled_episodes_are_left_out(tmp_path, dataset_name):
    from sardrone.sim.synthetic import write_synthetic_episode

    write_synthetic_episode(tmp_path / "episodes" / "plain.mcap", start_wall_ns=START_NS)
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=lambda m: None)
    assert len(dataset) == 0


def test_overlays_are_written_and_seeded(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap")
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=lambda m: None)
    first = save_overlays(dataset, tmp_path / "overlays", count=4, seed=1)
    assert first[-1].name == "sheet.jpg" and len(first) == 5
    overlay = Image.open(first[0])
    assert overlay.size == (640, 480 + 320 + 24)  # frame + caption + zoom strip
    # A magenta outline was drawn just outside the box of frame 0 (x0 = 300, y0 = 200).
    names = [p.name for p in first]
    again = [p.name for p in save_overlays(dataset, tmp_path / "overlays", count=4, seed=1)]
    assert names == again
    assert Image.open(tmp_path / "overlays" / "sheet.jpg").width == 4 * 480


def test_episodes_already_in_the_dataset_keep_what_was_computed_on_them(tmp_path, dataset_name):
    write_labelled_episode(tmp_path / "episodes" / "ep-one.mcap")
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=lambda m: None)
    dataset.set_values("clip_embedding", [[1.0, 0.0]] * len(dataset))  # stands for anything computed later
    write_labelled_episode(tmp_path / "episodes" / "ep-two.mcap", seconds=1.0)
    messages = []
    dataset = build_frames_dataset(tmp_path / "episodes", tmp_path / "frames", dataset_name, log=messages.append)
    assert len(messages) == 1 and messages[0].startswith("ep-two")  # only the new episode was extracted
    old = dataset.match(fo.ViewField("episode_id") == "ep-one")
    assert old.values("clip_embedding") == [[1.0, 0.0]] * 7, "existing samples were not replaced"
    assert dataset.count_values("episode_id") == {"ep-one": 7, "ep-two": 3}


def test_voc_loader_maps_every_class_to_person_and_uses_the_test_list(tmp_path, dataset_name):
    from sardrone.fo import voc

    base = tmp_path / "voc"  # lowercase sorts AFTER "__MACOSX", as in the real archive
    (base / "Annotations").mkdir(parents=True)
    (base / "JPEGImages").mkdir()
    (base / "ImageSets" / "Main").mkdir(parents=True)
    for stem, boxes in (("a", [("human", 11, 21, 30, 60)]), ("b", [("person", 1, 1, 10, 10), ("human", 50, 50, 70, 90)]), ("c", [])):
        Image.new("RGB", (100, 100)).save(base / "JPEGImages" / f"{stem}.jpg")
        objects = "".join(
            f"<object><name>{n}</name><difficult>0</difficult><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
            f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>" for n, x0, y0, x1, y1 in boxes
        )
        (base / "Annotations" / f"{stem}.xml").write_text(
            f"<annotation><size><width>100</width><height>100</height><depth>3</depth></size>{objects}</annotation>"
        )
    (base / "ImageSets" / "Main" / "test.txt").write_text("a\nb\n")  # c is not in the test split
    # A __MACOSX mirror, which sorts before "voc", must be ignored.
    junk = tmp_path / "__MACOSX" / "voc"
    (junk / "Annotations").mkdir(parents=True)
    (junk / "JPEGImages").mkdir()
    (junk / "Annotations" / "._a.xml").write_bytes(b"\x00\x05\x16\x07 not xml")
    dataset = voc.build(dataset_name, tmp_path, log=lambda m: None)
    assert len(dataset) == 2 and dataset.count("ground_truth.detections") == 3
    assert set(dataset.distinct("ground_truth.detections.label")) == {"person"}
    assert dataset.info["source"]["classes"] == {"human": 2, "person": 1}
    first = dataset.match(fo.ViewField("filepath").ends_with("a.jpg")).first().ground_truth.detections[0]
    # VOC pixels 11..30 -> edges 10..30, so x = 0.10, width = 0.20.
    assert first.bounding_box == pytest.approx([0.10, 0.20, 0.20, 0.40])
    assert len(dataset.load_saved_view(f"{dataset_name}-test")) == 2
