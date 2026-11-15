"""Turn recorded episodes into a labelled image dataset: `sar-frames`.

    make frames        # every labelled episode in data/episodes

An episode is a video-like recording; a detector trains on single pictures.
This step takes pictures out of each episode at 2 per second, saves them as
ordinary JPEG files, and attaches the simulator's exact label boxes as
FiftyOne `Detections`. Each picture keeps a link back to its episode and
timestamp, so any training image can be traced to the moment it came from.
"""

from pathlib import Path

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import argparse  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import fiftyone as fo  # noqa: E402

from sardrone.mcap.reader import (  # noqa: E402
    FrameBox,
    LabelledFrame,
    has_labels,
    read_calibration,
    read_labelled_frames,
)
from sardrone.mission.oracle import box_corners  # noqa: E402
from sardrone.mission.projection import CameraModel, DronePose, project  # noqa: E402

DATASET_NAME = "sar-frames"
DEFAULT_EPISODE_DIR = PROJECT_ROOT / "data" / "episodes"
DEFAULT_FRAME_DIR = PROJECT_ROOT / "data" / "frames"

MIN_HEIGHT_M = 2.0  # skip pictures taken on or near the ground
MIN_BOX_PX = 4.0  # a box smaller than this on its longer side is not a usable label
EDGE_PX = 1.0  # a box this close to the image border is "truncated"

# Episode facts copied onto every frame, so frames can be filtered by them.
EPISODE_FIELDS = ("world", "world_id", "weather", "time_of_day", "seed", "altitude_m", "occlusion")


@dataclass
class ExtractStats:
    frames: int = 0
    boxes: int = 0
    dropped_small: int = 0
    skipped_low: int = 0


def normalise_box(box: FrameBox, width: int, height: int) -> list[float]:
    """Pixel edges -> FiftyOne's [x, y, w, h], each as a fraction of the image.

    FiftyOne stores boxes relative to the image size (0..1), with x, y the
    top-left corner. That makes a label independent of the image resolution.
    """
    return [box.x0 / width, box.y0 / height, box.width / width, box.height / height]


def is_truncated(box: FrameBox, width: int, height: int) -> bool:
    """The box touches the image border: part of the person is out of view."""
    return box.x0 <= EDGE_PX or box.y0 <= EDGE_PX or box.x1 >= width - EDGE_PX or box.y1 >= height - EDGE_PX


def people_expected_by_geometry(frame: LabelledFrame, people: list[dict], model: CameraModel) -> int:
    """How many people SHOULD be in view, from 3D positions and the drone's pose.

    An independent cross-check on the label count: the centre of each
    person's true 3D box is projected into the image, and people whose
    centre lands inside the picture are counted. It uses PX4's ESTIMATED
    pose and ignores trees, so it cannot agree with the labels on every
    frame; large or systematic disagreement would mean something is wrong.
    """
    if frame.position is None:
        return 0
    pose = DronePose(frame.position, frame.orientation)
    count = 0
    for person in people:
        pixel = project(model, pose, person["box_centre"])
        if pixel and 0 <= pixel[0] < model.width and 0 <= pixel[1] < model.height:
            count += 1
    return count


def detections_for(frame: LabelledFrame, people_by_id: dict, width: int, height: int, stats: ExtractStats):
    detections = []
    for box in frame.boxes:
        if max(box.width, box.height) < MIN_BOX_PX:
            stats.dropped_small += 1
            continue
        person = people_by_id.get(box.id, {})
        detections.append(
            fo.Detection(
                label="person",
                bounding_box=normalise_box(box, width, height),
                person_id=box.id,
                pose=person.get("pose"),
                clothing=person.get("clothing"),
                occlusion=person.get("occlusion"),
                width_px=float(box.width),
                height_px=float(box.height),
                truncated=is_truncated(box, width, height),
            )
        )
    return fo.Detections(detections=detections)


def extract_episode(
    mcap_path: Path, dataset: fo.Dataset, frame_dir: Path = DEFAULT_FRAME_DIR, rate_hz: float = 2.0
) -> ExtractStats:
    """Add one episode's frames to the dataset (replacing any it had before)."""
    mcap_path = Path(mcap_path)
    sidecar = json.loads(mcap_path.with_suffix(".json").read_text())
    episode_id = sidecar["episode_id"]
    people = sidecar.get("people", [])
    people_by_id = {person["id"]: person for person in people}
    calibration = read_calibration(mcap_path)
    width, height = calibration["width"], calibration["height"]
    model = CameraModel(
        width, height, calibration["fx"], calibration["fy"], calibration["cx"], calibration["cy"],
        tuple(sidecar.get("camera_mount_m", (0.0, 0.0, 0.0))), sidecar.get("camera_pitch_deg", 0.0),
    )  # fmt: skip

    if dataset.has_field("episode_id"):
        dataset.delete_samples(dataset.match(fo.ViewField("episode_id") == episode_id))
    out_dir = Path(frame_dir) / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    stats = ExtractStats()
    samples = []
    for frame in read_labelled_frames(mcap_path, rate_hz):
        if frame.height_m < MIN_HEIGHT_M:
            stats.skipped_low += 1
            continue
        # The camera's JPEG bytes are written unchanged: no second compression.
        filepath = out_dir / f"{episode_id}_{round(frame.time_s * 1000):07d}.jpg"
        filepath.write_bytes(frame.jpeg)

        sample = fo.Sample(filepath=str(filepath))
        sample["episode_id"] = episode_id
        sample["timestamp_ns"] = frame.time_ns
        sample["time_s"] = float(frame.time_s)
        sample["height_m"] = float(frame.height_m)
        sample["mission_state"] = frame.mission_state
        for key in EPISODE_FIELDS:
            if key in sidecar:
                value = sidecar[key]
                sample[key] = float(value) if key == "altitude_m" else value
        sample["ground_truth"] = detections_for(frame, people_by_id, width, height, stats)
        sample["num_people"] = len(sample["ground_truth"].detections)
        sample["expected_by_geometry"] = people_expected_by_geometry(frame, people, model)
        samples.append(sample)
        stats.frames += 1
        stats.boxes += sample["num_people"]
    if samples:
        dataset.add_samples(samples, progress=False)
    return stats


def build_frames_dataset(
    episode_dir: Path = DEFAULT_EPISODE_DIR,
    frame_dir: Path = DEFAULT_FRAME_DIR,
    name: str = DATASET_NAME,
    rate_hz: float = 2.0,
    log=print,
) -> fo.Dataset:
    """Create or refresh the frames dataset from every labelled episode."""
    dataset = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name, persistent=True)
    for mcap_path in sorted(Path(episode_dir).glob("*.mcap")):
        if not mcap_path.with_suffix(".json").exists() or not has_labels(mcap_path):
            continue  # generated test images or older recordings: no label boxes
        stats = extract_episode(mcap_path, dataset, frame_dir, rate_hz)
        log(
            f"{mcap_path.stem}: {stats.frames} frames, {stats.boxes} boxes "
            f"({stats.dropped_small} tiny boxes dropped, {stats.skipped_low} ground frames skipped)"
        )
    return dataset


def count_agreement(dataset) -> dict:
    """Compare label counts with the geometric expectation, frame by frame."""
    labelled = dataset.values("num_people")
    expected = dataset.values("expected_by_geometry")
    pairs = list(zip(labelled, expected))
    total = len(pairs)
    return {
        "frames": total,
        "same": sum(a == b for a, b in pairs),
        "within_one": sum(abs(a - b) <= 1 for a, b in pairs),
        "labels_fewer": sum(a < b for a, b in pairs),
        "labels_more": sum(a > b for a, b in pairs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract labelled frames into FiftyOne.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--name", default=DATASET_NAME)
    parser.add_argument("--rate", type=float, default=2.0, help="frames per second to keep")
    args = parser.parse_args(argv)

    dataset = build_frames_dataset(args.dir, args.frames, args.name, args.rate)
    if len(dataset) == 0:
        print("no labelled episodes found - record one with `make search` or `make batch`")
        return 1
    print(f"dataset {dataset.name!r}: {len(dataset)} frames from "
          f"{len(dataset.distinct('episode_id'))} episodes, {dataset.count('ground_truth.detections')} boxes")  # fmt: skip
    print("people per frame:", dict(sorted(dataset.count_values("num_people").items())))
    agreement = count_agreement(dataset)
    print(
        f"label count vs geometry: same in {agreement['same']} of {agreement['frames']} frames, "
        f"within one in {agreement['within_one']}; labels fewer in {agreement['labels_fewer']}, "
        f"more in {agreement['labels_more']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
