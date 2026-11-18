"""Load the real test set (SARD) into FiftyOne.

    make sard          # data/real/sard/SARD -> FiftyOne dataset "sard"

SARD is 1,981 frames of real drone video of people in fields, quarries and
forest, labelled by hand (Sambolek and Ivasic-Kos, CC BY 4.0). It is the
yardstick for the sim-to-real gap: a detector trained only on simulation is
scored on these pictures.

Two rules protect that yardstick:

* `sard-test` is FROZEN. Nothing is ever trained or tuned on it.
* The one experiment that does use real images ("+5 % real", Phase 3C)
  takes them from a separate part, `sard-train-pool`.

The two parts are cut by FRAME NUMBER, not at random. The frames come from
video, so frame 812 and frame 813 are nearly the same picture. A random cut
would put near-copies of test images into the training pool, and the test
would then partly measure memory. Cutting the sequence in one place leaves
only one seam between the parts.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import re  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402

DATASET_NAME = "sard"
DEFAULT_DIR = env.PROJECT_ROOT / "data" / "real" / "sard" / "SARD"
VIEW_TEST = "sard-test"
VIEW_POOL = "sard-train-pool"
POOL_FRACTION = 0.30  # the first 30 % of the frame sequence may be trained on

# SARD's six posture labels, as spelled in its files -> our pose names.
POSES = {
    "stands": "standing",
    "Walking": "walking",
    "Running": "running",
    "seated": "sitting",
    "laying_down": "lying",
    "not_defined": "unknown",
}


@dataclass(frozen=True)
class RealBox:
    pose: str
    x0: float
    y0: float
    x1: float
    y1: float
    difficult: bool


def frame_number(filename: str) -> int:
    """gss1299.jpg -> 1299. The number is the frame's place in the video sequence."""
    match = re.search(r"(\d+)", Path(filename).stem)
    if not match:
        raise ValueError(f"no frame number in {filename!r}")
    return int(match.group(1))


def read_voc_boxes(xml_path: Path) -> tuple[int, int, list[RealBox]]:
    """Any Pascal VOC file: width, height and its boxes, the class name kept
    as-is in `pose` (for datasets whose classes are not SARD's postures)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(float(size.find("width").text)), int(float(size.find("height").text))
    boxes = []
    for item in root.findall("object"):
        box = item.find("bndbox")
        x0, y0 = float(box.find("xmin").text) - 1.0, float(box.find("ymin").text) - 1.0
        x1, y1 = float(box.find("xmax").text), float(box.find("ymax").text)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        difficult = (item.findtext("difficult") or "0").strip() == "1"
        boxes.append(RealBox(item.findtext("name", "").strip(), x0, y0, x1, y1, difficult))
    return width, height, boxes


def read_annotation(xml_path: Path) -> tuple[int, int, list[RealBox]]:
    """One Pascal VOC file: image width, height and its boxes.

    VOC stores the first and last pixel of a box, counted from 1. Shifting
    the minimum by one pixel turns that into the edge convention used
    everywhere else here (0-based, width = x1 - x0).
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for item in root.findall("object"):
        name = item.find("name").text.strip()
        if name not in POSES:
            raise ValueError(f"{xml_path.name}: unknown class {name!r}")
        box = item.find("bndbox")
        x0, y0 = float(box.find("xmin").text) - 1.0, float(box.find("ymin").text) - 1.0
        x1, y1 = float(box.find("xmax").text), float(box.find("ymax").text)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            continue  # an empty box cannot be a label
        difficult = (item.findtext("difficult") or "0").strip() == "1"
        boxes.append(RealBox(POSES[name], x0, y0, x1, y1, difficult))
    return width, height, boxes


def split_of(number: int, boundary: int) -> str:
    return "train_pool" if number <= boundary else "test"


def split_boundary(numbers: list[int], pool_fraction: float = POOL_FRACTION) -> int:
    """The frame number at which the sequence is cut: frames up to and
    including it form the training pool."""
    ordered = sorted(numbers)
    return ordered[max(0, int(len(ordered) * pool_fraction) - 1)]


def build_sard(folder: Path = DEFAULT_DIR, name: str = DATASET_NAME, log=print) -> fo.Dataset:
    """Create (or replace) the dataset from the extracted SARD folder."""
    folder = Path(folder)
    annotations = sorted(folder.glob("*.xml"), key=lambda p: frame_number(p.name))
    if not annotations:
        raise FileNotFoundError(f"no SARD annotation files in {folder}")
    boundary = split_boundary([frame_number(p.name) for p in annotations])

    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)
    samples, missing = [], 0
    for xml_path in annotations:
        image_path = xml_path.with_suffix(".jpg")
        if not image_path.exists():
            missing += 1
            continue
        width, height, boxes = read_annotation(xml_path)
        number = frame_number(xml_path.name)
        sample = fo.Sample(filepath=str(image_path))
        sample["frame_number"] = number
        sample["split"] = split_of(number, boundary)
        sample["source"] = "sard"
        sample["num_people"] = len(boxes)
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(
                    label="person",  # every SARD posture class is a person
                    bounding_box=[
                        box.x0 / width, box.y0 / height, (box.x1 - box.x0) / width, (box.y1 - box.y0) / height,
                    ],
                    pose=box.pose,
                    difficult=box.difficult,
                    width_px=box.x1 - box.x0,
                    height_px=box.y1 - box.y0,
                )
                for box in boxes
            ]  # fmt: skip
        )
        samples.append(sample)
    dataset.add_samples(samples, progress=False)
    dataset.save_view(VIEW_TEST, dataset.match(F("split") == "test"), overwrite=True)
    dataset.save_view(VIEW_POOL, dataset.match(F("split") == "train_pool"), overwrite=True)
    dataset.info["source"] = {
        "name": "SARD - Search and Rescue Image Dataset for Person Detection",
        "authors": "Sasa Sambolek, Marina Ivasic-Kos",
        "licence": "CC BY 4.0",
        "doi": "10.21227/ahxm-k331",
        "split_boundary_frame": boundary,
        "pool_fraction": POOL_FRACTION,
    }
    dataset.save()
    if missing:
        log(f"{missing} annotation files had no image and were skipped")
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load SARD into FiftyOne.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--name", default=DATASET_NAME)
    args = parser.parse_args(argv)
    try:
        dataset = build_sard(args.dir, args.name)
    except FileNotFoundError as error:
        print(f"error: {error}")
        return 1
    test, pool = dataset.load_saved_view(VIEW_TEST), dataset.load_saved_view(VIEW_POOL)
    print(f"dataset {dataset.name!r}: {len(dataset)} images, {dataset.count('ground_truth.detections')} people")
    print(f"  {VIEW_TEST}: {len(test)} images, {test.count('ground_truth.detections')} people (frozen)")
    print(f"  {VIEW_POOL}: {len(pool)} images, {pool.count('ground_truth.detections')} people")
    print(f"  cut after frame {dataset.info['source']['split_boundary_frame']}")
    print("  poses:", dict(sorted(dataset.count_values("ground_truth.detections.pose").items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
