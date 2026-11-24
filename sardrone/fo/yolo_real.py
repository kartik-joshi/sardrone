"""Load a real test set stored YOLO-style: each image with a .txt beside it,
one line per box, "class x_centre y_centre width height" in fractions.

    python -m sardrone.fo.yolo_real --name wisard-vis \\
        --dir data/real/wisard/sample/210417_MtErie_Enterprise_VIS_0003

Used for the WiSARD sample (issue #9): one real search flight from the
University of Washington, colour and thermal, frames in time order. Every
class becomes `person` (the sample's only class is 0). The whole folder is
a test set: saved view `<name>-test`, so sardrone.train.extra_real and
sardrone.train.tiled can score models on it. Nothing is trained on it.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def read_yolo_boxes(text: str) -> list[list[float]]:
    """[[x, y, w, h], ...] in FiftyOne's top-left format, from YOLO lines."""
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = (float(p) for p in parts)
        boxes.append([xc - w / 2, yc - h / 2, w, h])
    return boxes


def frame_number(path: Path) -> int | None:
    """The trailing number of a file name (its place in the flight), if any."""
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def build(name: str, folder: Path, overwrite: bool = True):
    import fiftyone as fo

    folder = Path(folder)
    if fo.dataset_exists(name):
        if not overwrite:
            raise SystemExit(f"{name} exists")
        fo.delete_dataset(name)
    dataset = fo.Dataset(name)
    dataset.persistent = True
    samples = []
    for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = image.with_suffix(".txt")
        boxes = read_yolo_boxes(label.read_text()) if label.exists() else []
        sample = fo.Sample(filepath=str(image.resolve()))
        sample["ground_truth"] = fo.Detections(detections=[fo.Detection(label="person", bounding_box=b) for b in boxes])
        sample["frame"] = frame_number(image)
        sample["labelled"] = label.exists()
        samples.append(sample)
    dataset.add_samples(samples, progress=False)
    dataset.compute_metadata()
    dataset.save_view(f"{name}-test", dataset.match(fo.ViewField("labelled")).sort_by("frame"))
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a YOLO-style real test set into FiftyOne.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = build(args.name, args.dir)
    view = dataset.load_saved_view(f"{args.name}-test")
    print(f"{args.name}: {len(view)} images, {view.count('ground_truth.detections')} people")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
