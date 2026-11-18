"""Load any Pascal VOC detection dataset as an extra real test set.

    python -m sardrone.fo.voc --name heridal --root data/real/heridal/<folder>

A second real test set answers a question SARD alone cannot: the models
that saw 300 SARD images in training were then tested on more SARD, from
the same footage. A dataset from other cameras, places and people says
whether anything carries over beyond "more of the same".

Every class in the files is mapped to `person` (these datasets label
only people). The dataset's own test split is used when it has one
(ImageSets/Main/test.txt); otherwise all images. Nothing from it is ever
trained on.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402

from sardrone.fo.real import read_voc_boxes  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def find_layout(root: Path) -> dict:
    """Locate Annotations/, the image folder and an optional test list under `root`."""
    root = Path(root)
    # Archives made on a Mac carry a __MACOSX folder that mirrors the real
    # one with metadata files ("._name.xml"); it must never be picked.
    annotations = next(
        (p for p in sorted(root.rglob("Annotations")) if p.is_dir() and "__MACOSX" not in p.parts), None
    )
    if annotations is None:
        raise FileNotFoundError(f"no Annotations folder under {root}")
    base = annotations.parent
    images = next((base / name for name in ("JPEGImages", "images", "Images") if (base / name).is_dir()), None)
    if images is None:
        raise FileNotFoundError(f"no image folder next to {annotations}")
    test_list = base / "ImageSets" / "Main" / "test.txt"
    return {"annotations": annotations, "images": images, "test_list": test_list if test_list.exists() else None}


def build(name: str, root: Path, log=print) -> fo.Dataset:
    """Create (or replace) dataset `name` from the VOC folder under `root`."""
    layout = find_layout(root)
    stems = (
        [line.strip() for line in layout["test_list"].read_text().splitlines() if line.strip()]
        if layout["test_list"]
        else sorted(p.stem for p in layout["annotations"].glob("*.xml"))
    )
    by_stem = {p.stem: p for p in layout["images"].iterdir() if p.suffix in IMAGE_SUFFIXES}
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    dataset = fo.Dataset(name, persistent=True)
    samples, classes, missing = [], Counter(), 0
    for stem in stems:
        xml_path = layout["annotations"] / f"{stem}.xml"
        image_path = by_stem.get(stem)
        if image_path is None or not xml_path.exists():
            missing += 1
            continue
        width, height, boxes = read_voc_boxes(xml_path)
        classes.update(box.pose for box in boxes)
        sample = fo.Sample(filepath=str(image_path))
        sample["source"] = name
        sample["split"] = "test"
        sample["num_people"] = len(boxes)
        sample["ground_truth"] = fo.Detections(
            detections=[
                fo.Detection(
                    label="person",
                    bounding_box=[b.x0 / width, b.y0 / height, (b.x1 - b.x0) / width, (b.y1 - b.y0) / height],
                    original_class=b.pose, difficult=b.difficult,
                    width_px=b.x1 - b.x0, height_px=b.y1 - b.y0,
                )  # fmt: skip
                for b in boxes
            ]
        )
        samples.append(sample)
    dataset.add_samples(samples, progress=False)
    dataset.save_view(f"{name}-test", dataset.match(fo.ViewField("split") == "test"), overwrite=True)
    dataset.info["source"] = {"root": str(root), "used_test_list": bool(layout["test_list"]), "classes": dict(classes)}
    dataset.save()
    if missing:
        log(f"{missing} listed images had no image or annotation file and were skipped")
    return dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a Pascal VOC dataset as an extra real test set.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    dataset = build(args.name, args.root)
    sizes = [max(w, h) for w, h in zip(dataset.values("ground_truth.detections.width_px", unwind=True),
                                       dataset.values("ground_truth.detections.height_px", unwind=True))]  # fmt: skip
    sizes.sort()
    print(f"dataset {dataset.name!r}: {len(dataset)} images, {len(sizes)} people, "
          f"test list used: {dataset.info['source']['used_test_list']}, classes {dataset.info['source']['classes']}")  # fmt: skip
    if sizes:
        print(f"person size (longer side, px): median {sizes[len(sizes) // 2]:.0f}, "
              f"5th percentile {sizes[len(sizes) // 20]:.0f}, 95th {sizes[len(sizes) * 19 // 20]:.0f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
