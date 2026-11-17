"""Find a model's false alarms and make them reviewable.

    python -m sardrone.fo.false_alarms --version v4

A FALSE ALARM is a box the model drew, at or above its chosen confidence
cut-off, that overlaps no true person (IoU below 0.5). They are worth
looking at one by one: a pattern in them (a shadow, a rock, a path edge)
is something the next round of data should contain.

* On the simulated frames, each frame gets `false_alarms` = how many such
  boxes it has. The saved view `sar-false-alarms` (created in Phase 2E,
  empty until now) is every frame with at least one.
* On the real test images, the most confident false alarms are cut out
  and saved as a contact sheet, so a person can see what fools the model.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402

from sardrone.train.metrics import ImageResult, match_image  # noqa: E402
from sardrone.train.scorecard import DEFAULT_CSV  # noqa: E402

DEFAULT_SHEET = env.PROJECT_ROOT / "docs" / "images"


def corners(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)


def false_alarm_boxes(truths, predictions, confidence: float, iou: float = 0.5) -> list:
    """The predicted Detections (at or above `confidence`) that match no true person.

    Uses the same matching as the scorecard: most confident first, each
    true person claimed once.
    """
    kept = [d for d in (predictions.detections if predictions else []) if d.label == "person" and d.confidence >= confidence]
    kept.sort(key=lambda d: -d.confidence)
    image = ImageResult(
        truths=tuple(corners(d.bounding_box) for d in (truths.detections if truths else []) if d.label == "person"),
        predictions=tuple((d.confidence, corners(d.bounding_box)) for d in kept),
    )
    outcome = match_image(image, iou)
    # match_image sorts by confidence the same way, so positions line up.
    return [detection for detection, (_, is_true) in zip(kept, outcome) if not is_true]


def scorecard_cutoff(version: str, path: Path = DEFAULT_CSV) -> float:
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["version"] == version:
                return float(row["confidence"])
    raise ValueError(f"{version} is not in {path}")


def mark_frames(view, field: str, confidence: float) -> dict:
    """Write `false_alarms` (a count) on every frame of the view."""
    ids = view.values("id")
    counts = [
        len(false_alarm_boxes(truths, predictions, confidence))
        for truths, predictions in zip(view.values("ground_truth"), view.values(field))
    ]
    view._dataset.set_values("false_alarms", dict(zip(ids, counts)), key_field="id")
    return {"frames": len(ids), "frames_with_false_alarms": sum(c > 0 for c in counts), "false_alarm_boxes": sum(counts)}


def contact_sheet(view, field: str, confidence: float, out: Path, count: int = 24, crop_px: int = 160) -> list[dict]:
    """The `count` most confident false alarms, cropped around the box, in a grid."""
    from PIL import Image, ImageDraw

    found = []
    for filepath, truths, predictions in zip(view.values("filepath"), view.values("ground_truth"), view.values(field)):
        for detection in false_alarm_boxes(truths, predictions, confidence):
            found.append((detection.confidence, filepath, detection.bounding_box))
    found.sort(key=lambda item: -item[0])
    chosen = found[:count]
    tile = 240
    columns = 6
    sheet = Image.new("RGB", (columns * tile, -(-len(chosen) // columns) * (tile + 18)), (20, 20, 20))
    rows = []
    for index, (score, filepath, (x, y, w, h)) in enumerate(chosen):
        image = Image.open(filepath).convert("RGB")
        width, height = image.size
        cx, cy = (x + w / 2) * width, (y + h / 2) * height
        half = max(crop_px / 2, max(w * width, h * height))
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        crop = image.crop(box).resize((tile, tile))
        draw = ImageDraw.Draw(crop)
        scale = tile / (box[2] - box[0])
        draw.rectangle([(x * width - box[0]) * scale, (y * height - box[1]) * scale,
                        ((x + w) * width - box[0]) * scale, ((y + h) * height - box[1]) * scale],
                       outline=(255, 230, 0), width=2)  # fmt: skip
        left, top = (index % columns) * tile, (index // columns) * (tile + 18)
        sheet.paste(crop, (left, top + 18))
        ImageDraw.Draw(sheet).text((left + 4, top + 3), f"{Path(filepath).name}  {score:.2f}", fill=(255, 255, 255))
        rows.append({"file": Path(filepath).name, "confidence": round(score, 3),
                     "width_px": round(w * width, 1), "height_px": round(h * height, 1)})  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return rows


def main(argv: list[str] | None = None) -> int:
    from sardrone.fo import curate, real
    from sardrone.fo.frames import DATASET_NAME

    parser = argparse.ArgumentParser(description="Mark false alarms and save the most confident ones.")
    parser.add_argument("--version", default="v4")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args(argv)
    field = f"pred_{args.version}"
    confidence = scorecard_cutoff(args.version)

    frames = fo.load_dataset(DATASET_NAME)
    marked = {}
    for name in (curate.VIEW_VAL, curate.VIEW_TEST):
        view = frames.load_saved_view(name)
        marked[name] = mark_frames(view, field, confidence)
    frames.info["false_alarms"] = {"version": args.version, "confidence": confidence, "iou": 0.5, **marked}
    frames.save()
    review = frames.load_saved_view(curate.VIEW_FALSE_ALARMS)
    print(f"{args.version} at confidence {confidence}: {marked}")
    print(f"saved view {curate.VIEW_FALSE_ALARMS!r} now holds {len(review)} frames")

    sard_test = fo.load_dataset(real.DATASET_NAME).load_saved_view(real.VIEW_TEST)
    real_count = sum(
        len(false_alarm_boxes(t, p, confidence)) for t, p in zip(sard_test.values("ground_truth"), sard_test.values(field))
    )
    sheet = DEFAULT_SHEET / f"{args.version}_false_alarms_real.jpg"
    rows = contact_sheet(sard_test, field, confidence, sheet, args.count)
    print(f"real test images: {real_count} false alarms at that cut-off; the {len(rows)} most confident -> {sheet}")
    for row in rows[:8]:
        print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
