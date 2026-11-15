"""Draw the labels on random frames, for a human to check.

    make overlay       # -> data/overlays/*.jpg and data/overlays/sheet.jpg

Numbers can pass while the labels are still wrong (shifted, mirrored, on the
wrong object). Looking at a random sample is the only real check, so this
saves 20 frames with their boxes drawn, plus a zoomed crop of each box.
The same seed always picks the same frames.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("FIFTYONE_DATABASE_DIR", str(PROJECT_ROOT / ".fiftyone" / "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

import argparse  # noqa: E402

import fiftyone as fo  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sardrone.fo.frames import DATASET_NAME  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "data" / "overlays"
MAGENTA = (255, 0, 255)


def box_pixels(detection, width: int, height: int) -> tuple[float, float, float, float]:
    """FiftyOne's relative [x, y, w, h] back to pixel edges x0, y0, x1, y1."""
    x, y, w, h = detection.bounding_box
    return x * width, y * height, (x + w) * width, (y + h) * height


def draw_overlay(sample) -> Image.Image:
    """The frame with its boxes, and a strip of 4x zoomed crops underneath."""
    image = Image.open(sample.filepath).convert("RGB")
    width, height = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    crops = []
    for detection in sample.ground_truth.detections:
        x0, y0, x1, y1 = box_pixels(detection, width, height)
        draw.rectangle([x0 - 2, y0 - 2, x1 + 1, y1 + 1], outline=MAGENTA, width=2)
        draw.text((x0, max(0, y0 - 14)), detection.person_id or detection.label, fill=(255, 255, 255))

        # Crop from the CLEAN image, so the zoom shows what the box really contains.
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left, top = int(cx - 40), int(cy - 40)
        crop = image.crop((left, top, left + 80, top + 80)).resize((320, 320), Image.NEAREST)
        zoom = ImageDraw.Draw(crop)
        zoom.rectangle([(x0 - left) * 4, (y0 - top) * 4, (x1 - left) * 4, (y1 - top) * 4], outline=MAGENTA)
        crops.append(crop)

    caption = (
        f"{sample.episode_id}  t={sample.time_s:.1f}s  {sample.mission_state}  "
        f"h={sample.height_m:.0f}m  labels={sample.num_people}  geometry={sample.expected_by_geometry}"
    )
    out = Image.new("RGB", (width, height + 320 + 24), (20, 20, 20))
    out.paste(canvas, (0, 24))
    ImageDraw.Draw(out).text((6, 6), caption, fill=(255, 255, 255))
    for index, crop in enumerate(crops[:4]):
        out.paste(crop, (index * 320, height + 24))
    return out


def save_overlays(dataset, out_dir: Path = DEFAULT_OUT, count: int = 20, seed: int = 51) -> list[Path]:
    """`count` random frames; half of them chosen among frames that HAVE
    people (random frames alone are mostly empty ground)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    with_people = dataset.match(fo.ViewField("num_people") > 0).take(count // 2, seed=seed)
    chosen = {sample.id: sample for sample in with_people}
    for sample in dataset.take(count * 3, seed=seed + 1):
        if len(chosen) >= count:
            break
        chosen.setdefault(sample.id, sample)

    paths, thumbs = [], []
    for sample in sorted(chosen.values(), key=lambda s: s.filepath):
        overlay = draw_overlay(sample)
        path = out_dir / f"overlay_{Path(sample.filepath).stem}.jpg"
        overlay.save(path, quality=88)
        paths.append(path)
        thumbs.append(overlay.resize((480, round(480 * overlay.height / overlay.width))))
    if thumbs:
        columns = 4
        rows = -(-len(thumbs) // columns)
        sheet = Image.new("RGB", (columns * 480, rows * thumbs[0].height), (20, 20, 20))
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, ((index % columns) * 480, (index // columns) * thumb.height))
        sheet.save(out_dir / "sheet.jpg", quality=85)
        paths.append(out_dir / "sheet.jpg")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save random frames with their labels drawn.")
    parser.add_argument("--name", default=DATASET_NAME)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args(argv)
    if not fo.dataset_exists(args.name):
        print(f"dataset {args.name!r} does not exist - run `make frames` first")
        return 1
    paths = save_overlays(fo.load_dataset(args.name), args.out, args.count, args.seed)
    print(f"wrote {len(paths) - 1} overlays and {paths[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
