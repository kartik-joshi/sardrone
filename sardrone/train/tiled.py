"""Tiled detection: run the detector on full-resolution tiles as well as the whole image.

    python -m sardrone.train.tiled --dataset heridal --versions v0 v4 v6 v8
    python -m sardrone.train.tiled --dataset sard --versions v0 v4 v6 v8

Why: the detector looks at images shrunk to 1280 px. A HERIDAL photo is
4000 px wide, so a person of about 56 px becomes about 18 px, near the
limit of what the model can see. Cutting the photo into 1280 px tiles at
full resolution shows the model each person at its real size.

The method is fixed in advance (TILE, OVERLAP, NMS_IOU below), not tuned on
the test sets:

  1. Cut the image into TILE x TILE tiles that overlap by OVERLAP pixels and
     cover it exactly (the last tile in each row or column is aligned with
     the image edge).
  2. Run the model on every tile at TILE resolution (no shrinking), and once
     on the whole image as before, so large, close people are still found.
  3. Drop tile boxes that touch a tile's inner cut line: the person is cut in
     two there, and the overlapping neighbour tile holds them whole.
  4. Merge everything with non-maximum suppression at IoU NMS_IOU and keep the
     MAX_BOXES most confident boxes, as for untiled predictions.

Each model keeps its scorecard confidence cut-off for precision and recall;
mAP does not depend on it. Results go to docs/results/tiled.csv, with the
untiled score of the same model on the same images next to it.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from sardrone.train.predict import MAX_BOXES, MIN_CONFIDENCE, field_for  # noqa: E402

TILE = 1280
OVERLAP = 256
NMS_IOU = 0.5
EDGE_PX = 2.0  # a box this close to an inner cut line counts as touching it

DEFAULT_CSV = env.PROJECT_ROOT / "docs" / "results" / "tiled.csv"
VIEWS = {"heridal": ("heridal", "heridal-test"), "sard": ("sard", "sard-test")}
FIELDS = ["dataset", "version", "method", "images", "people", "confidence", "map50", "map50_low", "map50_high",
          "map50_95", "precision", "recall", "max_recall", "tile", "overlap", "commit", "dirty", "evaluated_utc"]  # fmt: skip


def tile_starts(length: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    """Start offsets along one axis: tiles of `tile` px, overlapping by at least
    `overlap`, the first at 0 and the last ending exactly at `length`."""
    if length <= tile:
        return [0]
    count = math.ceil((length - overlap) / (tile - overlap))
    last = length - tile
    return [round(i * last / (count - 1)) for i in range(count)]


def tile_grid(width: int, height: int, tile: int = TILE, overlap: int = OVERLAP) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) pixel boxes of every tile; tiles never extend past the image."""
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in tile_starts(height, tile, overlap)
        for x in tile_starts(width, tile, overlap)
    ]


def keep_off_inner_edges(boxes, tile_box, width: int, height: int, margin: float = EDGE_PX) -> list[bool]:
    """For boxes in IMAGE pixels found in `tile_box`: False for boxes touching
    an edge of the tile that is a cut line inside the image (not the image border)."""
    x0, y0, x1, y1 = tile_box
    keep = []
    for bx0, by0, bx1, by1 in boxes:
        touches = (
            (x0 > 0 and bx0 <= x0 + margin)
            or (y0 > 0 and by0 <= y0 + margin)
            or (x1 < width and bx1 >= x1 - margin)
            or (y1 < height and by1 >= y1 - margin)
        )
        keep.append(not touches)
    return keep


def merge(boxes, scores, iou: float = NMS_IOU, limit: int = MAX_BOXES):
    """Non-maximum suppression over boxes from all tiles and the full image.
    Returns (boxes, scores) sorted by confidence, at most `limit`."""
    import torch
    from torchvision.ops import nms

    if len(boxes) == 0:
        return [], []
    b = torch.tensor(boxes, dtype=torch.float32)
    s = torch.tensor(scores, dtype=torch.float32)
    kept = nms(b, s, iou)[:limit]
    return b[kept].tolist(), s[kept].tolist()


def _person_boxes(result, label: str = "person"):
    """(xyxy boxes, scores) of one Ultralytics result, person class only. The
    untrained COCO model also detects cars, dogs and so on; scoring only ever
    counts `person`, as for untiled predictions."""
    names = result.names
    keep = [names[int(c)] == label for c in result.boxes.cls.cpu().tolist()]
    boxes = [b for b, k in zip(result.boxes.xyxy.cpu().tolist(), keep) if k]
    scores = [s for s, k in zip(result.boxes.conf.cpu().tolist(), keep) if k]
    return boxes, scores


def predict_image(model, image, imgsz: int, device: str, tile: int = TILE, overlap: int = OVERLAP):
    """All merged person boxes for one PIL image, in image pixels: (boxes, scores)."""
    width, height = image.size
    tiles = tile_grid(width, height, tile, overlap)
    crops = [image.crop(t) for t in tiles]
    full = model.predict(image, imgsz=imgsz, device=device, conf=MIN_CONFIDENCE, max_det=MAX_BOXES, verbose=False)[0]
    boxes, scores = _person_boxes(full)
    results = model.predict(crops, imgsz=tile, device=device, conf=MIN_CONFIDENCE, max_det=MAX_BOXES, verbose=False)
    for (tx0, ty0, tx1, ty1), result in zip(tiles, results):
        local, local_scores = _person_boxes(result)
        placed = [(bx0 + tx0, by0 + ty0, bx1 + tx0, by1 + ty0) for bx0, by0, bx1, by1 in local]
        for box, score, keep in zip(placed, local_scores,
                                    keep_off_inner_edges(placed, (tx0, ty0, tx1, ty1), width, height)):  # fmt: skip
            if keep:
                boxes.append(box)
                scores.append(score)
    return merge(boxes, scores)


def predict_view_tiled(view, weights: Path, version: str, imgsz: int = 1280, device: str = "mps", log=print) -> str:
    """Write merged tiled boxes into field `pred_<version>_tiled` of every sample."""
    import fiftyone as fo
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    model = YOLO(str(weights))
    name = "person"
    field = field_for(version) + "_tiled"
    values = {}
    paths, ids = view.values("filepath"), view.values("id")
    for n, (sample_id, path) in enumerate(zip(ids, paths)):
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        boxes, scores = predict_image(model, image, imgsz, device)
        detections = []
        for (x0, y0, x1, y1), score in zip(boxes, scores):
            x0, x1 = max(0.0, x0 / width), min(1.0, x1 / width)
            y0, y1 = max(0.0, y0 / height), min(1.0, y1 / height)
            if x1 > x0 and y1 > y0:
                detections.append(fo.Detection(label=name, confidence=float(score), bounding_box=[x0, y0, x1 - x0, y1 - y0]))
        values[sample_id] = fo.Detections(detections=detections)
        if n % 50 == 0:
            log(f"  {version}: tiled {n + 1}/{len(paths)}")
    view._dataset.set_values(field, values, key_field="id")
    return field


def score(view, field: str, version: str, repeats: int = 1000) -> dict:
    from sardrone.fo.false_alarms import scorecard_cutoff
    from sardrone.train import scorecard
    from sardrone.train.curves import envelope_curve

    images = scorecard.images_from_view(view, field)
    confidence = scorecard_cutoff(version)
    result = scorecard.evaluate(images, confidence, repeats)
    recall, _ = envelope_curve(images)
    s = result.scores
    return {
        "images": s.images, "people": s.truths, "confidence": round(confidence, 4), "map50": round(s.map50, 4),
        "map50_low": round(result.map50_interval[0], 4), "map50_high": round(result.map50_interval[1], 4),
        "map50_95": round(s.map50_95, 4), "precision": round(s.precision, 4), "recall": round(s.recall, 4),
        "max_recall": round(float(recall.max()), 4),
    }  # fmt: skip


GAIN_CSV = env.PROJECT_ROOT / "docs" / "results" / "tiled_gain.csv"
GAIN_FIELDS = ["dataset", "version", "images", "full_map50", "tiled_map50", "gain", "gain_low", "gain_high",
               "share_tiled_better", "repeats", "commit", "dirty", "evaluated_utc"]  # fmt: skip


def gains(datasets: list[str], versions: list[str], repeats: int = 1000, path: Path = GAIN_CSV) -> list[dict]:
    """Paired bootstrap of (tiled - full) mAP@0.5 from the stored predictions; no model is run."""
    import fiftyone as fo
    import numpy as np

    from sardrone.batch import git_commit
    from sardrone.train import scorecard
    from sardrone.train.metrics import interval, match_dataset, paired_difference

    commit, dirty = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for name in datasets:
        dataset_name, view_name = VIEWS[name]
        view = fo.load_dataset(dataset_name).load_saved_view(view_name)
        for version in versions:
            full = match_dataset(scorecard.images_from_view(view, field_for(version)))
            tiled = match_dataset(scorecard.images_from_view(view, field_for(version) + "_tiled"))
            samples = paired_difference(full, tiled, repeats)
            low, high = interval(samples)
            rows.append({
                "dataset": name, "version": version, "images": len(full.truths_per_image),
                "full_map50": round(score(view, field_for(version), version, 1)["map50"], 4),
                "tiled_map50": round(score(view, field_for(version) + "_tiled", version, 1)["map50"], 4),
                "gain": round(float(np.median(samples)), 4), "gain_low": round(low, 4), "gain_high": round(high, 4),
                "share_tiled_better": round(float((samples > 0).mean()), 4), "repeats": repeats,
                "commit": commit, "dirty": dirty, "evaluated_utc": stamp,
            })  # fmt: skip
            print(json.dumps(rows[-1]), flush=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAIN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


CUTOFF_CSV = env.PROJECT_ROOT / "docs" / "results" / "tiled_cutoffs.csv"
CUTOFF_FIELDS = ["version", "method", "val_images", "cutoff", "val_precision", "val_recall", "val_f1",
                 "heridal_precision", "heridal_recall", "sard_precision", "sard_recall", "commit", "dirty",
                 "evaluated_utc"]  # fmt: skip
POOL_SEED, POOL_TAKEN = 51, 300  # how the 300 real training images were drawn (export.py defaults)


def real_validation_view():
    """SARD pool photos that no model trained on: the pool minus the 300 that
    `export.py --add-real 300` drew with its default seed (v4, v6, v7, v8)."""
    import fiftyone as fo

    from sardrone.fo import real

    pool = fo.load_dataset(real.DATASET_NAME).load_saved_view(real.VIEW_POOL)
    used = pool.take(POOL_TAKEN, seed=POOL_SEED).values("id")
    return pool.exclude(used)


def cutoffs(versions: list[str], device: str = "mps", path: Path = CUTOFF_CSV, log=print) -> list[dict]:
    """Choose each model's cut-off (best F1) on real validation photos, for
    untiled and tiled detection alike, then apply it unchanged to the test sets."""
    import fiftyone as fo

    from sardrone.batch import git_commit
    from sardrone.train import scorecard
    from sardrone.train.metrics import match_dataset, operating_point
    from sardrone.train.predict import predict_view
    from sardrone.train.train import DEFAULT_TRAIN_ROOT

    val = real_validation_view()
    tests = {name: fo.load_dataset(d).load_saved_view(v) for name, (d, v) in VIEWS.items()}
    commit, dirty = git_commit()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for version in versions:
        run = json.loads((Path(DEFAULT_TRAIN_ROOT) / version / "run.json").read_text())
        weights, imgsz = Path(run["weights"]), run["config"]["imgsz"]
        log(f"{version}: predicting {len(val)} validation photos, untiled and tiled")
        predict_view(val, weights, version, imgsz=imgsz, device=device, log=log)
        predict_view_tiled(val, weights, version, imgsz, device, log=log)
        for method, field in (("full", field_for(version)), ("tiled", field_for(version) + "_tiled")):
            cut, p, r, f1 = operating_point(match_dataset(scorecard.images_from_view(val, field)))
            row = {"version": version, "method": method, "val_images": len(val), "cutoff": round(cut, 4),
                   "val_precision": round(p, 4), "val_recall": round(r, 4), "val_f1": round(f1, 4)}  # fmt: skip
            for name, view in tests.items():
                _, tp, tr, _ = operating_point(match_dataset(scorecard.images_from_view(view, field)), cut)
                row[f"{name}_precision"], row[f"{name}_recall"] = round(tp, 4), round(tr, 4)
            row.update({"commit": commit, "dirty": dirty, "evaluated_utc": stamp})
            rows.append(row)
            log(json.dumps(row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CUTOFF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def upsert(rows_new: list[dict], path: Path = DEFAULT_CSV) -> None:
    keys = {(r["dataset"], r["version"], r["method"]) for r in rows_new}
    rows = []
    if path.exists():
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if (r["dataset"], r["version"], r["method"]) not in keys]
    rows += [{k: str(v) for k, v in r.items()} for r in rows_new]
    rows.sort(key=lambda r: (r["dataset"], r["version"], r["method"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    import fiftyone as fo

    from sardrone.batch import git_commit
    from sardrone.train.train import DEFAULT_TRAIN_ROOT

    parser = argparse.ArgumentParser(description="Score models with and without tiled detection.")
    parser.add_argument("--dataset", choices=sorted(VIEWS))
    parser.add_argument("--cutoffs", action="store_true",
                        help="choose cut-offs on real validation photos (SARD pool minus the 300 trained on)")
    parser.add_argument("--gains", nargs="+", choices=sorted(VIEWS),
                        help="paired bootstrap of the tiled gain on these datasets, from stored predictions")
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, help="only the first N images (smoke test; the CSV is not written)")
    args = parser.parse_args(argv)
    if args.cutoffs:
        cutoffs(args.versions, args.device, log=lambda m: print(m, flush=True))
        print(f"results: {CUTOFF_CSV}")
        return 0
    if args.gains:
        gains(args.gains, args.versions)
        print(f"results: {GAIN_CSV}")
        return 0
    if not args.dataset:
        parser.error("--dataset is required unless --gains is given")

    dataset_name, view_name = VIEWS[args.dataset]
    view = fo.load_dataset(dataset_name).load_saved_view(view_name)
    if args.limit:
        view = view.limit(args.limit)
    commit, dirty = git_commit()
    for version in args.versions:
        run = json.loads((Path(DEFAULT_TRAIN_ROOT) / version / "run.json").read_text())
        full_field = field_for(version)
        if not view._dataset.has_field(full_field):
            raise SystemExit(f"{full_field} missing on {dataset_name}: score the untiled model first")
        tiled_field = predict_view_tiled(view, Path(run["weights"]), version, run["config"]["imgsz"], args.device,
                                         log=lambda m: print(m, flush=True))  # fmt: skip
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for method, field in (("full", full_field), ("tiled", tiled_field)):
            row = {"dataset": args.dataset, "version": version, "method": method, **score(view, field, version),
                   "tile": TILE if method == "tiled" else "", "overlap": OVERLAP if method == "tiled" else "",
                   "commit": commit, "dirty": dirty, "evaluated_utc": stamp}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
        if not args.limit:
            upsert(rows)
    if not args.limit:
        print(f"results: {DEFAULT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
