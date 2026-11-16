"""Write FiftyOne views to disk as a YOLO dataset, with a version hash.

    python -m sardrone.train.export --name v1      # train-v1 + val-v1 -> runs/exports/v1

YOLO expects one text file per image, one line per box:

    <class> <x centre> <y centre> <width> <height>      all as fractions 0..1

FiftyOne stores a box as [x left, y top, width, height], also in fractions,
so only the corner has to become a centre.

Images are not copied: the export holds symbolic links to the frames, so it
costs almost no disk. Its HASH is computed from every image's bytes and
every label line. Two exports with the same hash are the same training data,
bit for bit; a training run records the hash, so a model can always be
traced to exactly the data it saw.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "runs" / "exports"
CLASSES = ("person",)


def yolo_line(label: str, bounding_box, classes=CLASSES) -> str:
    """One label line. `bounding_box` is FiftyOne's [x, y, w, h] in 0..1."""
    x, y, w, h = bounding_box
    return f"{classes.index(label)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}"


def label_text(detections, classes=CLASSES) -> str:
    """The whole label file for one image ('' for an image with no people,
    which YOLO uses as a background example)."""
    lines = [
        yolo_line(d.label, d.bounding_box, classes)
        for d in (detections.detections if detections else [])
        if d.label in classes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def subsample_by_episode(view, count: int, seed: int = 51):
    """A seeded sample of about `count` frames, spread over episodes in
    proportion to their size.

    Frames from one flight are closely related, so a plain random sample
    could by chance take many frames from a few flights. Sampling within
    each episode keeps every world represented. Used when the full training
    set is too large for the available compute; the choice is recorded in
    the export.
    """
    import random

    ids, episodes = view.values("id"), view.values("episode_id")
    if count >= len(ids):
        return view
    by_episode: dict = {}
    for sample_id, episode in zip(ids, episodes):
        by_episode.setdefault(episode, []).append(sample_id)
    rng = random.Random(seed)
    chosen = []
    for episode in sorted(by_episode):
        members = sorted(by_episode[episode])
        share = max(1, round(count * len(members) / len(ids)))
        chosen += rng.sample(members, min(share, len(members)))
    return view.select(chosen)


def export_split(view, out_dir: Path, split: str, gt_field: str = "ground_truth") -> list[tuple[str, str, str]]:
    """Link images and write labels for one split.
    Returns [(relative image path, sha256 of the image, label text)]."""
    images, labels = out_dir / "images" / split, out_dir / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records = []
    seen: set[str] = set()
    for filepath, detections in zip(view.values("filepath"), view.values(gt_field)):
        source = Path(filepath)
        if source.name in seen:
            raise ValueError(f"two images named {source.name} in split {split!r}")
        seen.add(source.name)
        os.symlink(source, images / source.name)
        text = label_text(detections)
        (labels / f"{source.stem}.txt").write_text(text)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        records.append((f"{split}/{source.name}", digest, text))
    return records


def export_hash(records: list[tuple[str, str, str]]) -> str:
    """One hash over all images and labels, independent of export order."""
    total = hashlib.sha256()
    for name, digest, text in sorted(records):
        total.update(f"{name}\t{digest}\t{text}\n".encode())
    return total.hexdigest()[:16]


def export_dataset(splits: dict, out_dir: Path, name: str, view_names: dict | None = None, commit=("unknown", False)) -> dict:
    """Export several views. `splits` maps split name ("train", "val") to a view.

    Writes dataset.yaml (what Ultralytics reads) and export.json (the
    record: which views, how many images and boxes, the hash, the commit).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # an export is always rebuilt whole
    out_dir.mkdir(parents=True)
    records, summary = [], {}
    for split, view in splits.items():
        part = export_split(view, out_dir, split)
        records += part
        summary[split] = {
            "view": (view_names or {}).get(split, ""),
            "images": len(part),
            "boxes": sum(text.count("\n") for _, _, text in part),
            "empty_images": sum(1 for _, _, text in part if not text),
        }
    yaml_lines = [f"path: {out_dir}"]
    yaml_lines += [f"{split}: images/{split}" for split in splits]
    yaml_lines += ["names:"] + [f"  {index}: {label}" for index, label in enumerate(CLASSES)]
    (out_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n")
    manifest = {
        "name": name,
        "hash": export_hash(records),
        "classes": list(CLASSES),
        "splits": summary,
        "commit": commit[0],
        "dirty": commit[1],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

    import fiftyone as fo

    from sardrone.batch import git_commit
    from sardrone.fo.curate import VIEW_TRAIN, VIEW_VAL
    from sardrone.fo.frames import DATASET_NAME

    parser = argparse.ArgumentParser(description="Export saved views as a YOLO dataset.")
    parser.add_argument("--name", required=True, help="export name, e.g. v1")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--train", default=VIEW_TRAIN, help="saved view to train on")
    parser.add_argument("--val", default=VIEW_VAL, help="saved view to validate on")
    parser.add_argument("--out", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--max-train", type=int, help="sample this many training frames, spread over episodes")
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args(argv)

    dataset = fo.load_dataset(args.dataset)
    views = {"train": dataset.load_saved_view(args.train), "val": dataset.load_saved_view(args.val)}
    if not len(views["train"]) or not len(views["val"]):
        print(f"error: view {args.train!r} or {args.val!r} is empty - run `make curate` first")
        return 1
    names = {"train": args.train, "val": args.val}
    if args.max_train:
        full = len(views["train"])
        views["train"] = subsample_by_episode(views["train"], args.max_train, args.seed)
        names["train"] = f"{args.train} (sample of {len(views['train'])} from {full}, by episode, seed {args.seed})"
    manifest = export_dataset(views, args.out / args.name, args.name, names, git_commit())
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
