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


def export_split(view, out_dir: Path, split: str, gt_field: str = "ground_truth", realism_seed: int | None = None,
                 skip_realism=lambda path: False, copy: int = 0) -> list[tuple[str, str, str]]:  # fmt: skip
    """Link images and write labels for one split.
    Returns [(relative image path, sha256 of the image, label text)].

    With `realism_seed`, images are not linked but written as degraded
    copies (see realism.py), except those for which `skip_realism(path)`
    is true (real photographs need no fake camera effects).

    `copy` > 0 writes the same images again under names ending in `_r<copy>`,
    so a small set can be repeated to raise its share of the training data.
    """
    images, labels = out_dir / "images" / split, out_dir / "labels" / split
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    records = []
    seen: set[str] = set()
    for filepath, detections in zip(view.values("filepath"), view.values(gt_field)):
        source = Path(filepath)
        target = source.name if not copy else f"{source.stem}_r{copy}{source.suffix}"
        if source.name in seen:
            raise ValueError(f"two images named {source.name} in split {split!r}")
        seen.add(source.name)
        if realism_seed is not None and not skip_realism(source):
            from sardrone.train.realism import degrade_file

            data, _ = degrade_file(source, source.name, realism_seed)
            (images / target).write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
        else:
            os.symlink(source, images / target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
        text = label_text(detections)
        (labels / f"{Path(target).stem}.txt").write_text(text)
        records.append((f"{split}/{target}", digest, text))
    return records


def export_hash(records: list[tuple[str, str, str]]) -> str:
    """One hash over all images and labels, independent of export order."""
    total = hashlib.sha256()
    for name, digest, text in sorted(records):
        total.update(f"{name}\t{digest}\t{text}\n".encode())
    return total.hexdigest()[:16]


def export_dataset(splits: dict, out_dir: Path, name: str, view_names: dict | None = None, commit=("unknown", False),
                   realism_seed: int | None = None, extra_train=None, notes: dict | None = None,
                   extra_repeat: int = 1) -> dict:  # fmt: skip
    """Export several views. `splits` maps split name ("train", "val") to a view.

    Writes dataset.yaml (what Ultralytics reads) and export.json (the
    record: which views, how many images and boxes, the hash, the commit).

    `realism_seed` degrades the TRAINING images only; validation images are
    left as rendered so that scores stay comparable between versions.
    `extra_train` is a second view (e.g. real photographs) added to the
    training split as it is, `extra_repeat` times (copies get new names).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # an export is always rebuilt whole
    out_dir.mkdir(parents=True)
    records, summary = [], {}
    extra_paths = set(extra_train.values("filepath")) if extra_train is not None else set()
    for split, view in splits.items():
        part = export_split(view, out_dir, split, realism_seed=realism_seed if split == "train" else None)
        if split == "train" and extra_train is not None:
            for copy in range(extra_repeat):  # linked, never degraded
                part += export_split(extra_train, out_dir, split, copy=copy)
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
        "realism_seed": realism_seed,
        "extra_train_images": len(extra_paths),
        "extra_repeat": extra_repeat,
        "notes": notes or {},
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
    parser.add_argument("--realism", action="store_true", help="degrade training images like a real camera")
    parser.add_argument("--add-real", type=int, default=0, help="add this many images from sard-train-pool")
    parser.add_argument("--real-repeat", type=int, default=1, help="include the added real images this many times")
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
    extra, notes = None, {}
    if args.add_real:
        from sardrone.fo import real

        pool = fo.load_dataset(real.DATASET_NAME).load_saved_view(real.VIEW_POOL)
        extra = pool.take(args.add_real, seed=args.seed)
        notes["real"] = f"{len(extra)} images from {real.VIEW_POOL} (of {len(pool)}), seed {args.seed}"
    manifest = export_dataset(
        views, args.out / args.name, args.name, names, git_commit(),
        realism_seed=args.seed if args.realism else None, extra_train=extra, notes=notes,
        extra_repeat=args.real_repeat,
    )  # fmt: skip
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
