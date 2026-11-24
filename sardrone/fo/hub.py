"""Publish the simulated frames as a FiftyOne dataset on the Hugging Face Hub.

    python -m sardrone.fo.hub build                 # clean copy: sardrone-sim (no upload)
    HF_TOKEN=... python -m sardrone.fo.hub push     # upload (needs your Hugging Face token)

What goes in: the plain simulated worlds behind v1 (splits train / val /
test, split by whole flight), with exact labels and flight metadata, and
the CLIP embeddings. What stays out: model predictions, auto-labels, and
every real photograph (those belong to their own datasets and licences).
The dataset card is docs/hub/README.md.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

HUB_NAME = "sardrone-sim"
CARD = env.PROJECT_ROOT / "docs" / "hub" / "README.md"
SPLITS = {"train": "train-v1", "val": "val-v1", "test": "sim-test"}
FIELDS = [  # everything else (predictions, auto-labels, review fields) is dropped
    "ground_truth", "episode_id", "world", "world_id", "seed", "time_s", "altitude_m", "height_m",
    "time_of_day", "occlusion", "num_people", "expected_by_geometry", "mission_state", "clip_embedding",
]  # fmt: skip
LICENSE = "cc-by-4.0"
TAGS = ["fiftyone", "object-detection", "simulation", "sim-to-real", "drone", "search-and-rescue", "aerial"]


def build(source, splits: dict = SPLITS, name: str = HUB_NAME, overwrite: bool = True):
    """A clean, persistent copy of `source`'s split views, tagged by split."""
    import fiftyone as fo

    if fo.dataset_exists(name):
        if not overwrite:
            raise SystemExit(f"{name} exists")
        fo.delete_dataset(name)
    out = fo.Dataset(name)
    out.persistent = True
    keep = [f for f in FIELDS if source.has_field(f)]
    for split, view_name in splits.items():
        view = source.load_saved_view(view_name).select_fields(keep)
        ids = out.add_collection(view, include_info=False, new_ids=True)
        out.select(ids).tag_samples(split)
        out.select(ids).set_values("split", [split] * len(ids))
    out.info = {
        "source": "https://github.com/kartik-joshi/sardrone",
        "label": "person boxes from the simulator's bounding-box sensor: visible pixels only",
        "splits": {split: len(out.match_tags(split)) for split in splits},
    }
    out.save()
    return out


def fo_field(name):
    from fiftyone import ViewField

    return ViewField(name)


def check(dataset) -> dict:
    """Facts the dataset card quotes; recomputed, not typed by hand."""
    summary = {
        "samples": len(dataset),
        "splits": dataset.count_values("split"),
        "near_duplicate_frames": len(dataset.match_tags("duplicate")),
        "people_boxes": dataset.count("ground_truth.detections"),
        "frames_without_people": sum(1 for boxes in dataset.values("ground_truth.detections") if not boxes),
        "episodes": len(set(dataset.values("episode_id"))),
        "fields": sorted(dataset.get_field_schema().keys()),
    }
    by_split = {}
    for split in summary["splits"]:
        episodes = set(dataset.match(fo_field("split") == split).values("episode_id"))
        by_split[split] = episodes
    shared = set.intersection(*by_split.values()) if len(by_split) > 1 else set()
    summary["episodes_in_more_than_one_split"] = len(shared)
    return summary


def push(name: str = HUB_NAME, repo: str = HUB_NAME, private: bool = False) -> str:
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh
    from huggingface_hub import HfApi

    dataset = fo.load_dataset(name)
    fouh.push_to_hub(dataset, repo, description="Simulated drone frames of people in search-and-rescue worlds, "
                     "with exact person boxes", license=LICENSE, tags=TAGS, private=private,
                     min_fiftyone_version="1.0", exist_ok=True)  # fmt: skip
    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{repo}"
    card = CARD.read_text().replace("{repo_id}", repo_id)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    return repo_id


def main(argv: list[str] | None = None) -> int:
    import fiftyone as fo

    from sardrone.fo.frames import DATASET_NAME

    parser = argparse.ArgumentParser(description="Build or publish the simulated dataset on the Hugging Face Hub.")
    parser.add_argument("action", choices=["build", "check", "push"])
    parser.add_argument("--repo", default=HUB_NAME)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "build":
        build(fo.load_dataset(DATASET_NAME))
    if args.action in ("build", "check"):
        print(json.dumps(check(fo.load_dataset(HUB_NAME)), indent=2))
    if args.action == "push":
        print("published:", push(repo=args.repo, private=args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
