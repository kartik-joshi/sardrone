"""Curate the frames dataset: embed, map, de-duplicate, split, save views.

    make curate        # works on the FiftyOne dataset "sar-frames"

An EMBEDDING turns a picture into a list of numbers (here 512, from the
CLIP model) such that pictures that look alike get lists that are close
together. With that, a computer can answer "which images are similar?"
without anyone labelling anything. This module uses embeddings three ways:

* a 2D MAP of the whole dataset (UMAP squeezes 512 numbers into 2, keeping
  neighbours together), to see what the data contains and what is missing
* a SIMILARITY INDEX, to find images like a given one
* NEAR-DUPLICATE removal: frames that are almost the same picture add
  training time but no information. (Done with a few lines of explicit
  code on the embeddings rather than FiftyOne's own duplicate finder,
  whose distance threshold I could not reconcile with cosine distance.)

It then marks which episodes are held out, and saves named views
(`train-v1`, `sar-hard-cases`, ...) that later phases load by name.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402
import fiftyone.brain as fob  # noqa: E402
import fiftyone.zoo as foz  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402

from sardrone.fo.frames import DATASET_NAME  # noqa: E402

EMBEDDING_MODEL = "clip-vit-base32-torch"
EMBEDDING_FIELD = "clip_embedding"
SIMILARITY_KEY = "clip_sim"
MAP_KEY = "clip_umap"
DUPLICATE_TAG = "duplicate"

# Cosine distance (1 - cosine similarity; 0 = identical direction) below
# which two frames of one episode count as the same picture. Chosen from a
# sweep (docs/DEVLOG.md): the largest value at which under 1 % of the
# removed frames show a different number of people than the frame kept in
# their place. On the pilot data that was 0.006; on the v1 data 0.006 gave
# 1.8 %, and the rule gives 0.004 (0.7 %).
DUPLICATE_THRESHOLD = 0.004
SWEEP_THRESHOLDS = (0.002, 0.003, 0.004, 0.006, 0.01, 0.015)

SMALL_BOX_PX = 16.0  # "small" person: longer side of the box below this
VAL_EVERY = 10  # every 10th training episode is held out for validation

VIEW_TRAIN = "train-v1"
VIEW_VAL = "val-v1"
VIEW_TEST = "sim-test"
VIEW_TRAIN_DR = "train-v2-dr"
VIEW_TRAIN_VEH = "train-v7-veh"
VIEW_TRAIN_RICH = "train-v13-rich"
VIEW_HARD = "sar-hard-cases"
VIEW_FALSE_ALARMS = "sar-false-alarms"


def ensure_embeddings(dataset, log=print) -> int:
    """Embed every frame that has no embedding yet. Returns how many were computed.

    Embeddings are stored with length 1 ("unit vectors"). Then the cosine
    similarity of two frames is simply their dot product, and nothing
    downstream has to remember to normalise.
    """
    import numpy as np

    missing = dataset.exists(EMBEDDING_FIELD, False) if dataset.has_field(EMBEDDING_FIELD) else dataset
    count = len(missing)
    if count:
        log(f"embedding {count} frames with {EMBEDDING_MODEL} ...")
        model = foz.load_zoo_model(EMBEDDING_MODEL)
        # num_workers=0: image loading in this process. Worker processes
        # crashed here when the program was started from a pipe.
        ids = missing.values("id")
        raw = missing.compute_embeddings(model, batch_size=32, num_workers=0, progress=False)
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        dataset.set_values(EMBEDDING_FIELD, {i: v.tolist() for i, v in zip(ids, unit)}, key_field="id")
    return count


def find_duplicates(embeddings, episodes, times, threshold: float) -> dict[int, int]:
    """Near-duplicate frames WITHIN each episode: {duplicate index: kept index}.

    Frames are visited in time order. A frame is kept unless it is closer
    than `threshold` (cosine distance) to a frame already kept from the
    same episode; then it is a duplicate of the nearest such frame. The
    first frame of every group of look-alikes is therefore the one that
    stays. Frames of DIFFERENT episodes are never merged: similar-looking
    scenes from different worlds are variety, not repetition.

    `embeddings` must be unit vectors (see ensure_embeddings).
    """
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=float)
    order = sorted(range(len(episodes)), key=lambda i: (episodes[i], times[i]))
    kept: dict = {}  # episode -> list of kept indices
    duplicates: dict[int, int] = {}
    for index in order:
        mine = kept.setdefault(episodes[index], [])
        if mine:
            distances = 1.0 - embeddings[mine] @ embeddings[index]
            nearest = int(np.argmin(distances))
            if distances[nearest] < threshold:
                duplicates[index] = mine[nearest]
                continue
        mine.append(index)
    return duplicates


def build_index_and_map(dataset, seed: int = 51):
    """(Re)build the similarity index and the 2D map from the stored embeddings."""
    for key in (SIMILARITY_KEY, MAP_KEY):
        if key in dataset.list_brain_runs():
            dataset.delete_brain_run(key)
    fob.compute_similarity(
        dataset, embeddings=EMBEDDING_FIELD, brain_key=SIMILARITY_KEY, backend="sklearn", metric="cosine"
    )
    fob.compute_visualization(
        dataset, embeddings=EMBEDDING_FIELD, method="umap", brain_key=MAP_KEY, num_dims=2,
        seed=seed, progress=False, verbose=False,
    )  # fmt: skip


def duplicate_stats(dataset, threshold: float) -> tuple[dict, list[str]]:
    """What removing near-duplicates at `threshold` does, and their sample ids."""
    ids = dataset.values("id")
    people, states = dataset.values("num_people"), dataset.values("mission_state")
    duplicates = find_duplicates(
        dataset.values(EMBEDDING_FIELD), dataset.values("episode_id"), dataset.values("time_s"), threshold
    )
    stats = {
        "threshold": threshold,
        "duplicates": len(duplicates),
        "share": round(len(duplicates) / len(ids), 4) if ids else 0.0,
        "different_people_count": sum(people[dup] != people[keep] for dup, keep in duplicates.items()),
        "by_state": dict(Counter(states[dup] for dup in duplicates)),
    }
    return stats, [ids[dup] for dup in duplicates]


def tag_duplicates(dataset, threshold: float = DUPLICATE_THRESHOLD) -> dict:
    """Tag near-duplicates (the first frame of each group stays untagged).

    Frames are tagged, not deleted: the pictures stay on disk and in the
    dataset, and a view without the tag is what training uses. That keeps
    the decision reversible and visible.
    """
    stats, duplicate_ids = duplicate_stats(dataset, threshold)
    dataset.untag_samples(DUPLICATE_TAG)
    if duplicate_ids:
        dataset.select(duplicate_ids).tag_samples(DUPLICATE_TAG)
    return stats


def split_of(episode_id: str) -> str:
    """Which split an episode belongs to, from its id alone.

        test-007   -> "test"   the sim-test set: scored once, never tuned on
        train-020  -> "val"    every 10th training episode: for choosing
                               settings and deciding when to stop training
        train-021  -> "train"
        dr-012     -> "dr"     domain-randomized training episodes (v2)
        veh-004    -> "veh"    randomized episodes with parked vehicles (v7)
        rich-004   -> "rich"   randomized episodes with richer content (v13)
        pilot-003  -> "pilot"  pipeline debugging runs, not part of v1

    Deriving the split from the id means it cannot drift: the same episode
    is in the same split on every machine and every rerun.
    """
    prefix, _, number = episode_id.partition("-")
    if prefix == "test":
        return "test"
    if prefix in ("dr", "veh", "rich"):
        return prefix
    if prefix == "train" and number.isdigit():
        return "val" if int(number) % VAL_EVERY == 0 else "train"
    return "pilot"


def assign_splits(dataset) -> dict:
    """Mark whole EPISODES as train, val or test.

    Splitting by episode, not by frame, matters: frames from one flight
    over one world are closely related. If some went to training and others
    to validation, the model would be tested on scenes it has effectively
    seen, and the score would be too good ("leakage").
    """
    dataset.set_values("split", [split_of(e) for e in dataset.values("episode_id")])
    return dict(dataset.count_values("split"))


def hard_case_filter():
    """A frame is hard if it shows a person who is lying, under a tree,
    or small in the image, or if it shows anyone at dusk."""
    longer_side = (F("width_px") > F("height_px")).if_else(F("width_px"), F("height_px"))
    hard_person = (F("pose") == "lying") | (F("occlusion") == "partial") | (longer_side < SMALL_BOX_PX)
    has_hard_person = F("ground_truth.detections").filter(hard_person).length() > 0
    anyone_at_dusk = (F("time_of_day") == "dusk") & (F("num_people") > 0)
    return has_hard_person | anyone_at_dusk


def save_views(dataset) -> dict:
    """Create (or replace) the named views later phases load."""
    if not dataset.has_field("false_alarms"):
        # Filled in when the first trained model is evaluated (Phase 3B).
        dataset.add_sample_field("false_alarms", fo.IntField)
        dataset.set_values("false_alarms", [0] * len(dataset))
    unique = dataset.match_tags(DUPLICATE_TAG, bool=False)
    views = {
        VIEW_TRAIN: unique.match(F("split") == "train"),
        VIEW_VAL: unique.match(F("split") == "val"),
        VIEW_TRAIN_DR: unique.match(F("split") == "dr"),
        VIEW_TRAIN_VEH: unique.match(F("split") == "veh"),
        VIEW_TRAIN_RICH: unique.match(F("split") == "rich"),
        # The test set keeps its near-duplicates: it should look like what a
        # flight really produces, not like a cleaned-up version of it.
        VIEW_TEST: dataset.match(F("split") == "test"),
        VIEW_HARD: unique.match(hard_case_filter()),
        VIEW_FALSE_ALARMS: dataset.match(F("false_alarms") > 0),
    }
    for name, view in views.items():
        dataset.save_view(name, view, overwrite=True)
    return {name: len(view) for name, view in views.items()}


def plot_map(dataset, path: Path) -> None:
    """Save the 2D embedding map three times over: coloured by world, by
    time of day, and by whether the frame shows a person."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = dataset.load_brain_results(MAP_KEY).points
    panels = [
        ("world", dataset.values("world")),
        ("time_of_day", dataset.values("time_of_day")),
        ("people in frame", ["yes" if n else "no" for n in dataset.values("num_people")]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (title, values) in zip(axes, panels):
        for group in sorted(set(values)):
            mask = [value == group for value in values]
            axis.scatter(points[mask, 0], points[mask, 1], s=3, alpha=0.5, label=f"{group} ({sum(mask)})")
        axis.set_title(f"CLIP embeddings (UMAP), by {title}")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(markerscale=4, fontsize=9)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def curate(dataset, map_path: Path | None = None, log=print) -> dict:
    """Run every step. Safe to run again; results are replaced, not doubled."""
    ensure_embeddings(dataset, log)
    build_index_and_map(dataset)
    duplicates = tag_duplicates(dataset)
    splits = assign_splits(dataset)
    views = save_views(dataset)
    if map_path is not None:
        plot_map(dataset, map_path)
    summary = {"frames": len(dataset), "duplicates": duplicates, "splits": splits, "views": views}
    dataset.info["curation"] = {
        "embedding_model": EMBEDDING_MODEL,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "val_every": VAL_EVERY,
        **summary,
    }
    dataset.save()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed, de-duplicate, split and save views.")
    parser.add_argument("--name", default=DATASET_NAME)
    parser.add_argument("--map", type=Path, default=env.PROJECT_ROOT / "docs/images/phase2e_umap.png")
    parser.add_argument("--sweep", action="store_true", help="also print the threshold sweep")
    args = parser.parse_args(argv)
    if not fo.dataset_exists(args.name):
        print(f"dataset {args.name!r} does not exist - run `make frames` first")
        return 1
    dataset = fo.load_dataset(args.name)
    summary = curate(dataset, args.map)
    if args.sweep:
        for threshold in SWEEP_THRESHOLDS:
            print("sweep:", duplicate_stats(dataset, threshold)[0])
    duplicates = summary["duplicates"]
    print(f"{summary['frames']} frames; {duplicates['duplicates']} near-duplicates tagged "
          f"(threshold {duplicates['threshold']}; {duplicates['different_people_count']} of them show a "
          f"different number of people than the frame kept)")  # fmt: skip
    print("removed by mission state:", duplicates["by_state"])
    print("splits:", summary["splits"])
    print("saved views:", summary["views"])
    print(f"map: {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
