"""Score model versions on an extra real test set (e.g. HERIDAL).

    python -m sardrone.train.extra_real --dataset heridal --versions v0 v4 v6

Each model keeps the confidence cut-off chosen for it on simulated
validation data (from the scorecard), so nothing is tuned on this set.
Results go to docs/results/extra_real.csv, one row per (dataset, version).
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402

from sardrone.fo.false_alarms import scorecard_cutoff  # noqa: E402
from sardrone.train import scorecard  # noqa: E402
from sardrone.train.predict import field_for, predict_view  # noqa: E402
from sardrone.train.train import DEFAULT_TRAIN_ROOT  # noqa: E402

DEFAULT_CSV = env.PROJECT_ROOT / "docs" / "results" / "extra_real.csv"
FIELDS = ["dataset", "version", "images", "people", "confidence", "map50", "map50_low", "map50_high",
          "map50_95", "precision", "recall", "max_recall", "evaluated_utc"]  # fmt: skip


def evaluate(dataset_name: str, version: str, device: str = "mps", repeats: int = 1000, log=print) -> dict:
    from sardrone.train.curves import envelope_curve

    run = json.loads((Path(DEFAULT_TRAIN_ROOT) / version / "run.json").read_text())
    dataset = fo.load_dataset(dataset_name)
    view = dataset.load_saved_view(f"{dataset_name}-test")
    log(f"{version}: predicting {len(view)} {dataset_name} images ...")
    predict_view(view, Path(run["weights"]), version, imgsz=run["config"]["imgsz"], device=device, log=log)
    images = scorecard.images_from_view(view, field_for(version))
    confidence = scorecard_cutoff(version)
    result = scorecard.evaluate(images, confidence, repeats)
    recall, _ = envelope_curve(images)
    s = result.scores
    return {
        "dataset": dataset_name, "version": version, "images": s.images, "people": s.truths,
        "confidence": round(confidence, 4), "map50": round(s.map50, 4),
        "map50_low": round(result.map50_interval[0], 4), "map50_high": round(result.map50_interval[1], 4),
        "map50_95": round(s.map50_95, 4), "precision": round(s.precision, 4), "recall": round(s.recall, 4),
        "max_recall": round(float(recall.max()), 4),
        "evaluated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }  # fmt: skip


def upsert(row: dict, path: Path = DEFAULT_CSV) -> list[dict]:
    rows = []
    if path.exists():
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if (r["dataset"], r["version"]) != (row["dataset"], row["version"])]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: (r["dataset"], r["version"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score models on an extra real test set.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--versions", nargs="+", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args(argv)
    for version in args.versions:
        row = evaluate(args.dataset, version, args.device, log=lambda m: print(m, flush=True))
        upsert(row)
        print(json.dumps(row), flush=True)
    print(f"results: {DEFAULT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
