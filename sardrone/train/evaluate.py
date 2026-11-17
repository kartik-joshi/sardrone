"""Score a trained model on simulation and on real images; update the scorecard.

    make evaluate VERSION=v1 CHANGE="baseline: simulation only"

Three datasets, three jobs:

* val-v1      (simulated)  choose the confidence cut-off here, nowhere else
* sim-test    (simulated)  the score on unseen simulated worlds
* sard-test   (real)       the score on real drone photos, never trained on

The sim-to-real GAP is sim-test mAP@0.5 minus sard-test mAP@0.5, with a
95 % bootstrap interval. Precision and recall are reported at the cut-off
picked on val-v1, so the test sets play no part in tuning anything.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402

from sardrone.fo import curate, real  # noqa: E402
from sardrone.fo.frames import DATASET_NAME  # noqa: E402
from sardrone.train import scorecard  # noqa: E402
from sardrone.train.metrics import match_dataset, operating_point  # noqa: E402
from sardrone.train.predict import field_for, predict_view  # noqa: E402
from sardrone.train.train import DEFAULT_TRAIN_ROOT  # noqa: E402


def evaluate_version(version: str, change: str, *, train_root: Path = DEFAULT_TRAIN_ROOT, device: str = "mps",
                     limit: int | None = None, repeats: int = 1000, csv_path: Path = scorecard.DEFAULT_CSV,
                     commit=("unknown", False), log=print) -> dict:  # fmt: skip
    run = json.loads((Path(train_root) / version / "run.json").read_text())
    weights, imgsz = Path(run["weights"]), run["config"]["imgsz"]

    frames = fo.load_dataset(DATASET_NAME)
    views = {
        "val": frames.load_saved_view(curate.VIEW_VAL),
        "sim": frames.load_saved_view(curate.VIEW_TEST),
        "real": fo.load_dataset(real.DATASET_NAME).load_saved_view(real.VIEW_TEST),
    }
    if limit:  # a quick try only: such numbers must never reach the scorecard
        views = {name: view.limit(limit) for name, view in views.items()}
    field = field_for(version)
    images = {}
    for name, view in views.items():
        if len(view) == 0:
            raise ValueError(f"the {name} view is empty")
        log(f"predicting {len(view)} {name} images ...")
        predict_view(view, weights, version, imgsz=imgsz, device=device, log=log)
        images[name] = scorecard.images_from_view(view, field)

    # The cut-off with the best F1 on validation data, then held fixed.
    confidence = operating_point(match_dataset(images["val"]))[0]
    sim = scorecard.evaluate(images["sim"], confidence, repeats)
    real_scores = scorecard.evaluate(images["real"], confidence, repeats)
    row = scorecard.scorecard_row(
        version, change, sim, real_scores,
        train_hash=run["export_hash"], train_images=run["train_images"], commit=commit[0],
        evaluated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )  # fmt: skip
    if limit is None:
        scorecard.upsert(row, csv_path)
    return row


def main(argv: list[str] | None = None) -> int:
    from sardrone.batch import git_commit

    parser = argparse.ArgumentParser(description="Evaluate a model version and update the scorecard.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--change", required=True, help="what is different in this version, in a few words")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, help="quick try on N images per set; the scorecard is NOT written")
    parser.add_argument("--scorecard", type=Path, default=scorecard.DEFAULT_CSV,
                        help="CSV to update (default: the main scorecard)")  # fmt: skip
    args = parser.parse_args(argv)
    row = evaluate_version(
        args.version, args.change, device=args.device, limit=args.limit, csv_path=args.scorecard,
        commit=git_commit(), log=lambda message: print(message, flush=True),
    )  # fmt: skip
    print(json.dumps(row, indent=2))
    print("scorecard NOT written (--limit)" if args.limit else f"scorecard: {args.scorecard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
