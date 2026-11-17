"""Train the person detector (Ultralytics YOLO11n) on an exported dataset.

    make smoke-train                     # 1 epoch on a handful of images: does it run?
    make train EXPORT=v1 VERSION=v1      # the real run

FINE-TUNING: we do not start from random weights. `yolo11n.pt` was already
trained on everyday photos (the COCO dataset) and knows edges, textures and
"person-ness". Training continues from there on our images, which needs far
less data and time than starting from nothing.

Ultralytics is licensed AGPL-3.0. That is fine for this portfolio project
and is stated in the README; it would matter for a closed-source product.

Every run writes `run.json` next to its weights: the hash of the training
data, the settings, the commit and the library versions. A set of weights
without that file is not used for any reported number.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = PROJECT_ROOT / "runs" / "exports"
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "runs" / "train"
BASE_WEIGHTS = "yolo11n.pt"


@dataclass
class TrainConfig:
    epochs: int = 50
    # People are about 30 px tall in our 1280 px wide frames. Ultralytics'
    # default of 640 would halve that to 15 px, too small to learn from.
    imgsz: int = 1280
    batch: int = 8
    device: str = "mps"  # Apple GPU; "cpu" also works, slowly
    workers: int = 4
    patience: int = 15  # stop early if validation has not improved for this many epochs
    seed: int = 51
    # How many of the network's first modules keep their pretrained weights
    # untouched. YOLO11's "backbone" (the part that turns pixels into
    # general features) is modules 0 to 10, so 11 freezes exactly that and
    # trains only the detection head. None = train everything.
    # (Ultralytics: an int n freezes model.0 ... model.n-1; checked in
    # engine/trainer.py of 8.4.214.)
    freeze: int | None = None
    # Augmentation. Ultralytics' defaults, written out so a run's record is
    # complete and so later versions can change exactly one of them.
    mosaic: float = 1.0
    # Mosaic (four images glued into one) is switched off for the last few
    # epochs so the model finishes on normal-looking pictures. Ultralytics'
    # default of 10 would switch it off for a whole 10-epoch run.
    close_mosaic: int = 2
    fliplr: float = 0.5
    flipud: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    scale: float = 0.5
    translate: float = 0.1


def weights_dir() -> Path:
    """Where downloaded base weights are kept (inside the project)."""
    folder = PROJECT_ROOT / ".cache" / "ultralytics"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def train(export_dir: Path, version: str, config: TrainConfig | None = None,
          train_root: Path = DEFAULT_TRAIN_ROOT, commit=("unknown", False)) -> dict:  # fmt: skip
    """Run one training and write run.json. Returns the run record."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    config = config or TrainConfig()
    export_dir = Path(export_dir)
    export = json.loads((export_dir / "export.json").read_text())
    run_dir = Path(train_root) / version

    model = YOLO(str(weights_dir() / BASE_WEIGHTS))  # downloaded here on first use
    started = time.monotonic()
    results = model.train(
        data=str(export_dir / "dataset.yaml"),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        deterministic=True,
        plots=True,
        verbose=False,
        **asdict(config),
    )
    metrics = {key: round(float(value), 4) for key, value in (results.results_dict or {}).items()}
    record = {
        "version": version,
        "export_name": export["name"],
        "export_hash": export["hash"],
        "train_images": export["splits"]["train"]["images"],
        "val_images": export["splits"]["val"]["images"],
        "base_weights": BASE_WEIGHTS,
        "config": asdict(config),
        "val_metrics": metrics,  # Ultralytics' own validation numbers, on the val split
        "minutes": round((time.monotonic() - started) / 60, 1),
        "weights": str(run_dir / "weights" / "best.pt"),
        "commit": commit[0],
        "dirty": commit[1],
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    from sardrone.batch import git_commit

    parser = argparse.ArgumentParser(description="Train YOLO11n on an exported dataset.")
    parser.add_argument("--export", required=True, help="export name under runs/exports")
    parser.add_argument("--version", required=True, help="model version, e.g. v1")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, help="training seed (default 51); other seeds measure run-to-run variation")
    parser.add_argument("--freeze", type=int, help="freeze the first N modules (11 = the whole backbone)")
    args = parser.parse_args(argv)

    config = TrainConfig()
    for key in ("epochs", "imgsz", "batch", "device", "workers", "freeze", "seed"):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    record = train(DEFAULT_EXPORT_ROOT / args.export, args.version, config, commit=git_commit())
    print(json.dumps({k: record[k] for k in ("version", "export_hash", "train_images", "val_metrics", "minutes", "weights")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
