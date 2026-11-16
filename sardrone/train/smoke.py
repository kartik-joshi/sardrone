"""A two-minute check that training works at all on this machine.

    make smoke-train

Exports 48 frames of one episode and 16 of another, trains two epochs at a
small image size on the CPU, and predicts on the 16. The resulting numbers
mean nothing; the point is that every step runs before hours are spent on
the real training.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import fiftyone as fo  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402

from sardrone.fo.frames import DATASET_NAME  # noqa: E402
from sardrone.train import export, predict, train  # noqa: E402
from sardrone.train.metrics import score  # noqa: E402
from sardrone.train.scorecard import images_from_view  # noqa: E402


def main() -> int:
    dataset = fo.load_dataset(DATASET_NAME)
    with_people = dataset.match(F("num_people") > 0)
    episodes = sorted(with_people.distinct("episode_id"))
    if len(episodes) < 2:
        print("need frames from at least two episodes - run `make frames` first")
        return 1
    views = {
        "train": with_people.match(F("episode_id") == episodes[0]).take(48, seed=1),
        "val": with_people.match(F("episode_id") == episodes[-1]).take(16, seed=1),
    }
    out = export.DEFAULT_EXPORT_ROOT / "smoke"
    manifest = export.export_dataset(views, out, "smoke", {"train": episodes[0], "val": episodes[-1]})
    print(f"export {manifest['hash']}: {manifest['splits']}")
    config = train.TrainConfig(epochs=2, imgsz=640, batch=8, device="cpu", workers=0, patience=5)
    record = train.train(out, "smoke", config)
    print(f"trained in {record['minutes']} min; Ultralytics validation: {record['val_metrics']}")
    field = predict.predict_view(views["val"], record["weights"], "smoke", imgsz=640, device="cpu")
    scores, _ = score(images_from_view(views["val"], field))
    print(f"own metrics on the 16 validation frames: mAP@0.5 {scores.map50:.3f} (meaningless at this size)")
    print("smoke test passed: export, train, predict and score all ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
