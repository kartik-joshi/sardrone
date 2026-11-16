"""Phase 3 groundwork: detection metrics, YOLO export and the scorecard,
checked against numbers worked out by hand."""

import json

import numpy as np
import pytest

from sardrone.train import export, scorecard
from sardrone.train.metrics import (
    COCO_IOUS,
    ImageResult,
    ap_per_iou,
    average_precision,
    bootstrap_map,
    gap_interval,
    interval,
    iou,
    match_dataset,
    match_image,
    operating_point,
    score,
)

A = (0.0, 0.0, 10.0, 10.0)


def shifted(dx):
    return (dx, 0.0, 10.0 + dx, 10.0)


# ---- IoU and matching --------------------------------------------------------


def test_iou_by_hand():
    assert iou(A, A) == 1.0
    assert iou(A, (20.0, 20.0, 30.0, 30.0)) == 0.0
    assert iou(A, (10.0, 0.0, 20.0, 10.0)) == 0.0  # touching edges share no area
    # Shifted by half its width: shared 50, combined 150.
    assert iou(A, shifted(5.0)) == pytest.approx(1 / 3)
    # A 5 x 5 box inside a 10 x 10 one: 25 / 100.
    assert iou(A, (0.0, 0.0, 5.0, 5.0)) == pytest.approx(0.25)
    # Stretching both axes by different factors does not change IoU.
    stretch = lambda b: (b[0] * 3, b[1] * 0.5, b[2] * 3, b[3] * 0.5)  # noqa: E731
    assert iou(stretch(A), stretch(shifted(5.0))) == pytest.approx(1 / 3)


def test_each_true_box_is_claimed_once_by_the_most_confident_prediction():
    image = ImageResult(truths=(A,), predictions=((0.6, A), (0.9, shifted(1.0))))
    # Both overlap the one person. The 0.9 box is taken first and claims it
    # (IoU 9/11 = 0.82); the 0.6 box, although a perfect fit, is then a duplicate.
    assert match_image(image, 0.5) == [(0.9, True), (0.6, False)]


def test_a_prediction_takes_the_best_overlapping_true_box():
    left, right = A, (8.0, 0.0, 18.0, 10.0)
    image = ImageResult(truths=(left, right), predictions=((0.9, (7.0, 0.0, 17.0, 10.0)), (0.8, A)))
    # The first box overlaps `right` far more (IoU 0.82) than `left` (0.18).
    assert match_image(image, 0.5) == [(0.9, True), (0.8, True)]


def test_stricter_iou_turns_a_loose_box_into_a_miss():
    image = ImageResult(truths=(A,), predictions=((0.9, shifted(2.0)),))  # IoU 8/12 = 0.667
    assert match_image(image, 0.5) == [(0.9, True)]
    assert match_image(image, 0.7) == [(0.9, False)]


# ---- average precision -------------------------------------------------------


def test_average_precision_of_a_perfect_and_a_useless_detector():
    ones = np.ones(4)
    assert average_precision(ones, np.array([0.25, 0.5, 0.75, 1.0])) == pytest.approx(1.0)
    assert average_precision(np.zeros(4), np.zeros(4)) == pytest.approx(0.0, abs=0.01)
    assert average_precision(np.array([]), np.array([])) == 0.0


def test_average_precision_by_hand():
    # 2 people. Predictions in confidence order: correct, wrong, correct.
    # After each: precision 1/1, 1/2, 2/3; recall 0.5, 0.5, 1.0.
    precision, recall = np.array([1.0, 0.5, 2 / 3]), np.array([0.5, 0.5, 1.0])
    # Envelope: recall up to 0.5 -> precision 1.0; above 0.5 up to 1.0 -> 2/3.
    # 101 points: 51 of them (0.00..0.50) at 1.0, 50 (0.51..1.00) at 2/3.
    assert average_precision(precision, recall) == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_recall_never_reached_counts_as_zero_precision():
    # 4 people, one perfect prediction: recall stops at 0.25.
    # 26 points (0.00..0.25) at precision 1.0, the other 75 at 0.
    assert average_precision(np.array([1.0]), np.array([0.25])) == pytest.approx(26 / 101)


def dataset():
    """Three images, 4 people in total."""
    return [
        ImageResult(truths=(A,), predictions=((0.9, A),)),  # found
        ImageResult(truths=(A, (50.0, 50.0, 60.0, 60.0)), predictions=((0.8, shifted(2.0)), (0.7, (80.0, 80.0, 90.0, 90.0)))),
        ImageResult(truths=(A,), predictions=()),  # missed
    ]  # fmt: skip


def test_score_by_hand():
    scores, matched = score(dataset())
    assert (scores.images, scores.truths, scores.predictions) == (3, 4, 3)
    # At IoU 0.5: correct (0.9), correct (0.8, IoU 0.667), wrong (0.7).
    # Precision 1, 1, 2/3; recall 0.25, 0.5, 0.5. AP = 51 points at 1.0 / 101.
    assert scores.map50 == pytest.approx(51 / 101)
    # At IoU 0.70 and above the 0.8 box no longer counts: correct, wrong, wrong.
    # AP = 26 / 101 for the 6 thresholds 0.70..0.95; 51 / 101 for 0.50..0.65.
    assert scores.map50_95 == pytest.approx((4 * 51 / 101 + 6 * 26 / 101) / 10)
    assert list(ap_per_iou(matched)[:4]) == pytest.approx([51 / 101] * 4)
    assert matched.ious == COCO_IOUS and list(matched.confidence) == [0.9, 0.8, 0.7]
    # Best F1 keeps the two correct boxes: precision 1.0, recall 2/4.
    assert (scores.confidence, scores.precision, scores.recall) == (0.8, 1.0, 0.5)
    assert scores.f1 == pytest.approx(2 / 3)


def test_operating_point_at_a_given_cutoff():
    _, matched = score(dataset())
    assert operating_point(matched, 0.75)[1:] == pytest.approx((1.0, 0.5, 2 / 3))
    assert operating_point(matched, 0.85)[1:] == pytest.approx((1.0, 0.25, 0.4))
    confidence, precision, recall, _ = operating_point(matched, 0.1)  # everything kept
    assert (precision, recall) == pytest.approx((2 / 3, 0.5))
    assert operating_point(matched, 0.99) == (0.99, 0.0, 0.0, 0.0)  # nothing kept


def test_no_predictions_at_all():
    scores, _ = score([ImageResult(truths=(A,), predictions=())])
    assert (scores.map50, scores.map50_95, scores.precision, scores.recall) == (0.0, 0.0, 0.0, 0.0)


# ---- bootstrap ---------------------------------------------------------------


def test_weights_of_one_reproduce_the_plain_score_and_weights_reweight_images():
    scores, matched = score(dataset())
    assert ap_per_iou(matched, np.ones(3))[0] == pytest.approx(scores.map50)
    # Only the first image, three times: one person, found -> AP 1.
    assert ap_per_iou(matched, np.array([3.0, 0.0, 0.0]))[0] == pytest.approx(1.0)
    # Only the third image: a person and no prediction -> AP 0.
    assert ap_per_iou(matched, np.array([0.0, 0.0, 2.0]))[0] == 0.0


def test_bootstrap_is_seeded_and_brackets_the_score():
    rng = np.random.default_rng(1)
    images = []
    for _ in range(200):  # a detector that finds about 70 % of people, with some false alarms
        predictions = []
        if rng.random() < 0.7:
            predictions.append((float(rng.uniform(0.5, 1.0)), A))
        if rng.random() < 0.2:
            predictions.append((float(rng.uniform(0.0, 0.6)), (40.0, 40.0, 50.0, 50.0)))
        images.append(ImageResult(truths=(A,), predictions=tuple(predictions)))
    scores, matched = score(images)
    samples = bootstrap_map(matched, repeats=300, seed=5)
    assert samples.shape == (300, 2)
    assert np.array_equal(samples, bootstrap_map(matched, repeats=300, seed=5))
    low, high = interval(samples[:, 0])
    assert low < scores.map50 < high
    assert 0.02 < high - low < 0.25  # 200 images: an interval several points wide
    # Four times the data -> about half the width.
    _, big = score(images * 4)
    wide_low, wide_high = interval(bootstrap_map(big, repeats=300, seed=5)[:, 0])
    assert (wide_high - wide_low) < 0.7 * (high - low)


def test_interval_and_gap():
    samples = np.arange(1000) / 1000  # 0.000 .. 0.999
    assert interval(samples) == pytest.approx((0.025, 0.974), abs=0.002)
    sim, real = np.full(500, 0.9), np.full(500, 0.3)
    assert gap_interval(sim, real) == pytest.approx((0.6, 0.6))


# ---- YOLO export -------------------------------------------------------------


def test_yolo_line_turns_a_corner_into_a_centre():
    # FiftyOne [x, y, w, h] = left 0.25, top 0.25, 0.05 wide, 0.1 high.
    assert export.yolo_line("person", [0.25, 0.25, 0.05, 0.1]) == "0 0.275000 0.300000 0.050000 0.100000"
    with pytest.raises(ValueError):
        export.yolo_line("tent", [0, 0, 1, 1])


fo = pytest.importorskip("fiftyone", reason="fiftyone not installed - run `make view` once")


@pytest.fixture
def frames(tmp_path):
    import uuid

    from PIL import Image

    name = f"test-export-{uuid.uuid4().hex[:8]}"
    data = fo.Dataset(name)
    samples = []
    for index in range(4):
        path = tmp_path / "src" / f"frame_{index}.jpg"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 48), (index * 40, 120, 60)).save(path)
        sample = fo.Sample(filepath=str(path))
        boxes = [] if index == 3 else [fo.Detection(label="person", bounding_box=[0.1 * index, 0.2, 0.05, 0.1])]
        if index == 1:
            boxes.append(fo.Detection(label="tent", bounding_box=[0.5, 0.5, 0.1, 0.1]))  # not a trained class
        sample["ground_truth"] = fo.Detections(detections=boxes)
        sample["predictions"] = fo.Detections(
            detections=[fo.Detection(label="person", bounding_box=[0.1 * index, 0.2, 0.05, 0.1], confidence=0.9 - 0.1 * index)]
        )
        samples.append(sample)
    data.add_samples(samples, progress=False)
    yield data
    fo.delete_dataset(name)


def test_export_writes_links_labels_yaml_and_manifest(frames, tmp_path):
    views = {"train": frames.limit(3), "val": frames.skip(3)}
    out = tmp_path / "export"
    manifest = export.export_dataset(views, out, "v-test", {"train": "train-v1", "val": "val-v1"}, ("abc1234", False))
    assert manifest["splits"]["train"] == {"view": "train-v1", "images": 3, "boxes": 3, "empty_images": 0}
    assert manifest["splits"]["val"] == {"view": "val-v1", "images": 1, "boxes": 0, "empty_images": 1}
    assert manifest["commit"] == "abc1234" and len(manifest["hash"]) == 16

    link = out / "images" / "train" / "frame_1.jpg"
    assert link.is_symlink() and link.resolve() == (tmp_path / "src" / "frame_1.jpg").resolve()
    # The tent is not a trained class and is left out; the person line remains.
    assert (out / "labels" / "train" / "frame_1.txt").read_text() == "0 0.125000 0.250000 0.050000 0.100000\n"
    assert (out / "labels" / "val" / "frame_3.txt").read_text() == ""  # a background image
    yaml_text = (out / "dataset.yaml").read_text()
    assert "train: images/train" in yaml_text and "val: images/val" in yaml_text and "0: person" in yaml_text
    assert json.loads((out / "export.json").read_text())["hash"] == manifest["hash"]


def test_export_hash_identifies_the_data(frames, tmp_path):
    views = {"train": frames.limit(3), "val": frames.skip(3)}
    first = export.export_dataset(views, tmp_path / "a", "v")["hash"]
    assert export.export_dataset(views, tmp_path / "b", "v")["hash"] == first  # same data, same hash
    # One fewer image: a different dataset.
    assert export.export_dataset({"train": frames.limit(2), "val": frames.skip(3)}, tmp_path / "c", "v")["hash"] != first
    # One label changed by a hair: a different dataset.
    sample = frames.first()
    sample.ground_truth.detections[0].bounding_box = [0.0, 0.2, 0.05, 0.1001]
    sample.save()
    assert export.export_dataset(views, tmp_path / "d", "v")["hash"] != first
    # Exporting again replaces the folder instead of failing on existing links.
    export.export_dataset(views, tmp_path / "d", "v")


# ---- scorecard ---------------------------------------------------------------


def test_images_from_view(frames):
    images = scorecard.images_from_view(frames, "predictions")
    assert len(images) == 4
    assert images[1].truths == (pytest.approx((0.1, 0.2, 0.15, 0.3)),)  # the tent is not a person
    assert images[3].truths == () and images[3].predictions[0][0] == pytest.approx(0.6)
    scores, _ = score(images)
    # 3 people, 4 predictions in confidence order: correct x3, then a false alarm.
    assert scores.map50 == pytest.approx(1.0) and scores.precision == 1.0 and scores.recall == 1.0


def test_scorecard_row_and_files(frames, tmp_path):
    images = scorecard.images_from_view(frames, "predictions")
    sim = scorecard.evaluate(images, repeats=50)
    real = scorecard.evaluate([ImageResult(truths=(A,), predictions=())] * 5, confidence=sim.scores.confidence, repeats=50)
    row = scorecard.scorecard_row(
        "v1", "baseline", sim, real, train_hash="deadbeef", train_images=3, commit="abc1234",
        evaluated_utc="2026-11-15T00:00:00Z",
    )  # fmt: skip
    assert row["sim_map50"] == 1.0 and row["real_map50"] == 0.0 and row["gap_map50"] == 1.0
    assert row["sim_map50_low"] <= row["sim_map50"] <= row["sim_map50_high"]
    assert (row["sim_images"], row["real_images"]) == (4, 5)

    path = tmp_path / "scorecard.csv"
    scorecard.upsert(row, path)
    scorecard.upsert({**row, "version": "v2", "change": "randomization"}, path)
    rows = scorecard.upsert({**row, "sim_map50": 0.5}, path)  # v1 evaluated again
    assert [r["version"] for r in rows] == ["v1", "v2"], "a version has one row, replaced on re-evaluation"
    assert rows[0]["sim_map50"] == "0.5"
    text = path.with_suffix(".md").read_text()
    assert "| v1 | baseline | 0.500 (" in text and "`deadbeef`" in text and "| v2 | randomization |" in text


# ---- predictions -------------------------------------------------------------


def test_model_boxes_become_fiftyone_detections():
    from sardrone.train.predict import boxes_to_detections, field_for

    assert field_for("v1") == "pred_v1"
    detections = boxes_to_detections(
        [[0.25, 0.25, 0.30, 0.35], [-0.02, 0.9, 0.05, 1.03], [0.5, 0.5, 0.5, 0.6]],
        [0.8, 0.3, 0.9],
        [0.0, 0.0, 0.0],
        {0: "person"},
    ).detections
    assert len(detections) == 2  # the zero-width box is dropped
    # Corners (0.25, 0.25)-(0.30, 0.35) -> FiftyOne [x, y, w, h].
    assert detections[0].bounding_box == pytest.approx([0.25, 0.25, 0.05, 0.10])
    assert detections[0].label == "person" and detections[0].confidence == 0.8
    # A box sticking out of the image is clipped to it.
    assert detections[1].bounding_box == pytest.approx([0.0, 0.9, 0.05, 0.1])


def test_training_record_keeps_every_setting():
    from dataclasses import asdict

    from sardrone.train.train import TrainConfig

    config = asdict(TrainConfig())
    assert config["imgsz"] == 1280, "640 would shrink a 30 px person to 15 px"
    assert config["seed"] == 51 and config["flipud"] == 0.0
    assert {"mosaic", "fliplr", "hsv_h", "hsv_s", "hsv_v", "scale", "translate"} <= set(config)


def test_subsample_is_seeded_and_spread_over_episodes(frames):
    frames.set_values("episode_id", ["a", "a", "a", "b"])
    sample = export.subsample_by_episode(frames, 2, seed=1)
    # About half of each episode: 2 of a's 3 frames (rounded from 1.5), and b's only frame.
    assert sorted(sample.values("episode_id")) == ["a", "a", "b"]
    assert sample.values("id") == export.subsample_by_episode(frames, 2, seed=1).values("id")
    assert len(export.subsample_by_episode(frames, 10)) == 4  # asking for more than exists


# ---- camera realism ----------------------------------------------------------


def test_realism_draw_is_seeded_by_name_and_within_range():
    from sardrone.train.realism import RealismConfig, draw_for

    config = RealismConfig()
    first, _ = draw_for("frame_1.jpg", 51)
    assert first == draw_for("frame_1.jpg", 51)[0]
    assert first != draw_for("frame_2.jpg", 51)[0] and first != draw_for("frame_1.jpg", 52)[0]
    draws = [draw_for(f"f{i}.jpg", 51)[0] for i in range(400)]
    assert all(config.gain[0] <= d.gain <= config.gain[1] for d in draws)
    assert all(config.jpeg_quality[0] <= d.jpeg_quality <= config.jpeg_quality[1] for d in draws)
    assert all(d.blur_radius_px == 0.0 or 0.3 <= d.blur_radius_px <= 1.2 for d in draws)
    blurred = sum(d.blur_radius_px > 0 for d in draws)
    assert 160 <= blurred <= 240  # about half of 400


def test_realism_effects_are_measurable():
    import io

    import numpy as np
    from PIL import Image

    from sardrone.train.realism import RealismDraw, apply

    grey = Image.new("RGB", (128, 96), (128, 128, 128))

    def result(draw):
        return np.asarray(Image.open(io.BytesIO(apply(grey, draw, np.random.default_rng(0)))), dtype=float)

    plain = RealismDraw(gain=1.0, gamma=1.0, blur_radius_px=0.0, noise_sigma=0.0, jpeg_quality=95)
    assert result(plain).mean() == pytest.approx(128.0, abs=1.0)
    assert result(plain).shape == (96, 128, 3)  # size is unchanged, so the boxes still fit
    # Gain 1.2 brightens by 20 %: 128 -> 153.6.
    brighter = RealismDraw(1.2, 1.0, 0.0, 0.0, 95)
    assert result(brighter).mean() == pytest.approx(153.6, abs=1.5)
    # Gamma 1.15 darkens mid-grey: (128/255) ** 1.15 * 255 = 115.4.
    assert result(RealismDraw(1.0, 1.15, 0.0, 0.0, 95)).mean() == pytest.approx(115.4, abs=1.5)
    # Noise of sigma 6 shows up as about that much spread (JPEG smooths a little).
    noisy = result(RealismDraw(1.0, 1.0, 0.0, 6.0, 95))
    assert 3.5 < noisy.std() < 7.0 and noisy.mean() == pytest.approx(128.0, abs=1.0)
    # Stronger compression gives a smaller file for the same noisy picture.
    rng = np.random.default_rng(1)
    busy = Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
    low = apply(busy, RealismDraw(1.0, 1.0, 0.0, 0.0, 35), np.random.default_rng(0))
    high = apply(busy, RealismDraw(1.0, 1.0, 0.0, 0.0, 85), np.random.default_rng(0))
    assert len(low) < 0.6 * len(high)


def test_export_with_realism_changes_training_images_only(frames, tmp_path):
    views = {"train": frames.limit(3), "val": frames.skip(3)}
    out = tmp_path / "realism"
    manifest = export.export_dataset(views, out, "v3", realism_seed=51)
    train_image, val_image = out / "images" / "train" / "frame_0.jpg", out / "images" / "val" / "frame_3.jpg"
    assert not train_image.is_symlink() and val_image.is_symlink()
    assert train_image.read_bytes() != (tmp_path / "src" / "frame_0.jpg").read_bytes()
    assert manifest["realism_seed"] == 51
    # Same seed, same bytes, same hash; and the hash differs from the plain export.
    assert export.export_dataset(views, tmp_path / "again", "v3", realism_seed=51)["hash"] == manifest["hash"]
    assert export.export_dataset(views, tmp_path / "plain", "v1")["hash"] != manifest["hash"]
    # Labels are untouched.
    assert (out / "labels" / "train" / "frame_1.txt").read_text() == "0 0.125000 0.250000 0.050000 0.100000\n"


def test_extra_real_images_are_added_to_training_unchanged(frames, tmp_path):
    views = {"train": frames.limit(2), "val": frames.skip(3)}
    extra = frames.skip(2).limit(1)  # stands in for a real photograph
    out = tmp_path / "mixed"
    manifest = export.export_dataset(views, out, "v4", realism_seed=51, extra_train=extra, notes={"real": "1 image"})
    assert manifest["splits"]["train"]["images"] == 3 and manifest["extra_train_images"] == 1
    assert (out / "images" / "train" / "frame_2.jpg").is_symlink(), "real images are linked, never degraded"
    assert not (out / "images" / "train" / "frame_0.jpg").is_symlink()
    assert manifest["notes"] == {"real": "1 image"}
