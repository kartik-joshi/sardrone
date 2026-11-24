"""Phase 2E: curation and auto-label scoring, on a tiny hand-made dataset.

No model is downloaded or run here: embeddings and "predictions" are made
up, so the bookkeeping (duplicates, splits, views, precision / recall) can
be checked against numbers worked out by hand.
"""

import uuid

import pytest
from PIL import Image

fo = pytest.importorskip("fiftyone", reason="fiftyone not installed - run `make view` once")
pytest.importorskip("sklearn")

from sardrone.fo import autolabel, curate  # noqa: E402
from sardrone.fo.autolabel import (  # noqa: E402
    FIELD,
    precision_recall,
    recall_by,
    recall_by_size,
    save_review_queue,
    score,
    to_detections,
)


def person(x, y, w=0.05, h=0.1, **attributes):
    defaults = dict(pose="standing", occlusion="none", width_px=64.0, height_px=96.0, person_id="person_1")
    return fo.Detection(label="person", bounding_box=[x, y, w, h], **{**defaults, **attributes})


def prediction(x, y, confidence, w=0.05, h=0.1, label="person"):
    return fo.Detection(label=label, bounding_box=[x, y, w, h], confidence=confidence)


@pytest.fixture
def dataset(tmp_path):
    """Six frames from three episodes; frames 0 and 1 are almost the same picture."""
    name = f"test-curate-{uuid.uuid4().hex[:8]}"
    data = fo.Dataset(name)
    rows = [
        # episode, time of day, embedding, ground truth
        ("train-001", "noon", [1.0, 0.0, 0.0], [person(0.1, 0.1)]),
        ("train-001", "noon", [0.9999, 0.01, 0.0], [person(0.1, 0.1)]),
        ("train-001", "noon", [0.0, 1.0, 0.0], []),
        ("train-002", "dusk", [0.0, 0.0, 1.0], [person(0.5, 0.5)]),
        ("train-010", "noon", [0.7, 0.7, 0.0], [person(0.3, 0.3, pose="lying"), person(0.6, 0.6, occlusion="partial")]),
        ("test-001", "golden_hour", [0.0, 0.7, 0.7], [person(0.2, 0.7, width_px=10.0, height_px=12.0)]),
    ]  # fmt: skip
    samples = []
    for index, (episode, time_of_day, embedding, truth) in enumerate(rows):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (64, 48), (90, 120, 60)).save(path)
        sample = fo.Sample(filepath=str(path))
        sample["episode_id"], sample["time_of_day"] = episode, time_of_day
        sample["world"], sample["mission_state"] = "meadow", "SEARCH"
        sample["num_people"] = len(truth)
        sample[curate.EMBEDDING_FIELD] = embedding
        sample["ground_truth"] = fo.Detections(detections=truth)
        samples.append(sample)
    data.add_samples(samples, progress=False)
    yield data
    fo.delete_dataset(name)


def frame(dataset, index):
    return dataset.match(fo.ViewField("filepath").ends_with(f"frame_{index}.jpg")).first()


# ---- curation ----------------------------------------------------------------


def test_find_duplicates_by_hand():
    import math

    def unit(degrees):
        return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]

    # Cosine distance between directions d degrees apart is 1 - cos(d):
    # 2 deg -> 0.0006, 5 deg -> 0.0038, 10 deg -> 0.0152.
    embeddings = [unit(0), unit(2), unit(4), unit(10), unit(1), unit(0)]
    episodes = ["a", "a", "a", "a", "b", "b"]
    times = [0.0, 0.5, 1.0, 1.5, 0.0, 0.5]

    # Threshold 0.001: frame 1 (2 deg from frame 0) is a duplicate of frame 0.
    # Frame 2 is 4 deg from frame 0 (0.0024): kept. Episode b: frame 5 is
    # 1 deg from frame 4 (0.00015): duplicate. Frame 4 is NOT merged with
    # frame 0 although they are 1 deg apart: different episodes.
    assert curate.find_duplicates(embeddings, episodes, times, 0.001) == {1: 0, 5: 4}
    # Threshold 0.004: frame 2 (0.0024 from frame 0) goes too; frame 3 is
    # 10 deg from frame 0 (0.0152): stays.
    assert curate.find_duplicates(embeddings, episodes, times, 0.004) == {1: 0, 2: 0, 5: 4}
    # Each duplicate points at the NEAREST kept frame: with frames 0 and 3
    # kept, a frame at 9 deg belongs to frame 3.
    found = curate.find_duplicates([unit(0), unit(10), unit(9)], ["a"] * 3, [0, 1, 2], 0.004)
    assert found == {2: 1}
    assert curate.find_duplicates(embeddings, episodes, times, 0.0) == {}


def test_order_follows_time_not_position_in_the_list():
    # The frame recorded first is the one kept, whatever order they arrive in.
    found = curate.find_duplicates([[1.0, 0.0], [1.0, 0.0]], ["a", "a"], [5.0, 1.0], 0.001)
    assert found == {0: 1}


def test_near_duplicates_are_tagged_and_the_first_of_each_group_is_kept(dataset):
    dataset.set_values("time_s", [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    stats = curate.tag_duplicates(dataset, threshold=0.004)
    assert stats["duplicates"] == 1 and stats["different_people_count"] == 0
    assert stats["by_state"] == {"SEARCH": 1}
    tagged = dataset.match_tags(curate.DUPLICATE_TAG)
    assert [s.filepath[-11:] for s in tagged] == ["frame_1.jpg"]
    assert len(dataset) == 6, "duplicates are tagged, never deleted"


def test_a_stricter_threshold_keeps_everything_and_retagging_replaces(dataset):
    dataset.set_values("time_s", [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    curate.tag_duplicates(dataset, threshold=0.004)
    stats = curate.tag_duplicates(dataset, threshold=1e-7)
    assert stats["duplicates"] == 0
    assert len(dataset.match_tags(curate.DUPLICATE_TAG)) == 0


def test_split_comes_from_the_episode_id():
    assert curate.split_of("test-007") == "test"
    assert curate.split_of("train-021") == "train"
    assert [curate.split_of(f"train-{n:03d}") for n in (10, 20, 60)] == ["val"] * 3
    assert curate.split_of("dr-012") == "dr"
    assert curate.split_of("veh-004") == "veh"
    assert curate.split_of("rich-004") == "rich"
    assert curate.split_of("pilot-003") == "pilot" and curate.split_of("pilot-meadow-search") == "pilot"
    # 60 training episodes: 6 for validation, 54 to train on.
    splits = [curate.split_of(f"train-{n:03d}") for n in range(1, 61)]
    assert splits.count("val") == 6 and splits.count("train") == 54


def test_whole_episodes_go_to_one_split(dataset):
    counts = curate.assign_splits(dataset)
    assert counts == {"train": 4, "val": 1, "test": 1}
    for episode in dataset.distinct("episode_id"):
        splits = dataset.match(fo.ViewField("episode_id") == episode).distinct("split")
        assert len(splits) == 1, "frames of one flight must never be in two splits"
    assert frame(dataset, 4).split == "val" and frame(dataset, 5).split == "test"


def test_saved_views(dataset):
    dataset.set_values("time_s", [0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    curate.tag_duplicates(dataset, threshold=0.004)
    curate.assign_splits(dataset)
    counts = curate.save_views(dataset)
    # 4 train frames minus 1 duplicate; 1 val frame; 1 test frame; nothing flagged yet.
    assert counts[curate.VIEW_TRAIN] == 3 and counts[curate.VIEW_VAL] == 1
    assert counts[curate.VIEW_TEST] == 1 and counts[curate.VIEW_FALSE_ALARMS] == 0
    # Hard: the dusk frame, the lying + occluded frame, and the small (12 px) person.
    hard = dataset.load_saved_view(curate.VIEW_HARD)
    assert sorted(s.filepath[-11:] for s in hard) == ["frame_3.jpg", "frame_4.jpg", "frame_5.jpg"]
    train = dataset.load_saved_view(curate.VIEW_TRAIN)
    assert train.distinct("split") == ["train"]
    assert not set(train.distinct("episode_id")) & set(dataset.load_saved_view(curate.VIEW_TEST).distinct("episode_id"))
    assert curate.save_views(dataset) == counts  # saving again replaces, no error


def test_embeddings_are_only_computed_when_missing(dataset):
    messages = []
    assert curate.ensure_embeddings(dataset, log=messages.append) == 0
    assert messages == [], "every frame already has an embedding: no model is loaded"


# ---- auto-label scoring ------------------------------------------------------


def test_to_detections_converts_pixels_and_clips_to_the_image():
    found = [("person", 0.3, 320.0, 240.0, 384.0, 336.0), ("tent", 0.2, -10.0, 900.0, 50.0, 1000.0),
             ("person", 0.1, 700.0, 100.0, 700.0, 150.0)]  # fmt: skip
    detections = to_detections(found, 1280, 960).detections
    assert len(detections) == 2  # the zero-width box is dropped
    assert detections[0].bounding_box == pytest.approx([0.25, 0.25, 0.05, 0.1])
    assert detections[0].confidence == 0.3 and detections[1].label == "tent"
    # A box sticking out of the image is cut at the border: x 0..50, y 900..960.
    assert detections[1].bounding_box == pytest.approx([0.0, 0.9375, 50 / 1280, 60 / 960])


@pytest.fixture
def predicted(dataset):
    """Hand-made predictions with a known score.

    5 true people in total (frames 0, 1, 3 have one; frame 4 has two; frame 5
    has one small one = 6 boxes, of which frame 1 duplicates frame 0).
    """
    predictions = {
        0: [prediction(0.1, 0.1, 0.30)],  # exact hit
        1: [prediction(0.1, 0.1, 0.25), prediction(0.7, 0.7, 0.08)],  # hit + false alarm
        2: [prediction(0.4, 0.4, 0.12), prediction(0.4, 0.4, 0.2, label="tent")],  # false alarm on empty frame
        3: [prediction(0.52, 0.5, 0.18)],  # shifted by 40 % of the width: IoU 0.43
        4: [prediction(0.3, 0.3, 0.06)],  # finds the lying person, misses the occluded one
        5: [],  # misses the small person
    }
    for index, found in predictions.items():
        sample = frame(dataset, index)
        sample[FIELD] = fo.Detections(detections=found)
        sample.save()
    return dataset


def test_precision_and_recall_by_hand(predicted):
    summary = score(predicted)
    assert summary["frames"] == 6 and summary["true_people"] == 6
    assert summary["predicted_person_boxes"] == 6  # the tent box is not a person box
    assert summary["other_classes"] == {"tent": 1}

    rows = {row["threshold"]: row for row in summary["iou_0.5"]}
    # All 6 person boxes: 3 correct (frames 0, 1, 4), 3 wrong (two false
    # alarms and the shifted box). 6 people, so 3 are missed.
    assert (rows[0.05]["tp"], rows[0.05]["fp"], rows[0.05]["fn"]) == (3, 3, 3)
    assert rows[0.05]["precision"] == 0.5 and rows[0.05]["recall"] == 0.5
    # Confidence >= 0.1 drops the 0.08 false alarm and the 0.06 correct box.
    assert (rows[0.1]["tp"], rows[0.1]["fp"], rows[0.1]["fn"]) == (2, 2, 4)
    # Confidence >= 0.2: only the 0.30 and 0.25 boxes, both correct.
    assert (rows[0.2]["tp"], rows[0.2]["fp"]) == (2, 0)
    assert rows[0.2]["precision"] == 1.0 and rows[0.2]["recall"] == pytest.approx(2 / 6, abs=1e-4)
    assert rows[0.4]["tp"] == 0 and rows[0.4]["precision"] == 0.0

    # At IoU 0.3 the shifted box (IoU 0.43) counts as correct too.
    loose = {row["threshold"]: row for row in summary["iou_0.3"]}
    assert (loose[0.05]["tp"], loose[0.05]["fp"]) == (4, 2)
    assert precision_recall(predicted, "auto_eval") == summary["iou_0.5"]


def test_recall_breakdowns(predicted):
    score(predicted)
    by_pose = recall_by(predicted, "auto_eval", "pose")
    assert by_pose["lying"] == {"found": 1, "total": 1, "recall": 1.0}
    assert by_pose["standing"] == {"found": 2, "total": 5, "recall": 0.4}
    by_occlusion = recall_by(predicted, "auto_eval", "occlusion")
    assert by_occlusion["partial"] == {"found": 0, "total": 1, "recall": 0.0}
    by_size = recall_by_size(predicted, "auto_eval")
    assert by_size["under 16 px"] == {"found": 0, "total": 1, "recall": 0.0}
    assert by_size["64 px and over"]["total"] == 5


def test_review_queue_holds_the_least_confident_boxes(predicted, monkeypatch):
    score(predicted)
    monkeypatch.setattr(autolabel, "REVIEW_FRACTION", 0.34)  # 2 of the 6 person boxes
    queue = save_review_queue(predicted, predicted)
    assert queue["boxes"] == 2 and queue["cutoff"] == 0.08
    assert queue["correct_in_queue"] == 1  # the 0.06 box was right, the 0.08 one was not
    view = predicted.load_saved_view(autolabel.VIEW_REVIEW)
    assert sorted(view.values(f"{FIELD}.detections.confidence", unwind=True)) == [0.06, 0.08]


def test_report_is_written(predicted, tmp_path):
    summary = score(predicted)
    summary["review_queue"] = save_review_queue(predicted, predicted)
    summary["commit"], summary["dirty"] = "abc1234", False
    path = tmp_path / "report.md"
    autolabel.write_report(summary, path)
    text = path.read_text()
    assert "| 0.05 | 0.500 | 0.500 | 0.500 | 3 | 3 | 3 |" in text
    assert "commit `abc1234`" in text and "uncommitted" not in text
    assert path.with_suffix(".json").exists()


# ---- domain gap ---------------------------------------------------------------


def test_domain_separability_is_half_for_one_domain_and_one_for_two():
    import numpy as np

    from sardrone.fo.gapmap import domain_separability

    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(400, 16))
    # Two halves of the same cloud cannot be told apart: accuracy near chance.
    assert domain_separability(cloud[:200], cloud[200:]) == pytest.approx(0.5, abs=0.08)
    # The same cloud shifted far away is trivially separable.
    assert domain_separability(cloud[:200], cloud[200:] + 10.0) == pytest.approx(1.0, abs=0.01)
    # Unequal sizes are balanced first, so "always say the big one" cannot score above 50 %.
    assert domain_separability(cloud[:50], cloud[50:]) == pytest.approx(0.5, abs=0.15)


def test_false_alarm_boxes_use_the_scorecard_matching(predicted):
    from sardrone.fo.false_alarms import false_alarm_boxes

    # Frame 1 (see the `predicted` fixture): a hit at 0.25 and a false box at 0.08.
    sample = frame(predicted, 1)
    found = false_alarm_boxes(sample.ground_truth, sample[autolabel.FIELD], confidence=0.05)
    assert [round(d.confidence, 2) for d in found] == [0.08]
    assert false_alarm_boxes(sample.ground_truth, sample[autolabel.FIELD], confidence=0.1) == []  # cut off
    # Frame 3: the box is shifted (IoU 0.43 < 0.5), so it counts as a false alarm.
    sample = frame(predicted, 3)
    assert len(false_alarm_boxes(sample.ground_truth, sample[autolabel.FIELD], confidence=0.05)) == 1
    # Frame 5: no prediction, nothing to count.
    assert false_alarm_boxes(frame(predicted, 5).ground_truth, frame(predicted, 5)[autolabel.FIELD], 0.05) == []


def test_mark_frames_fills_the_false_alarm_view(predicted):
    from sardrone.fo.false_alarms import mark_frames

    curate.assign_splits(predicted)
    curate.save_views(predicted)
    summary = mark_frames(predicted, autolabel.FIELD, confidence=0.05)
    # False boxes: frame 1 (0.08), frame 2 (0.12, nobody there), frame 3 (shifted).
    assert summary == {"frames": 6, "frames_with_false_alarms": 3, "false_alarm_boxes": 3}
    review = predicted.load_saved_view(curate.VIEW_FALSE_ALARMS)
    assert sorted(s.filepath[-11:] for s in review) == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]
