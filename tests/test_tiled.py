"""Tiled detection: tile geometry, edge rule, merging, coordinate mapping. No model needed."""

import itertools
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from sardrone.train.tiled import keep_off_inner_edges, merge, predict_image, tile_grid, tile_starts  # noqa: E402


@pytest.mark.parametrize("length", [500, 1280, 1281, 1920, 2250, 3000, 4000, 5000])
def test_tiles_cover_the_axis_with_the_promised_overlap(length):
    tile, overlap = 1280, 256
    starts = tile_starts(length, tile, overlap)
    assert starts[0] == 0
    ends = [min(s + tile, length) for s in starts]
    assert ends[-1] == length  # last tile ends exactly at the image edge
    assert all(e <= length for e in ends)  # never past the image
    for (s0, e0), (s1, _) in zip(zip(starts, ends), zip(starts[1:], ends[1:])):
        assert s1 > s0
        assert e0 - s1 >= overlap  # neighbours overlap by at least `overlap`


def test_every_pixel_of_a_heridal_photo_is_in_some_tile():
    width, height = 4000, 3000
    tiles = tile_grid(width, height)
    for x, y in itertools.product(range(0, width, 37), range(0, height, 37)):
        assert any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in tiles)


def test_a_person_smaller_than_the_overlap_is_whole_in_some_tile():
    # 56 px is the median HERIDAL person; the overlap is 256 px.
    width, height, size = 4000, 3000, 56
    tiles = tile_grid(width, height)
    for x, y in itertools.product(range(0, width - size, 53), range(0, height - size, 53)):
        assert any(x0 <= x and y0 <= y and x + size <= x1 and y + size <= y1 for x0, y0, x1, y1 in tiles)


def test_boxes_on_inner_cut_lines_are_dropped_but_image_borders_are_kept():
    width, height = 4000, 3000
    tile = (1000, 0, 2280, 1280)  # an interior column, top row
    boxes = [
        (1000, 500, 1060, 560),  # touches the inner left cut: drop
        (1500, 1250, 1560, 1280),  # touches the inner bottom cut: drop
        (1500, 0, 1560, 40),  # touches the top, which is the image border: keep
        (1500, 500, 1560, 560),  # well inside: keep
    ]
    assert keep_off_inner_edges(boxes, tile, width, height) == [False, False, True, True]


def test_merge_keeps_the_most_confident_of_duplicates():
    boxes = [(100, 100, 150, 160), (102, 101, 151, 161), (400, 400, 450, 460)]
    scores = [0.6, 0.9, 0.5]
    kept, kept_scores = merge(boxes, scores)
    assert len(kept) == 2
    assert kept_scores == pytest.approx([0.9, 0.5])
    assert merge([], []) == ([], [])


def _result(boxes, scores, classes, names):
    return SimpleNamespace(
        names=names,
        boxes=SimpleNamespace(
            xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            conf=torch.tensor(scores, dtype=torch.float32),
            cls=torch.tensor(classes, dtype=torch.float32),
        ),
    )


class FakeModel:
    """Answers the full-image call, then one result per tile; records what it saw."""

    names = {0: "person", 2: "car"}

    def __init__(self, person_at):
        self.person_at = person_at  # (x0, y0, x1, y1) in image pixels
        self.calls = []

    def predict(self, source, imgsz, **_):
        self.calls.append((source, imgsz))
        if not isinstance(source, list):  # full image: misses the small person, sees a car
            return [_result([(10, 10, 300, 200)], [0.8], [2], self.names)]
        results = []
        for crop in source:
            x0, y0 = crop.origin
            px0, py0, px1, py1 = self.person_at
            w, h = crop.size
            if x0 <= px0 and y0 <= py0 and px1 <= x0 + w and py1 <= y0 + h:
                results.append(_result([(px0 - x0, py0 - y0, px1 - x0, py1 - y0)], [0.7], [0], self.names))
            else:
                results.append(_result([], [], [], self.names))
        return results


class FakeImage:
    def __init__(self, width, height):
        self.size = (width, height)

    def crop(self, box):
        crop = FakeImage(box[2] - box[0], box[3] - box[1])
        crop.origin = (box[0], box[1])
        return crop


def test_tile_boxes_come_back_in_image_pixels_and_non_people_are_ignored():
    person = (2500, 1700, 2556, 1753)  # a 56 x 53 px person
    model = FakeModel(person)
    boxes, scores = predict_image(model, FakeImage(4000, 3000), imgsz=1280, device="cpu")
    # The car from the full-image pass is not a person and is dropped; the
    # person, seen by several overlapping tiles, is merged to one box.
    assert len(boxes) == 1
    assert boxes[0] == pytest.approx(person)
    assert scores == pytest.approx([0.7])
    full_call, tile_call = model.calls
    assert full_call[1] == 1280 and tile_call[1] == 1280
    assert len(tile_call[0]) == len(tile_grid(4000, 3000))


def _matched(correct_flags_per_image, truths_per_image):
    """A tiny Matched: one prediction per flag, confidences descending."""
    import numpy as np

    from sardrone.train.metrics import COCO_IOUS, Matched

    image, correct = [], []
    for number, flags in enumerate(correct_flags_per_image):
        for flag in flags:
            image.append(number)
            correct.append(flag)
    n = len(correct)
    return Matched(
        confidence=np.linspace(1.0, 0.1, n),
        correct=np.array([correct] * len(COCO_IOUS), dtype=bool),
        image=np.array(image),
        truths_per_image=np.array(truths_per_image),
        ious=COCO_IOUS,
    )


def test_paired_difference_is_zero_for_identical_methods_and_positive_for_a_better_one():
    import numpy as np

    from sardrone.train.metrics import paired_difference

    truths = [2, 1, 2, 1]
    worse = _matched([[True, False], [False], [True, False], [False]], truths)
    better = _matched([[True, True], [True], [True, True], [True]], truths)
    assert np.allclose(paired_difference(worse, worse, 200), 0.0)
    gains = paired_difference(worse, better, 200)
    assert (gains > 0).mean() > 0.9


def test_paired_difference_refuses_different_test_sets():
    from sardrone.train.metrics import paired_difference

    a = _matched([[True], [False]], [1, 1])
    b = _matched([[True], [False]], [1, 2])
    with pytest.raises(ValueError):
        paired_difference(a, b, 10)
