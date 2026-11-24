"""Detection metrics, written out in full so every number can be explained.

For each image we have the true boxes and the predicted boxes, each
prediction with a confidence. The steps:

1. MATCH. Go through the predictions from most to least confident. A
   prediction is a TRUE POSITIVE if it overlaps a not-yet-claimed true box
   by at least the IoU threshold (IoU = shared area / combined area);
   otherwise it is a FALSE POSITIVE. Each true box can be claimed once.
2. PRECISION-RECALL CURVE. Lower the confidence cut-off step by step. At
   each step: precision = correct / drawn, recall = correct / existing.
3. AVERAGE PRECISION (AP) is the area under that curve, a single number in
   0..1 that does not depend on choosing a cut-off. "mAP@0.5" uses IoU 0.5;
   "mAP@0.5:0.95" averages AP over IoU 0.50, 0.55, ... 0.95 and so also
   rewards tight boxes. (With one class, mAP and AP are the same thing.)

Uncertainty: a score from 1,400 test images would come out a little
different on another 1,400. The BOOTSTRAP estimates how much: draw 1,400
images at random WITH replacement from the test set, recompute the score,
repeat 1,000 times; the middle 95 % of those scores is the confidence
interval. Whole images are resampled, never single boxes, because boxes in
one image are not independent.
"""

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 (any consistent unit)
COCO_IOUS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class ImageResult:
    truths: tuple[Box, ...]
    predictions: tuple[tuple[float, Box], ...]  # (confidence, box)


def iou(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    combined = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / combined


def match_image(image: ImageResult, iou_threshold: float) -> list[tuple[float, bool]]:
    """[(confidence, is_true_positive)] for one image, most confident first."""
    claimed: set[int] = set()
    outcome = []
    for confidence, box in sorted(image.predictions, key=lambda p: -p[0]):
        best, best_iou = -1, iou_threshold
        for index, truth in enumerate(image.truths):
            if index in claimed:
                continue
            overlap = iou(box, truth)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best >= 0:
            claimed.add(best)
        outcome.append((confidence, best >= 0))
    return outcome


@dataclass
class Matched:
    """All predictions of a dataset, flattened, ready for fast re-scoring."""

    confidence: np.ndarray  # (n,) sorted from most to least confident
    correct: np.ndarray  # (n_ious, n) bool: true positive at each IoU threshold
    image: np.ndarray  # (n,) index of the image each prediction belongs to
    truths_per_image: np.ndarray  # (n_images,)
    ious: tuple[float, ...]


def match_dataset(images: list[ImageResult], ious: tuple[float, ...] = COCO_IOUS) -> Matched:
    confidences, image_index, correct = [], [], [[] for _ in ious]
    for number, image in enumerate(images):
        for level, threshold in enumerate(ious):
            matches = match_image(image, threshold)
            correct[level].extend(is_tp for _, is_tp in matches)
            if level == 0:
                confidences.extend(confidence for confidence, _ in matches)
                image_index.extend([number] * len(matches))
    order = np.argsort(-np.asarray(confidences, dtype=float), kind="stable")
    return Matched(
        confidence=np.asarray(confidences, dtype=float)[order],
        correct=np.asarray(correct, dtype=bool).reshape(len(ious), -1)[:, order],
        image=np.asarray(image_index, dtype=int)[order],
        truths_per_image=np.asarray([len(image.truths) for image in images], dtype=float),
        ious=tuple(ious),
    )


def _curve(correct: np.ndarray, weights: np.ndarray, total_truths: float):
    """Precision and recall after each prediction, most confident first."""
    true_positives = np.cumsum(correct * weights)
    drawn = np.cumsum(weights)
    recall = true_positives / total_truths if total_truths > 0 else np.zeros_like(true_positives)
    precision = np.divide(true_positives, drawn, out=np.zeros_like(true_positives), where=drawn > 0)
    return precision, recall


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the precision-recall curve, the COCO way.

    The raw curve zig-zags. First it is smoothed into its upper envelope:
    the precision at a recall level becomes the best precision at that
    recall OR ANY HIGHER one. Then it is read off at 101 evenly spaced
    recall levels (0.00, 0.01, ... 1.00) and averaged. Recall levels the
    detector never reaches count as precision 0.
    """
    if len(precision) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recall, RECALL_POINTS, side="left")
    reached = positions < len(envelope)
    values = np.zeros(len(RECALL_POINTS))
    values[reached] = envelope[positions[reached]]
    return float(values.mean())


def ap_per_iou(matched: Matched, image_weights: np.ndarray | None = None) -> np.ndarray:
    """AP at each IoU threshold. `image_weights` says how many times each
    image counts (all ones normally; bootstrap resampling changes it)."""
    if image_weights is None:
        image_weights = np.ones(len(matched.truths_per_image))
    weights = image_weights[matched.image] if len(matched.image) else np.zeros(0)
    total = float(image_weights @ matched.truths_per_image)
    return np.asarray(
        [average_precision(*_curve(matched.correct[level], weights, total)) for level in range(len(matched.ious))]
    )


@dataclass(frozen=True)
class Scores:
    images: int
    truths: int
    predictions: int
    map50: float
    map50_95: float
    confidence: float  # the cut-off the next three numbers are measured at
    precision: float
    recall: float
    f1: float


def operating_point(matched: Matched, confidence: float | None = None) -> tuple[float, float, float, float]:
    """(confidence, precision, recall, F1) at IoU 0.5.

    With `confidence=None` the cut-off with the best F1 on THIS data is
    used. That is fine for choosing a cut-off on validation data; on a test
    set, pass the cut-off chosen elsewhere, or the result is flattering.
    """
    level = matched.ious.index(0.5)
    precision, recall = _curve(
        matched.correct[level], np.ones(len(matched.confidence)), float(matched.truths_per_image.sum())
    )
    if len(precision) == 0:
        return (confidence if confidence is not None else 0.0), 0.0, 0.0, 0.0
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    if confidence is None:
        index = int(np.argmax(f1))
    else:
        kept = int(np.sum(matched.confidence >= confidence))
        if kept == 0:
            return confidence, 0.0, 0.0, 0.0
        index = kept - 1
    chosen = confidence if confidence is not None else float(matched.confidence[index])
    return chosen, float(precision[index]), float(recall[index]), float(f1[index])


def score(images: list[ImageResult], confidence: float | None = None) -> tuple[Scores, Matched]:
    matched = match_dataset(images)
    aps = ap_per_iou(matched)
    chosen, precision, recall, f1 = operating_point(matched, confidence)
    scores = Scores(
        images=len(images),
        truths=int(matched.truths_per_image.sum()),
        predictions=len(matched.confidence),
        map50=float(aps[matched.ious.index(0.5)]),
        map50_95=float(aps.mean()),
        confidence=chosen,
        precision=precision,
        recall=recall,
        f1=f1,
    )
    return scores, matched


def bootstrap_map(matched: Matched, repeats: int = 1000, seed: int = 0) -> np.ndarray:
    """`repeats` x 2 array of (mAP@0.5, mAP@0.5:0.95) on resampled image sets."""
    rng = np.random.default_rng(seed)
    count = len(matched.truths_per_image)
    half = matched.ious.index(0.5)
    out = np.zeros((repeats, 2))
    for repeat in range(repeats):
        # How many times each image was drawn when picking `count` with replacement.
        weights = np.bincount(rng.integers(0, count, count), minlength=count).astype(float)
        aps = ap_per_iou(matched, weights)
        out[repeat] = aps[half], aps.mean()
    return out


def paired_difference(a: Matched, b: Matched, repeats: int = 1000, seed: int = 0) -> np.ndarray:
    """`repeats` samples of mAP@0.5(b) - mAP@0.5(a), resampling the SAME images
    for both. Use when two methods ran on the same test set: the paired
    difference is far tighter than comparing two separate intervals."""
    count = len(a.truths_per_image)
    if len(b.truths_per_image) != count or not np.array_equal(a.truths_per_image, b.truths_per_image):
        raise ValueError("paired_difference needs the same images, in the same order, with the same truths")
    rng = np.random.default_rng(seed)
    ha, hb = a.ious.index(0.5), b.ious.index(0.5)
    out = np.zeros(repeats)
    for repeat in range(repeats):
        weights = np.bincount(rng.integers(0, count, count), minlength=count).astype(float)
        out[repeat] = ap_per_iou(b, weights)[hb] - ap_per_iou(a, weights)[ha]
    return out


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def gap_interval(sim: np.ndarray, real: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval of (sim score - real score).

    The two test sets are separate, so their bootstrap draws are paired up
    arbitrarily: replicate i of one minus replicate i of the other.
    """
    return interval(sim - real, level)
