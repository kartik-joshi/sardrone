"""Auto-label frames with an open-vocabulary detector, then MEASURE it.

    make autolabel     # labels unique frames, scores them, writes the report

An OPEN-VOCABULARY detector finds objects named in a text prompt ("a
person", "a tent") without having been trained on our data. That is how
real, unlabelled drone footage would get its first labels. But a labeller
is only useful if you know how wrong it is. In simulation we have exact
ground truth, so here the auto-labels are scored against it:

* PRECISION: of the boxes it drew, what share is really a person?
* RECALL:    of the people really there, what share did it find?

Both depend on the confidence cut-off, so they are reported for several.
A predicted box counts as correct when it overlaps a true box with IoU
("intersection over union": shared area / combined area) of at least 0.5.

The least confident 5 % of its person boxes are saved as a review queue:
those are the ones a human should look at first.
"""

from sardrone.fo import env  # noqa: F401  (must come before `import fiftyone`)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import fiftyone as fo  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402

from sardrone.fo.curate import DUPLICATE_TAG  # noqa: E402
from sardrone.fo.frames import DATASET_NAME  # noqa: E402

MODEL_NAME = "google/owlv2-base-patch16-ensemble"
# Prompt -> class name. The plan's prompt is "person . backpack . tent . jacket".
PROMPTS = {"a person": "person", "a backpack": "backpack", "a tent": "tent", "a jacket": "jacket"}
FIELD = "auto_labels"
MIN_SCORE = 0.05  # keep low-confidence boxes too; cut-offs are studied afterwards
NMS_IOU = 0.5
VIEW_REVIEW = "auto-label-review"
REVIEW_FRACTION = 0.05
THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
DEFAULT_REPORT = env.PROJECT_ROOT / "docs" / "results" / "auto_labeler.md"


class OwlLabeler:
    """OWLv2 behind a tiny interface: image in, labelled boxes out."""

    def __init__(self, device: str | None = None):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).eval().to(self.device)
        self.prompts = list(PROMPTS)

    def detect(self, image) -> list[tuple[str, float, float, float, float, float]]:
        """[(class, score, x0, y0, x1, y1)] in pixels of `image`."""
        from torchvision.ops import batched_nms

        torch = self.torch
        inputs = self.processor(text=[self.prompts], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Checked on 12 frames against ground truth: boxes come back in the
        # ORIGINAL image's pixels when given its (height, width).
        result = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=MIN_SCORE, target_sizes=torch.tensor([[image.height, image.width]])
        )[0]
        boxes, scores, labels = result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()
        if len(boxes) == 0:
            return []
        # The model proposes many overlapping boxes for one object.
        # Non-maximum suppression keeps the most confident of each cluster.
        keep = batched_nms(boxes, scores, labels, NMS_IOU)
        return [
            (PROMPTS[self.prompts[int(labels[i])]], float(scores[i]), *[float(v) for v in boxes[i]])
            for i in keep
        ]


def to_detections(found, width: int, height: int) -> fo.Detections:
    detections = []
    for label, score, x0, y0, x1, y1 in found:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            fo.Detection(
                label=label,
                confidence=score,
                bounding_box=[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
            )
        )
    return fo.Detections(detections=detections)


def auto_label(view, labeler=None, log=print, every: int = 250) -> int:
    """Label every frame in `view` that has no auto-labels yet. Returns how many."""
    from PIL import Image

    todo = view.exists(FIELD, False) if view._dataset.has_field(FIELD) else view
    total = len(todo)
    if total == 0:
        return 0
    labeler = labeler or OwlLabeler()
    started = time.monotonic()
    # autosave: results are written as we go, so an interrupted run resumes.
    for count, sample in enumerate(todo.iter_samples(autosave=True, batch_size=20), start=1):
        image = Image.open(sample.filepath).convert("RGB")
        sample[FIELD] = to_detections(labeler.detect(image), image.width, image.height)
        if count % every == 0:
            rate = (time.monotonic() - started) / count
            log(f"  {count}/{total} frames, {rate:.2f} s each, about {rate * (total - count) / 60:.0f} min left")
    return total


# ---- scoring -----------------------------------------------------------------


def precision_recall(view, eval_key: str, thresholds=THRESHOLDS) -> list[dict]:
    """Precision and recall for `person` at several confidence cut-offs.

    After one evaluation every predicted person box is marked "tp" (true
    positive: matches a real person) or "fp" (false positive). Predictions
    are matched most-confident first, so raising the cut-off only drops
    the tail: a count over the remaining boxes gives the numbers directly.
    """
    people = view.filter_labels(FIELD, F("label") == "person", only_matches=False)
    confidences = people.values(f"{FIELD}.detections.confidence", unwind=True)
    marks = people.values(f"{FIELD}.detections.{eval_key}", unwind=True)
    total_truth = view.count("ground_truth.detections")
    rows = []
    for threshold in thresholds:
        tp = sum(1 for c, m in zip(confidences, marks) if c >= threshold and m == "tp")
        fp = sum(1 for c, m in zip(confidences, marks) if c >= threshold and m == "fp")
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_truth if total_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": threshold, "tp": tp, "fp": fp, "fn": total_truth - tp,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        })  # fmt: skip
    return rows


def recall_by(view, eval_key: str, attribute: str) -> dict:
    """Recall split by a ground-truth attribute (pose, occlusion, ...), at
    the lowest cut-off: the best the labeller can do for each group."""
    values = view.values(f"ground_truth.detections.{attribute}", unwind=True)
    marks = view.values(f"ground_truth.detections.{eval_key}", unwind=True)
    groups: dict = {}
    for value, mark in zip(values, marks):
        found, total = groups.get(value, (0, 0))
        groups[value] = (found + (mark == "tp"), total + 1)
    return {str(k): {"found": f, "total": t, "recall": round(f / t, 4)} for k, (f, t) in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def recall_by_size(view, eval_key: str) -> dict:
    widths = view.values("ground_truth.detections.width_px", unwind=True)
    heights = view.values("ground_truth.detections.height_px", unwind=True)
    marks = view.values(f"ground_truth.detections.{eval_key}", unwind=True)
    buckets = [("under 16 px", 0, 16), ("16 to 32 px", 16, 32), ("32 to 64 px", 32, 64), ("64 px and over", 64, 1e9)]
    out = {}
    for name, low, high in buckets:
        chosen = [m for w, h, m in zip(widths, heights, marks) if low <= max(w, h) < high]
        if chosen:
            found = sum(m == "tp" for m in chosen)
            out[name] = {"found": found, "total": len(chosen), "recall": round(found / len(chosen), 4)}
    return out


def score(view) -> dict:
    """Evaluate the auto-labels in `view` against ground truth."""
    people_only = view.filter_labels(FIELD, F("label") == "person", only_matches=False)
    results = people_only.evaluate_detections(
        FIELD, gt_field="ground_truth", eval_key="auto_eval", classes=["person"],
        iou=0.5, compute_mAP=True, progress=False,
    )  # fmt: skip
    loose = people_only.evaluate_detections(
        FIELD, gt_field="ground_truth", eval_key="auto_eval_iou30", classes=["person"],
        iou=0.3, progress=False,
    )  # fmt: skip
    del loose
    other = view.filter_labels(FIELD, F("label") != "person", only_matches=False)
    return {
        "model": MODEL_NAME,
        "frames": len(view),
        "true_people": view.count("ground_truth.detections"),
        "predicted_person_boxes": people_only.count(f"{FIELD}.detections"),
        "mAP_50_95": round(float(results.mAP()), 4),
        "iou_0.5": precision_recall(view, "auto_eval"),
        "iou_0.3": precision_recall(view, "auto_eval_iou30"),
        "recall_by_pose": recall_by(view, "auto_eval", "pose"),
        "recall_by_occlusion": recall_by(view, "auto_eval", "occlusion"),
        "recall_by_size": recall_by_size(view, "auto_eval"),
        "recall_by_time_of_day": _recall_by_sample_field(view, "time_of_day"),
        "other_classes": dict(other.count_values(f"{FIELD}.detections.label")),
    }


def _recall_by_sample_field(view, field: str) -> dict:
    out = {}
    for value in sorted(view.distinct(field)):
        marks = view.match(F(field) == value).values("ground_truth.detections.auto_eval", unwind=True)
        if marks:
            found = sum(m == "tp" for m in marks)
            out[value] = {"found": found, "total": len(marks), "recall": round(found / len(marks), 4)}
    return out


def save_review_queue(dataset, view) -> dict:
    """Save the least confident 5 % of person boxes as a named view."""
    confidences = sorted(
        view.filter_labels(FIELD, F("label") == "person", only_matches=False).values(
            f"{FIELD}.detections.confidence", unwind=True
        )
    )
    if not confidences:
        return {"boxes": 0, "frames": 0, "cutoff": None}
    cutoff = confidences[max(0, int(len(confidences) * REVIEW_FRACTION) - 1)]
    queue = view.filter_labels(FIELD, (F("label") == "person") & (F("confidence") <= cutoff))
    dataset.save_view(VIEW_REVIEW, queue, overwrite=True)
    marks = queue.values(f"{FIELD}.detections.auto_eval", unwind=True)
    return {
        "boxes": len(marks),
        "frames": len(queue),
        "cutoff": round(cutoff, 4),
        "correct_in_queue": sum(m == "tp" for m in marks),
    }


def write_report(summary: dict, path: Path) -> None:
    def table(rows):
        lines = ["| Confidence at least | Precision | Recall | F1 | Correct | False alarms | Missed |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]  # fmt: skip
        lines += [
            f"| {r['threshold']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} |"
            for r in rows
        ]
        return "\n".join(lines)

    def groups(title, data):
        lines = [f"| {title} | Found | Of | Recall |", "| --- | --- | --- | --- |"]
        lines += [f"| {k} | {v['found']} | {v['total']} | {v['recall']:.3f} |" for k, v in data.items()]
        return "\n".join(lines)

    review = summary["review_queue"]
    text = f"""# Auto-labeller scored against ground truth

Generated by `make autolabel`. Do not edit by hand.

- Model: `{summary['model']}`, prompts: {', '.join(f'"{p}"' for p in PROMPTS)}
- Data: {summary['frames']} unique frames of `sar-frames` (near-duplicates left out),
  {summary['true_people']} true person boxes from the simulator
- Code: commit `{summary['commit']}`{' with uncommitted changes' if summary['dirty'] else ''}
- The model drew {summary['predicted_person_boxes']} person boxes with confidence at least {MIN_SCORE}
- mAP@0.5:0.95 for `person`: **{summary['mAP_50_95']:.3f}**

## Precision and recall, a box counting as correct at IoU 0.5

{table(summary['iou_0.5'])}

## The same at IoU 0.3 (looser overlap)

{table(summary['iou_0.3'])}

## Who it finds (recall at the lowest cut-off, IoU 0.5)

{groups('Pose', summary['recall_by_pose'])}

{groups('Occlusion', summary['recall_by_occlusion'])}

{groups('Box size (longer side)', summary['recall_by_size'])}

{groups('Time of day', summary['recall_by_time_of_day'])}

## Other prompted classes

The worlds contain no backpacks or tents, and jackets are not labelled, so
there is no ground truth for these. Boxes drawn: {summary['other_classes']}.

## Review queue

The least confident {REVIEW_FRACTION:.0%} of person boxes (confidence at most {review['cutoff']})
are saved as the view `{VIEW_REVIEW}`: {review['boxes']} boxes in {review['frames']} frames.
{review.get('correct_in_queue', 0)} of them are in fact correct.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    from sardrone.batch import git_commit

    parser = argparse.ArgumentParser(description="Auto-label frames and score the labels.")
    parser.add_argument("--name", default=DATASET_NAME)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, help="only this many frames (for a quick try)")
    parser.add_argument("--score-only", action="store_true", help="skip labelling")
    args = parser.parse_args(argv)
    if not fo.dataset_exists(args.name):
        print(f"dataset {args.name!r} does not exist - run `make frames` first")
        return 1
    dataset = fo.load_dataset(args.name)
    view = dataset.match_tags(DUPLICATE_TAG, bool=False).sort_by("filepath")
    if args.limit:
        view = view.limit(args.limit)

    if not args.score_only:
        print(f"auto-labelling up to {len(view)} frames with {MODEL_NAME} ...", flush=True)
        done = auto_label(view, log=lambda message: print(message, flush=True))
        print(f"labelled {done} frames")
    labelled = view.exists(FIELD)
    if len(labelled) == 0:
        print("nothing is labelled yet")
        return 1
    summary = score(labelled)
    summary["review_queue"] = save_review_queue(dataset, labelled)
    summary["commit"], summary["dirty"] = git_commit()
    write_report(summary, args.report)
    best = max(summary["iou_0.5"], key=lambda row: row["f1"])
    print(f"scored {summary['frames']} frames: mAP {summary['mAP_50_95']:.3f}; best F1 {best['f1']:.3f} at "
          f"confidence {best['threshold']} (precision {best['precision']:.3f}, recall {best['recall']:.3f})")  # fmt: skip
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
