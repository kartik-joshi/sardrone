"""Run a trained detector over a FiftyOne view and store its boxes.

Predictions are stored with a very low confidence cut-off on purpose.
Average precision needs the WHOLE ranking of boxes, including the unsure
ones; cutting them off early would lower the score for no reason. A cut-off
for actual use is chosen later, on validation data.
"""

from pathlib import Path

MIN_CONFIDENCE = 0.001
MAX_BOXES = 300


def field_for(version: str) -> str:
    return f"pred_{version}"


def boxes_to_detections(xyxyn, confidences, classes, names: dict):
    """Ultralytics boxes (corners as fractions of the image) -> fo.Detections."""
    import fiftyone as fo

    detections = []
    for (x0, y0, x1, y1), confidence, cls in zip(xyxyn, confidences, classes):
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in (x0, y0, x1, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            fo.Detection(
                label=names[int(cls)], confidence=float(confidence), bounding_box=[x0, y0, x1 - x0, y1 - y0]
            )
        )
    return fo.Detections(detections=detections)


def predict_view(view, weights: Path, version: str, imgsz: int = 1280, device: str = "mps",
                 batch: int = 16, log=print) -> str:  # fmt: skip
    """Write the model's boxes into field `pred_<version>` of every sample."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    field = field_for(version)
    paths = view.values("filepath")
    ids = view.values("id")
    values = {}
    for start in range(0, len(paths), batch):
        results = model.predict(
            paths[start : start + batch], imgsz=imgsz, device=device, conf=MIN_CONFIDENCE,
            max_det=MAX_BOXES, verbose=False,
        )  # fmt: skip
        for sample_id, result in zip(ids[start : start + batch], results):
            boxes = result.boxes
            values[sample_id] = boxes_to_detections(
                boxes.xyxyn.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), result.names
            )
        if (start // batch) % 20 == 0:
            log(f"  predicted {min(start + batch, len(paths))}/{len(paths)}")
    view._dataset.set_values(field, values, key_field="id")
    return field
