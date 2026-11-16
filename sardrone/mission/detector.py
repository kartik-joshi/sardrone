"""The trained detector behind the same interface as the oracle.

    detector = YoloDetector("runs/train/v1/weights/best.pt", camera_model, confidence=0.27)
    sightings = detector.detect(pose, jpeg)

Unlike the oracle it looks at the PICTURE. It does not know who a person
is (`person_id` is None), cannot see through trees, and makes both kinds
of mistake: boxes where nobody is, and people without a box. The mission
has to cope with all of that, which is what the tracker and the
confirmation orbit are for.
"""

import io
from pathlib import Path

from sardrone.mission.oracle import Sighting
from sardrone.mission.projection import BBox, CameraModel, DronePose, pixel_to_ground


class YoloDetector:
    def __init__(
        self,
        weights: Path | str,
        model: CameraModel,
        *,
        confidence: float,
        imgsz: int = 1280,
        device: str = "mps",
        ground_z: float = 0.0,
        name: str | None = None,
    ):
        from ultralytics import YOLO

        self.network = YOLO(str(weights))
        self.model = model
        self.confidence = confidence  # the cut-off chosen on validation data
        self.imgsz = imgsz
        self.device = device
        self.ground_z = ground_z
        self.name = name or f"yolo:{Path(weights).parent.parent.name}"

    def boxes(self, jpeg: bytes) -> list[tuple[float, BBox]]:
        """[(confidence, box in pixels)] for every person found in the picture."""
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        result = self.network.predict(
            image, imgsz=self.imgsz, device=self.device, conf=self.confidence, verbose=False
        )[0]
        found = []
        for (x0, y0, x1, y1), confidence, cls in zip(
            result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()
        ):
            if result.names[int(cls)] == "person":
                found.append((float(confidence), BBox(x0, y0, x1, y1)))
        return found

    def detect(self, pose: DronePose, jpeg: bytes | None = None) -> list[Sighting]:
        if jpeg is None:
            return []
        sightings = []
        for confidence, bbox in self.boxes(jpeg):
            # Bottom-centre of the box onto flat ground, as for the oracle.
            ground = pixel_to_ground(self.model, pose, (bbox.x0 + bbox.x1) / 2, bbox.y1, self.ground_z)
            sightings.append(Sighting(None, bbox, ground, confidence))
        return sightings
