"""Object memory: keep following a person through frames where the detector misses.

A detector looks at each frame on its own, and it flickers: a person found
in frame 10 may be missed in frames 11 and 12 and found again in 13. Acting
on single frames would make the drone lose and re-find the same person. A
TRACKER fixes that by giving each person an identity that lasts over time.

Each track keeps a small MOTION MODEL (a Kalman filter): where the box is
and how fast it is moving in the image. Every frame, in two steps:

1. PREDICT: move every track forward by its own velocity. This is the
   tracker's guess of where each person is NOW, before looking.
2. UPDATE: match the detector's boxes to those guesses (by overlap), and
   correct each matched track towards its detection.

A track with no matching detection is NOT dropped at once. It "coasts" on
its prediction for up to `max_age` frames, which is the memory: when the
detector finds the person again near the predicted spot, the same identity
continues. (The idea of keeping objects alive through brief misses with a
motion model is the one behind Waabi's MAD paper; this is the classic,
much simpler "SORT" form of it.)

Boxes are (x0, y0, x1, y1) in pixels.
"""

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]


def box_iou(a: Box, b: Box) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    return shared / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared)


def centre(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class BoxFilter:
    """A Kalman filter for one box: centre x, centre y, width, height, and
    the velocity of the centre. "Constant velocity" model: between frames
    the centre is assumed to keep moving as it did; size is assumed steady.

    A Kalman filter keeps, besides its best guess, a measure of how UNSURE
    it is (the covariance P). Predicting makes it less sure; a measurement
    makes it more sure. How far a measurement pulls the guess depends on
    the two uncertainties: a noisy detector is trusted less than a
    confident prediction, and the other way round.
    """

    def __init__(self, box: Box, position_noise: float = 1.0, velocity_noise: float = 0.5, measurement_noise: float = 2.0):
        cx, cy = centre(box)
        # State: [cx, cy, w, h, vx, vy]
        self.x = np.array([cx, cy, box[2] - box[0], box[3] - box[1], 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])  # velocity starts unknown
        self.F = np.eye(6)
        self.F[0, 4] = self.F[1, 5] = 1.0  # centre += velocity * one frame
        self.H = np.eye(4, 6)  # we measure cx, cy, w, h (not velocity)
        self.Q = np.diag([position_noise] * 4 + [velocity_noise] * 2) ** 2
        self.R = np.eye(4) * measurement_noise**2

    def predict(self) -> Box:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.box

    def update(self, box: Box) -> None:
        cx, cy = centre(box)
        z = np.array([cx, cy, box[2] - box[0], box[3] - box[1]])
        residual = z - self.H @ self.x  # how far the detection is from the guess
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # the "gain": how much to trust the detection
        self.x = self.x + K @ residual
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def box(self) -> Box:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4]), float(self.x[5])


@dataclass
class Track:
    id: int
    filter: BoxFilter
    hits: int = 1  # frames with a matching detection
    age: int = 1  # frames since the track was created
    misses: int = 0  # frames in a row WITHOUT a detection (coasting)
    predicted: Box | None = None  # where the motion model expected it this frame, before looking
    detection: int | None = None  # index of the detection matched this frame (None while coasting)
    history: list = field(default_factory=list)  # (frame index, box, "detected" | "coasted")

    @property
    def box(self) -> Box:
        return self.filter.box

    @property
    def coasting(self) -> bool:
        return self.misses > 0


def associate(tracks: list[Box], detections: list[Box], min_iou: float) -> list[tuple[int, int]]:
    """Pair predicted track boxes with detections: [(track index, detection index)].

    Uses the Hungarian method, which finds the pairing with the largest
    TOTAL overlap. A greedy "best match first" can pair two people the
    wrong way round when they stand close together.
    """
    if not tracks or not detections:
        return []
    from scipy.optimize import linear_sum_assignment

    overlap = np.array([[box_iou(t, d) for d in detections] for t in tracks])
    rows, columns = linear_sum_assignment(-overlap)
    return [(int(r), int(c)) for r, c in zip(rows, columns) if overlap[r, c] >= min_iou]


class Tracker:
    """SORT-style multi-object tracker with a coasting memory.

        tracker = Tracker()
        for boxes in detections_per_frame:
            for track in tracker.step(boxes):
                ...  # track.id, track.box, track.coasting
    """

    def __init__(self, max_age: int = 10, min_hits: int = 3, min_iou: float = 0.1):
        # max_age: how many frames a track may coast before it is forgotten.
        # min_hits: detections needed before a track is reported at all,
        #   which keeps one-frame false alarms from becoming "people".
        # min_iou: small on purpose. People are about 30 px; a few pixels of
        #   motion or box jitter already costs a lot of overlap.
        self.max_age, self.min_hits, self.min_iou = max_age, min_hits, min_iou
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1
        self.lost: list[Track] = []  # tracks that were dropped, for analysis

    def step(self, detections: list[Box]) -> list[Track]:
        """Advance one frame. Returns the tracks that are currently confirmed
        (seen often enough), including those that are coasting."""
        predicted = [track.filter.predict() for track in self.tracks]
        for track, box in zip(self.tracks, predicted):
            track.predicted, track.detection = box, None
        pairs = associate(predicted, detections, self.min_iou)
        matched_tracks = {t for t, _ in pairs}
        matched_detections = {d for _, d in pairs}

        for track_index, detection_index in pairs:
            track = self.tracks[track_index]
            track.filter.update(detections[detection_index])
            track.hits += 1
            track.misses = 0
            track.detection = detection_index
        for index, track in enumerate(self.tracks):
            track.age += 1
            if index not in matched_tracks:
                track.misses += 1
        for index, box in enumerate(detections):
            if index not in matched_detections:
                self.tracks.append(Track(self._next_id, BoxFilter(box), detection=index))
                self._next_id += 1

        kept = []
        for track in self.tracks:
            if track.misses > self.max_age:
                self.lost.append(track)
                continue
            track.history.append((self.frame, track.box, "coasted" if track.coasting else "detected"))
            kept.append(track)
        self.tracks = kept
        self.frame += 1
        return [track for track in self.tracks if track.hits >= self.min_hits]
