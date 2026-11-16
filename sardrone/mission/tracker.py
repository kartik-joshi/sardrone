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


def gated_distance(a: Box, b: Box) -> float:
    """Distance between two box centres, in units of the boxes' own size.

    1.0 means "one body-length apart". Measuring in body-lengths makes one
    threshold work for a person 15 px tall and one 80 px tall.
    """
    (ax, ay), (bx, by) = centre(a), centre(b)
    size = (max(a[2] - a[0], a[3] - a[1]) + max(b[2] - b[0], b[3] - b[1])) / 2
    return float(np.hypot(ax - bx, ay - by) / max(size, 1.0))


def associate(tracks: list[Box], detections: list[Box], gate: float = 1.5) -> list[tuple[int, int]]:
    """Pair predicted track boxes with detections: [(track index, detection index)].

    Boxes are compared by how far apart their CENTRES are, not by overlap.
    People here are about 30 px tall while the picture shifts by tens of
    pixels per frame when the drone turns, so a box often does not overlap
    its own previous position at all. (The first closed-loop flight, which
    matched by overlap, broke 3 people into 70 tracks.) A pair is allowed
    when the centres are within `gate` body-lengths.

    The Hungarian method then picks the pairing with the smallest TOTAL
    distance. A greedy "nearest first" can pair two people the wrong way
    round when they are close together.
    """
    if not tracks or not detections:
        return []
    from scipy.optimize import linear_sum_assignment

    cost = np.array([[gated_distance(t, d) for d in detections] for t in tracks])
    blocked = cost > gate
    cost = np.where(blocked, 1e6, cost)  # forbidden pairs: possible only if nothing else is left
    rows, columns = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, columns) if not blocked[r, c]]


class Tracker:
    """SORT-style multi-object tracker with a coasting memory.

        tracker = Tracker()
        for boxes in detections_per_frame:
            for track in tracker.step(boxes):
                ...  # track.id, track.box, track.coasting
    """

    def __init__(self, max_age: int = 10, min_hits: int = 3, gate: float = 1.5):
        # max_age: how many frames a track may coast before it is forgotten.
        # min_hits: detections needed before a track is reported at all,
        #   which keeps one-frame false alarms from becoming "people".
        # gate: how far (in body-lengths) a detection may be from where the
        #   motion model expected the person and still be the same person.
        self.max_age, self.min_hits, self.gate = max_age, min_hits, gate
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
        pairs = associate(predicted, detections, self.gate)
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
