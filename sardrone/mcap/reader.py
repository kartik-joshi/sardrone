"""Read labelled camera frames back out of an episode file.

The writer uses the Foxglove SDK; this reader uses the independent `mcap`
library, the same one the tests use to check the files.

A "labelled frame" is one camera picture together with everything recorded
at that instant: the label boxes, the drone's pose, and which state the
mission was in.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

from sardrone.mcap.episode_writer import (
    TOPIC_CALIBRATION,
    TOPIC_EVENTS,
    TOPIC_GROUND_TRUTH_BOXES,
    TOPIC_IMAGE,
    TOPIC_TF,
)
from sardrone.mcap.frames import Quat, Vec3
from sardrone.mcap.timeutil import NS_PER_SEC


@dataclass(frozen=True)
class FrameBox:
    id: str  # e.g. "person_2"
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class LabelledFrame:
    time_ns: int
    time_s: float  # seconds since the episode started
    jpeg: bytes
    boxes: tuple[FrameBox, ...]
    position: Vec3 | None  # drone body in `map` (ENU), as PX4 estimated it
    orientation: Quat | None
    mission_state: str  # "SEARCH", "CONFIRM", ...; "" if the episode has no states

    @property
    def height_m(self) -> float:
        return self.position.z if self.position else 0.0


def has_labels(path: Path | str) -> bool:
    """True if the episode carries per-frame label boxes."""
    with open(path, "rb") as handle:
        summary = make_reader(handle).get_summary()
        return any(c.topic == TOPIC_GROUND_TRUTH_BOXES for c in summary.channels.values())


def read_calibration(path: Path | str) -> dict:
    """Image size and pinhole intrinsics: width, height, fx, fy, cx, cy."""
    with open(path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, _, _, message in reader.iter_decoded_messages(topics=[TOPIC_CALIBRATION]):
            k = message.K
            return {"width": message.width, "height": message.height,
                    "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5]}  # fmt: skip
    raise ValueError(f"{path} has no camera calibration")


def _boxes(annotations) -> tuple[FrameBox, ...]:
    boxes = []
    for outline, text in zip(annotations.points, annotations.texts):
        xs = [point.x for point in outline.points]
        ys = [point.y for point in outline.points]
        boxes.append(FrameBox(text.text, min(xs), min(ys), max(xs), max(ys)))
    return tuple(boxes)


def read_labelled_frames(path: Path | str, rate_hz: float | None = 2.0) -> Iterator[LabelledFrame]:
    """Yield frames that have labels, thinned to about `rate_hz` per second.

    The camera is recorded at 10 Hz. Neighbouring frames are almost the same
    picture, which adds little for training and makes near-duplicates. So
    by default only the first frame in each half second is kept.
    `rate_hz=None` yields every labelled frame.

    A picture and its labels are matched by IDENTICAL timestamps; a picture
    without a label message is skipped rather than treated as "nobody here".
    """
    period_ns = None if rate_hz is None else int(NS_PER_SEC / rate_hz)
    topics = [TOPIC_IMAGE, TOPIC_GROUND_TRUTH_BOXES, TOPIC_TF, TOPIC_EVENTS]
    start_ns = next_due = None
    position = orientation = None
    state = ""
    pending: dict[int, dict] = {}  # log time -> {"jpeg": ..., "boxes": ...}

    with open(path, "rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _, channel, raw, message in reader.iter_decoded_messages(topics=topics, log_time_order=True):
            time_ns = raw.log_time
            if start_ns is None:
                start_ns = next_due = time_ns
            if channel.topic == TOPIC_TF:
                if message.child_frame_id == "base_link":
                    t, r = message.translation, message.rotation
                    position, orientation = Vec3(t.x, t.y, t.z), Quat(r.w, r.x, r.y, r.z)
                continue
            if channel.topic == TOPIC_EVENTS:
                if message.message.startswith("state: "):
                    # "state: SEARCH -> CONFIRM (person_1 sighted)" -> "CONFIRM"
                    state = message.message.split("-> ", 1)[1].split(" ", 1)[0]
                continue

            entry = pending.setdefault(time_ns, {})
            if channel.topic == TOPIC_IMAGE:
                entry["jpeg"] = bytes(message.data)
            else:
                entry["boxes"] = _boxes(message)
            for old in [t for t in pending if t < time_ns]:
                del pending[old]  # an older picture never got labels: drop it
            if "jpeg" not in entry or "boxes" not in entry:
                continue
            del pending[time_ns]
            if period_ns is not None:
                if time_ns < next_due:
                    continue
                while next_due <= time_ns:
                    next_due += period_ns
            yield LabelledFrame(
                time_ns=time_ns,
                time_s=(time_ns - start_ns) / NS_PER_SEC,
                jpeg=entry["jpeg"],
                boxes=entry["boxes"],
                position=position,
                orientation=orientation,
                mission_state=state,
            )
