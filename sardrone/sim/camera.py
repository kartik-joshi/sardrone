"""Camera sources.

The recorder only knows the `CameraSource` interface: "give me the next frame
as JPEG bytes, and tell me your image size and field of view". That lets us
record with generated images (SyntheticCamera) or with the simulator's camera
(GazeboCamera) without touching the recorder.
"""

import io
import json
import math
import os
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from PIL import Image, ImageDraw


class CameraSource(Protocol):
    width: int
    height: int
    hfov_deg: float  # horizontal field of view
    mount_m: tuple[float, float, float]  # position on the drone: forward, left, up

    def read(self) -> bytes:
        """Return the next frame as JPEG bytes."""
        ...


def pinhole_intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) for an ideal pinhole camera with square pixels.

    The focal length in pixels follows from the field of view:
        tan(hfov / 2) = (width / 2) / fx
    """
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, width / 2.0, height / 2.0


class SyntheticCamera:
    """Generates test images: a ground grid that scrolls as the drone moves,
    a frame counter, and an orange "person" marker when `person_visible`."""

    def __init__(self, width: int = 640, height: int = 480, hfov_deg: float = 80.0):
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg
        self.mount_m = (0.0, 0.0, 0.0)
        self.frame_index = 0
        self.person_visible = False
        self.caption = ""
        # Set these from the drone's position to make the grid scroll.
        self.east_m = 0.0
        self.north_m = 0.0

    def read(self) -> bytes:
        image = Image.new("RGB", (self.width, self.height), (70, 110, 60))
        draw = ImageDraw.Draw(image)

        # Scrolling grid: one line every 40 px, shifted 8 px per metre moved.
        spacing = 40
        shift_x = int(self.east_m * 8) % spacing
        shift_y = int(self.north_m * 8) % spacing
        for x in range(-shift_x, self.width, spacing):
            draw.line([(x, 0), (x, self.height)], fill=(90, 135, 80))
        for y in range(shift_y - spacing, self.height, spacing):
            draw.line([(0, y), (self.width, y)], fill=(90, 135, 80))

        if self.person_visible:
            cx, cy = self.width // 2, self.height // 2
            draw.ellipse([cx - 8, cy - 34, cx + 8, cy - 18], fill=(255, 140, 0))  # head
            draw.rectangle([cx - 10, cy - 16, cx + 10, cy + 24], fill=(255, 140, 0))  # body

        draw.rectangle([0, 0, self.width, 22], fill=(0, 0, 0))
        draw.text((6, 5), f"frame {self.frame_index:05d}  {self.caption}", fill=(255, 255, 255))

        self.frame_index += 1
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()


# ---- Gazebo ------------------------------------------------------------------
#
# Gazebo's Python bindings live in Gazebo's own Python, not in this venv. So a
# small helper program (gz_image_bridge.py) runs there and pipes frames to us.
# Everything below up to GazeboCamera is plain byte parsing and is unit-tested
# without a simulator.

FRAME_MAGIC = b"SARF"
FRAME_HEADER = struct.Struct("<4sIIIIq")  # magic, width, height, format, length, sim ns

# gz.msgs.PixelFormatType values we accept (checked in the installed gz.msgs10).
GZ_L_INT8 = 1
GZ_RGB_INT8 = 3
_PIL_MODES = {GZ_L_INT8: ("L", 1), GZ_RGB_INT8: ("RGB", 3)}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GZ_PYTHON = (
    PROJECT_ROOT.parent / "PX4-Autopilot/Tools/setup/macos/.pixi/envs/default/bin/python"
)
BRIDGE_SCRIPT = Path(__file__).with_name("gz_image_bridge.py")


class CameraError(RuntimeError):
    pass


@dataclass
class RawFrame:
    width: int
    height: int
    pixel_format: int
    sim_time_ns: int
    data: bytes


def hfov_from_fx(width: int, fx: float) -> float:
    """Inverse of pinhole_intrinsics: field of view in degrees from fx in pixels."""
    return math.degrees(2.0 * math.atan((width / 2.0) / fx))


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    chunks = []
    while count > 0:
        chunk = stream.read(count)
        if not chunk:
            break
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def read_lens(stream: BinaryIO) -> dict:
    """First line from the bridge: {"width", "height", "fx", "fy", "cx", "cy"}."""
    line = stream.readline()
    if not line:
        raise CameraError("the Gazebo image bridge closed before sending the lens")
    lens = json.loads(line)
    missing = {"width", "height", "fx", "fy", "cx", "cy"} - set(lens)
    if missing:
        raise CameraError(f"lens line is missing {sorted(missing)}")
    return lens


def read_frame(stream: BinaryIO) -> RawFrame | None:
    """Next frame from the bridge, or None at a clean end of stream."""
    header = _read_exact(stream, FRAME_HEADER.size)
    if not header:
        return None
    if len(header) < FRAME_HEADER.size:
        raise CameraError("stream ended in the middle of a frame header")
    magic, width, height, pixel_format, length, sim_time_ns = FRAME_HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise CameraError(f"bad frame marker {magic!r}: the stream is out of step")
    data = _read_exact(stream, length)
    if len(data) < length:
        raise CameraError("stream ended in the middle of a frame")
    return RawFrame(width, height, pixel_format, sim_time_ns, data)


def frame_to_jpeg(frame: RawFrame, quality: int = 85) -> bytes:
    if frame.pixel_format not in _PIL_MODES:
        raise CameraError(f"unsupported Gazebo pixel format {frame.pixel_format}")
    mode, channels = _PIL_MODES[frame.pixel_format]
    expected = frame.width * frame.height * channels
    if len(frame.data) != expected:
        raise CameraError(
            f"frame has {len(frame.data)} bytes, expected {expected} "
            f"for {frame.width}x{frame.height} {mode}"
        )
    image = Image.frombytes(mode, (frame.width, frame.height), frame.data)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class GazeboCamera:
    """The simulated camera on the PX4 `gz_x500_mono_cam` model.

        with GazeboCamera() as camera:      # starts the bridge, reads the lens
            jpeg = camera.read()            # newest frame, as JPEG bytes

    read() returns the NEWEST frame Gazebo has rendered, so an image can be
    up to one render interval (1/30 s) older than the pose it is logged with.
    `sim_time_ns` holds the simulator time of the last frame returned.
    """

    # Where PX4's x500_mono_cam model puts the camera on the drone
    # (Tools/simulation/gz/models/x500_mono_cam/model.sdf, <pose>.12 .03 .242).
    mount_m = (0.12, 0.03, 0.242)

    def __init__(
        self,
        model: str = "x500_mono_cam_0",
        world: str = "default",
        *,
        gz_python: Path | str | None = None,
        rate_hz: float = 10.0,
        jpeg_quality: int = 85,
        start_timeout_s: float = 15.0,
        stale_after_s: float = 3.0,
    ):
        base = f"/world/{world}/model/{model}/link/camera_link/sensor/camera"
        self.topic = f"{base}/image"
        self.info_topic = f"{base}/camera_info"
        self.gz_python = Path(
            gz_python or os.environ.get("SARDRONE_GZ_PYTHON") or DEFAULT_GZ_PYTHON
        )
        self.rate_hz = rate_hz
        self.jpeg_quality = jpeg_quality
        self.start_timeout_s = start_timeout_s
        self.stale_after_s = stale_after_s

        self.width = 0
        self.height = 0
        self.hfov_deg = 0.0
        self.sim_time_ns = 0
        self.frames_received = 0

        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._latest: RawFrame | None = None
        self._latest_at = 0.0
        self._jpeg: bytes | None = None  # cache: encode each frame once
        self._error: Exception | None = None

    # ---- lifecycle ----------------------------------------------------------

    def open(self) -> "GazeboCamera":
        if not self.gz_python.exists():
            raise CameraError(
                f"Gazebo's Python was not found at {self.gz_python}. "
                "Set SARDRONE_GZ_PYTHON to the python inside the Gazebo environment."
            )
        # PX4 runs Gazebo on the loopback interface; a subscriber must match.
        env = dict(os.environ)
        env.setdefault("GZ_IP", "127.0.0.1")
        self._process = subprocess.Popen(
            [
                str(self.gz_python),
                str(BRIDGE_SCRIPT),
                "--topic", self.topic,
                "--info-topic", self.info_topic,
                "--rate", str(self.rate_hz),
            ],
            stdout=subprocess.PIPE,
            env=env,
        )  # fmt: skip
        threading.Thread(target=self._pump, daemon=True).start()

        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self
                if self._error:
                    raise CameraError(f"Gazebo camera failed to start: {self._error}")
            time.sleep(0.05)
        self.close()
        raise CameraError(
            f"no image on {self.topic} after {self.start_timeout_s:g} s. Is the simulator "
            "running a model WITH a camera (`make sim`)? Plain gz_x500 has none; "
            "use --camera synthetic for that."
        )

    def _pump(self) -> None:
        """Background thread: keep only the newest frame from the bridge."""
        stream = self._process.stdout
        try:
            lens = read_lens(stream)
            self.width, self.height = int(lens["width"]), int(lens["height"])
            self.hfov_deg = hfov_from_fx(self.width, float(lens["fx"]))
            while (frame := read_frame(stream)) is not None:
                with self._lock:
                    self._latest, self._jpeg = frame, None
                    self._latest_at = time.monotonic()
                    self.frames_received += 1
            raise CameraError("the Gazebo image bridge stopped")
        except Exception as error:  # noqa: BLE001 - handed to the reading thread
            with self._lock:
                self._error = error

    def close(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "GazeboCamera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- CameraSource -------------------------------------------------------

    def read(self) -> bytes:
        with self._lock:
            frame, jpeg, error, age = (
                self._latest,
                self._jpeg,
                self._error,
                time.monotonic() - self._latest_at,
            )
        if frame is None:
            raise CameraError("GazeboCamera.read() called before open()")
        if error is not None:
            raise CameraError(f"Gazebo camera stopped: {error}")
        if age > self.stale_after_s:
            # Never keep logging the same old picture as if it were live.
            raise CameraError(f"no new camera frame for {age:.1f} s")
        if jpeg is None:
            jpeg = frame_to_jpeg(frame, self.jpeg_quality)
            with self._lock:
                if self._latest is frame:
                    self._jpeg = jpeg
        self.sim_time_ns = frame.sim_time_ns
        return jpeg
