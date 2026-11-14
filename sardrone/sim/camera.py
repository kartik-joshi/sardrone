"""Camera sources.

The recorder only knows the `CameraSource` interface: "give me the next frame
as JPEG bytes, and tell me your image size and field of view". That lets us
record with generated images today and swap in the real Gazebo camera later
without touching the recorder.
"""

import io
import math
from typing import Protocol

from PIL import Image, ImageDraw


class CameraSource(Protocol):
    width: int
    height: int
    hfov_deg: float  # horizontal field of view

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


class GazeboCamera:
    """TODO(camera): read frames from the Gazebo simulator.

    Plan:
      1. Launch a PX4 gz model that actually has a camera. The plain
         `gz_x500` does NOT. According to the PX4 Gazebo docs the camera
         variants are `gz_x500_mono_cam` (forward RGB), `gz_x500_mono_cam_down`
         (downward RGB), `gz_x500_depth` (depth) and `gz_x500_gimbal`.
         Confirm against YOUR checkout:
             ls PX4-Autopilot/Tools/simulation/gz/models | grep -i -E "cam|depth|gimbal"
      2. Find the image topic while the sim runs:   gz topic -l | grep image
      3. Subscribe with the gz-transport Python bindings (gz.transport /
         gz.msgs Image), convert each frame to JPEG with Pillow, and return
         it from read().
      4. Read hfov/width/height from the model's SDF <camera> element so the
         CameraCalibration matches the simulated lens.
    """

    def __init__(self, topic: str, width: int = 1280, height: int = 960, hfov_deg: float = 100.0):
        self.topic = topic
        self.width = width
        self.height = height
        self.hfov_deg = hfov_deg

    def read(self) -> bytes:
        raise NotImplementedError("GazeboCamera is not implemented yet - see the class docstring")
