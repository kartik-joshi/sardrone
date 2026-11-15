"""The Gazebo camera, tested without Gazebo.

The byte protocol between gz_image_bridge.py and GazeboCamera is checked with
in-memory streams, and the whole GazeboCamera (subprocess, reader thread,
JPEG encoding) is run against a tiny stand-in bridge script.
"""

import io
import json
import math
import sys
import textwrap

import pytest
from PIL import Image

from sardrone.mcap.episode_writer import TOPIC_TF, EpisodeMeta, EpisodeWriter
from sardrone.mcap.frames import Quat
from sardrone.sim import camera as camera_module
from sardrone.sim.camera import (
    FRAME_HEADER,
    FRAME_MAGIC,
    GZ_L_INT8,
    GZ_RGB_INT8,
    CameraError,
    GazeboCamera,
    RawFrame,
    frame_to_jpeg,
    hfov_from_fx,
    pinhole_intrinsics,
    read_frame,
    read_lens,
)
from tests.conftest import START_NS, Episode

LENS = {"width": 1280, "height": 960, "fx": 539.9363327, "fy": 539.9363708, "cx": 640.0, "cy": 480.0}


def pack_frame(width, height, pixel_format, sim_time_ns, data) -> bytes:
    header = FRAME_HEADER.pack(FRAME_MAGIC, width, height, pixel_format, len(data), sim_time_ns)
    return header + data


# ---- lens -------------------------------------------------------------------


def test_hfov_matches_the_sdf_value():
    # mono_cam/model.sdf says <horizontal_fov>1.74</horizontal_fov> (radians);
    # Gazebo reported fx = 539.936 px for the 1280 px wide image.
    assert hfov_from_fx(1280, LENS["fx"]) == pytest.approx(math.degrees(1.74), abs=1e-3)


@pytest.mark.parametrize("hfov_deg", [60.0, 80.0, 99.7, 120.0])
def test_hfov_from_fx_is_the_inverse_of_pinhole_intrinsics(hfov_deg):
    fx, _, _, _ = pinhole_intrinsics(1280, 960, hfov_deg)
    assert hfov_from_fx(1280, fx) == pytest.approx(hfov_deg, abs=1e-9)


def test_read_lens():
    stream = io.BytesIO(json.dumps(LENS).encode() + b"\nrest")
    assert read_lens(stream) == LENS
    assert stream.read() == b"rest"  # nothing past the line was consumed


def test_read_lens_errors():
    with pytest.raises(CameraError, match="closed before"):
        read_lens(io.BytesIO(b""))
    with pytest.raises(CameraError, match="missing"):
        read_lens(io.BytesIO(b'{"width": 4}\n'))


# ---- frames -----------------------------------------------------------------


def test_read_frame_round_trip():
    pixels = bytes(range(24))  # 4 x 2 RGB
    stream = io.BytesIO(
        pack_frame(4, 2, GZ_RGB_INT8, 132_200_000_000, pixels)
        + pack_frame(4, 2, GZ_RGB_INT8, 132_300_000_000, pixels[::-1])
    )
    first, second = read_frame(stream), read_frame(stream)
    assert first == RawFrame(4, 2, GZ_RGB_INT8, 132_200_000_000, pixels)
    assert second.data == pixels[::-1]
    assert second.sim_time_ns - first.sim_time_ns == 100_000_000  # 0.1 s apart
    assert read_frame(stream) is None  # clean end of stream


def test_read_frame_rejects_broken_streams():
    whole = pack_frame(4, 2, GZ_RGB_INT8, 0, bytes(24))
    with pytest.raises(CameraError, match="frame header"):
        read_frame(io.BytesIO(whole[:10]))
    with pytest.raises(CameraError, match="middle of a frame"):
        read_frame(io.BytesIO(whole[:-1]))
    with pytest.raises(CameraError, match="out of step"):
        read_frame(io.BytesIO(b"XXXX" + whole[4:]))


def test_frame_to_jpeg_keeps_size_and_colours():
    width, height = 64, 48
    # Left half red, right half blue.
    row = bytes([255, 0, 0]) * (width // 2) + bytes([0, 0, 255]) * (width // 2)
    jpeg = frame_to_jpeg(RawFrame(width, height, GZ_RGB_INT8, 0, row * height), quality=95)
    assert jpeg.startswith(b"\xff\xd8")

    image = Image.open(io.BytesIO(jpeg))
    assert image.size == (width, height)
    red, blue = image.getpixel((8, 24)), image.getpixel((56, 24))
    assert red[0] > 200 and red[2] < 60, "left side must stay red (channel order RGB)"
    assert blue[2] > 200 and blue[0] < 60


def test_frame_to_jpeg_greyscale():
    jpeg = frame_to_jpeg(RawFrame(8, 8, GZ_L_INT8, 0, bytes([128]) * 64))
    image = Image.open(io.BytesIO(jpeg))
    assert image.mode == "L" and image.size == (8, 8)


def test_frame_to_jpeg_rejects_bad_input():
    with pytest.raises(CameraError, match="unsupported"):
        frame_to_jpeg(RawFrame(4, 2, 99, 0, bytes(24)))
    with pytest.raises(CameraError, match="expected 24"):
        frame_to_jpeg(RawFrame(4, 2, GZ_RGB_INT8, 0, bytes(23)))


# ---- GazeboCamera against a stand-in bridge ---------------------------------

FAKE_BRIDGE = """
    import json, struct, sys, time
    out = sys.stdout.buffer
    out.write(json.dumps({lens}).encode() + b"\\n")
    header = struct.Struct("<4sIIIIq")
    for index in range({frames}):
        data = bytes([index * 40 % 256, 0, 0]) * (16 * 12)
        out.write(header.pack(b"SARF", 16, 12, 3, len(data), (index + 1) * 100_000_000) + data)
        out.flush()
        time.sleep(0.02)
    {ending}
"""


def fake_camera(tmp_path, monkeypatch, *, frames=5, ending="time.sleep(30)", **kwargs):
    lens = {"width": 16, "height": 12, "fx": 8.0, "fy": 8.0, "cx": 8.0, "cy": 6.0}
    script = tmp_path / "fake_bridge.py"
    script.write_text(
        textwrap.dedent(FAKE_BRIDGE.format(lens=lens, frames=frames, ending=ending))
    )
    monkeypatch.setattr(camera_module, "BRIDGE_SCRIPT", script)
    return GazeboCamera(gz_python=sys.executable, start_timeout_s=10.0, **kwargs)


def test_gazebo_camera_reads_lens_and_newest_frame(tmp_path, monkeypatch):
    with fake_camera(tmp_path, monkeypatch) as camera:
        assert (camera.width, camera.height) == (16, 12)
        # fx = 8 px on a 16 px wide image: tan(hfov/2) = 8/8, so hfov = 90 deg.
        assert camera.hfov_deg == pytest.approx(90.0)

        deadline = 200
        while camera.frames_received < 5 and deadline:
            camera_module.time.sleep(0.02)
            deadline -= 1
        jpeg = camera.read()
        assert Image.open(io.BytesIO(jpeg)).size == (16, 12)
        assert camera.sim_time_ns == 500_000_000  # the 5th and newest frame
        assert camera.read() is jpeg  # same frame is encoded only once


def test_gazebo_camera_topics_follow_model_and_world():
    camera = GazeboCamera(model="x500_mono_cam_0", world="default")
    assert camera.topic == (
        "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
    )
    assert camera.info_topic.endswith("/sensor/camera/camera_info")


def test_gazebo_camera_reports_a_dead_bridge(tmp_path, monkeypatch):
    with fake_camera(tmp_path, monkeypatch, frames=1, ending="") as camera:
        deadline = 200
        while deadline:
            try:
                camera.read()
            except CameraError as error:
                assert "stopped" in str(error)
                return
            camera_module.time.sleep(0.02)
            deadline -= 1
    pytest.fail("a bridge that exited was not reported")


def test_gazebo_camera_refuses_stale_frames(tmp_path, monkeypatch):
    with fake_camera(tmp_path, monkeypatch, frames=1, stale_after_s=0.2) as camera:
        camera.read()
        camera_module.time.sleep(0.4)
        with pytest.raises(CameraError, match="no new camera frame"):
            camera.read()


def test_gazebo_camera_times_out_without_images(tmp_path, monkeypatch):
    camera = fake_camera(tmp_path, monkeypatch, frames=0)
    camera.start_timeout_s = 0.5
    with pytest.raises(CameraError, match="WITH a camera"):
        camera.open()


def test_missing_gazebo_python_is_a_clear_error(tmp_path):
    with pytest.raises(CameraError, match="SARDRONE_GZ_PYTHON"):
        GazeboCamera(gz_python=tmp_path / "nope").open()


# ---- mount position reaches the file ----------------------------------------


def test_camera_mount_position_is_written_to_tf(tmp_path):
    path = tmp_path / "ep.mcap"
    meta = EpisodeMeta(world="w", weather="clear", time_of_day="noon", mission_prompt="p")
    with EpisodeWriter(path, meta, camera_mount_m=GazeboCamera.mount_m) as episode:
        episode.log_pose_ned(START_NS, 0.0, 0.0, -10.0, Quat(1.0, 0.0, 0.0, 0.0))

    mounts = [t for t in Episode(path).decoded(TOPIC_TF) if t.child_frame_id == "camera_front"]
    assert len(mounts) == 1
    t = mounts[0].translation
    # 12 cm forward, 3 cm left, 24.2 cm up (x500_mono_cam/model.sdf).
    assert (t.x, t.y, t.z) == pytest.approx((0.12, 0.03, 0.242))
